"""Explicit packet API and bounded, event-owned two-GPU column phase."""
import torch
import torch.nn.functional as F
import comfy_kitchen.backends.cuda as ck


def linear_packet(q, scales, z, weight, lora, epilogue):
    assert q.ndim == 2 and q.dtype == torch.int8 and scales.shape == (len(q), 1)
    y = ck._int8_linear_turing_quantized(q, weight._qdata, scales, weight._params.scale, None, torch.float16)
    assert y is not None, 'Unqualified GEMM fallback'
    delta = F.linear(z, lora[0])
    return epilogue(y, delta, lora[2])


class Pipeline:
    def __init__(self, shared, slots=2):
        assert slots in (2, 3)
        self.shared, self.slots = shared, slots
        self.producer = torch.cuda.Stream(device=0)
        self.source_copy = torch.cuda.Stream(device=0)
        self.copy = torch.cuda.Stream(device=1)
        self.streams = [torch.cuda.Stream(device=r) for r in (0, 1)]
        self.last_events = []
        self.trace = False
        self.wait_events = []

    def phase(self, sh, h, phase, early=False):
        weights, loras = sh[phase], sh['l'+phase]
        assert (loras[0][1].data_ptr(), loras[1][1].data_ptr()) in self.shared.validated
        ready = torch.cuda.Event()
        ready.record(torch.cuda.current_stream(0))
        self.producer.wait_event(ready)
        for rank in (0, 1):
            self.streams[rank].wait_stream(torch.cuda.current_stream(rank))
        producer = self.streams[0] if early else self.producer
        h.record_stream(producer)
        h.record_stream(self.source_copy)
        producer.wait_event(ready)
        copy = self.copy
        slots = [dict(source=None, destination=None, done0=None, done1=None, copied=None) for _ in range(self.slots)]
        output = [[], []]
        events = []
        waits = []
        for index, start in enumerate(range(0, len(h), 8192)):
            x = h[start:start+8192]
            m = len(x)
            slot = slots[index % self.slots]
            if slot['done0'] is not None:
                producer.wait_event(slot['done0'])
                producer.wait_event(slot['copied'])
                copy.wait_event(slot['done1'])
            with torch.cuda.device(0), torch.cuda.stream(producer):
                # All former side-stream reads precede the fences above; source
                # allocations belong to this producer stream and stay in the slot.
                slot['source'] = None
                if m >= 3:
                    q, s = self.shared.call(self.shared.quant, x, weights[0])
                    z = F.linear(x, loras[0][1])
                    slot['source'] = (q, s, z)
                else:
                    slot['source'] = (x,)
                produced = torch.cuda.Event()
                produced.record(producer)
            copy.wait_event(produced)
            self.source_copy.wait_event(produced)
            with torch.cuda.stream(self.source_copy), torch.cuda.device(1), torch.cuda.stream(copy):
                source = slot['source']
                if slot['destination'] is None or len(slot['destination']) != len(source):
                    slot['destination'] = tuple(torch.empty((8192, *t.shape[1:]), dtype=t.dtype, device=1) for t in source)
                local = tuple(t[:m] for t in slot['destination'])
                for dst, src in zip(local, source):
                    dst.copy_(src, non_blocking=True)
                copied = torch.cuda.Event()
                copied.record(copy)
                slot['copied'] = copied
            for rank, packet, event in ((0, slot['source'], produced), (1, local, copied)):
                stream = self.streams[rank]
                if self.trace:
                    wa=torch.cuda.Event(enable_timing=True);wb=torch.cuda.Event(enable_timing=True)
                    wa.record(stream)
                stream.wait_event(event)
                if self.trace:
                    wb.record(stream);waits.append((rank,wa,wb))
                with torch.cuda.device(rank), torch.cuda.stream(stream):
                    if m >= 3:
                        y = linear_packet(*packet, weights[rank], loras[rank], self.shared.epilogue)
                    else:
                        y = self.shared.original(packet[0], weights[rank], loras[rank])
                    output[rank].append(y)
                    done = torch.cuda.Event()
                    done.record(stream)
                    slot['done'+str(rank)] = done
            events.append((produced, copied, slot['done0'], slot['done1']))
        # Preserve all source/destination ownership through the enqueued work.
        # A bounded number of packet slots live here; output tensors are separate.
        for rank in (0, 1):
            torch.cuda.current_stream(rank).wait_stream(self.streams[rank])
        self.producer.wait_stream(self.streams[0])
        self.producer.wait_stream(self.copy)
        self.producer.wait_stream(self.source_copy)
        self.copy.wait_stream(self.streams[1])
        self.last_events = events
        self.wait_events = waits
        # Returning the slots makes lifetime explicit to callers until completion.
        return output, slots
