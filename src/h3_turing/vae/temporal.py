"""Schedule the owner's existing temporal chunks; let its assembler consume them.

Raylight e82cad3 demonstrates independent H3 temporal chunks. This experiment
keeps the installed H3 decode_temporal, blending, finalization and tile batches.
It adds no Ray/FSDP dependency and retains at most one queued chunk per rank.
"""
import copy
from concurrent.futures import ThreadPoolExecutor
import types


def temporal_class(adapter):
    class TemporalParallel:
        def __init__(self, owner, replica, peer_device, *, baseline_batch=4):
            import torch
            self.torch, self.owner, self.replica = torch, owner, replica
            self.primary_device = next(owner.decoder.parameters()).device
            self.peer_device = torch.device(peer_device)
            self.original_temporal = owner.decode_temporal
            self.original_adaptive = owner._adaptive_decode
            self.old_temporal = owner.__dict__.get('decode_temporal', adapter._MISSING)
            self.old_adaptive = owner.__dict__.get('_adaptive_decode', adapter._MISSING)
            assert baseline_batch == 4 and owner.comfy_has_chunked_io and owner.tiling
            assert not any(k in owner.__dict__ for k in ('tiled_decode', '_decode_tile_row'))
            self.peer_owner = copy.copy(owner)
            self.peer_owner._modules = dict(owner._modules)
            self.peer_owner.post_quant_conv, self.peer_owner.decoder = replica
            self.compute = [torch.cuda.Stream(device=d) for d in (self.primary_device, self.peer_device)]
            self.copy0 = torch.cuda.Stream(device=self.primary_device)
            self.copy1 = torch.cuda.Stream(device=self.peer_device)
            self.pool = []
            self.active = None
            self.group_calls = self.tiles = 0

        def __enter__(self):
            self.pool = [ThreadPoolExecutor(max_workers=1, thread_name_prefix=f'h3-temporal-{r}')
                         for r in (0, 1)]
            self.owner.decode_temporal = types.MethodType(
                lambda _, z, output_buffer=None: self._temporal(z, output_buffer), self.owner)
            self.owner._adaptive_decode = types.MethodType(
                lambda _, z, frames=slice(None): self._consume(z, frames), self.owner)
            return self

        def __exit__(self, *_):
            try:
                for executor in self.pool:
                    executor.shutdown(wait=True, cancel_futures=True)
                for stream in (*self.compute, self.copy0, self.copy1):
                    stream.synchronize()
            finally:
                adapter._restore_instance_attribute(self.owner, 'decode_temporal', self.old_temporal)
                adapter._restore_instance_attribute(self.owner, '_adaptive_decode', self.old_adaptive)
                self.active = None
                self.peer_owner = None

        def _run(self, index):
            torch = self.torch
            rank = index % 2
            device = (self.primary_device, self.peer_device)[rank]
            frame = self.active
            start = index * self.owner.tokens_chunk_size
            stop = start + self.owner.tokens_chunk_size + self.owner.token_overlap
            source = frame['z'][:, :, start:stop]
            compute = self.compute[rank]
            with torch.inference_mode(), torch.cuda.device(device):
                compute.wait_event(frame['ready'])
                if rank:
                    self.copy0.wait_event(frame['ready'])
                    with torch.cuda.stream(self.copy0), torch.cuda.device(device), torch.cuda.stream(compute):
                        local = source.to(device, non_blocking=True)
                else:
                    local = source
                with torch.cuda.stream(compute):
                    import comfy.model_management as mm
                    batch = int(max(1, min(4, mm.get_free_memory(device) // (128 * 2**20 * local.shape[0]))))
                    assert batch == 4, 'Temporal scheduling would change the original spatial tile batch'
                    decoded = (self.peer_owner._adaptive_decode(local) if rank
                               else self.original_adaptive(local))
                    ready = torch.cuda.Event()
                    ready.record(compute)
                if rank:
                    self.copy1.wait_event(ready)
                    self.copy0.wait_event(ready)
                    with torch.cuda.stream(self.copy1), torch.cuda.device(self.primary_device), torch.cuda.stream(self.copy0):
                        result = decoded.to(self.primary_device, non_blocking=True)
                        copied = torch.cuda.Event()
                        copied.record(self.copy0)
                        decoded.record_stream(self.copy1)
                    return result, copied
                return decoded, ready

        def _submit(self, index):
            if index < self.active['count']:
                self.active['futures'][index] = self.pool[index % 2].submit(self._run, index)

        def _consume(self, z, frames):
            assert frames == slice(None)
            active = self.active
            index = active['cursor']
            start = index * self.owner.tokens_chunk_size
            expected = active['z'][:, :, start:start + self.owner.tokens_chunk_size + self.owner.token_overlap]
            assert z.shape == expected.shape and z.stride() == expected.stride()
            result, ready = active['futures'].pop(index).result()
            stream = self.torch.cuda.current_stream(self.primary_device)
            stream.wait_event(ready)
            result.record_stream(stream)
            active['cursor'] += 1
            self._submit(index + 2)
            return result

        def _temporal(self, z, output_buffer):
            assert self.active is None and z.device == self.primary_device
            pad, count = self.owner._decode_temporal_chunks(z.shape[2])
            # The original assembler also pads its own local view. This small
            # duplicate preserves its source verbatim and is included in timing.
            padded = self.torch.cat((z, z[:, :, -1:].repeat(1, 1, pad, 1, 1)), dim=2) if pad else z
            ready = self.torch.cuda.Event()
            ready.record(self.torch.cuda.current_stream(self.primary_device))
            self.active = dict(z=padded, count=count, cursor=0, futures={}, ready=ready)
            try:
                self._submit(0)
                self._submit(1)
                result = self.original_temporal(z, output_buffer)
                assert self.active['cursor'] == count and not self.active['futures']
                self.group_calls += count
                height, width = [int(n) * self.owner.vae_ratio for n in z.shape[-2:]]
                self.tiles += count * sum(g.cost for g in adapter.make_groups(self.owner, height, width, 4))
                return result
            finally:
                for future in self.active['futures'].values():
                    try:
                        future.result()
                    except BaseException:
                        pass
                for stream in (*self.compute, self.copy0, self.copy1):
                    stream.synchronize()
                self.active = None
    return TemporalParallel
