"""Balance the last odd temporal chunk with the already qualified spatial path."""
from .temporal import temporal_class


def hybrid_class(adapter, spatial_class):
    class HybridTemporal(temporal_class(adapter)):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self.last_peer_future = None
            self.joint_last_chunks = 0

        def _submit(self, index):
            super()._submit(index)
            if index < self.active['count'] and index % 2:
                self.last_peer_future = self.active['futures'][index]

        def _run(self, index):
            frame = self.active
            if frame['count'] % 2 == 0 or index != frame['count'] - 1 or index == 0:
                return super()._run(index)
            # The peer's previous decoder invocation must finish before the same
            # replica is used by the joint spatial decode. Wait for host enqueue
            # completion first, then cover both compute and P2P source reads.
            assert self.last_peer_future is not None
            self.last_peer_future.result()
            for stream in (self.compute[1], self.copy1, self.copy0):
                stream.synchronize()
            self.last_peer_future = None
            torch = self.torch
            start = index * self.owner.tokens_chunk_size
            clip = frame['z'][:, :, start:start + self.owner.tokens_chunk_size + self.owner.token_overlap]
            with torch.inference_mode(), torch.cuda.device(self.primary_device), torch.cuda.stream(self.compute[0]):
                self.compute[0].wait_event(frame['ready'])
                with spatial_class(self.owner, self.replica, self.peer_device):
                    result = self.original_adaptive(clip)
                ready = torch.cuda.Event()
                ready.record(self.compute[0])
            self.joint_last_chunks += 1
            return result, ready
    return HybridTemporal
