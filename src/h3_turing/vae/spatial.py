"""Guarded, reversible H3 FP16 VAE spatial-group parallelism prototype.

Importing does not initialize CUDA. This is not an installed ComfyUI custom node.
The caller must already have a fully resident, ordinary FP16 H3 VAE on GPU 0.
Only complete existing tile batches are scheduled; the original assembly runs.
"""

from __future__ import annotations

import copy
import dataclasses
import logging
import threading
import time
import types
from concurrent.futures import ThreadPoolExecutor


_MISSING = object()


def _restore_instance_attribute(obj, name, old):
    if old is _MISSING:
        obj.__dict__.pop(name, None)
    else:
        setattr(obj, name, old)


@dataclasses.dataclass(frozen=True)
class Group:
    row: int
    index: int
    y_start: int
    y_length: int
    x_windows: tuple[tuple[int, int], ...]

    @property
    def cost(self):
        return len(self.x_windows)


def make_groups(owner, height, width, batch=4):
    """Use the owner's actual split_tiles, without inventing another tile plan."""
    if batch < 1:
        raise ValueError("batch must be positive")
    ys, yl, _ = owner.split_tiles(height)
    xs, xl, _ = owner.split_tiles(width)
    if len(ys) != len(yl) or len(xs) != len(xl):
        raise RuntimeError("Invalid owner tile plan")
    groups = []
    for row, (start, length) in enumerate(zip(ys, yl)):
        windows = tuple(zip(xs, xl))
        for index, offset in enumerate(range(0, len(windows), batch)):
            groups.append(Group(row, index, start, length, windows[offset:offset + batch]))
    return tuple(groups)


def partition_groups(groups):
    """Exact two-bin partition for this small plan; keep first group on primary."""
    if not groups or len(groups) > 20:
        raise RuntimeError("Prototype supports 1..20 whole tile groups per chunk")
    costs = [g.cost for g in groups]
    total = sum(costs)
    best = None
    for bits in range(1 << (len(groups) - 1)):
        assignment = (0,) + tuple((bits >> i) & 1 for i in range(len(groups) - 1))
        a = sum(c for c, rank in zip(costs, assignment) if rank == 0)
        b = total - a
        score = (max(a, b), abs(a - b), assignment)
        if best is None or score < best[0]:
            best = (score, assignment)
    return best[1]


def module_storage_bytes(modules):
    """Ordinary tensors only. Deduplicate aliased source storage for the estimate."""
    seen = set()
    total = 0
    for module in modules:
        for value in tuple(module.parameters()) + tuple(module.buffers()):
            if hasattr(value, "_qdata"):
                raise RuntimeError("FP16 experiment refuses quantized VAE parameters")
            storage = value.untyped_storage()
            key = (str(value.device), storage.data_ptr(), storage.nbytes())
            if key not in seen:
                seen.add(key)
                total += storage.nbytes()
    return total


def clone_decode_modules(owner, device, primary_device):
    """Deepcopy module objects with tensors pre-copied directly to the peer.

    A normal deepcopy followed by .to(peer) briefly duplicates the full decoder
    on primary. Memoizing each parameter/buffer avoids that unnecessary peak.
    No dtype conversion, dequantization, or new initialization is performed.
    """
    import torch

    modules = (owner.post_quant_conv, owner.decoder)
    memo = {}
    for root in modules:
        for module in root.modules():
            if module._forward_hooks or module._forward_pre_hooks or module._backward_hooks or module._backward_pre_hooks:
                raise RuntimeError("Prototype refuses decoder modules with hooks")
            for name in ("weight_function", "bias_function", "weight_lowvram_function", "bias_lowvram_function"):
                if getattr(module, name, None):
                    raise RuntimeError(f"Prototype refuses active Comfy weight callbacks: {name}")
            registered = {id(t) for t in module._parameters.values() if t is not None}
            registered.update(id(t) for t in module._buffers.values() if t is not None)
            for name, value in vars(module).items():
                if isinstance(value, torch.Tensor) and id(value) not in registered:
                    raise RuntimeError(f"Unregistered decoder tensor attribute: {name}")
        for parameter in root.parameters():
            if hasattr(parameter, "_qdata") or parameter.dtype != torch.float16:
                raise RuntimeError("Source decode parameters must be ordinary FP16")
            if parameter.device != primary_device:
                raise RuntimeError("Source decoder is partially offloaded; fully load it first")
            if id(parameter) not in memo:
                memo[id(parameter)] = torch.nn.Parameter(
                    parameter.detach().to(device=device, copy=True),
                    requires_grad=parameter.requires_grad,
                )
        for buffer in root.buffers():
            if id(buffer) not in memo:
                memo[id(buffer)] = buffer.detach().to(device=device, copy=True)
    post = copy.deepcopy(owner.post_quant_conv, memo)
    decoder = copy.deepcopy(owner.decoder, memo)
    return post, decoder


class SpatialGroupParallel:
    """Context which replaces only tiled_decode and its row result supplier."""

    def __init__(self, owner, replica, peer_device, *, baseline_batch=4):
        import torch

        self.torch = torch
        self.owner = owner
        self.replica = replica
        self.primary_device = next(owner.decoder.parameters()).device
        self.peer_device = torch.device(peer_device)
        self.batch = baseline_batch
        self.original_tiled = owner.tiled_decode
        self.original_row = owner._decode_tile_row
        self.old_tiled_attribute = owner.__dict__.get("tiled_decode", _MISSING)
        self.old_row_attribute = owner.__dict__.get("_decode_tile_row", _MISSING)
        self.executors = None
        self.active = None
        self.installed = False
        self.group_calls = 0
        self.tiles = 0

    def __enter__(self):
        owner = self.owner
        if owner.__class__.__name__ != "MiniMaxH3VideoVAE":
            raise RuntimeError("Prototype is restricted to MiniMaxH3VideoVAE")
        if (owner.tile_size, owner.tile_overlap_min, owner.vae_ratio) != (256, 64, 16):
            raise RuntimeError("Unexpected H3 spatial tile configuration")
        if not owner.tiling or not owner.comfy_has_chunked_io:
            raise RuntimeError("Expected H3-owned tiled chunked decoding")
        if self.primary_device == self.peer_device or self.primary_device.type != "cuda":
            raise RuntimeError("Two different CUDA devices are required")
        if self.batch != 4:
            raise RuntimeError("This candidate requires an observed baseline tile batch of four")
        self.executors = []
        try:
            for index in range(2):
                self.executors.append(ThreadPoolExecutor(max_workers=1, thread_name_prefix=f"h3-vae-{index}"))
            self.installed = True
            owner.tiled_decode = types.MethodType(lambda _, z, frames=slice(None): self._tiled(z, frames), owner)
            owner._decode_tile_row = types.MethodType(lambda _, z, xs, xl: self._row(z, xs, xl), owner)
            return self
        except BaseException:
            self.__exit__(None, None, None)
            raise

    def __exit__(self, exc_type, exc, tb):
        # Drain first: no module/storage may be freed while a worker still reads it.
        try:
            if self.executors:
                for executor in self.executors:
                    executor.shutdown(wait=True, cancel_futures=True)
        finally:
            if self.installed:
                _restore_instance_attribute(self.owner, "tiled_decode", self.old_tiled_attribute)
                _restore_instance_attribute(self.owner, "_decode_tile_row", self.old_row_attribute)
                self.installed = False
            self.active = None

    def _run(self, z, group, rank, ready):
        torch = self.torch
        device = (self.primary_device, self.peer_device)[rank]
        with torch.inference_mode(), torch.cuda.device(device):
            stream = torch.cuda.Stream(device=device)
            with torch.cuda.stream(stream):
                stream.wait_event(ready)
                ratio = self.owner.vae_ratio
                row = z[..., group.y_start // ratio:(group.y_start + group.y_length) // ratio, :]
                pieces = [row[..., x // ratio:(x + length) // ratio].to(device=device)
                          for x, length in group.x_windows]
                value = torch.cat(pieces)
                if rank == 0:
                    result = self.owner._decode_pixels(value)
                else:
                    post, decoder = self.replica
                    result = decoder(post(value))
                # Blocking peer copy and stream completion deliberately simplify
                # readiness/lifetime ownership for the first measured prototype.
                result = result.to(device=self.primary_device, non_blocking=False)
            stream.synchronize()
            return result

    def _tiled(self, z, frames):
        import comfy.model_management as mm

        if self.active is not None:
            raise RuntimeError("Concurrent/reentrant H3 spatial decoding is unsupported")
        if tuple(z.shape[-2:]) not in ((32,60),(48,84)) or z.device != self.primary_device:
            raise RuntimeError("This experiment is limited to the 960x512 clip on primary")
        free = mm.get_free_memory(z.device)
        selected_batch = int(max(1, min(4, free // (128 * 2**20 * z.shape[0]))))
        if selected_batch != self.batch:
            raise RuntimeError("Current free VRAM would change the baseline's tile batching")
        height,width=(int(d)*self.owner.vae_ratio for d in z.shape[-2:])
        groups = make_groups(self.owner, height, width, self.batch)
        assignment = partition_groups(groups)
        ready = self.torch.cuda.Event()
        ready.record(self.torch.cuda.current_stream(z.device))
        futures = {}
        try:
            for group, rank in zip(groups, assignment):
                futures[(group.row, group.index)] = self.executors[rank].submit(
                    self._run, z, group, rank, ready
                )
            self.active = {"groups": groups, "futures": futures, "row": 0, "z": z}
            result = self.original_tiled(z, frames)
            if self.active["row"] != len(self.owner.split_tiles(height)[0]):
                raise RuntimeError("Original spatial assembly consumed an unexpected row count")
            self.group_calls += len(groups)
            self.tiles += sum(g.cost for g in groups)
            return result
        finally:
            # Also drain errors here, before the next temporal chunk can start.
            for future in futures.values():
                try:
                    future.result()
                except BaseException:
                    pass
            self.active = None

    def _row(self, z_row, x_idx, x_len):
        if self.active is None:
            raise RuntimeError("Tile row requested outside the owned spatial assembly")
        row = self.active["row"]
        self.active["row"] += 1
        expected_x, expected_len, _ = self.owner.split_tiles(self.active["z"].shape[-1]*self.owner.vae_ratio)
        if list(x_idx) != list(expected_x) or list(x_len) != list(expected_len):
            raise RuntimeError("Original spatial assembly changed its x tile plan")
        matching = [group for group in self.active["groups"] if group.row == row]
        if not matching:
            raise RuntimeError("Unexpected spatial assembly row")
        ratio = self.owner.vae_ratio
        group0 = matching[0]
        expected_row = self.active["z"][..., group0.y_start // ratio:
                                        (group0.y_start + group0.y_length) // ratio, :]
        if (z_row.shape != expected_row.shape or z_row.stride() != expected_row.stride()
                or z_row.data_ptr() != expected_row.data_ptr()):
            raise RuntimeError("Original spatial assembly changed its y tile view")
        for group in matching:
            result = self.active["futures"][(row, group.index)].result()
            yield from result.chunk(len(group.x_windows))


_OWNED_LOCK = threading.Lock()


def install_owned_decode_wrapper(vae, peer_device="cuda:1", *, min_free_after_copy=1024 * 2**20, context_class=None):
    """Install on one Comfy VAE object; return a callable restoring its method.

    Wrapper creation itself does not load models or initialize CUDA. The original
    Comfy call must load the VAE before invoking _decode_tiled_owned. Peer replica
    lifetime covers one complete video decode, not subsequent sampling jobs.
    Replica construction/release time is inside the measured VAE node.
    """
    context_class = context_class or SpatialGroupParallel
    original = vae._decode_tiled_owned
    old_attribute = vae.__dict__.get("_decode_tiled_owned", _MISSING)
    old_marker = vae.__dict__.get("_h3_parallel_owned_installed", _MISSING)
    if getattr(vae, "_h3_parallel_owned_installed", False):
        raise RuntimeError("Owned decode adapter is already installed")

    def wrapped(self, samples, **kwargs):
        import torch
        import comfy.model_management as mm

        if not _OWNED_LOCK.acquire(blocking=False):
            raise RuntimeError("Overlapping H3 owned decode calls are unsupported")
        try:
            owner = self.first_stage_model
            if not torch.is_inference_mode_enabled():
                raise RuntimeError("Expected the same inference_mode as the baseline Comfy execution")
            if self.vae_dtype != torch.float16:
                raise RuntimeError("This experiment requires the existing FP16 VAE compute dtype")
            primary = next(owner.decoder.parameters()).device
            peer = torch.device(peer_device)
            if peer.type != "cuda" or peer.index is None:
                raise RuntimeError("Explicit CUDA peer device index is required")
            if primary != self.device or primary.type != "cuda" or peer == primary:
                raise RuntimeError("VAE must be fully loaded on a different CUDA device")
            if samples.shape[0] != 1 or tuple(owner.decode_output_shape(samples.shape)[2:]) not in ((192,512,960),(362,512,960),(192,768,1344)):
                raise RuntimeError("This experiment is restricted to the exact 192-frame 960x512 video")
            size = module_storage_bytes((owner.post_quant_conv, owner.decoder))
            if mm.get_free_memory(peer) < size + min_free_after_copy:
                raise RuntimeError("Insufficient peer VRAM for FP16 decoder plus headroom")
            t0 = time.perf_counter()
            with torch.inference_mode():
                torch.cuda.synchronize(primary)
                replica = clone_decode_modules(owner, peer, primary)
                torch.cuda.synchronize(peer)
                clone_seconds = time.perf_counter() - t0
                adapter = None
                try:
                    with context_class(owner, replica, peer) as adapter:
                        result = original(samples, **kwargs)
                finally:
                    # SpatialGroupParallel has drained all workers before this.
                    if adapter is not None:
                        adapter.replica = None
                    del replica
            logging.info(
                "[H3 FP16 VAE parallel] replica_bytes=%d replica_seconds=%.6f groups=%d tiles=%d",
                size, clone_seconds, adapter.group_calls, adapter.tiles,
            )
            logging.info("[H3 FP16 VAE parallel numerics] %s", {
                "cudnn_benchmark": torch.backends.cudnn.benchmark,
                "cudnn_deterministic": torch.backends.cudnn.deterministic,
                "deterministic_algorithms": torch.are_deterministic_algorithms_enabled(),
                "fp16_reduced_precision_reduction": torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction,
                "cudnn_version": torch.backends.cudnn.version(),
            })
            return result
        finally:
            _OWNED_LOCK.release()

    vae._decode_tiled_owned = types.MethodType(wrapped, vae)
    vae._h3_parallel_owned_installed = True

    def uninstall():
        if _OWNED_LOCK.locked():
            raise RuntimeError("Cannot remove the adapter during an active decode")
        _restore_instance_attribute(vae, "_decode_tiled_owned", old_attribute)
        _restore_instance_attribute(vae, "_h3_parallel_owned_installed", old_marker)

    return uninstall
