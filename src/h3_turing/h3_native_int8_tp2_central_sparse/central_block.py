"""TP2 with the FP32 residual stream and modulation only on the primary GPU."""
import time

import torch

from . import tp2 as core


def _peer_copy(tensor, destination, producer, consumer):
    consumer.wait_stream(producer)
    with torch.cuda.device(destination.index), torch.cuda.stream(consumer):
        result = torch.empty_like(tensor, device=destination)
        result.copy_(tensor, non_blocking=True)
    return result


def tp2_block(sh, x, t_emb, mod_segments, rope_freqs,
              transformer_options=None, attention=None, **kwargs):
    from comfy.ldm.minimax.model import _mod_gate, _mod_scale_shift
    d0, d1 = sh['dev0'], sh['dev1']
    assert x.device == d0, (x.device, d0)
    if attention is not None:
        raise RuntimeError('TP2 Central requires dense native CK attention')
    st0, st1 = torch.cuda.current_stream(d0.index), torch.cuda.current_stream(d1.index)
    core.NCALL[0] += 1
    started = time.perf_counter()
    with torch.cuda.device(d0.index), torch.cuda.stream(st0):
        x0 = x.to(torch.float32)
        m = sh['adaln'](t_emb)
        h0 = _mod_scale_shift(core._rms(x0, sh['n1'][0], sh['eps1']), m[0], m[1],
                             mod_segments).to(torch.float16)
    h1 = _peer_copy(h0, d1, st0, st1)

    def front(rank, stream, h):
        device = d0 if rank == 0 else d1
        with torch.cuda.device(device.index), torch.cuda.stream(stream):
            owned = core._qkv_rope(sh, h, rope_freqs, rank)
            out = core._attn(owned, sh, transformer_options, mod_segments)
            return core._lin((out / core.KO).to(torch.float16), sh['op'][rank], sh['lop'][rank])

    p0, p1 = core._par(lambda: front(0, st0, h0), lambda: front(1, st1, h1))
    r1 = _peer_copy(p1, d0, st1, st0)
    # The primary wait is queued before releasing its broadcast source h0.
    del h0, h1
    with torch.cuda.device(d0.index), torch.cuda.stream(st0):
        a = core._sum_scaled_fp32(p0, r1, core.KO)
        x0 = _mod_gate(x0, m[2], a, mod_segments)
        del a
        h0 = _mod_scale_shift(core._rms(x0, sh['n2'][0], sh['eps2']), m[3], m[4],
                             mod_segments).to(torch.float16)
    h1 = _peer_copy(h0, d1, st0, st1)
    # GPU1's wait includes the primary's p1 read, before p1 storage can be reused.
    del p0, p1, r1
    core._snap('central_qkv+attn+reduce', started)
    started = time.perf_counter()

    def half(rank, stream, h):
        device = d0 if rank == 0 else d1
        with torch.cuda.device(device.index), torch.cuda.stream(stream):
            return core._mlp_chunked(sh, h, rank)

    mm0, mm1 = core._par(lambda: half(0, st0, h0), lambda: half(1, st1, h1))
    r1b = _peer_copy(mm1, d0, st1, st0)
    del h0, h1
    with torch.cuda.device(d0.index), torch.cuda.stream(st0):
        reduced = core._sum_scaled_fp32(mm0, r1b, core.KC2)
        result = _mod_gate(x0, m[5], reduced, mod_segments)
    # Protect GPU1's mm1 source if its allocator reuses this storage next block.
    st1.wait_stream(st0)
    core._snap('central_mlp+reduce', started)
    return result
