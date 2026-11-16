"""Fallback INT32 dequant + PDD with both original Half boundaries."""
import torch
import triton
import triton.language as tl


@triton.jit
def _finish(A, XS, WS, D, O, N: tl.constexpr, TOTAL: tl.constexpr,
            SCALE: tl.constexpr, B: tl.constexpr):
    i = tl.program_id(0) * B + tl.arange(0, B)
    acc = tl.load(A + i, i < TOTAL, 0).to(tl.float32)
    xs = tl.load(XS + i // N, i < TOTAL, 0).to(tl.float32)
    ws = tl.load(WS + i % N, i < TOTAL, 0).to(tl.float32)
    delta = tl.load(D + i, i < TOTAL, 0).to(tl.float32)
    main_h = ((acc * xs) * ws).to(tl.float16).to(tl.float32)
    pdd_h = (delta * SCALE).to(tl.float16).to(tl.float32)
    tl.store(O + i, (main_h + pdd_h).to(tl.float16), i < TOTAL)


def finish(acc, xs, ws, delta, scale, out=None):
    assert acc.ndim == 2 and acc.dtype == torch.int32
    assert delta.shape == acc.shape and delta.dtype == torch.float16
    assert xs.dtype == ws.dtype == torch.float32
    assert xs.numel() == len(acc) and ws.numel() == acc.shape[1]
    out = torch.empty_like(delta) if out is None else out
    assert out.shape == delta.shape and out.dtype == delta.dtype
    assert all(t.is_cuda and t.is_contiguous() and t.device == acc.device
               for t in (acc, xs, ws, delta, out))
    assert all(out.untyped_storage().data_ptr() != tensor.untyped_storage().data_ptr()
               for tensor in (acc, xs, ws)), 'Output must not overlap accumulator/scales'
    # Only the whole, fresh delta allocation may be reused. Offset views can
    # overlap unread elements belonging to a different program.
    if out.untyped_storage().data_ptr() == delta.untyped_storage().data_ptr():
        assert out.data_ptr() == delta.data_ptr() and out.stride() == delta.stride()
    with torch.cuda.device(acc.device):
        compiled = _finish[(triton.cdiv(acc.numel(), 256),)](
            acc, xs, ws, delta, out, acc.shape[1], acc.numel(), float(scale), 256,
            num_warps=4, enable_fp_fusion=False)
    return out, compiled
