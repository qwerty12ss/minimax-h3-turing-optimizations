"""One FP32 add and power-of-two multiply, with FP fusion disabled."""
import torch
import triton
import triton.language as tl


@triton.jit
def _sum_scaled(A, B, C, N: tl.constexpr, SCALE: tl.constexpr,
                BLOCK: tl.constexpr):
    offset = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    a = tl.load(A + offset, offset < N, other=0).to(tl.float32)
    b = tl.load(B + offset, offset < N, other=0).to(tl.float32)
    added = a + b
    result = added * SCALE
    tl.store(C + offset, result, offset < N)


def sum_scaled(a, b, scale):
    if (a.dtype != torch.float16 or b.dtype != torch.float16
            or a.device != b.device or a.device.type != 'cuda'
            or a.shape != b.shape or not a.is_contiguous() or not b.is_contiguous()
            or tuple(a.shape) != (29226, 5376) or scale not in (64, 256)):
        raise RuntimeError('Unqualified TP2 sum/reduce contract')
    result = torch.empty(a.shape, dtype=torch.float32, device=a.device)
    with torch.cuda.device(a.device):
        _sum_scaled[(triton.cdiv(a.numel(), 1024),)](
            a, b, result, a.numel(), scale, 1024,
            num_warps=4, enable_fp_fusion=False)
    return result
