import torch
import triton
import triton.language as tl


@triton.jit
def _scale_add(Y,D,O,N:tl.constexpr,S:tl.constexpr,B:tl.constexpr):
    i=tl.program_id(0)*B+tl.arange(0,B)
    delta=tl.load(D+i,i<N,0).to(tl.float32)
    scaled=(delta*S).to(tl.float16).to(tl.float32)
    base=tl.load(Y+i,i<N,0).to(tl.float32)
    tl.store(O+i,(base+scaled).to(tl.float16),i<N)


def scale_add(y,delta,scale,inplace=False,block=1024,warps=4):
    assert y.dtype==delta.dtype==torch.float16 and y.is_contiguous() and delta.is_contiguous()
    out=y if inplace else torch.empty_like(y)
    with torch.cuda.device(y.device):
        compiled=_scale_add[(triton.cdiv(y.numel(),block),)](y,delta,out,y.numel(),float(scale),block,
                                                           num_warps=warps,enable_fp_fusion=False)
    return out,compiled
