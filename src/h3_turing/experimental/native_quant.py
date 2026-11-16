"""Standalone CUDA ABI: no installed backend patching or model duplication."""
import ctypes
from pathlib import Path
import torch

LIB = ctypes.CDLL(str(Path(__file__).with_name('libh3_quant.so')))
LIB.h3_quant.argtypes = [ctypes.c_void_p] * 3 + [ctypes.c_int, ctypes.c_int, ctypes.c_float, ctypes.c_int, ctypes.c_void_p]
LIB.h3_quant.restype = ctypes.c_int


def quantize(x, clip=1., warps=4):
    assert x.is_contiguous() and x.ndim == 2 and x.dtype == torch.float16
    with torch.cuda.device(x.device):
        q = torch.empty_like(x, dtype=torch.int8)
        s = torch.empty((len(x), 1), device=x.device, dtype=torch.float32)
        code = LIB.h3_quant(x.data_ptr(), q.data_ptr(), s.data_ptr(),
                            x.shape[0], x.shape[1], float(clip), warps,
                            torch.cuda.current_stream(x.device).cuda_stream)
        assert code == 0, code
    return q, s
