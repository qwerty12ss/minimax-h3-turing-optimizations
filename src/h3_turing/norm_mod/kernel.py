"""Explicit non-FTZ PTX RN stages; original Torch denominator stays outside.

V1 libdevice helpers lowered to FTZ on the installed toolchain. Explicit PTX
is required here; source-level mul_rn names alone are not sufficient evidence.
"""
import hashlib
import torch
import triton
import triton.language as tl
from .contract import SHAPE, require, validate_configuration

_COMPILED = {}


@triton.jit
def _pointwise(X, Inv, W, Scale, Shift, Rows, Y, TOTAL, C: tl.constexpr,
               SCALE_ROW: tl.constexpr, SCALE_COL: tl.constexpr,
               SHIFT_ROW: tl.constexpr, SHIFT_COL: tl.constexpr, BLOCK: tl.constexpr):
    index = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    valid = index < TOTAL
    row, column = index // C, index % C
    modulation_row = tl.load(Rows + row, mask=valid, other=0)
    x = tl.load(X + index, mask=valid, other=0).to(tl.float32)
    inverse = tl.load(Inv + row, mask=valid, other=0).to(tl.float32)
    weight = tl.load(W + column, mask=valid, other=0).to(tl.float32)
    scale = tl.load(Scale + modulation_row * SCALE_ROW + column * SCALE_COL, mask=valid, other=0).to(tl.float32)
    shift = tl.load(Shift + modulation_row * SHIFT_ROW + column * SHIFT_COL, mask=valid, other=0).to(tl.float32)
    normalized = tl.inline_asm_elementwise('mul.rn.f32 $0, $1, $2;', constraints='=f,f,f', args=[x, inverse], dtype=tl.float32, is_pure=True, pack=1)
    weighted = tl.inline_asm_elementwise('mul.rn.f32 $0, $1, $2;', constraints='=f,f,f', args=[normalized, weight], dtype=tl.float32, is_pure=True, pack=1)
    multiplier = tl.inline_asm_elementwise('add.rn.f32 $0, $1, $2;', constraints='=f,f,f', args=[tl.full((), 1.0, tl.float32), scale], dtype=tl.float32, is_pure=True, pack=1)
    scaled = tl.inline_asm_elementwise('mul.rn.f32 $0, $1, $2;', constraints='=f,f,f', args=[weighted, multiplier], dtype=tl.float32, is_pure=True, pack=1)
    shifted = tl.inline_asm_elementwise('add.rn.f32 $0, $1, $2;', constraints='=f,f,f', args=[scaled, shift], dtype=tl.float32, is_pure=True, pack=1)
    output = shifted.to(tl.float16, fp_downcast_rounding='rtne')
    tl.store(Y + index, output, mask=valid)


def fused_pointwise(x, inverse, weight, shift, scale, row_map, block, num_warps, strict_shape=True):
    validate_configuration(block, num_warps)
    require(x.ndim == 2 and x.shape[1] == SHAPE[1] and (not strict_shape or tuple(x.shape) == SHAPE), 'Unqualified residual shape')
    require(x.is_cuda and x.dtype is torch.float32 and x.is_contiguous(), 'CUDA contiguous FP32 residual required')
    require(tuple(inverse.shape) == (x.shape[0], 1) and inverse.dtype is torch.float32 and inverse.is_contiguous()
            and inverse.device == x.device, 'Original Torch FP32 inverse required')
    require(tuple(weight.shape) == (SHAPE[1],) and weight.dtype in (torch.float16, torch.float32)
            and weight.is_contiguous() and weight.device == x.device, 'Unqualified norm weight')
    require(shift.ndim == scale.ndim == 2 and shift.shape == scale.shape and shift.shape[1] == SHAPE[1]
            and shift.device == scale.device == x.device and shift.dtype in (torch.float16, torch.float32)
            and scale.dtype is shift.dtype and tuple(shift.stride()) == tuple(scale.stride())
            and min(*shift.stride(), *scale.stride()) > 0,
            'Unqualified modulation tables')
    require(tuple(row_map.shape) == (x.shape[0],) and row_map.dtype is torch.int64 and row_map.is_contiguous()
            and row_map.device == x.device, 'Source-derived Torch row mapping required')
    previous = torch.cuda.current_device()
    stream = int(torch.cuda.current_stream(x.device).cuda_stream)
    with torch.cuda.device(x.device):
        output = torch.empty_like(x, dtype=torch.float16)
        compiled = _pointwise[(triton.cdiv(x.numel(), block),)](x, inverse, weight, scale, shift, row_map, output,
              TOTAL=x.numel(), C=SHAPE[1], SCALE_ROW=scale.stride(0), SCALE_COL=scale.stride(1),
              SHIFT_ROW=shift.stride(0), SHIFT_COL=shift.stride(1), BLOCK=int(block),
              num_warps=int(num_warps), num_stages=1, enable_fp_fusion=False)
        key = (int(block), int(num_warps), str(weight.dtype), str(shift.dtype), str(scale.dtype),
               tuple(shift.stride()), tuple(scale.stride()), int(x.numel()), x.device.index)
        _COMPILED[key] = compiled
        require(int(torch.cuda.current_stream(x.device).cuda_stream) == stream, 'Pointwise kernel changed caller stream')
    require(torch.cuda.current_device() == previous, 'Pointwise kernel changed caller device')
    return output


def compiled_metadata(include_ptx=False):
    results = []
    for key, compiled in _COMPILED.items():
        ptx = compiled.asm.get('ptx', '')
        instructions = '\n'.join(line for line in ptx.splitlines() if not line.lstrip().startswith('//'))
        results.append({'specialization': repr(key), 'registers': getattr(compiled, 'n_regs', None),
            'spills': getattr(compiled, 'n_spills', None), 'shared_bytes': getattr(compiled.metadata, 'shared', None),
            'ptx_sha256': hashlib.sha256(ptx.encode()).hexdigest(), 'has_fma': 'fma.' in instructions,
            'has_ftz': '.ftz' in instructions, 'has_fp16_rn_cast': 'cvt.rn.f16' in instructions,
            'rn_fp32_mul_instructions': instructions.count('mul.rn.f32'),
            'rn_fp32_add_instructions': instructions.count('add.rn.f32'), 'device_index': key[-1]})
        if include_ptx:
            require(len(ptx.encode()) <= 256*1024, 'Bounded compiled PTX size exceeded')
            results[-1]['ptx_text'] = ptx
    return results
