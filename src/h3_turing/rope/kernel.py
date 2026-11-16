"""Explicit staged pointwise RoPE only. Imported only by an explicit GPU call."""
import triton
import triton.language as tl


@triton.jit
def _h2f(bits):
    return tl.inline_asm_elementwise('cvt.f32.f16 $0, $1;',constraints='=f,h',
        args=[bits],dtype=tl.float32,is_pure=False,pack=1)


@triton.jit
def _f2h(value):
    return tl.inline_asm_elementwise('cvt.rn.f16.f32 $0, $1;',constraints='=h,f',
        args=[value],dtype=tl.uint16,is_pure=False,pack=1)


@triton.jit
def _mul(a,b):
    return tl.inline_asm_elementwise('mul.rn.f32 $0, $1, $2;',constraints='=f,f,f',
        args=[a,b],dtype=tl.float32,is_pure=False,pack=1)


@triton.jit
def _fma(a,b,c):
    return tl.inline_asm_elementwise('fma.rn.f32 $0, $1, $2, $3;',constraints='=f,f,f,f',
        args=[a,b,c],dtype=tl.float32,is_pure=False,pack=1)


@triton.jit
def _rope(N,F,O,C:tl.constexpr,START,NS1:tl.constexpr,NS2:tl.constexpr,
          FS1:tl.constexpr,FS3:tl.constexpr,FS4:tl.constexpr,FS5:tl.constexpr,
          NORMAL_HALF:tl.constexpr,FREQ_HALF:tl.constexpr,OS1:tl.constexpr,BLOCK:tl.constexpr):
    # N: original F.rms_norm result [1,C,28,128]. This kernel performs no
    # normalization, reduction, approximation, denominator or scale operation.
    offset=tl.program_id(0)*BLOCK+tl.arange(0,BLOCK)
    token=offset//3584
    head=offset//128%28
    dim=offset%128
    valid=token<C
    rotate=dim<96
    first=dim<48
    opposite=tl.where(first,dim+48,dim-48)
    npos=token*NS1+head*NS2+dim
    opos=token*NS1+head*NS2+opposite
    if NORMAL_HALF:
        raw=tl.load(N.to(tl.pointer_type(tl.uint16))+npos,valid,other=0)
        otherraw=tl.load(N.to(tl.pointer_type(tl.uint16))+opos,valid&rotate,other=0)
        current=_h2f(raw)
        other=_h2f(otherraw)
        tail=raw
    else:
        current=tl.load(N+npos,valid,other=0)
        other=tl.load(N+opos,valid&rotate,other=0)
        tail=_f2h(current)
    pair=dim%48
    fpos=token*FS1+pair*FS3
    diagonal=fpos+tl.where(first,0,FS4+FS5)
    offdiagonal=fpos+tl.where(first,FS5,FS4)
    if FREQ_HALF:
        dbits=tl.load(F.to(tl.pointer_type(tl.uint16))+diagonal,valid&rotate,other=0)
        obits=tl.load(F.to(tl.pointer_type(tl.uint16))+offdiagonal,valid&rotate,other=0)
        dvalue=_h2f(dbits)
        ovalue=_h2f(obits)
        # eager casts N32 to table-half before multiplication.
        if not NORMAL_HALF:
            current=_h2f(_f2h(current))
            other=_h2f(_f2h(other))
        product=_h2f(_f2h(_mul(current,dvalue)))
        result=_f2h(_fma(other,ovalue,product))
        if not NORMAL_HALF:
            # eager .type_as(N32), followed by original packed-Q FP16 copy.
            # Keep these conversions explicit even for NaN payloads.
            result=_f2h(_h2f(result))
    else:
        dvalue=tl.load(F+diagonal,valid&rotate,other=0)
        ovalue=tl.load(F+offdiagonal,valid&rotate,other=0)
        product=_mul(current,dvalue)
        result=_f2h(_fma(other,ovalue,product))
    bits=tl.where(rotate,result,tail)
    target=head*OS1+(START+token)*128+dim
    tl.store(O.to(tl.pointer_type(tl.uint16))+target,bits,valid)


def launch(normal,freqs,output,start):
    count=normal.shape[1]
    return _rope[(triton.cdiv(count*3584,256),)](normal,freqs,output,count,start,
        normal.stride(1),normal.stride(2),freqs.stride(1),freqs.stride(3),freqs.stride(4),freqs.stride(5),
        normal.dtype.itemsize==2,freqs.dtype.itemsize==2,output.stride(1),256,num_warps=4,enable_fp_fusion=False)
