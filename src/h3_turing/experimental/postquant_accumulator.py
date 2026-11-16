"""Extract the installed CK fallback body without changing its GEMM call."""
import ast
import copy
import hashlib
from pathlib import Path

import torch
import comfy_kitchen.backends.cuda as ck


PIN = 'c5b1c3bc70869e418728114e8c27a889cf5b56ccb31ccf82a001386aef12ee8e'
BINARY_PIN = 'ffbc49b384dfb9e75c946f442a8888b9bbc7a7d0d05223f7c8ef1a1bc9602652'


def extract():
    source = Path(ck.__file__).read_bytes()
    digest = hashlib.sha256(source).hexdigest()
    assert digest == PIN, f'Unverified CK source: {digest}'
    binary_digest = hashlib.sha256(Path(ck._C.__file__).read_bytes()).hexdigest()
    assert binary_digest == BINARY_PIN, f'Unverified CK binary: {binary_digest}'
    tree = ast.parse(source)
    linear = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == 'int8_linear')
    fallback = next(n for n in linear.body if isinstance(n, ast.If)
                    and ast.unparse(n.test) == 'not used_cutlass')
    body = copy.deepcopy(fallback.body)
    dequant = body.pop()
    assert isinstance(dequant, ast.Expr)
    assert ast.unparse(dequant.value.func) == '_C.dequantize_int8_linear'
    assert ast.dump(ast.Module(body=body + [dequant], type_ignores=[]), include_attributes=False) == ast.dump(
        ast.Module(body=fallback.body, type_ignores=[]), include_attributes=False)
    function = ast.parse('def accumulator(x_qdata, x_scale, weight, weight_scale):\n'
                         '    m, k = x_qdata.shape\n'
                         '    n = weight.shape[0]\n'
                         '    stream_ptr = torch.cuda.current_stream(x.device).cuda_stream\n').body[0]
    # The original fallback refers to x only for its device.
    function.args.args.insert(0, ast.arg(arg='x'))
    function.body += body
    function.body += ast.parse('return out_int32, x_scale, weight_scale').body
    namespace = dict(vars(ck))
    exec(compile(ast.fix_missing_locations(ast.Module(body=[function], type_ignores=[])),
                 '<pinned-ck-accumulator>', 'exec'), namespace)
    return namespace['accumulator'], dict(ck_source_sha256=digest, ck_binary_sha256=binary_digest,
        fallback_body_unchanged=True, removed='dequantize_int8_linear only',
        no_Half_base_allocation=True)


_accumulator = None
PROOF = None


def admitted(x, weight, scale):
    m, k = x.shape
    n = weight.shape[0]
    return (x.is_cuda and x.dtype == torch.float16 and x.is_contiguous()
            and m > 128 and n == 5376 and k in (3584, 7168)
            and weight.dtype == torch.int8 and weight.is_contiguous()
            and scale.dtype == torch.float32 and scale.numel() == n
            and torch.cuda.get_device_capability(x.device) == (7, 5)
            and not ck._prefer_turing_fused_int8(m, n, k)
            and (ck._prefer_cublas_int8_fallback(m, n, k, x.get_device())
                 or ck._DISABLE_CUTLASS_INT8
                 or not ck._cuda_device_supports_cutlass_int8_dequant(x)))


def accumulate(x, weight, scale, packet=None):
    global _accumulator, PROOF
    assert admitted(x, weight, scale), 'Use baseline for unsupported dispatch/shape'
    if _accumulator is None:
        _accumulator, PROOF = extract()
    with torch.cuda.device(x.device):
        q, xs = ck.quantize_int8_rowwise_convrot64(x, 256) if packet is None else packet
        ws = ck._int8_weight_scale_arg(scale, x.device)
        return _accumulator(x, q, xs, weight, ws)
