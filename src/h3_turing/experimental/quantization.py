"""Quantization experiments. Installed CK files are never modified."""
import ast
import copy
import hashlib
import os
from pathlib import Path

import torch
import triton
import triton.language as tl
from triton.language.extra.cuda import libdevice
import comfy_kitchen.backends.cuda as ck


@triton.jit
def _quant(X, Q, S, K: tl.constexpr, B: tl.constexpr, CLIP: tl.constexpr):
    row = tl.program_id(0)
    i = tl.arange(0, B)
    v = tl.load(X + row * K + i, i < K, 0).to(tl.float32)
    # The installed ConvRot uses the regular H4 Kronecker product, not Sylvester H.
    for shift in tl.static_range(0, 8, 2):
        b = tl.gather(v, i ^ (1 << shift), 0)
        c = tl.gather(v, i ^ (2 << shift), 0)
        d = tl.gather(v, i ^ (3 << shift), 0)
        v = ((v + b) + c) - d
    v = v * 0.0625
    amax = tl.max(tl.where(i < K, tl.abs(v), 0), 0)
    scale = tl.maximum(amax * CLIP / 127., 1.e-12)
    q = libdevice.nearbyint(v / scale)
    q = tl.minimum(tl.maximum(q, -127.), 127.).to(tl.int8)
    tl.store(Q + row * K + i, q, i < K)
    tl.store(S + row, scale)


def quantize(x, clip=1.):
    if os.environ.get('H3_ACCURACY_IMPL') == 'native':
        from .native_quant import quantize as native
        return native(x, clip)
    assert x.ndim == 2 and x.is_contiguous() and x.dtype == torch.float16
    m, k = x.shape
    assert k % 256 == 0
    q = torch.empty((m, k), device=x.device, dtype=torch.int8)
    s = torch.empty((m, 1), device=x.device, dtype=torch.float32)
    _quant[(m,)](x, q, s, k, triton.next_power_of_2(k), float(clip),
                 num_warps=8, enable_fp_fusion=False)
    return q, s


def postquant():
    source = Path(ck.__file__).read_bytes()
    tree = ast.parse(source)
    node = copy.deepcopy(next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == 'int8_linear'))
    sites = [i for i, n in enumerate(node.body) if isinstance(n, ast.If) and isinstance(n.test, ast.Name) and n.test.id == 'convrot']
    assert len(sites) == 1
    original = ast.dump(node, include_attributes=False)
    index = sites[0]
    quant_branch = node.body[index]
    node.name = '_accuracy_postquant'
    node.args.kwonlyargs += [ast.arg(arg='x_qdata'), ast.arg(arg='x_scale')]
    node.args.kw_defaults += [None, None]
    node.body[index] = ast.copy_location(ast.Pass(), quant_branch)
    inverse = copy.deepcopy(node)
    inverse.name = 'int8_linear'
    inverse.args.kwonlyargs = inverse.args.kwonlyargs[:-2]
    inverse.args.kw_defaults = inverse.args.kw_defaults[:-2]
    inverse.body[index] = quant_branch
    assert ast.dump(inverse, include_attributes=False) == original
    namespace = dict(vars(ck))
    exec(compile(ast.fix_missing_locations(ast.Module(body=[node], type_ignores=[])), '<accuracy-postquant>', 'exec'), namespace)
    return namespace[node.name], {'ck_sha256': hashlib.sha256(source).hexdigest(), 'inverse_AST_exact': True}
