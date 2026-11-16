"""Two-output FC2 preparation; no installed module mutations."""
import ast
import copy
import ctypes
import hashlib
from pathlib import Path
import threading

import torch
import torch.nn.functional as F
import comfy_kitchen.backends.cuda as ck

HERE = Path(__file__).resolve().parent
CK_SHA = 'c5b1c3bc70869e418728114e8c27a889cf5b56ccb31ccf82a001386aef12ee8e'
CORE_SHA = 'e7c45a7f44d4ce81c9b20786a5d5eb838ad89c284859f000a65fa4a3fbcac4a5'
LIB = ctypes.CDLL(str(HERE / 'libjoint.so'))
LIB.h3_joint.argtypes = [ctypes.c_void_p] * 5 + [ctypes.c_int] * 3 + [ctypes.c_void_p]
LIB.h3_joint.restype = ctypes.c_int


class Tables:
    def __init__(self):
        self.tables = {}
        self.lock = threading.Lock()

    def get(self, device):
        with torch.cuda.device(device):
            stream = torch.cuda.current_stream(device)
            with self.lock:
                if device.index not in self.tables:
                    bits = torch.arange(65536, device=device, dtype=torch.int32).to(torch.int16)
                    table = F.silu(bits.view(torch.float16).to(torch.float32))
                    ready = torch.cuda.Event()
                    ready.record(stream)
                    self.tables[device.index] = table, ready
                table, ready = self.tables[device.index]
            stream.wait_event(ready)
            table.record_stream(stream)
            return table


def prepare(g, tables, warps=8):
    assert g.dtype is torch.float16 and g.is_cuda and g.is_contiguous()
    assert g.ndim == 2 and g.shape[1] == 14336
    m, k = len(g), 7168
    with torch.cuda.device(g.device):
        lut = tables.get(g.device)
        a = torch.empty((m, k), dtype=torch.float16, device=g.device)
        q = torch.empty((m, k), dtype=torch.int8, device=g.device)
        s = torch.empty((m, 1), dtype=torch.float32, device=g.device)
        status = LIB.h3_joint(g.data_ptr(), a.data_ptr(), q.data_ptr(), s.data_ptr(),
                              lut.data_ptr(), m, k, warps,
                              torch.cuda.current_stream(g.device).cuda_stream)
        assert status == 0, status
        return a, q, s


def ck_functions():
    source = Path(ck.__file__).read_bytes()
    assert hashlib.sha256(source).hexdigest() == CK_SHA
    original = next(n for n in ast.parse(source).body if isinstance(n, ast.FunctionDef) and n.name == 'int8_linear')
    index, = [i for i, n in enumerate(original.body) if isinstance(n, ast.If) and isinstance(n.test, ast.Name) and n.test.id == 'convrot']
    prefix = copy.deepcopy(original)
    prefix.name = '_joint_quant_reference'
    prefix.body = prefix.body[:index + 1] + ast.parse('return x_qdata, x_scale').body
    post = copy.deepcopy(original)
    post.name = '_joint_postquant'
    post.args.kwonlyargs += [ast.arg(arg='x_qdata'), ast.arg(arg='x_scale')]
    post.args.kw_defaults += [None, None]
    removed = post.body[index]
    post.body[index] = ast.copy_location(ast.Pass(), removed)
    inverse = copy.deepcopy(post)
    inverse.name = original.name
    inverse.args.kwonlyargs = inverse.args.kwonlyargs[:-2]
    inverse.args.kw_defaults = inverse.args.kw_defaults[:-2]
    inverse.body[index] = removed
    assert ast.dump(inverse, include_attributes=False) == ast.dump(original, include_attributes=False)
    namespace = dict(vars(ck))
    exec(compile(ast.fix_missing_locations(ast.Module(body=[prefix, post], type_ignores=[])), '<joint-CK-derivation>', 'exec'), namespace)
    return namespace[prefix.name], namespace[post.name], dict(ck_sha256=CK_SHA, postquant_inverse_AST_exact=True)


def mlp_function(core, preparation, fc2):
    source = Path(core.__file__).read_bytes()
    assert hashlib.sha256(source).hexdigest() == CORE_SHA
    node = copy.deepcopy(next(n for n in ast.parse(source).body if isinstance(n, ast.FunctionDef) and n.name == '_mlp_chunked'))
    original = copy.deepcopy(node)
    loop, = [n for n in node.body if isinstance(n, ast.For)]
    assert [ast.unparse(n) for n in loop.body[2:6]] == [
        'gate, up = g.chunk(2, dim=-1)',
        'act = F.silu(gate.to(torch.float32)) * up.to(torch.float32)',
        'del g, gate, up',
        "y = _lin((act / KC2).to(torch.float16), sh['fc2'][rank], sh['lfc2'][rank])"]
    saved = copy.deepcopy(loop.body[2:6])
    loop.body[2:6] = ast.parse("act, q, scales = _joint_prepare(g, sh['fc2'][rank])\ndel g\ny = _joint_fc2(act, q, scales, sh['fc2'][rank], sh['lfc2'][rank])").body
    assert ast.unparse(loop.body[-1]) == 'del act, y'
    loop.body[-1] = ast.parse('del act, q, scales, y').body[0]
    inverse = copy.deepcopy(node)
    inv_loop, = [n for n in inverse.body if isinstance(n, ast.For)]
    inv_loop.body[2:5] = saved
    inv_loop.body[-1] = ast.parse('del act, y').body[0]
    assert ast.dump(inverse, include_attributes=False) == ast.dump(original, include_attributes=False)
    # Use the original live namespace: _lin is selected temporarily by the parent.
    scope = {}
    factory = ast.parse('def _factory(_joint_prepare, _joint_fc2):\n    pass').body[0]
    factory.body = [node, ast.Return(ast.Name(id=node.name, ctx=ast.Load()))]
    exec(compile(ast.fix_missing_locations(ast.Module(body=[factory], type_ignores=[])), '<joint-MLP>', 'exec'), vars(core), scope)
    return scope['_factory'](preparation, fc2), dict(core_sha256=CORE_SHA, mlp_inverse_AST_exact=True, KC2=float(core.KC2), g_released_before_FC2=True)
