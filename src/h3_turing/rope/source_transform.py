"""Frozen original function plus exactly three AST statement edits; no imports/GPU."""
import ast
import copy
import hashlib
from .contract import CORE_SHA, require

def dump(value):
    return ast.dump(value, include_attributes=False)

def function_tree(source):
    tree = ast.parse(source)
    matches = [x for x in tree.body if isinstance(x, ast.FunctionDef) and x.name == '_qkv_rope']
    require(len(matches) == 1, 'Exactly one original QKV function required')
    return matches[0]

def transform(source):
    raw = source if isinstance(source, bytes) else source.encode('utf-8')
    require(hashlib.sha256(raw).hexdigest() == CORE_SHA, 'Original full native source differs')
    original = function_tree(source)
    value = copy.deepcopy(original)
    loop = [x for x in value.body if isinstance(x, ast.For)]
    require(len(loop) == 1, 'Original QKV chunk loop differs')
    expected = ast.parse('\nqo.ck.rms_rope_split_half_(q, k, rf[:, start:end, ...], qw, kw, epsilon=att.q_norm.eps, rot_dim=rf.shape[-3]*2)\noutput[0][:, :, start:end].copy_(q.permute(0, 2, 1, 3))\noutput[1][:, :, start:end].copy_(k.permute(0, 2, 1, 3))\n').body
    positions = [i for i, x in enumerate(loop[0].body) if dump(x) == dump(expected[0])]
    require(len(positions) == 1, 'Original RMS call differs')
    pos = positions[0]
    require([dump(x) for x in loop[0].body[pos:pos + 3]] == [dump(x) for x in expected], 'Original Q/K copy statements differ')
    replacement = ast.parse('\n_h3_qk_direct(q, k, rf[:, start:end, ...], qw, kw, output[:2], start, att.q_norm.eps, rf.shape[-3]*2)\n').body[0]
    loop[0].body[pos:pos + 3] = [replacement]
    check = copy.deepcopy(value)
    restored_loop = [x for x in check.body if isinstance(x, ast.For)][0]
    restored_loop.body[pos:pos + 1] = copy.deepcopy(expected)
    require(dump(check) == dump(original), 'Unexpected QKV source edit')
    return ast.fix_missing_locations(ast.Module(body=[value], type_ignores=[]))

def compile_candidate(source, live_globals, helper):
    value = transform(source).body[0]
    factory = ast.FunctionDef(name='_h3_private_factory', args=ast.arguments(posonlyargs=[], args=[ast.arg(arg='_h3_qk_direct')], vararg=None, kwonlyargs=[], kw_defaults=[], kwarg=None, defaults=[]), body=[value, ast.Return(value=ast.Name(id='_qkv_rope', ctx=ast.Load()))], decorator_list=[], returns=None, type_comment=None, type_params=[])
    module = ast.fix_missing_locations(ast.Module(body=[factory], type_ignores=[]))
    local = {}
    exec(compile(module, '<h3-qk-direct-output-source>', 'exec'), live_globals, local)
    candidate = local['_h3_private_factory'](helper)
    require(candidate.__globals__ is live_globals, 'Original live global namespace was not preserved')
    return candidate
