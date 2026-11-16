"""Derive only instance preparation from the pinned original runtime source.

This module imports no Torch and executes no preparation. It proves a complete
inverse for all semantic changes; core math/streams and the 108-patch loop remain
original. The private owner supplies transactional publication and rollback.
"""
import ast
import copy
import hashlib
from pathlib import Path

SOURCE_SHA='863df6b6f9ff04618c0936b964f7976cbb0d2d3d1feea4fbd1e03bd096106d10'
SOURCE=Path(__file__).resolve().parent/'source/runtime_pdd.py'


def require(value,message):
    if not value:raise RuntimeError(message)


def dump(value):return ast.dump(value,include_attributes=False)
def stmt(text):return ast.parse(text).body[0]
def expr(text):return ast.parse(text,mode='eval').body


CHANGES=(
    ('stmt','counts = [0, 0]','counts, pending, source_owners = _candidate_pending(state)'),
    ('stmt','weight = state.patcher.patch_weight_to_device(key, device_to=state.devices[0], return_weight=True)',
            'weight = original'),
    ('expr','not weight._params.convrot or weight.device != state.devices[0]',
            "not weight._params.convrot or weight.device.type != 'cpu'"),
    ('expr','state._runtime_core.shard_block(shadow, *state.devices, lora=None)',
            'state._runtime_core.shard_block(shadow, *state.devices, lora=_candidate_loras(state,index,source_owners))'),
    ('expr','any(item is not None for key in ("lqkv", "lop", "lfc1", "lfc2") for item in shard[key])',
            'any(item is None for key in ("lqkv", "lop", "lfc1", "lfc2") for item in shard[key])'),
    ('expr','"Activation LoRA bypass unexpectedly enabled"','"Every original Half A/B bypass must be enabled"'),
    ('stmt','state.shards.append(shard)','_candidate_append(state,pending,counts,shard)'),
    ('stmt','state.bytes_per_rank = tuple(counts)','_candidate_validate_pending(state,pending,counts,source_owners)'),
    ('expr','"prepared"','"prepared_unpublished"'),
    ('expr','len(state.shards)','len(pending)'),
    ('expr','list(state.bytes_per_rank)','list(counts)'),
    ('expr','"Comfy resident runtime merge/requant API; dynamic offload rounding may differ"',
            '"Original INT8 base plus original core Half A/B activation bypass; changed math"'),
    ('stmt','_release(state)','_candidate_rollback(state,locals().get("pending"),locals().get("source_owners"))'),
)


def _replace(tree,old,new,count=1):
    wanted=dump(old);hits=[]
    class Replace(ast.NodeTransformer):
        def visit(self,node):
            if dump(node)==wanted:
                hits.append(node)
                return ast.copy_location(copy.deepcopy(new),node)
            return super().visit(node)
    result=Replace().visit(tree)
    require(len(hits)==count,'Exact preparation mutation count changed: '+wanted+' got '+str(len(hits)))
    return result


def original():
    require(SOURCE.is_file() and not SOURCE.is_symlink() and 0<SOURCE.stat().st_size<1024**2,
            'Pinned bounded runtime preparation source required')
    raw=SOURCE.read_bytes();require(hashlib.sha256(raw).hexdigest()==SOURCE_SHA,'Original preparation source changed')
    nodes=[n for n in ast.parse(raw).body if isinstance(n,ast.FunctionDef) and n.name=='_prepare']
    require(len(nodes)==1,'One original runtime preparation required')
    return nodes[0]


def derived():
    source=original();candidate=copy.deepcopy(source);pairs=[]
    # len(state.shards) appears once in cached admission and once in its final
    # log. Only the final logging keyword is replaced; cached checks stay exact.
    deferred=('len(state.shards)','len(pending)')
    for kind,old,new in CHANGES:
        if (old,new)==deferred:continue
        parse=stmt if kind=='stmt' else expr
        before,after=parse(old),parse(new)
        candidate=_replace(candidate,before,after);pairs.append((before,after))
    emit=[n for n in ast.walk(candidate) if isinstance(n,ast.Call) and isinstance(n.func,ast.Name)
          and n.func.id=='_emit' and n.args and isinstance(n.args[0],ast.Constant)
          and n.args[0].value=='prepared_unpublished']
    require(len(emit)==1,'One final unpublished event required')
    keywords={k.arg:k for k in emit[0].keywords}
    for name,value in (('full_weight_patches_before_sharding',False),('full_projection_return_weight',False)):
        require(dump(keywords[name].value)==dump(ast.Constant(True)),'Original merge flag changed')
        keywords[name].value=ast.Constant(value)
    require(dump(keywords['blocks'].value)==dump(expr(deferred[0])),'Original final block count changed')
    keywords['blocks'].value=expr(deferred[1])
    inverse=copy.deepcopy(candidate)
    reverse_emit=next(n for n in ast.walk(inverse) if isinstance(n,ast.Call) and isinstance(n.func,ast.Name)
                     and n.func.id=='_emit' and n.args and isinstance(n.args[0],ast.Constant)
                     and n.args[0].value=='prepared_unpublished')
    reverse_keywords={k.arg:k for k in reverse_emit.keywords}
    for name in ('full_weight_patches_before_sharding','full_projection_return_weight'):
        reverse_keywords[name].value=ast.Constant(True)
    reverse_keywords['blocks'].value=expr(deferred[0])
    for before,after in reversed(pairs):inverse=_replace(inverse,after,before)
    require(dump(inverse)==dump(source),'Whole original preparation inverse failed')
    return ast.fix_missing_locations(candidate)


def compile_prepare(runtime,helpers):
    """A private function namespace; never assign original module globals."""
    require(set(helpers)=={'_candidate_pending','_candidate_loras','_candidate_append',
                          '_candidate_validate_pending','_candidate_rollback','_emit'},
            'Exact private preparation helpers required')
    namespace=dict(vars(runtime));namespace.update(helpers)
    tree=ast.Module(body=[derived()],type_ignores=[])
    exec(compile(tree,str(SOURCE),'exec',dont_inherit=True),namespace)
    return namespace['_prepare']


def compile_release(runtime,emit):
    # Identical release body/protocol, with only the private diagnostic emitter
    # bound differently so original COMMON does not falsely label A/B as merged.
    raw=SOURCE.read_bytes();require(hashlib.sha256(raw).hexdigest()==SOURCE_SHA,'Original release source changed')
    selected=[n for n in ast.parse(raw).body if isinstance(n,ast.FunctionDef) and n.name=='_release']
    require(len(selected)==1,'One original release required')
    namespace=dict(vars(runtime));namespace['_emit']=emit
    exec(compile(ast.Module(body=selected,type_ignores=[]),str(SOURCE),'exec',dont_inherit=True),namespace)
    return namespace['_release']


def audit():
    a=original();b=derived()
    return dict(original_source_sha256=SOURCE_SHA,original_prepare_AST_sha256=hashlib.sha256(dump(a).encode()).hexdigest(),
                candidate_prepare_AST_sha256=hashlib.sha256(dump(b).encode()).hexdigest(),whole_inverse_equal=True,
                original_108_patch_loop_preserved=True,full_projection_merge_calls_removed=1,
                runtime_math_or_stream_implementation_changed=False,preparation_executed=False)
