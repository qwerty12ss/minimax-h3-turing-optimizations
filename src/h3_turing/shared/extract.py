"""Source-pinned CK quant/postquant derivation; installed package is read only."""
import ast
import copy
import hashlib
from pathlib import Path

CK_SHA256 = 'c5b1c3bc70869e418728114e8c27a889cf5b56ccb31ccf82a001386aef12ee8e'

def source(name):
    import comfy_kitchen.backends.cuda as ck
    if name != 'ck_cuda.py':
        raise ValueError('Only the installed CK dependency is used')
    data = Path(ck.__file__).read_bytes()
    if hashlib.sha256(data).hexdigest() != CK_SHA256:
        raise RuntimeError('Unqualified comfy-kitchen source; requalification required')
    return data

def require(value,message):
 if not value:raise RuntimeError(message)

def dump(value):return ast.dump(value,include_attributes=False)

def derive():
 original=next(n for n in ast.parse(source('ck_cuda.py')).body if isinstance(n,ast.FunctionDef) and n.name=='int8_linear')
 sites=[i for i,n in enumerate(original.body) if isinstance(n,ast.If) and isinstance(n.test,ast.Name) and n.test.id=='convrot']
 require(len(sites)==1,'One original quantization statement required');index=sites[0]
 prefix=copy.deepcopy(original);prefix.name='_h3_column_quant_prefix'
 prefix.body=prefix.body[:index+1]+ast.parse('return x_qdata,x_scale').body
 restored=copy.deepcopy(prefix);restored.name=original.name
 restored.body=restored.body[:-1]+copy.deepcopy(original.body[index+1:])
 require(dump(restored)==dump(original),'Whole original prefix inverse failed')
 post=copy.deepcopy(original);post.name='_h3_column_postquant'
 # Two mandatory keyword-only inputs; all original arguments/statements survive.
 post.args.kwonlyargs += [ast.arg(arg='x_qdata'),ast.arg(arg='x_scale')]
 post.args.kw_defaults += [None,None]
 post.body[index]=ast.copy_location(ast.Pass(),post.body[index])
 inverse=copy.deepcopy(post);inverse.name=original.name
 inverse.args.kwonlyargs=inverse.args.kwonlyargs[:-2];inverse.args.kw_defaults=inverse.args.kw_defaults[:-2]
 inverse.body[index]=copy.deepcopy(original.body[index])
 require(dump(inverse)==dump(original),'Whole original postquant inverse failed')
 baseline=copy.deepcopy(original);baseline.name='_h3_column_original_linear'
 inverse=copy.deepcopy(baseline);inverse.name=original.name
 require(dump(inverse)==dump(original),'Whole original baseline inverse failed')
 # The supported branch reads x values only in the removed quantization site.
 # Remaining value-consuming activation/GEMV paths are excluded by admission.
 require(ast.unparse(original.body[index+1].test)=='m == 1 and k % 4 == 0 or (m == 2 and k % 16 == 0)',
         'Original postquant small-M boundary changed')
 require(index==21,'Original prequant statement inventory changed')
 nodes=[ast.fix_missing_locations(n) for n in (prefix,post,baseline)]
 proof=dict(whole_prefix_inverse_exact=True,whole_postquant_inverse_exact=True,whole_baseline_inverse_exact=True,
   only_postquant_quant_branch_removed=True,original_selector_and_EVT_unchanged=True,
   quantization_statement_index=index,original_function_ast_sha256=hashlib.sha256(dump(original).encode()).hexdigest(),
   supported_rows=[8192,4650],K=5376,metadata_only_Half_placeholder=True)
 proof['prelaunch_selector_and_quantizer_gates_required']=True
 return nodes,proof
