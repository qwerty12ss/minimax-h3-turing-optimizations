"""Replace exactly two central-block norm/mod/cast expressions, nothing else."""
import ast
import copy
import hashlib
from .contract import require

HELPER = '_h3_exact_norm_mod_v2_dispatch'
COMPILED_NAME = '_h3_exact_norm_mod_v2_central_block'


def dump(node):
    return ast.dump(node, include_attributes=False)


def function(source, name):
    nodes = [node for node in ast.parse(source).body if isinstance(node, ast.FunctionDef) and node.name == name]
    require(len(nodes) == 1, 'Expected one source-derived function: ' + name)
    return copy.deepcopy(nodes[0])


def denominator_node(core_source):
    node = function(core_source, '_rms')
    expected = function('def _rms(t,w,eps):\n return t * torch.rsqrt(t.float().pow(2).mean(-1,keepdim=True)+eps).to(t.dtype) * w\n', '_rms')
    require(dump(node) == dump(expected), 'Original Torch RMS expression changed')
    inverse = copy.deepcopy(node.body[0].value.left.right)
    result = function('def denominator(t,eps):\n return None\n', 'denominator')
    result.body[0].value = inverse
    return ast.fix_missing_locations(result)


def make_denominator(core_source, torch):
    node = denominator_node(core_source)
    namespace = {'torch': torch}
    exec(compile(ast.Module(body=[node], type_ignores=[]), '<source-derived-original-H3-denominator>', 'exec'), namespace)
    return namespace['denominator']


def transform_central(source):
    original = function(source, 'tp2_block')
    replacements = []
    for norm, shift, scale in ((1, 0, 1), (2, 3, 4)):
        expression = f"_mod_scale_shift(core._rms(x0, sh['n{norm}'][0], sh['eps{norm}']), m[{shift}], m[{scale}], mod_segments).to(torch.float16)"
        replacements.append((ast.parse(expression, mode='eval').body,
            ast.parse(f"{HELPER}(x0, sh['n{norm}'][0], sh['eps{norm}'], m[{shift}], m[{scale}], mod_segments)", mode='eval').body))
    changed = copy.deepcopy(original)
    counts = [0, 0]
    class Replace(ast.NodeTransformer):
        def visit_Assign(self, node):
            for index, (before, after) in enumerate(replacements):
                if dump(node.value) == dump(before):
                    require(len(node.targets) == 1 and isinstance(node.targets[0], ast.Name) and node.targets[0].id == 'h0',
                            'Expected h0 assignment only')
                    counts[index] += 1
                    node.value = copy.deepcopy(after)
            return node
    changed = Replace().visit(changed)
    require(counts == [1, 1], 'Exactly one norm1 and norm2 expression must change')
    # Undo the two changed RHS nodes and compare every remaining AST node.
    reverted = copy.deepcopy(changed)
    class Revert(ast.NodeTransformer):
        def visit_Assign(self, node):
            for before, after in replacements:
                if dump(node.value) == dump(after):
                    node.value = copy.deepcopy(before)
            return node
    require(dump(Revert().visit(reverted)) == dump(original), 'Central-block mutation escaped the two expressions')
    changed.name = COMPILED_NAME
    ast.fix_missing_locations(changed)
    provenance = {'replaced_rhs': 2, 'original_function_ast_sha256': hashlib.sha256(dump(original).encode()).hexdigest(),
                  'transformed_function_ast_sha256': hashlib.sha256(dump(changed).encode()).hexdigest(),
                  'unchanged_ast_after_reverting_two_rhs': True}
    return changed, provenance


def source_reference_functions(core_source, model_source, torch):
    nodes = [function(core_source, '_rms'), function(model_source, '_mod_row'), function(model_source, '_mod_scale_shift')]
    namespace = {'torch': torch}
    exec(compile(ast.fix_missing_locations(ast.Module(body=nodes, type_ignores=[])), '<unchanged-H3-reference-functions>', 'exec'), namespace)
    return namespace['_rms'], namespace['_mod_scale_shift']
