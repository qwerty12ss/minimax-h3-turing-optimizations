"""Connect the published operators to the original TP2 block with scoped ownership."""
import ast
import hashlib
import threading
from pathlib import Path

import torch
import torch.nn.functional as F

from h3_turing.norm_mod import kernel as norm_kernel, source_transform as norm_transform
from h3_turing.norm_mod.row_map import RowMapScope
from h3_turing.pointwise import scale_add
from h3_turing.rope import direct_output, source_transform as rope_transform
from h3_turing.shared.streamed import Shared
from h3_turing.sum_reduce import _sum_scaled
from h3_turing.swiglu_convrot.joint import Tables, prepare, ck_functions, mlp_function

_BLOCK_LOCK = threading.Lock()


class Optimized:
    def __init__(self, state):
        from h3_turing.h3_native_int8_tp2_central_sparse import central_block, tp2
        self.state, self.core = state, tp2
        self.rows = RowMapScope()
        self.blocks = self.norm_calls = self.rope_calls = self.joint_calls = 0
        self.original_linear = tp2._lin
        self.shared = Shared(self.linear)
        self.shared.epilogue = lambda y, d, s: scale_add(y, d, s, True, 256, 4)[0]
        source = Path(tp2.__file__).read_bytes()
        self.denominator = norm_transform.make_denominator(source, torch)
        self.rope = rope_transform.compile_candidate(source, vars(tp2), self.rope_dispatch)
        self.tables = Tables()
        _, self.post, self.ck_proof = ck_functions()
        self.mlp, self.mlp_proof = mlp_function(tp2, self.joint_prepare, self.joint_fc2)
        central_source = Path(central_block.__file__).read_bytes()
        if hashlib.sha256(central_source).hexdigest() != "2e0c2a2ca37b299042616cd1ec1f68ea9328888d6ef239f8a67e364b61beaafc":
            raise RuntimeError("Original central-block source changed")
        node, self.norm_proof = norm_transform.transform_central(central_source)
        count = 0
        for statement in node.body:
            if isinstance(statement, ast.Assign) and len(statement.targets) == 1 and isinstance(statement.targets[0], ast.Name) and statement.targets[0].id == "h1":
                if ast.unparse(statement.value) != "_peer_copy(h0, d1, st0, st1)":
                    raise RuntimeError("The original central broadcast changed")
                phase = "qkv" if count == 0 else "fc1"
                statement.value = ast.parse(f"_shared_prepare(sh, h0, {phase!r}, st0, st1)", mode="eval").body
                count += 1
        if count != 2:
            raise RuntimeError("Expected exactly two original central broadcasts")
        factory = ast.parse(f"def factory({norm_transform.HELPER}, _shared_prepare):\n    pass").body[0]
        factory.body = [node, ast.Return(ast.Name(id=node.name, ctx=ast.Load()))]
        scope = {}
        exec(compile(ast.fix_missing_locations(ast.Module(body=[factory], type_ignores=[])), "<public-shared-central>", "exec"), vars(central_block), scope)
        self.block = scope["factory"](self.norm, self.shared.prepare)

    def linear(self, x, weight, lora):
        if lora is None:
            return self.original_linear(x, weight, None)
        up, down, scale = lora
        y = F.linear(x, weight)
        delta = F.linear(F.linear(x, down), up)
        return scale_add(y, delta, scale, True, 256, 4)[0]

    def norm(self, x, weight, epsilon, shift, scale, segments):
        inverse = self.denominator(x, epsilon)
        mapping = self.rows.prepare(segments, x.shape[0], shift.shape[0], x.device, torch)
        self.norm_calls += 1
        return norm_kernel.fused_pointwise(x, inverse, weight, shift, scale, mapping, 512, 4, strict_shape=False)

    def rope_dispatch(self, q, k, frequencies, q_scale, k_scale, owners, start, epsilon, rot_dim):
        self.rope_calls += 1
        return direct_output.direct(torch, q, k, frequencies, q_scale, k_scale, owners, start, epsilon, rot_dim)

    def joint_prepare(self, g, weight):
        if len(g) <= 128:
            gate, up = g.chunk(2, dim=-1)
            return (F.silu(gate.float()) * up.float() / 256.0).half(), None, None
        self.joint_calls += 1
        return prepare(g, self.tables, 8)

    def joint_fc2(self, activation, q, scales, weight, lora):
        if q is None:
            return self.shared.original(activation, weight, lora)
        y = self.post(activation, weight._qdata, weight._params.scale, out_dtype=torch.float16,
                      convrot=True, convrot_groupsize=256, x_qdata=q, x_scale=scales)
        if lora is None:
            return y
        up, down, scale = lora
        return scale_add(y, F.linear(F.linear(activation, down), up), scale, True, 256, 4)[0]

    @staticmethod
    def sum(a, b, scale):
        if not (a.shape == b.shape and a.ndim == 2 and a.shape[1] == 5376 and len(a) >= 1
                and a.dtype == b.dtype == torch.float16 and a.is_cuda and a.device == b.device
                and a.is_contiguous() and b.is_contiguous() and scale in (64.0, 256.0)):
            raise RuntimeError("Unsupported TP2 SUM geometry/dtype/device")
        output = torch.empty(a.shape, dtype=torch.float32, device=a.device)
        with torch.cuda.device(a.device):
            _sum_scaled[((a.numel() + 1023) // 1024,)](a, b, output, a.numel(), scale, 1024,
                                                   num_warps=4, enable_fp_fusion=False)
        return output

    def __call__(self, sh, x, emb, segments, frequencies, **kwargs):
        if not _BLOCK_LOCK.acquire(blocking=False):
            raise RuntimeError("Concurrent TP2 model executions are unsupported")
        names = ("_lin", "_sum_scaled_fp32", "_qkv_rope", "_mlp_chunked")
        saved = [getattr(self.core, name) for name in names]
        try:
            if sh is self.state.shards[0]:
                self.rows.begin(segments)
            for name, function in zip(names, [self.shared.linear, self.sum, self.rope, self.mlp]):
                setattr(self.core, name, function)
            result = self.block(sh, x, emb, segments, frequencies, **kwargs)
            self.shared.finish()
            self.blocks += 1
            if sh is self.state.shards[-1]:
                from comfy_execution.utils import get_executing_context
                from .validation import record_optimizer
                context = get_executing_context()
                if context is None:
                    raise RuntimeError("Expected a ComfyUI execution context for the optimizer receipt")
                record_optimizer(context.prompt_id, self.report())
            return result
        finally:
            for name, function in zip(names, saved):
                setattr(self.core, name, function)
            _BLOCK_LOCK.release()

    def report(self):
        return dict(blocks=self.blocks, norm_calls=self.norm_calls, rope_calls=self.rope_calls,
                    joint_calls=self.joint_calls, shared_counts=dict(self.shared.counts))
