"""H3 原生 INT8 ConvRot + baked PDD8 双卡 TP2 实验节点。

算法基于 NorwinYu/minimax-h3-dual-2080ti (MIT), commit
047e06890fc599638b7ca07947f445a15f278713. 不修改 ComfyUI 核心文件。
"""
import contextvars
import functools
import logging
import os
import time
import weakref

import torch

from .tp2 import INT8_LAYOUT, shard_block
from .central_block import tp2_block
from .sparse_attention import MODES, SparsePolicy, resolve_step

LOG = "[H3 Native INT8 TP2 Central Sparse]"
_ACTIVE = contextvars.ContextVar("h3_native_int8_tp2_central_sparse_active", default=False)
_HOOKED = False


class _State:
    def __init__(self, patcher, dev0, dev1, reserve_gib, qkv_chunk_tokens, mlp_chunk_tokens, policy):
        self.patcher = patcher
        self.policy = policy
        self.qkv_chunk_tokens = int(qkv_chunk_tokens)
        self.mlp_chunk_tokens = int(mlp_chunk_tokens)
        self.devices = (dev0, dev1)
        self.reserve = int(reserve_gib * 1024**3)
        self.shards = []
        self.bytes_per_rank = (0, 0)
        self.started = 0.0

    def prepare(self, loaded):
        if self.shards:
            return
        import comfy.model_management as mm
        dm = self.patcher.model.diffusion_model
        # 先卸载文本编码器等缓存，再分配固定驻留的两半 backbone。
        for device in self.devices:
            mm.free_memory(self.reserve, device, keep_loaded=[loaded])
        t0 = time.perf_counter()
        try:
            for block in dm.blocks:
                shard = shard_block(block, *self.devices)
                shard["qkv_chunk_tokens"] = self.qkv_chunk_tokens
                shard["mlp_chunk_tokens"] = self.mlp_chunk_tokens
                shard["sparse_policy"] = self.policy
                self.shards.append(shard)
            counts = [0, 0]
            for sh in self.shards:
                for key in ("qkv", "op", "fc1", "fc2"):
                    for rank, weight in enumerate(sh[key]):
                        counts[rank] += weight._qdata.numel() * weight._qdata.element_size()
                        counts[rank] += weight._params.scale.numel() * weight._params.scale.element_size()
                for key in ("n1", "n2", "qn", "kn"):
                    for rank, weight in enumerate(sh[key]):
                        counts[rank] += weight.numel() * weight.element_size()
            self.bytes_per_rank = tuple(counts)
            for device in self.devices:
                torch.cuda.synchronize(device)
            logging.info("%s %d blocks ready in %.2fs; resident shards GiB=%s",
                         LOG, len(self.shards), time.perf_counter()-t0,
                         tuple(round(n/1024**3, 3) for n in counts))
        except BaseException:
            self.release()
            raise

    def release(self):
        self.shards.clear()
        self.bytes_per_rank = (0, 0)

    def block_forward(self, index):
        def forward(x, t_emb, mod_segments, rope_freqs,
                    transformer_options=None, attention=None, **kwargs):
            if len(self.shards) != 50:
                raise RuntimeError("TP2 shards were not prepared through model management")
            if index == 0:
                self.started = time.perf_counter()
            out = getattr(self, "block_impl", tp2_block)(self.shards[index], x, t_emb, mod_segments, rope_freqs,
                            transformer_options=transformer_options, attention=attention, **kwargs)
            if index == 49 and os.environ.get("H3_TP2_STEP_TIMING") == "1":
                for device in self.devices:
                    torch.cuda.synchronize(device)
                logging.info("%s 50-block loop %.3fs", LOG, time.perf_counter()-self.started)
            return out
        forward._h3_native_int8_tp2_central_sparse = index
        return forward


def _install_hooks():
    global _HOOKED
    if _HOOKED:
        return
    import comfy.model_management as mm
    import comfy.model_prefetch as mp

    orig_malloc = mp.malloc_graph_enabled

    @functools.wraps(orig_malloc)
    def malloc_graph_enabled(device):
        return False if _ACTIVE.get() else orig_malloc(device)

    mp.malloc_graph_enabled = malloc_graph_enabled
    orig_load = mm.LoadedModel.model_load
    orig_req = mm.LoadedModel.model_memory_required
    orig_loaded = mm.LoadedModel.model_loaded_memory
    orig_unload = mm.LoadedModel.model_unload

    @functools.wraps(orig_load)
    def model_load(loaded, *args, **kwargs):
        state = getattr(loaded.model, "_h3_native_int8_tp2_central_sparse", None)
        if state is None:
            return orig_load(loaded, *args, **kwargs)
        # 仅应用 object patches/injections，禁止把整模型搬到主卡。
        loaded.model.model_patches_to(loaded.device)
        loaded.model.model_patches_to(loaded.model.model_dtype())
        real_model = loaded.model.patch_model(load_weights=False)
        # 未分片模块保留 Comfy manual-cast/offload 路径，包括 PDD 输出头。
        for module in real_model.modules():
            if hasattr(module, "comfy_cast_weights"):
                module.comfy_cast_weights = True
        state.prepare(loaded)
        loaded.real_model = weakref.ref(real_model)
        loaded.model_finalizer = weakref.finalize(real_model, mm.cleanup_models)
        loaded.model_finalizer.atexit = False
        return real_model

    @functools.wraps(orig_req)
    def model_memory_required(loaded, device):
        state = getattr(loaded.model, "_h3_native_int8_tp2_central_sparse", None)
        if state is None:
            return orig_req(loaded, device)
        return 0 if state.shards else state.reserve

    @functools.wraps(orig_loaded)
    def model_loaded_memory(loaded):
        state = getattr(loaded.model, "_h3_native_int8_tp2_central_sparse", None)
        if state is None:
            return orig_loaded(loaded)
        return state.bytes_per_rank[0]

    @functools.wraps(orig_unload)
    def model_unload(loaded, memory_to_free=None, unpatch_weights=True):
        state = getattr(loaded.model, "_h3_native_int8_tp2_central_sparse", None)
        if state is None:
            return orig_unload(loaded, memory_to_free, unpatch_weights)
        # TP2 以整个双卡模型为卸载单位，不允许只清理主卡半片。
        state.release()
        return orig_unload(loaded, None, unpatch_weights)

    mm.LoadedModel.model_load = model_load
    mm.LoadedModel.model_memory_required = model_memory_required
    mm.LoadedModel.model_loaded_memory = model_loaded_memory
    mm.LoadedModel.model_unload = model_unload
    _HOOKED = True
    logging.info("%s installed scoped model-management and aimdo hooks", LOG)


def sparse_forward(original, policy):
    @functools.wraps(original)
    def forward(*args, **kwargs):
        # Star7's finite-output guard exposes (*args, **kwargs), so signature
        # binding hides the real options inside its VAR_KEYWORD argument.
        # Core H3 passes options by keyword; its native positional slot is 3.
        positional = 'transformer_options' not in kwargs and len(args) > 3
        current = args[3] if positional else kwargs.get('transformer_options')
        if not isinstance(current, dict):
            raise RuntimeError('TP2 hybrid requires actual H3 transformer_options from the sampler')
        options = dict(current)
        options['prefetch_dynamic_vbars'] = False
        options['_h3_tp2_sparse_step'] = resolve_step(options)
        logging.info('%s model step=%d/8 backend=%s mode=%s', LOG,
                     options['_h3_tp2_sparse_step'][0], options['_h3_tp2_sparse_step'][1], policy.mode)
        if positional:
            args = args[:3] + (options,) + args[4:]
        else:
            kwargs = dict(kwargs, transformer_options=options)
        token = _ACTIVE.set(True)
        try:
            return original(*args, **kwargs)
        finally:
            _ACTIVE.reset(token)
    return forward


class MiniMaxH3NativeINT8TP2CentralSparse:
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {
            "model": ("MODEL",),
            "attention_mode": (list(MODES), {"default": MODES[0]}),
            "sol_tau": ("FLOAT", {"default": 1.0, "min": -2.0, "max": 4.0, "step": 0.1}),
            "sla_sparsity": ("FLOAT", {"default": 0.85, "min": 0.0, "max": 0.95, "step": 0.05}),
            "qkv_chunk_tokens": ("INT", {"default": 8192, "min": 256, "max": 32768, "step": 256}),
            "mlp_chunk_tokens": ("INT", {"default": 8192, "min": 256, "max": 32768, "step": 256}),
            "replica_device": ("STRING", {"default": "cuda:1"}),
            "reserve_gib": ("FLOAT", {"default": 18.0, "min": 12.0, "max": 21.0, "step": 0.5}),
        }}

    RETURN_TYPES = ("MODEL",)
    FUNCTION = "apply"
    CATEGORY = "MiniMaxH3/Experimental"
    DESCRIPTION = "Native INT8 ConvRot baked-PDD8 TP2 with bounded QKV/MLP projection chunks. CK steps0,1,6,7; Sol/SLA steps2-5. Residual, norm and modulation on primary only."

    def apply(self, model, qkv_chunk_tokens=8192, mlp_chunk_tokens=8192, replica_device="cuda:1", reserve_gib=18.0, attention_mode=MODES[0], sol_tau=1.0, sla_sparsity=0.85):
        import comfy.model_management as mm
        if model.is_dynamic():
            raise RuntimeError("Native INT8 TP2 requires --disable-dynamic-vram")
        if getattr(model, "patches", {}):
            raise RuntimeError("Native INT8 TP2 requires baked backbone and object-patched PDD heads; runtime weight patches unsupported")
        if any(getattr(model, key, None) for key in ("_h3_native_int8_tp2", "_h3_native_int8_tp2_sparse", "_h3_native_int8_tp2_chunked", "_h3_native_int8_tp2_central", "_h3_native_int8_tp2_central_sparse")):
            raise RuntimeError("TP2 is already applied. Restart when switching single-GPU/TP2 graphs")
        result = model.clone()
        dm = result.model.diffusion_model
        dev0 = torch.device(str(result.load_device or mm.get_torch_device()))
        dev1 = torch.device(replica_device)
        if dev0.index is None:
            dev0 = torch.device("cuda", torch.cuda.current_device())
        assert dev0.type == dev1.type == "cuda" and dev0 != dev1, (dev0, dev1)
        assert torch.cuda.device_count() == 2, "Mask this process to the two 2080 Ti GPUs only"
        assert torch.cuda.can_device_access_peer(dev0.index, dev1.index), "CUDA P2P is required"
        assert len(dm.blocks) == 50, len(dm.blocks)
        for index, block in enumerate(dm.blocks):
            for name, layer in (("qkv", block.attn.qkv_proj), ("op", block.attn.out_proj),
                                ("fc1", block.mlp.fc1), ("fc2", block.mlp.fc2)):
                weight = layer.weight
                assert getattr(weight, "_layout_cls", None) == INT8_LAYOUT, (index, name, type(weight))
                assert weight._params.convrot, (index, name, "ConvRot required")
                assert not weight.is_meta and weight.device.type == "cpu", (index, name, weight.device)
        if not 256 <= int(qkv_chunk_tokens) <= 32768 or not 256 <= int(mlp_chunk_tokens) <= 32768:
            raise ValueError("Invalid chunk size")
        policy = SparsePolicy(attention_mode, sol_tau, sla_sparsity)
        policy.prepare_runtime((dev0, dev1))
        state = _State(result, dev0, dev1, reserve_gib, qkv_chunk_tokens, mlp_chunk_tokens, policy)
        result._h3_native_int8_tp2_central_sparse = state
        result.multi_device_managed = True
        result.model_options.setdefault("transformer_options", {})["prefetch_dynamic_vbars"] = False
        for index in range(50):
            result.add_object_patch("diffusion_model.blocks.%d.forward" % index, state.block_forward(index))

        original = result.object_patches.get("diffusion_model.forward", dm.forward)
        result.add_object_patch("diffusion_model.forward", sparse_forward(original, policy))
        _install_hooks()
        logging.info("%s configured %s/%s qkv_chunk=%d mlp_chunk=%d central_fp32_residual; shards initialize lazily at sampler model_load", LOG, dev0, dev1, qkv_chunk_tokens, mlp_chunk_tokens)
        return (result,)


NODE_CLASS_MAPPINGS = {"MiniMaxH3NativeINT8TP2CentralSparse": MiniMaxH3NativeINT8TP2CentralSparse}
NODE_DISPLAY_NAME_MAPPINGS = {"MiniMaxH3NativeINT8TP2CentralSparse": "MiniMax-H3 Native INT8 ConvRot TP2 Central FP32 Residual + Hybrid Sparse (experimental)"}
