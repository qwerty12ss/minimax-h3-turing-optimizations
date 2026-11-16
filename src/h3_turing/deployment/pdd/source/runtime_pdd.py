"""Private PDD runtime patching before native INT8 TP2 sharding.

No checkpoint is written. Importing this module performs no Torch/CUDA work.
Each full TP2 projection is patched by the installed Comfy API and requantized
before slicing. The remaining parameters use that API's normal backup/restore
path. This is not an activation LoRA bypass.
"""
import dataclasses
import inspect
import json
import logging
from pathlib import Path
import sys
import threading
import time
import types


PREFIX = "[H3 Runtime PDD TP2]"
STATE_ATTRIBUTE = "_h3_native_int8_tp2_central_sparse"
METHOD = "comfy_runtime_merge_requant_full_weight_then_tp2"
_LOCK = threading.Lock()
LARGE_SUFFIXES = {"qkv": "attn.qkv_proj.weight", "op": "attn.out_proj.weight",
                  "fc1": "mlp.fc1.weight", "fc2": "mlp.fc2.weight"}
COMMON = {"method": METHOD, "offline_bake": False, "activation_bypass": False,
          "lora_strength": 1.0, "head_strength": 1.0, "parameter_keys": 308,
          "converted_modules": 258, "tp2_weight_keys": 200,
          "adaln_weight_keys": 50, "adaln_bias_keys": 50,
          "refiner_weight_keys": 8, "raw_lora_modules": 362}


def _emit(event, **fields):
    logging.info("%s %s", PREFIX, json.dumps({"event": event, **COMMON, **fields}, sort_keys=True))


def expected_keys():
    large = {f"diffusion_model.blocks.{index}.{suffix}"
             for index in range(50) for suffix in LARGE_SUFFIXES.values()}
    adaln_weight = {f"diffusion_model.blocks.{index}.adaln_proj.linear.weight" for index in range(50)}
    adaln_bias = {f"diffusion_model.blocks.{index}.adaln_proj.linear.bias" for index in range(50)}
    refiner = {f"diffusion_model.token_refiner.blocks.{index}.{suffix}"
               for index in range(2) for suffix in LARGE_SUFFIXES.values()}
    return large, adaln_weight, adaln_bias, refiner


def _native_module():
    matches = []
    for module in tuple(sys.modules.values()):
        if not isinstance(module, types.ModuleType):
            continue
        namespace = vars(module)
        mappings = namespace.get("NODE_CLASS_MAPPINGS")
        if not isinstance(mappings, dict):
            continue
        cls = mappings.get("MiniMaxH3NativeINT8TP2CentralSparse")
        if cls is not None and getattr(cls, "__module__", None) == namespace.get("__name__"):
            matches.append(module)
    if len(matches) != 1:
        raise RuntimeError(f"Expected one loaded native TP2 module, found {len(matches)}")
    native = matches[0]
    if Path(native.__file__).resolve().parent.name != "h3_native_int8_tp2_central_sparse":
        raise RuntimeError("Unexpected native TP2 source directory")
    return native


def _signature(tensor):
    def version(value):
        try:
            return int(value._version)
        except RuntimeError:  # Comfy executes loaders under inference_mode.
            return None
    def storage(value):
        allocation = value.untyped_storage()
        return {"device": str(value.device), "dtype": str(value.dtype),
                "shape": tuple(value.shape), "stride": tuple(value.stride()),
                "storage_offset": int(value.storage_offset()),
                "data_ptr": int(value.data_ptr()),
                "storage_ptr": int(allocation.data_ptr()), "storage_nbytes": int(allocation.nbytes()),
                "version": version(value)}

    def metadata(value):
        if value is None or isinstance(value, (bool, int, float, str)):
            return value
        if callable(getattr(value, "untyped_storage", None)):
            return {"tensor_storage": storage(value)}
        if dataclasses.is_dataclass(value):
            return {field.name: metadata(getattr(value, field.name)) for field in dataclasses.fields(value)}
        if isinstance(value, dict):
            return {str(key): metadata(item) for key, item in value.items()}
        if isinstance(value, (tuple, list)):
            return tuple(metadata(item) for item in value)
        # torch.dtype and other immutable scalar layout descriptors.
        if hasattr(value, "__dict__"):
            return {key: metadata(item) for key, item in vars(value).items()}
        return str(value)

    return {"logical_shape": tuple(tensor.shape), "logical_dtype": str(tensor.dtype),
            "logical_device": str(tensor.device), "layout": str(getattr(tensor, "_layout_cls", None)),
            "data": storage(getattr(tensor, "_qdata", tensor)),
            "quant_params": metadata(getattr(tensor, "_params", None))}


def _signature_difference(expected, current, path="weight"):
    if isinstance(expected, dict) and isinstance(current, dict):
        if set(expected) != set(current):
            return f"{path} fields: baseline={sorted(expected)}, current={sorted(current)}"
        for key in expected:
            # CK Parameter detach/rewrap legitimately clones the tiny scale
            # tensor. Its content is checked against a private frozen copy by
            # _weight_difference; the multi-GB qdata must keep its storage.
            if path == "weight.quant_params.scale.tensor_storage" and key in ("data_ptr", "storage_ptr", "version"):
                continue
            result = _signature_difference(expected[key], current[key], path + "." + key)
            if result is not None:
                return result
        return None
    # Inference tensors have no version counter; require agreement whenever
    # both counters exist, and retain all storage/metadata checks regardless.
    if path.endswith(".version") and (expected is None or current is None):
        return None
    if expected != current:
        return f"{path}: baseline={expected!r}, current={current!r}"
    return None


def _weight_difference(state, key, tensor):
    difference = _signature_difference(state._runtime_signatures[key], _signature(tensor))
    if difference is not None:
        return difference
    frozen_scale = state._runtime_scale_baselines.get(key)
    if frozen_scale is not None:
        current_scale = tensor._params.scale
        if not state._runtime_torch.equal(current_scale, frozen_scale):
            return "weight.quant_params.scale.content: differs from frozen original CPU scale"
        if not bool(state._runtime_torch.isfinite(current_scale).all().item()):
            return "weight.quant_params.scale.content: nonfinite scale"
    return None


def _scale_pointer(tensor):
    scale = getattr(getattr(tensor, "_params", None), "scale", None)
    return int(scale.untyped_storage().data_ptr()) if callable(getattr(scale, "untyped_storage", None)) else None


def _bytes(tensor):
    data = getattr(tensor, "_qdata", tensor)
    size = data.numel() * data.element_size()
    params = getattr(tensor, "_params", None)
    scale = getattr(params, "scale", None)
    if scale is not None:
        size += scale.numel() * scale.element_size()
    return int(size)


def _sync(state):
    for device in state.devices:
        state._runtime_torch.cuda.synchronize(device)


def _assert_original_storage(state):
    for key, original in state._runtime_originals.items():
        difference = _weight_difference(state, key, original)
        if difference is not None:
            raise RuntimeError(f"Original CPU parameter storage changed during runtime patching: {key}; {difference}")


def _validate_current_base(state):
    """Allow Comfy Parameter rewrapping only when complete storages are retained.

    No large content hashes are calculated here. This is operational provenance
    using unchanged CPU qdata storage and the reviewed non-inplace Comfy API.
    New qdata fails closed; tiny scale clones must equal frozen CPU values.
    """
    import comfy.utils

    current = {}
    rebound = 0
    scale_rebound = 0
    for key, original in state._runtime_originals.items():
        tensor = comfy.utils.get_attr(state.patcher.model, key)
        if tensor.device.type != "cpu":
            raise RuntimeError(f"Original runtime PDD base must remain on CPU: {key}, {tensor.device}")
        difference = _weight_difference(state, key, tensor)
        if difference is not None:
            raise RuntimeError(f"Current CPU base storage/quant metadata differs from pinned runtime baseline: {key}; {difference}")
        current[key] = tensor
        rebound += int(tensor is not original)
        scale_rebound += int(_scale_pointer(tensor) != _scale_pointer(original))
    # All 308 checks must pass before refreshing any live Parameter reference.
    state._runtime_originals.update(current)
    return rebound, scale_rebound


def _release(state):
    import comfy.utils

    if not state.shards and not state._runtime_owned_backups:
        return
    _sync(state)
    started = time.perf_counter()
    restored = 0
    for key in sorted(state._runtime_owned_backups):
        backup = state.patcher.backup.pop(key, None)
        if backup is None or backup.inplace_update:
            raise RuntimeError(f"Missing/nonrestorable standard runtime backup: {key}")
        # This is the same restore operation used by ModelPatcher.unpatch_model.
        comfy.utils.set_attr_param(state.patcher.model, key, backup.weight)
        restored += 1
    state._runtime_owned_backups.clear()
    state.shards.clear()
    state.bytes_per_rank = (0, 0)
    state.patcher.model.current_weight_patches_uuid = None
    _assert_original_storage(state)
    rebound, scale_rebound = _validate_current_base(state)
    _sync(state)
    _emit("released", prepare_generation=state._runtime_generation,
          restored_parameter_keys=restored, base_parameters_preserved=True,
          wrapper_rebound_keys=rebound,
          scale_rebound_keys=scale_rebound,
          base_validation="same_cpu_qdata_storage_and_frozen_scale_equality",
          cryptographic_ram_content_check=False,
          elapsed_seconds=time.perf_counter() - started)


def _shadow_block(block, weights):
    attn = types.SimpleNamespace(heads=block.attn.heads, head_dim=block.attn.head_dim,
        q_norm=block.attn.q_norm, k_norm=block.attn.k_norm,
        qkv_proj=types.SimpleNamespace(weight=weights["qkv"], bias=None),
        out_proj=types.SimpleNamespace(weight=weights["op"], bias=None))
    mlp = types.SimpleNamespace(fc1=types.SimpleNamespace(weight=weights["fc1"], bias=None),
                               fc2=types.SimpleNamespace(weight=weights["fc2"], bias=None))
    return types.SimpleNamespace(attn=attn, mlp=mlp, norm1=block.norm1,
                                 norm2=block.norm2, adaln_proj=block.adaln_proj)


def _prepare(state, loaded):
    if state.shards:
        if len(state.shards) != 50 or len(state._runtime_owned_backups) != 108:
            raise RuntimeError("Incomplete cached runtime PDD state")
        return
    import comfy.model_management as mm
    import comfy.utils

    torch = state._runtime_torch
    if state._runtime_owned_backups or state.patcher.backup:
        raise RuntimeError("Runtime PDD preparation requires clean standard weight backups")
    _assert_original_storage(state)
    _sync(state)
    started = time.perf_counter()
    for device in state.devices:
        mm.free_memory(state.reserve, device, keep_loaded=[loaded])
    _sync(state)
    patch_started = time.perf_counter()
    rebound, scale_rebound = _validate_current_base(state)
    patched_large = set()
    patched_remaining = set()
    state._runtime_generation += 1
    try:
        with torch.inference_mode(), torch.cuda.device(state.devices[0]):
            # Standard resident Comfy path for all 50 curve weights, their 50
            # mandatory DC biases, and eight token-refiner projection weights.
            # Standard backups retain the original CPU values; release restores
            # them even if an unload caller asks to retain other model patches.
            for key in state._runtime_remaining_keys:
                original = comfy.utils.get_attr(state.patcher.model, key)
                try:
                    state.patcher.patch_weight_to_device(key, device_to=state.devices[0], return_weight=False)
                finally:
                    if key in state.patcher.backup:
                        state._runtime_owned_backups.add(key)
                if key not in state.patcher.backup:
                    raise RuntimeError(f"Comfy did not back up nonsharded runtime parameter {key}")
                if comfy.utils.get_attr(state.patcher.model, key) is original:
                    raise RuntimeError(f"Comfy did not replace nonsharded runtime parameter {key}")
                patched_remaining.add(key)
            dm = state.patcher.model.diffusion_model
            counts = [0, 0]
            for index, block in enumerate(dm.blocks):
                full_weights = {}
                for name, suffix in LARGE_SUFFIXES.items():
                    key = f"diffusion_model.blocks.{index}.{suffix}"
                    original = comfy.utils.get_attr(state.patcher.model, key)
                    difference = _weight_difference(state, key, original)
                    if difference is not None:
                        raise RuntimeError(f"TP2 base projection storage changed before patching: {key}; {difference}")
                    # One complete projection determines its quantization scales
                    # before output/input slicing. Never patch the halves.
                    weight = state.patcher.patch_weight_to_device(
                        key, device_to=state.devices[0], return_weight=True)
                    if comfy.utils.get_attr(state.patcher.model, key) is not original or key in state.patcher.backup:
                        raise RuntimeError(f"return_weight=True mutated the original projection: {key}")
                    if getattr(weight, "_layout_cls", None) != state._runtime_core.INT8_LAYOUT:
                        raise RuntimeError(f"Runtime patched projection lost native INT8 layout: {key}")
                    if not weight._params.convrot or weight.device != state.devices[0]:
                        raise RuntimeError(f"Runtime patched projection lost ConvRot/device contract: {key}")
                    full_weights[name] = weight
                    patched_large.add(key)
                shadow = _shadow_block(block, full_weights)
                shard = state._runtime_core.shard_block(shadow, *state.devices, lora=None)
                # shard_block retains its input attention object. Replace the
                # temporary facade so no full GPU projection survives sharding.
                shard["att"] = block.attn
                shard["qkv_chunk_tokens"] = state.qkv_chunk_tokens
                shard["mlp_chunk_tokens"] = state.mlp_chunk_tokens
                shard["sparse_policy"] = state.policy
                if any(item is not None for key in ("lqkv", "lop", "lfc1", "lfc2") for item in shard[key]):
                    raise RuntimeError("Activation LoRA bypass unexpectedly enabled")
                state.shards.append(shard)
                for key in ("qkv", "op", "fc1", "fc2"):
                    for rank, weight in enumerate(shard[key]):
                        counts[rank] += _bytes(weight)
                for key in ("n1", "n2", "qn", "kn"):
                    for rank, weight in enumerate(shard[key]):
                        counts[rank] += _bytes(weight)
                del full_weights, shadow, shard, weight
            counts[0] += sum(_bytes(comfy.utils.get_attr(state.patcher.model, key)) for key in patched_remaining)
            state.bytes_per_rank = tuple(counts)
            _assert_original_storage(state)
        if patched_large != state._runtime_large_keys or patched_remaining != set(state._runtime_remaining_keys):
            raise RuntimeError("Runtime PDD target coverage was incomplete")
        _sync(state)
        _emit("prepared", prepare_generation=state._runtime_generation, blocks=len(state.shards), ranks=2,
              all_keys_applied=True, full_weight_patches_before_sharding=True,
              base_parameters_preserved=True, elapsed_seconds=time.perf_counter() - started,
              wrapper_rebound_keys=rebound,
              scale_rebound_keys=scale_rebound,
              base_validation="same_cpu_qdata_storage_and_frozen_scale_equality",
              cryptographic_ram_content_check=False,
              patch_and_shard_elapsed_seconds=time.perf_counter() - patch_started,
              resident_bytes_per_rank=list(state.bytes_per_rank),
              native_patch_api="ModelPatcher.patch_weight_to_device", full_projection_return_weight=True,
              remaining_parameters_standard_backup=True,
              numeric_scope="Comfy resident runtime merge/requant API; dynamic offload rounding may differ")
    except BaseException:
        _release(state)
        raise


class MiniMaxH3RuntimePDDINT8TP2:
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {
            "model": ("MODEL",),
            "attention_mode": (["ck_sol_all_int8_hybrid"],),
            "sol_tau": ("FLOAT", {"default": 1.0, "min": -2.0, "max": 4.0, "step": 0.1}),
            "sla_sparsity": ("FLOAT", {"default": 0.85, "min": 0.0, "max": 0.95, "step": 0.05}),
            "qkv_chunk_tokens": ("INT", {"default": 8192, "min": 256, "max": 32768, "step": 256}),
            "mlp_chunk_tokens": ("INT", {"default": 8192, "min": 256, "max": 32768, "step": 256}),
            "replica_device": ("STRING", {"default": "cuda:1"}),
            "reserve_gib": ("FLOAT", {"default": 18.0, "min": 12.0, "max": 21.0, "step": 0.5}),
        }}

    RETURN_TYPES = ("MODEL",)
    FUNCTION = "apply"
    CATEGORY = "H3 isolated benchmark"

    def apply(self, model, qkv_chunk_tokens=8192, mlp_chunk_tokens=8192, replica_device="cuda:1",
              reserve_gib=18.0, attention_mode="ck_sol_all_int8_hybrid", sol_tau=1.0, sla_sparsity=0.85):
        import torch
        import comfy.utils

        if model.is_dynamic() or getattr(model, "weight_inplace_update", False) or model.backup:
            raise RuntimeError("Runtime PDD TP2 requires a fresh nondynamic, non-inplace CPU model")
        if attention_mode != "ck_sol_all_int8_hybrid":
            raise RuntimeError("Runtime PDD comparison requires the original Sol-All-INT8 policy before Sage control")
        large, adaln_w, adaln_b, refiner = expected_keys()
        expected = large | adaln_w | adaln_b | refiner
        if set(model.patches) != expected:
            raise RuntimeError(f"Expected308 PDD parameter patches; missing={sorted(expected-set(model.patches))}, extra={sorted(set(model.patches)-expected)}")
        for key, patches in model.patches.items():
            if len(patches) != 1 or len(patches[0]) != 5:
                raise RuntimeError(f"Expected exactly one ordinary PDD patch per parameter: {key}")
            strength, adapter, strength_model, offset, function = patches[0]
            if float(strength) != 1.0 or float(strength_model) != 1.0 or offset is not None or function is not None:
                raise RuntimeError(f"Unexpected PDD strength/offset/function: {key}")
        final = model.object_patches.get("diffusion_model.final_layer.forward")
        function = getattr(final, "__func__", final)
        if function is None:
            raise RuntimeError("Runtime PDD heads are not object-patched")
        closure = inspect.getclosurevars(function).nonlocals
        heads = [value for value in closure.values() if value.__class__.__name__ == "_PDDHeadLinear"]
        if len(heads) != 2 or any(float(head.strength) != 1.0 for head in heads):
            raise RuntimeError("Runtime PDD requires both trained output heads at strength1")
        native = _native_module()
        clean = model.clone()
        clean.patches = {}
        result = native.MiniMaxH3NativeINT8TP2CentralSparse().apply(
            clean, qkv_chunk_tokens=qkv_chunk_tokens, mlp_chunk_tokens=mlp_chunk_tokens,
            replica_device=replica_device, reserve_gib=reserve_gib, attention_mode=attention_mode,
            sol_tau=sol_tau, sla_sparsity=sla_sparsity)[0]
        result.patches = {key: list(patches) for key, patches in model.patches.items()}
        state = getattr(result, STATE_ATTRIBUTE)
        state._runtime_core = sys.modules[state.policy.__class__.__module__.rsplit(".", 1)[0] + ".tp2"]
        state._runtime_torch = torch
        state._runtime_large_keys = large
        state._runtime_remaining_keys = sorted(adaln_w | adaln_b | refiner)
        state._runtime_owned_backups = set()
        state._runtime_generation = 0
        state._runtime_originals = {key: comfy.utils.get_attr(result.model, key) for key in expected}
        if any(tensor.device.type != "cpu" for tensor in state._runtime_originals.values()):
            raise RuntimeError("All original PDD targets must be CPU-resident before runtime preparation")
        state._runtime_signatures = {key: _signature(value) for key, value in state._runtime_originals.items()}
        state._runtime_scale_baselines = {
            key: value._params.scale.detach().clone()
            for key, value in state._runtime_originals.items()
            if callable(getattr(getattr(getattr(value, "_params", None), "scale", None), "untyped_storage", None))
        }
        state.prepare = types.MethodType(_prepare, state)
        state.release = types.MethodType(_release, state)
        _emit("configured", native_source=str(Path(native.__file__).resolve()),
              devices=[str(device) for device in state.devices],
              full_weight_patches_before_sharding=True, base_parameters_preserved=True,
              return_weight_projection_keys=200, normal_backup_parameter_keys=108,
              base_validation="same_cpu_qdata_storage_and_frozen_scale_equality",
              cryptographic_ram_content_check=False,
              checkpoint_written=False, inference_only=True)
        return (result,)


NODE_CLASS_MAPPINGS = {"MiniMaxH3RuntimePDDINT8TP2": MiniMaxH3RuntimePDDINT8TP2}
NODE_DISPLAY_NAME_MAPPINGS = {"MiniMaxH3RuntimePDDINT8TP2": "H3 Original INT8 + Runtime PDD TP2"}
