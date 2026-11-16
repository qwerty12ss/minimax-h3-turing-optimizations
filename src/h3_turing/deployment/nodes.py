"""ComfyUI integration for the standalone, source-pinned dual-SM75 stack."""
import types

import torch

from .pdd.source import runtime_pdd as runtime
from .pdd.owner import Owner

LAST_VALIDATION = {}
_LAST_PROMPT_ID = None


def require(value, message):
    if not value:
        raise RuntimeError(message)


class H3TuringApply:
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {"model": ("MODEL",),
                "preset": (["optimized", "baseline"], {"default": "optimized"}),
                "sol_tau": ("FLOAT", {"default": 1.0, "min": -2.0, "max": 4.0, "step": 0.1}),
                "reserve_gib": ("FLOAT", {"default": 18.0, "min": 12.0, "max": 21.0})}}

    RETURN_TYPES = ("MODEL",)
    FUNCTION = "apply"
    CATEGORY = "MiniMaxH3/Turing"

    def apply(self, model, preset="optimized", sol_tau=1.0, reserve_gib=18.0):
        # Import the original native package before its explicit discovery gate.
        from h3_turing import h3_native_int8_tp2_central_sparse as native
        require(torch.cuda.device_count() == 2, "Select exactly two GPUs with tools/launch.py --gpus")
        require(all(torch.cuda.get_device_capability(i) == (7, 5) for i in range(2)), "This stack requires two SM75 GPUs")
        result = runtime.MiniMaxH3RuntimePDDINT8TP2().apply(model, sol_tau=sol_tau, reserve_gib=reserve_gib)[0]
        state = getattr(result, runtime.STATE_ATTRIBUTE)
        large, _, _, _ = runtime.expected_keys()
        adapters = {}
        for key in sorted(large):
            strength, adapter, base_strength, offset, function = result.patches[key][0]
            b, a, alpha, mid, dora, reshape = adapter.weights
            require(mid is dora is reshape is None and a.dtype == b.dtype == torch.bfloat16,
                    "Original converted CPU BF16 PDD adapters required")
            # Upstream joins the three rank-64 Q/K/V adapters into one
            # block-diagonal rank-192 projection and scales alpha likewise.
            rank = 192 if key.endswith("attn.qkv_proj.weight") else 64
            require(a.shape[0] == b.shape[1] == rank and float(alpha) == float(rank),
                    "Original FL2VA PDD rank/alpha required")
            adapters[key] = dict(A=a, B=b, adapter=adapter, rank=a.shape[0])
        def guard():
            require(state.patcher is result, "TP2 preparation owner changed")
            runtime._assert_original_storage(state)
        # This owner publishes all 50 shards transactionally and restores the
        # 108 ordinary Comfy patches when the two-GPU model unloads.
        owner = Owner(state, runtime, torch, adapters, guard, lambda *a, **k: None, inject=False)
        state.prepare = types.MethodType(lambda current, loaded: owner.prepare(loaded), state)
        state.release = types.MethodType(lambda current: owner.release(), state)
        state._h3_public_pdd_owner = owner
        state._h3_public_preset = preset
        if preset == "optimized":
            from .optimized import Optimized
            state._h3_public_optimizer = Optimized(state)
            state.block_impl = state._h3_public_optimizer
        elif preset != "baseline":
            raise ValueError("Unknown preset")
        return (result,)


class H3TuringVAE:
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {"vae": ("VAE",), "mode": (["serial", "temporal_hybrid"],)}}
    RETURN_TYPES = ("VAE",)
    FUNCTION = "apply"
    CATEGORY = "MiniMaxH3/Turing"

    def apply(self, vae, mode="serial"):
        if mode == "temporal_hybrid" and not getattr(vae, "_h3_parallel_owned_installed", False):
            from h3_turing.vae import spatial
            from h3_turing.vae.hybrid import hybrid_class
            selected = hybrid_class(spatial, spatial.SpatialGroupParallel)
            spatial.install_owned_decode_wrapper(vae, "cuda:1", context_class=selected)
        elif mode != "serial" and mode != "temporal_hybrid":
            raise ValueError("Unknown VAE scheduling mode")
        return (vae,)


class H3TuringTestImage:
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {"width": ("INT", {"default": 960, "min": 32, "max": 1344, "step": 32}),
                             "height": ("INT", {"default": 512, "min": 32, "max": 768, "step": 32})}}
    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "create"
    CATEGORY = "MiniMaxH3/Turing"

    def create(self, width=960, height=512):
        # A deterministic test pattern, unrelated to any user's source media.
        image = torch.full((1, height, width, 3), 0.65, dtype=torch.float32)
        image[:, height // 2:] = torch.tensor([0.25, 0.35, 0.45])
        image[:, height // 4:height * 3 // 4, width * 2 // 5:width * 3 // 5] = torch.tensor([0.9, 0.2, 0.1])
        return (image,)


class H3TuringValidateFrames:
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {"images": ("IMAGE",), "audio": ("AUDIO",)}}
    RETURN_TYPES = ("IMAGE", "AUDIO")
    RETURN_NAMES = ("images", "audio")
    FUNCTION = "check"
    CATEGORY = "MiniMaxH3/Turing"

    def check(self, images, audio):
        global _LAST_PROMPT_ID
        from comfy_execution.utils import get_executing_context
        from .validation import optimizer_reports
        from h3_turing.attention.sol_backend import flush_deferred_statistics
        flush_deferred_statistics()
        require(images.ndim == 4 and images.shape[-1] == 3 and images.shape[0] > 0, "Expected decoded RGB frames")
        require(bool(torch.isfinite(images).all()), "Nonfinite decoded frames")
        require(bool(torch.isfinite(audio["waveform"]).all()), "Nonfinite decoded audio")
        LAST_VALIDATION.clear()
        context = get_executing_context()
        require(context is not None, "Expected a ComfyUI execution context")
        _LAST_PROMPT_ID = context.prompt_id
        LAST_VALIDATION.update(video_frames=images.shape[0], height=images.shape[1], width=images.shape[2],
                               video_finite=True, audio_finite=True, audio_sample_rate=audio["sample_rate"],
                               audio_samples=audio["waveform"].shape[-1],
                               optimizers=optimizer_reports(context.prompt_id))
        return images, audio


NODE_CLASS_MAPPINGS = {c.__name__: c for c in [H3TuringApply, H3TuringVAE, H3TuringTestImage, H3TuringValidateFrames]}
NODE_DISPLAY_NAME_MAPPINGS = {"H3TuringApply": "H3 Turing: dual-GPU acceleration",
                            "H3TuringVAE": "H3 Turing: VAE scheduling",
                            "H3TuringTestImage": "H3 Turing: synthetic test image",
                            "H3TuringValidateFrames": "H3 Turing: validate decoded video/audio"}


def register_report_route():
    import server
    from aiohttp import web
    @server.PromptServer.instance.routes.get("/h3_turing/validation")
    async def report(request):
        running, pending = server.PromptServer.instance.prompt_queue.get_current_queue()
        if running or pending:
            return web.json_response({"error": "Wait for the queue to finish"}, status=409)
        return web.json_response({"prompt_id": _LAST_PROMPT_ID, "validation": dict(LAST_VALIDATION)})


register_report_route()
