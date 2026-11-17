"""Opt-in H3 recipe: adapted Qwen TP2, bounded CPU packs and scoped FC2.

The compute paths are the measured experiment. This public wrapper omits raw
media/conditioning dumps and exposes only numerical per-job statistics.
See README.md and LICENSE (GPL-3.0-only).
"""
import ctypes
import json
import os
from pathlib import Path
import threading
import time
import types

import torch

ROOT = Path(os.environ['H3_LUNA_TRIAL_ROOT'])
ACTIVE = {}
LOCAL = threading.local()
ORIGINAL = None
LIB = None


def install_gemm():
    global ORIGINAL
    if ORIGINAL is not None:
        return
    import comfy_kitchen.backends.cuda as ck
    ORIGINAL = ck._C.cublas_gemm_int8
    ck._C.cublas_gemm_int8 = gemm


def gemm(ac, wc, oc, workspace, stream):
    if not getattr(LOCAL, 'fc2', False):
        return ORIGINAL(ac, wc, oc, workspace, stream)
    a, w, out = [torch.from_dlpack(t) for t in (ac, wc, oc)]
    assert a.shape[1] == 7168 and w.shape == (5376, 7168) and out.shape == (len(a), 5376)
    key = f'{a.device.index}:{len(a)}'
    ACTIVE['fc2_counts'][key] = ACTIVE['fc2_counts'].get(key, 0) + 1
    tile = ACTIVE.get('tiles', {}).get(str(len(a))) if 'fc2' in ACTIVE['mode'] else None
    if tile is None:
        import comfy_kitchen.backends.cuda as ck
        return ORIGINAL(*(ck._wrap_for_dlpack(t) for t in (a, w, out)), workspace, stream)
    code = LIB.launch(tile, len(a), 5376, 7168, 1, a.data_ptr(), w.data_ptr(), out.data_ptr(), None, stream)
    if code:
        raise RuntimeError(f'CUTLASS FC2 launch failed: {code}')


class LunaBenchConfigure:
    @classmethod
    def INPUT_TYPES(cls):
        return {'required': {'model': ('MODEL',), 'clip': ('CLIP',), 'mode': ('STRING',), 'tag': ('STRING',)}}

    RETURN_TYPES = ('MODEL', 'CLIP')
    FUNCTION = 'apply'
    CATEGORY = 'h3_turing/recipes'

    def apply(self, model, clip, mode, tag):
        global ACTIVE, LIB
        assert mode in ['baseline', 'fc2', 'cache', 'fc2_cache', 'qwen', 'qwen_fc2_cache']
        ACTIVE = {'mode': mode, 'tag': tag, 'fc2_counts': {}, 'stages': {}}
        if 'fc2' in mode:
            # This is the measured tail-chunk policy, not an autotuning request.
            ACTIVE['tiles'] = {'4650': 12}
            if LIB is None:
                LIB = ctypes.CDLL(str(ROOT / 'libfc2_int8.so'))
                LIB.launch.argtypes = [ctypes.c_int] * 5 + [ctypes.c_void_p] * 5
                LIB.launch.restype = ctypes.c_int
        install_gemm()
        from h3_turing.deployment.pdd.source import runtime_pdd as runtime
        state = getattr(model, runtime.STATE_ATTRIBUTE)
        opt = state._h3_public_optimizer
        original = opt.joint_fc2
        if not getattr(original, '_luna_wrapped', False):
            def fc2(*args, **kwargs):
                old = getattr(LOCAL, 'fc2', False)
                LOCAL.fc2 = True
                try:
                    return original(*args, **kwargs)
                finally:
                    LOCAL.fc2 = old
            fc2._luna_wrapped = True
            from h3_turing.swiglu_convrot.joint import mlp_function
            opt.joint_fc2 = fc2
            opt.mlp, opt.mlp_proof = mlp_function(opt.core, opt.joint_prepare, fc2)
        csm = clip.cond_stage_model
        candidates = [m for _, m in csm.named_modules() if hasattr(m, 'model') and hasattr(m.model, 'layers')]
        assert len(candidates) == 1, 'The pinned Qwen3-VL architecture is required'
        transformer = candidates[0]
        if not hasattr(type(csm), '_luna_original_encode'):
            type(csm)._luna_original_encode = type(csm).encode_token_weights
            def encode(self, *args, **kwargs):
                t = time.perf_counter()
                out = type(self)._luna_original_encode(self, *args, **kwargs)
                for i in range(2):
                    torch.cuda.synchronize(i)
                ACTIVE['stages']['qwen_encode_seconds'] = time.perf_counter() - t
                return out
            type(csm).encode_token_weights = encode
        if not getattr(state, '_luna_timed', False):
            original_prepare = state.prepare
            def prepare(*args, **kwargs):
                t = time.perf_counter()
                result = original_prepare(*args, **kwargs)
                ACTIVE['stages']['tp_prepare_seconds'] = time.perf_counter() - t
                return result
            state.prepare = prepare
            state._luna_timed = True
        if 'cache' in mode:
            from .packed_shards import install
            install(state, lambda: ACTIVE['stages'], lambda: 'cache' in ACTIVE['mode'])
        if 'qwen' in mode:
            from .awq_tp import wrap
            clip = wrap(clip, transformer, ACTIVE['stages'])
        else:
            clip = clip.clone()
        original_encode = clip.encode_from_tokens.__func__
        def timed_encode(self, *args, **kwargs):
            t = time.perf_counter()
            out = original_encode(self, *args, **kwargs)
            for i in range(2):
                torch.cuda.synchronize(i)
            ACTIVE['stages']['clip_total_seconds'] = time.perf_counter() - t
            return out
        clip.encode_from_tokens = types.MethodType(timed_encode, clip)
        return model, clip


NODE_CLASS_MAPPINGS = {'LunaBenchConfigure': LunaBenchConfigure}
NODE_DISPLAY_NAME_MAPPINGS = {'LunaBenchConfigure': 'H3 optional Luna recipe (Qwen changes output)'}

import server
from aiohttp import web


@server.PromptServer.instance.routes.get('/h3_luna_recipe/stats')
async def recipe_stats(request):
    # No images, prompt, tokens, conditioning tensors or machine inventory.
    return web.json_response(ACTIVE)
