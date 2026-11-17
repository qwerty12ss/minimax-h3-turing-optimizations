# Optional Luna-derived H3 recipe

Run instructions: [Русский](../../docs/RECIPE_LUNA.ru.md) | [English](../../docs/RECIPE_LUNA.md).

**Enabling Qwen TP2 changes both generated video and source audio, even with the same seed.** `cache` alone preserved the original tensors in the measured case. This directory is opt-in; the base installer does not install these nodes.

The measured `awq_tp.py`, `packed_shards.py` and `kernel.cu` are preserved byte for byte. `__init__.py` is a publication wrapper around the same configuration path: raw tensor/media dumps and probe mode were removed, and a scalar statistics GET endpoint was added. The wrapper and public client have CPU/integration checks; this newly packaged launch has not been rerun end to end on GPUs. Numerical experiment results are in [luna_recipe.json](../../results/luna_recipe.json).

`test_cache_lifetime.py` checks wrapper rebinding, real tensor replacement, scale/norm invalidation and bounded retention over repeated generations, without loading H3 or starting CUDA. It needs PyTorch. This checks ownership, not GPU arithmetic.

FC2 is restricted to the measured policy in `fc2-policy.json`: M4650 uses tile12, M8192 falls back to the original CK kernel, with N5376/K7168. Build `kernel.cu` against CUTLASS 3.8.0 commit `afa1772203677c5118fcd82537a9c8fefbcc7008`; no binary or CUTLASS checkout is included.

The Qwen scheduling/TP adaptation and copied FC2 source derive from [Luna-AI-Infra/ComfyUI-H3-2080Ti-TP at 945c326](https://github.com/Luna-AI-Infra/ComfyUI-H3-2080Ti-TP/tree/945c326132dc182d27299a44c003e6b8f5849ab6), based on Márton Sági's Comfy-H3-MultiStream. This optional recipe is distributed under **GPL-3.0-only**; see [LICENSE](LICENSE), the preserved [upstream fork NOTICE](NOTICE) and [UPSTREAM-NOTICE.txt](UPSTREAM-NOTICE.txt). Modifications for the measured AWQ/DiT adaptation and this public wrapper were made in November 2026. Original weights are not redistributed. Paths such as `docs/MODELS.md` in the verbatim fork NOTICE refer to [the upstream fork](https://github.com/Luna-AI-Infra/ComfyUI-H3-2080Ti-TP/tree/945c326132dc182d27299a44c003e6b8f5849ab6/docs), not this directory. External CUTLASS templates have their own retained [BSD license](CUTLASS-LICENSE.txt). Model terms are linked from the base installation guide.
