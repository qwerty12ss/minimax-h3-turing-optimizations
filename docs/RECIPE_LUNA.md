# Additional recipe: Qwen TP2 + FC2 + DiT cache

**English** | [Русский](RECIPE_LUNA.ru.md)

**This recipe changes generated output.** Qwen TP2 changes floating-point results in text encoding, so **both generated video and source audio differ even with identical input, prompt, seed and settings**. In the measured case, lighting, dust and object position changed. A quality improvement or regression has not been established; review the new take by watching and listening. Use `cache` without Qwen when preserving the original take matters.

This extends our `preset=optimized` pipeline with adaptations from [Luna-AI-Infra at commit 945c326](https://github.com/Luna-AI-Infra/ComfyUI-H3-2080Ti-TP/tree/945c326132dc182d27299a44c003e6b8f5849ab6). The complete original Luna repository was **not** benchmarked. The native NVFP4 AWQ encoder remains; there is no GGUF conversion. FL2VA INT8 ConvRot, separate PDD, SigmaShift 12/3, CFG1, eight Euler steps and attention remain unchanged: CK INT8 on steps 1,2,7,8 and Sol INT8 on 3–6.

| `--luna-mode` | New operations | Output in the measured case |
|---|---|---|
| `baseline` | None; existing optimized TP2 | Reference take |
| `cache` | Bounded DiT CPU rank packs | Raw frames, source audio and conditioning bitwise identical |
| `fc2` / `fc2_cache` | Measured FC2 policy, alone / with DiT cache | FC2 alone and cache alone were exact; the combined non-Qwen mode was not separately measured |
| `qwen` | Row-sharded Qwen TP2, two-slot streaming, CPU packs | **Different video and audio** |
| `qwen_fc2_cache` | All three | **Different video and audio**; bitwise identical to `qwen` in the measured case |

## Observed performance and output

One scene and seed, 960×512, 192 frames, 24 fps, eight seconds, eight steps, two RTX 2080 Ti 22-GiB cards with NV2/P2P. Full ComfyUI execution includes text encoding, preparation, sampling, both VAEs and MP4 saving. Startup, queueing and subsequent lossless diagnostic dumps are excluded. Every timed job generated a fresh result.

| Mode | Full job, seconds | Comparison |
|---|---:|---|
| Two warm controls | 167.856, 174.832; median **171.344** | Fresh processes without unused experimental packs |
| Warm `cache` | **158.545** | −7.47% versus median |
| Warm `qwen` | **158.859** | −7.29% versus median |
| Warm `qwen_fc2_cache` | **140.554** | **−17.97%** versus median; −16.3…19.6% versus observed controls |
| `fc2` | **174.712** | Only −0.120 s / 0.07% versus adjacent 174.832-s control; no meaningful gain established |

The fastest-recorded-runs video used **167.317 → 140.554 s**, saving **26.763 s / 16.0%**. Its original run was the first job in that series and excluded from the warm-control estimate above. Do not mix these comparison bases or add percentages from unrelated experiments.

First DiT pack construction took **185.169 s**. First Qwen pack construction with an **already warm DiT cache** took **159.670 s**. A fully cold job with both fixed caches was not separately measured. **140.554 s requires both caches warm in the same process.** Sample counts are small, clocks were not locked, and the observed control range is not a confidence interval.

Cache alone and FC2 alone preserved all 192 float32 frames, source waveform, conditioning, token tags and reference latent bit for bit. This does not promise every model/geometry is exact or that MP4 container bytes match. Qwen conditioning relative L2 was **7.17×10⁻⁶**; full float RGB PSNR **24.09 dB**, MAE **0.02781** on a 0–1 scale; source audio relative L2 **0.1151**. These measure difference, not perceptual quality. Sampled frames showed no obvious detail collapse, but subjective motion/audio quality was not fully assessed; a pre-existing framing defect near six seconds remained.

## Install the optional recipe

Follow the system prerequisites in [INSTALL.md](INSTALL.md): Linux x86_64, Python 3.12, CUDA Toolkit 13.0, two **22-GiB** SM75 cards and bidirectional P2P. Plan for **128 GB RAM**. Additional pinned CPU packs retain **19.03 GiB DiT + 12.78 GiB Qwen = 31.81 GiB**, beyond the other models/buffers. An early unbounded cache caused OOM; this source contains the fixed one-slot-per-block cache, replacement before allocation and DiT limits of 50 blocks / 22 GiB.

From this repository root, create a separate runtime:

```bash
export H3_RUNTIME="$HOME/h3-turing-luna-runtime"
export H3_LUNA_TRIAL_ROOT="$H3_RUNTIME/luna-output"
python3.12 tools/setup_stack.py --root "$H3_RUNTIME"
python3.12 tools/models.py download --root "$H3_RUNTIME"
mkdir -p "$H3_LUNA_TRIAL_ROOT"
ln -s "$PWD/recipes/luna" "$H3_RUNTIME/ComfyUI/custom_nodes/h3_luna_recipe"
"$H3_RUNTIME/.venv/bin/python" recipes/luna/test_cache_lifetime.py
```

The base guide's SHA-verified `models.py link` is also supported. Create the custom-node link once; inspect any existing destination first. Keep this source checkout in place. The CPU test checks ownership/retention, not GPU arithmetic.

For `qwen_fc2_cache`, `fc2` and `fc2_cache`, build FC2 from source; skip this step for `cache` or `qwen`:

```bash
git init "$H3_RUNTIME/cutlass"
git -C "$H3_RUNTIME/cutlass" remote add origin https://github.com/NVIDIA/cutlass.git
git -C "$H3_RUNTIME/cutlass" fetch --depth=1 origin afa1772203677c5118fcd82537a9c8fefbcc7008
git -C "$H3_RUNTIME/cutlass" checkout --detach FETCH_HEAD
"${CUDA_HOME:-/usr/local/cuda}/bin/nvcc" -arch=sm_75 -O3 -std=c++17 \
  --shared -Xcompiler=-fPIC -I"$H3_RUNTIME/cutlass/include" \
  recipes/luna/kernel.cu -o "$H3_LUNA_TRIAL_ROOT/libfc2_int8.so"
```

On Linux with a user systemd session, launch with the experiment's **88-GiB process limit**:

```bash
systemd-run --user --scope -p MemoryMax=88G -p MemorySwapMax=0 \
  python3.12 tools/launch.py --root "$H3_RUNTIME" --gpus 0,1 --port 8197 --cache-classic
```

Select your physical GPU indices and a free local port. The launcher checks SM75, memory and P2P. Without systemd, provide an equivalent process/container memory limit. A process limit does not guarantee host-wide free RAM; stop your own test if memory becomes insufficient.

`--cache-classic` keeps loader/model objects alive. The client adds a fresh dependency tag on every invocation, forcing conditioning, all eight sampling steps, VAE and saving to run again. Use this client with an exclusive test queue, one job at a time. In another terminal from this repository root:

```bash
# First invocation creates packs; second invocation reuses them.
python3.12 tools/run_workflow.py workflows/h3_turing_8s_api.json \
  --url http://127.0.0.1:8197 --luna-mode qwen_fc2_cache
python3.12 tools/run_workflow.py workflows/h3_turing_8s_api.json \
  --url http://127.0.0.1:8197 --luna-mode qwen_fc2_cache
```

Use `--luna-mode cache` for the output-preserving alternative. For an uncached control, stop **this** ComfyUI, restart it with identical launch options and run `--luna-mode baseline` twice, excluding the first job as warmup. Switching a populated process to baseline is insufficient: the allocator can retain pinned RAM after Python references disappear.

The receipt includes full execution time, finite video/audio checks and numerical `luna_recipe.stages`. Warm DiT should report `pack_hits=50`, `pack_misses=0`, `retained_pack_bytes=20431001600`; warm Qwen should report `qwen_pack_hit=true`, `qwen_retained_pack_bytes=13722444800`. Add `--report /path/to/local-receipt.json` for a small local receipt. Videos are under this runtime's `ComfyUI/output/video`.

The public API workflow uses a synthetic image and neutral text, so it checks operation without private assets; the measured timings came from a different unpublished input. For your own input follow the base guide, preserving node structure and 960×512 / 192 frames. No additional GUI workflow is qualified here.

## Publication and validation scope

The measured `awq_tp.py`, `packed_shards.py` and `kernel.cu` are unchanged byte for byte. The public wrapper removes raw frame/conditioning dumps and exposes only scalar statistics. The wrapper/client have local integration checks; **this newly packaged GPU launch was not rerun end to end in this documentation update**. Numerical results and provenance: [luna_recipe.json](../results/luna_recipe.json). Source and retained GPL-3.0-only/upstream notices: [recipes/luna](../recipes/luna/README.md).

No private image, creative prompt, generated clip, conversation, model weight, credential or originating server address/path is included. Review your own artifacts before sharing them.
