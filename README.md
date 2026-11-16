# MiniMax H3: Turing / RTX 2080 Ti optimizations

**English** | [Русский](README.ru.md)

Research sources, operator tests, benchmark parameters and numeric results for MiniMax H3 on **two RTX 2080 Ti cards modified to 22 GiB each**. The recorded host has **2 × Xeon E5-2686 v4, 128 GB RAM, PCIe 3.0 ×16 per GPU and an NV2 link** between the pair.

This is a curated research release with a new Git history. It includes no input media, generated videos, creative prompts, conversations, model weights, deployment inventories or prebuilt binaries. Historical measurements are provided as numbers; synthetic tests provide an independent way to check the published operators.

## Included code

| Component | Source | Scope |
|---|---|---|
| SM75 Sol / SLA attention | `src/h3_turing/attention/` | CUDA routing, centroids, INT8 attention, bounded asynchronous route statistics; local source build |
| Exact PDD scale/add and SUM | `pointwise.py`, `sum_reduce.py` | Preserve the original Half rounding boundaries; SUM has a fixed geometry contract |
| Shared QKV / FC1 preparation | `shared/` | Source-pinned CK quantizer split and bounded two-GPU packets |
| Norm/modulation and RoPE | `norm_mod/`, `rope/` | Explicit arithmetic and indexing contracts; geometry-specific research kernels |
| SwiGLU + ConvRot | `swiglu_convrot/` | Two outputs: original Half activation and quantized FC2 input; local FP32 SiLU lookup table |
| VAE scheduling | `vae/` | Reversible spatial/temporal scheduling around the owner's existing decoder and assembler |
| Unpromoted experiments | `experimental/` | Exact dequant/PDD fusion, head-dependent thresholds and quantizer clipping; limitations in the benchmark notes |

Includes ComfyUI custom nodes, the original TP2 core, PDD integration, an isolated stack installer and two portable video/audio workflows. **[Deploy from scratch](docs/INSTALL.md)** covers hardware, SHA256-pinned public models, dependencies, configuration and execution. See [deployment validation](docs/DEPLOYMENT_VALIDATION.md) for actual checks. New synthetic workflows need no historical private assets. Source hashes and adaptation notes are in [the source manifest](results/source_manifest.json).

## Recorded full-job results

All rows below use a 960 × 512, 192-frame, 24-fps, eight-step workload on **2 × 2080 Ti**. Each row is a separate bracketed experiment, with its own seed and baseline. Cold runs are excluded. The exact estimators and raw checks are described in [benchmark notes](docs/BENCHMARKS.md).

| Experiment | Control, s | Candidate, s | Time reduction |
|---|---:|---:|---:|
| Bounded shared QKV / FC1 packet pipeline | 168.05375 | 165.007 | 1.813% |
| Selected temporal VAE scheduling | 165.307 | 163.619 | 1.021% |
| Joint SwiGLU + ConvRot preparation | 163.4205 | 159.739 | 2.253% |

These percentages must not be added. The final row's nine recorded jobs passed all 99 raw descriptor comparisons for that case. The earlier shared/VAE series have multiple control fingerprint groups; the ledger preserves those differences and does not claim universal full-graph determinism.

## RTX 3090 comparison

The retained historical RTX 3090 24-GB reference is **217.417 s for a cold job**, with Sage `auto` plus the memory-efficient Sage patch, 960 × 512, 192 frames and eight steps. It is **not a pure BF16-attention benchmark**. The actual kernel selected by `auto` was not captured, and a matched hot measurement is unavailable.

Our 2080 Ti profile uses CK INT8 on steps 1, 2, 7, 8 and sparse Sol INT8 on steps 3, 4, 5, 6. **A verified RTX 3090 timing with that same hybrid profile was not found.** The reference is therefore excluded from cross-GPU speedup claims. Public source research and the missing comparison are recorded in [the 3090 notes](docs/RTX3090.md) and [machine-readable data](results/rtx3090.json).

## Run the portable checks

Python 3.10 or newer is required for the CPU checks. The package has no automatic GPU dependency installation, and importing `h3_turing` does not initialize CUDA.

```bash
python -m pip install -e ".[test]"
python -m pytest -q
python tools/summarize_results.py
```

For native kernels, use Linux x86_64, an SM75 GPU, a CUDA toolkit supporting SM75, PyTorch and the matching ComfyUI / comfy-kitchen environment. Triton is required only for the Triton operators. See [hardware and captured software versions](docs/HARDWARE.md) and [reproduction boundaries](docs/REPRODUCIBILITY.md).

```bash
python tools/build_native.py attention swiglu
python benchmarks/attention_smoke.py
python benchmarks/verify_swiglu.py
H3_RUN_GPU_TESTS=1 python -m pytest -q -m gpu
```

`verify_swiglu.py` requires two CUDA devices and the pinned CK source. Native outputs and verification logs remain local and are ignored by Git. GPU correctness and performance must be requalified on the actual target environment; a CPU test pass does not establish those properties.

Third-party attribution and license texts are retained in [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md) and `THIRD_PARTY_LICENSES/`.
