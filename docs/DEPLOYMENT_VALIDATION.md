# Standalone deployment validation — November 16, 2026

**English** | [Русский](DEPLOYMENT_VALIDATION.ru.md)

The supplied optimized workflows completed in a freshly created Linux Python 3.12.3 venv on two RTX 2080 Ti cards modified to 22 GiB each. The tested implementation is [756f3d2](https://github.com/qwerty12ss/minimax-h3-turing-optimizations/commit/756f3d23f9abcefc49b84bbac229dbdcfcb68c72). This is a functional deployment check; historical performance results remain separate.

Machine-readable evidence, including execution counters and hashes of **92 tested source/config/workflow files**, is in [deployment_validation.json](../results/deployment_validation.json). Every listed hash was compared with the committed release.

## Installation and model sources

`tools/setup_stack.py` created a new venv without system site packages, checked out the public dependencies at the revisions in `configs/stack.lock.json`, installed the Python stack, passed `pip check`, and built native attention and joint SwiGLU from source. The joint PTX arithmetic gate passed. The final constraints record 107 resolved dependency versions. Re-running the installer with those final constraints also succeeded.

All five model files were existing copies of the public weights, fully SHA256-verified before linking into the isolated runtime. The test did **not** download another complete 43.84 GB copy. Each pinned source URL was separately accessed anonymously with an HTTP range request: all five returned HTTP 206 and valid safetensors headers. Downloader tests additionally checked corruption rejection, partial-file publication and directory traversal rejection. No hidden model files or private assets were used.

The launcher selected the physical GPUs independently of CUDA's default enumeration, checked SM75, memory capacity and bidirectional P2P, and started a separate localhost ComfyUI on an unused port. Required nodes were available. The tested launch uses the native caching allocator, dynamic VRAM disabled and `--cache-none`. Other existing applications were not stopped or reconfigured.

## Checks

| Check | Observed result |
|---|---|
| CPU and metadata tests in the clean Linux venv | 18 passed |
| Opt-in SM75 GPU tests | 8 passed; combined suite: **26 passed** |
| Native attention routing smoke | Passed; routing scope only |
| Joint SwiGLU/ConvRot | 44 fixtures, 88 variants across both 2080 Ti GPUs and default/nondefault streams; **zero bit mismatches** in activation, quantized data and scales; repeat/input-preservation checks passed |
| Fresh separate Git clone | 18 CPU tests passed; 8 opt-in GPU tests skipped on the publication workstation |
| Python wheel | Built; frozen TP2 source bytes and runtime integration preserved |
| Repeated isolated installer | Passed with the final dependency constraints |

## Complete video/audio workflows

Both graphs used the bundled synthetic image generator, the new neutral fixture prompt, seed 42, original FL2VA INT8 ConvRot plus PDD, Euler, CFG 1, SigmaShift 12/3 and eight PDD steps. Attention used CK INT8 on steps 1,2,7,8 and Sol all-INT8 on steps 3–6. Decoded image/audio tensors were finite and deferred routing checks were flushed before saving.

| API workflow | Video | VAE | Independently decoded MP4 |
|---|---|---|---|
| `h3_turing_smoke_api.json` | 640×352, 124 frames, 24 fps | Serial | All 124 H.264 frames and 162 AAC audio frames decoded; 32 kHz audio |
| `h3_turing_8s_api.json` | 960×512, 192 frames, 24 fps | Temporal hybrid | All 192 H.264 frames and 250 AAC audio frames decoded; 32 kHz audio |

Each final receipt recorded **400 TP2 block executions, 800 norm/mod calls and 800 shared preparation phases**. Smoke/full respectively recorded 1,600/3,200 direct-output RoPE calls and 1,600/3,200 joint SwiGLU calls. The full parallel VAE logged 11 groups and 165 tiles. Its replica occupied 4,847,070,912 bytes. Scalar receipts survive model unloading without retaining model objects; the client checks that the receipt belongs to its own job.

GUI counterparts were generated from the installed node schemas, with connection slots and optional widget defaults checked. The editor itself was not driven interactively; the API counterparts above are the executed workflows. Use the CLI recipe for the directly verified path.

## Scope and publication

Other GPU workloads were present. No new comparative latency measurement is published from these runs. Functional success, operator-level bit checks and universal full-graph bitwise equivalence are distinct; the last is not claimed. This check does not qualify arbitrary shapes, the baseline preset, stock 11 GiB cards, SM80+ GPUs or artistic scene quality.

Gitleaks 8.30.1 checked release files and the new Git history with full redaction and inline allow directives disabled: no secret findings. A separate identifier/path review found no private server addresses/paths or unrelated project material. Public files contain code, settings, synthetic workflows, documentation and numeric evidence. Model weights, media, raw logs, conversation exports, credentials and server inventories are excluded. The original repository remains private.

Start with [installation](INSTALL.md) or [the Russian installation recipe](INSTALL.ru.md).
