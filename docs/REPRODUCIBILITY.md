# Reproduction boundaries

The public package contains reviewed research snapshots, portable CPU contracts, synthetic GPU checks, local CUDA build recipes and curated numeric ledgers. It includes a new portable deployment and synthetic test workflows; see [installation](INSTALL.md). Model weights are downloaded separately from pinned public sources. Historical media and original creative prompts are excluded, so the historical videos cannot be recreated bit for bit using this repository.

## Dependencies and builds

Install the package in the Python environment used by your ComfyUI runtime. CPU checks need only Python and pytest. GPU operators use PyTorch; the Triton kernels additionally need Triton. CK integration imports an existing comfy-kitchen installation. The isolated installer and model downloader create a separate runtime; they do not change a running ComfyUI installation.

The captured CK Python source SHA256 is `c5b1c3bc70869e418728114e8c27a889cf5b56ccb31ccf82a001386aef12ee8e`. Source-pinned AST adapters refuse different bytes. The original TP core hash used by the optional SwiGLU MLP rewrite is `e7c45a7f44d4ce81c9b20786a5d5eb838ad89c284859f000a65fa4a3fbcac4a5`. The original TP2 core is now included under `h3_native_int8_tp2_central_sparse`, alongside an independently packaged deployment adapter and synthetic workflows. Updating these pins requires requalification, not merely editing the expected hash.

`python tools/build_native.py attention swiglu` locates local nvcc through CUDA_HOME/CUDA_ROOT/CUDA_PATH or PATH. The Linux x86_64 native recipes target SM75. The attention recipe retains the original fast-math profile; the joint SwiGLU recipe uses explicit non-fused, non-FTZ arithmetic flags and checks generated PTX for its specific contract. This does not make the entire attention algorithm bitwise equal to dense BF16 attention.

The attention build emits a local binary manifest with a freshly computed checksum; the loader also checks ABI 7. `benchmarks/attention_smoke.py` exercises synthetic routing only. It is not an accuracy test or a video latency benchmark. To use full attention, call the explicit `sol_sm75_backend.prepare` / `run` functions with their shape, dtype and routing contracts; caller code must preserve the chosen attention step schedule.

`python benchmarks/verify_swiglu.py` runs the historical operator fixture plan against the installed CK dependency on two CUDA devices. It writes local results under `benchmark-output/`. `H3_RUN_GPU_TESTS=1 python -m pytest -m gpu -q` runs smaller synthetic Triton rounding checks on SM75. An omitted CUDA/Triton toolchain is reported as a skip. CPU checks do not validate generated GPU machine code or historical speed claims.

The experimental `headtau` and `quant` targets can be built explicitly with the same build tool; neither is part of the default accepted profile. Never use `python -O` for qualification: several original operator admission gates use Python assertions.

## Source adaptations

The bundled source-pinned snapshots retain their original docstrings, including historical qualification status. Current deployment checks are recorded separately in [DEPLOYMENT_VALIDATION.md](DEPLOYMENT_VALIDATION.md); changing a snapshot comment would invalidate its full-byte source guard.

`results/source_manifest.json` records SHA256 of original snapshots and published files. Changes include package-relative imports, replacement of deployment paths by installed-package resolution, removal of site orchestration, local output directories and removal of machine-specific historical qualification loaders. Shape/index/arithmetic contracts remain visible. Historical qualification receipts are not substitutes for testing a new build.

GPU smoke tests, raw numerical equality, graph behavior and latency are distinct checks. Requalify all relevant layers after compiler, PyTorch, CK, model, shape or hardware changes. Keep cold jobs separate from hot jobs, retain before/after controls, and report the estimator alongside the timing.
