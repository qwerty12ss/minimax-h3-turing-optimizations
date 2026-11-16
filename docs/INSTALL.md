# Deploy MiniMax H3 from scratch

**English** | [Русский](INSTALL.ru.md)

This installs an isolated ComfyUI, FP16 H3 loader, PDD, dual-GPU TP2 and the published acceleration operators. The supplied workflows create a synthetic test image and use a new neutral prompt. No private assets, credentials or server configuration are needed.

## Requirements

Linux x86_64, Python **3.12** (Ubuntu 24.04 is a suitable starting point), **two SM75 RTX 2080 Ti cards modified to 22 GiB each**, bidirectional CUDA P2P, approximately **128 GB host RAM**, and **90 GB free disk space**. Stock 11 GiB cards are unsupported. The five weights occupy 43.84 GB. Install **CUDA Toolkit 13.0**, including `nvcc`, and a compatible NVIDIA driver using [NVIDIA's Linux installation guide](https://docs.nvidia.com/cuda/cuda-installation-guide-linux/) and the [CUDA 13.0 archive](https://developer.nvidia.com/cuda-13-0-0-download-archive). The recorded driver is from branch 580. Windows and WSL2 are not qualified deployment targets.

The historical host used 2 × Xeon E5-2686 v4, 128 GB RAM, PCIe 3.0 ×16 per GPU and NV2. See [hardware](HARDWARE.md) for recorded details. Other interconnects require fresh P2P and performance checks.

## Installation

```bash
sudo apt update
sudo apt install -y git build-essential python3.12 python3.12-venv ffmpeg
nvcc --version
nvidia-smi
git clone https://github.com/qwerty12ss/minimax-h3-turing-optimizations.git
cd minimax-h3-turing-optimizations
export CUDA_HOME=/usr/local/cuda
python3.12 tools/setup_stack.py --root "$HOME/h3-turing-runtime"
python3.12 tools/models.py download --root "$HOME/h3-turing-runtime"
python3.12 tools/launch.py --root "$HOME/h3-turing-runtime" --gpus 0,1 --port 8188
```

Set `CUDA_HOME` to your Toolkit location. The installer checks out [pinned Git revisions](../configs/stack.lock.json), creates its own venv, installs ComfyUI under [version constraints](../requirements/runtime-constraints.txt), runs `pip check`, and builds attention/SwiGLU locally for SM75. It refuses unowned nonempty runtime directories or modified dependency checkouts. Existing ComfyUI installations are untouched. Keep this source checkout in place because installation is editable.

The model downloader needs no token. It verifies sizes and full SHA256 before accepting each weight. Exact source revisions and filenames are in [models.lock.json](../configs/models.lock.json). Sources: [Comfy-Org/MiniMax-H3](https://huggingface.co/Comfy-Org/MiniMax-H3/tree/4b53a35191437de05370da548e59aeabe2a5a09a) (FL2VA INT8 ConvRot, Qwen3-VL NVFP4 encoder, FP16 video VAE and FP32 audio VAE) and [alibaba-pai/MiniMax-H3-Acc-LoRAs](https://huggingface.co/alibaba-pai/MiniMax-H3-Acc-LoRAs/tree/335001fb9e5455d68a0caa18ec2e319072150328) (FL2VA 8-step PDD). Model licenses remain those of the source repositories.

For an existing public model cache, replace `download` with:

```bash
python3.12 tools/models.py link --root "$HOME/h3-turing-runtime" --from-models /path/to/existing/ComfyUI/models
```

This hashes all five files before linking; the originals are untouched. Recheck with `tools/models.py verify`. Use the original FL2VA checkpoint plus the separate PDD file, not a PDD-baked checkpoint.

## Run the supplied workflows

Leave ComfyUI running and open another terminal in this repository:

```bash
python3.12 tools/run_workflow.py workflows/h3_turing_smoke_api.json
python3.12 tools/run_workflow.py workflows/h3_turing_8s_api.json
```

The smoke workflow uses 640×352, 124 frames, 8 steps and serial VAE. The full workflow uses 960×512, 192 frames, 24 fps, 8 steps and temporal hybrid VAE. Both save MP4 with audio under the runtime's `ComfyUI/output/video`. The client waits for completion and checks finite decoded frames/audio. Optional `--report /path/to/local-receipt.json` saves a compact numerical receipt without machine paths or prompts.

`--gpus 0,1` selects physical GPU indices; they become logical `cuda:0,1`. The launcher verifies SM75, memory capacity and bidirectional P2P. It listens on localhost only. Open `http://127.0.0.1:8188` for the UI; use your own SSH tunnel for a remote host. To avoid an occupied port, launch with e.g. `--port 8197` and add `--url http://127.0.0.1:8197` to the workflow client. Use `*_gui.json` in [workflows](../workflows) in the visual editor; the API files belong to the CLI client.

Physical selection is independent of CUDA's default ordering on mixed-GPU hosts. The launcher uses the native CUDA caching allocator (`--disable-cuda-malloc`), disables dynamic VRAM and disables graph caching. These settings are part of the tested recipe: the asynchronous allocator exhausted device memory in the integration test, while the native allocator completed the smoke profile.

## Configuration

Keep the trained chain: original FL2VA → SigmaShift **12/3** → FL2VA PDD **nfe=8**, strengths **1/1** → Activation Chunk → H3TuringApply → BasicGuider (CFG **1**) → **Euler**, with sigmas from PDD. Hybrid attention uses CK INT8 on steps 1,2,7,8 and Sol all-INT8 on steps 3–6, `tau=1`; QKV/MLP chunks are **8192**. An arbitrary scheduler or step count is not equivalent.

`H3TuringApply.preset=optimized` enables shared QKV/FC1 preparation, exact Half scale/add and FP32 SUM, norm/mod, direct-output RoPE and joint SwiGLU/ConvRot. `baseline` retains the original central TP2 block with Half A/B PDD and the same attention schedule. Temporal hybrid VAE is admitted only for the shapes listed in `vae/spatial.py`, including 960×512×192. Use serial VAE and new qualification for other shapes. H3 frame lengths follow **17k+5**. Concurrent TP2 model execution, different GPU architectures and arbitrary geometries are unsupported.

To use your own input after the smoke test, replace H3TuringTestImage with LoadImage and connect it to `MiniMaxH3ImageToVideo.first_frame`; edit that node's prompt. For API use, upload the image to your own ComfyUI first and update the graph. Keep assets, outputs and local logs outside Git.

## Checks and troubleshooting

```bash
"$HOME/h3-turing-runtime/.venv/bin/python" -m pytest -q
CUDA_VISIBLE_DEVICES=0,1 "$HOME/h3-turing-runtime/.venv/bin/python" benchmarks/attention_smoke.py
CUDA_VISIBLE_DEVICES=0,1 H3_RUN_GPU_TESTS=1 "$HOME/h3-turing-runtime/.venv/bin/python" -m pytest -q -m gpu
```

Missing nodes mean a plugin import failed: inspect startup errors and use the pinned installer. Hash mismatches reject corrupted or different files; remove/reacquire the incorrect file deliberately. Source guard failures require the pinned dependency rather than editing the expected hash. Native build failures require Toolkit 13.0, correct CUDA_HOME, a C++ compiler and disk space. P2P failures require a compatible GPU topology/driver. For OOM, check actual 22 GiB cards, other GPU processes and the supplied geometry. Do not run qualification with `python -O`.

The async attention log's `density=nan` is a deferred statistics placeholder. The validation node flushes routing checks and independently verifies finite video/audio tensors before saving.

See [the deployment validation receipt](DEPLOYMENT_VALIDATION.md) for actual checks and limits. A finite decoded output is not proof of universal bitwise equivalence or artistic quality. Keep other GPU workloads absent for performance measurements; shared-load functional checks do not establish historical latency.
