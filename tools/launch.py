"""Start the isolated ComfyUI on localhost with exactly two visible SM75 GPUs."""
from __future__ import annotations
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys


def physical_devices(indices):
    inventory = subprocess.check_output(["nvidia-smi", "--query-gpu=index,uuid", "--format=csv,noheader,nounits"], text=True)
    devices = dict(tuple(part.strip() for part in line.split(",", 1)) for line in inventory.splitlines() if line.strip())
    if any(i not in devices for i in indices):
        raise ValueError("A selected physical GPU index does not exist in nvidia-smi")
    return ",".join(devices[i] for i in indices)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path.home() / "h3-turing-runtime")
    parser.add_argument("--gpus", default="0,1", help="Two physical GPU indices, e.g. 0,1")
    parser.add_argument("--port", type=int, default=8188)
    parser.add_argument("--cache-classic", action="store_true",
                        help="Retain loader/model objects for the optional Luna recipe; its client forces fresh generation")
    args = parser.parse_args()
    indices = args.gpus.split(",")
    if len(indices) != 2 or len(set(indices)) != 2 or not all(i.isdigit() for i in indices):
        parser.error("--gpus must contain two distinct numeric GPU indices")
    root = args.root.expanduser().resolve()
    if not (root / "h3-stack.json").is_file():
        parser.error("Install the runtime first")
    python = root / ".venv/bin/python"
    if not python.is_file():
        parser.error("Runtime Python is missing; finish tools/setup_stack.py first")
    source = Path(__file__).resolve().parents[1]
    for entry in json.loads((source / "configs/models.lock.json").read_text())["models"]:
        path = root / "ComfyUI/models" / entry["path"]
        if not path.is_file() or path.stat().st_size != entry["bytes"]:
            parser.error(f"Missing model or wrong size: {entry['path']}; run tools/models.py download or verify")
    # CUDA's default FASTEST_FIRST order may differ from nvidia-smi indices on
    # mixed-GPU hosts. Resolve explicit NVML indices to UUIDs without logging them.
    environment = dict(os.environ, CUDA_VISIBLE_DEVICES=physical_devices(indices),
                       CUDA_DEVICE_ORDER="PCI_BUS_ID", H3_SOL_ASYNC_ROUTING="1")
    if "CUDA_HOME" not in environment and Path("/usr/local/cuda/bin/nvcc").is_file():
        environment["CUDA_HOME"] = "/usr/local/cuda"
    probe = '''import torch
import hashlib
from pathlib import Path
import comfy_kitchen.backends.cuda as ck
assert hashlib.sha256(Path(ck.__file__).read_bytes()).hexdigest() == "c5b1c3bc70869e418728114e8c27a889cf5b56ccb31ccf82a001386aef12ee8e", "Pinned comfy-kitchen source required"
assert torch.cuda.device_count() == 2, "Exactly two visible GPUs required"
for i in range(2):
    assert torch.cuda.get_device_capability(i) == (7, 5), "SM75 required"
    assert torch.cuda.get_device_properties(i).total_memory >= 21 * 1024**3, "22 GiB cards required; stock 11 GiB is unsupported"
assert all(torch.cuda.can_device_access_peer(i, 1-i) for i in range(2)), "Bidirectional CUDA P2P required"
print("Two SM75 22 GiB GPUs and bidirectional P2P verified", flush=True)
'''
    subprocess.run([python, "-c", probe], env=environment, check=True)
    command = [str(python), str(root / "ComfyUI/main.py"), "--listen", "127.0.0.1", "--port", str(args.port),
               "--disable-dynamic-vram", "--disable-cuda-malloc", "--reserve-vram", "1", "--preview-method", "none",
               "--cache-classic" if args.cache_classic else "--cache-none", "--disable-auto-launch"]
    raise SystemExit(subprocess.call(command, env=environment, cwd=root / "ComfyUI"))


if __name__ == "__main__":
    main()
