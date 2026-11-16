"""Install a pinned, isolated Linux stack. Existing ComfyUI installations are untouched."""
from __future__ import annotations
import argparse
import json
import os
from pathlib import Path
import platform
import shutil
import subprocess
import sys

REPO = Path(__file__).resolve().parents[1]


def run(argv, **kwargs):
    print("+", " ".join(map(str, argv)), flush=True)
    subprocess.run(list(map(str, argv)), check=True, **kwargs)


def checkout(path, entry):
    if path.exists():
        if not (path / ".git").is_dir():
            raise RuntimeError(f"Refusing to replace an existing directory: {path}")
        head = subprocess.check_output(["git", "-C", str(path), "rev-parse", "HEAD"], text=True).strip()
        if head != entry["commit"]:
            raise RuntimeError(f"Existing checkout has another revision: {path}")
        if subprocess.check_output(["git", "-C", str(path), "status", "--porcelain", "--untracked-files=no"], text=True).strip():
            raise RuntimeError(f"Existing checkout has modified tracked files: {path}")
        return
    path.mkdir(parents=True)
    run(["git", "init", "--quiet", path])
    run(["git", "-C", path, "remote", "add", "origin", entry["url"]])
    run(["git", "-C", path, "fetch", "--depth=1", "origin", entry["commit"]])
    run(["git", "-C", path, "checkout", "--quiet", "--detach", "FETCH_HEAD"])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path.home() / "h3-turing-runtime")
    parser.add_argument("--skip-native-build", action="store_true", help="Install Python stack only; build kernels before generation")
    args = parser.parse_args()
    if platform.system() != "Linux" or platform.machine().lower() not in {"x86_64", "amd64"}:
        raise RuntimeError("Use Linux x86_64 (tested Python 3.12); Windows is not a native deployment target")
    if sys.version_info[:2] != (3, 12):
        raise RuntimeError("Run this pinned installer with Python 3.12")
    if not shutil.which("git") or not shutil.which("c++"):
        raise RuntimeError("Install git and build-essential first; see docs/INSTALL.ru.md")
    root = args.root.expanduser().resolve()
    if root == REPO or root in REPO.parents or REPO in root.parents:
        raise RuntimeError("Choose a runtime root outside the source repository")
    lock = json.loads((REPO / "configs/stack.lock.json").read_text())
    root.mkdir(parents=True, exist_ok=True)
    marker = root / "h3-stack.json"
    if any(root.iterdir()) and not marker.exists():
        raise RuntimeError("Choose an empty runtime root; existing unowned directories are not modified")
    if marker.exists() and json.loads(marker.read_text())["stack"] != lock:
        raise RuntimeError("This runtime belongs to another stack revision; choose a fresh directory")
    marker.write_text(json.dumps({"stack": lock, "source": str(REPO)}, indent=2) + "\n")
    comfy = root / "ComfyUI"
    checkout(comfy, lock["repositories"]["ComfyUI"])
    for name, entry in lock["repositories"].items():
        if name != "ComfyUI":
            checkout(comfy / "custom_nodes" / name, entry)
    python = root / ".venv/bin/python"
    if not python.exists():
        run([sys.executable, "-m", "venv", root / ".venv"])
    pip = [python, "-m", "pip", "--disable-pip-version-check"]
    run(pip + ["install", "--no-cache-dir", "torch==2.13.0+cu130", "torchvision==0.28.0+cu130", "--index-url", lock["torch_index"], "-c", REPO / "requirements/runtime-constraints.txt"])
    run(pip + ["install", "--no-cache-dir", "-r", comfy / "requirements.txt", "-c", REPO / "requirements/runtime-constraints.txt", "pytest>=8"])
    run(pip + ["install", "--no-deps", "-e", REPO])
    run(pip + ["check"])
    node = comfy / "custom_nodes/h3_turing_optimizations"
    node.mkdir(exist_ok=True)
    shim = 'from h3_turing.deployment.nodes import NODE_CLASS_MAPPINGS, NODE_DISPLAY_NAME_MAPPINGS\n'
    init = node / "__init__.py"
    if init.exists() and init.read_text() != shim:
        raise RuntimeError("Refusing to replace a modified integration shim")
    init.write_text(shim)
    if not args.skip_native_build:
        environment = dict(os.environ)
        if "CUDA_HOME" not in environment and Path("/usr/local/cuda/bin/nvcc").is_file():
            environment["CUDA_HOME"] = "/usr/local/cuda"
        run([python, REPO / "tools/build_native.py", "attention", "swiglu"], env=environment)
    print(f"Installed. Next: {python} {REPO / 'tools/models.py'} download --root {root}")


if __name__ == "__main__":
    main()
