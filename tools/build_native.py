"""Build reviewed native operators locally; never downloads binaries or code."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import platform
import shutil
import subprocess

ROOT = Path(__file__).resolve().parents[1]
PACKAGE = ROOT / "src" / "h3_turing"


def compiler():
    for name in ("CUDA_HOME", "CUDA_ROOT", "CUDA_PATH"):
        if os.environ.get(name):
            candidate = Path(os.environ[name]) / "bin" / "nvcc"
            if candidate.is_file():
                return str(candidate)
    found = shutil.which("nvcc")
    if not found:
        raise RuntimeError("nvcc missing: set CUDA_HOME or add it to PATH")
    return found


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def build(target, nvcc):
    if platform.system() != "Linux":
        raise RuntimeError("These published native recipes target Linux x86_64")
    if platform.machine().lower() not in {"x86_64", "amd64"}:
        raise RuntimeError("The native ABI recipe requires x86_64")
    common = [nvcc, "-O3", "-std=c++17", "-arch=sm_75"]
    ieee = ["--fmad=false", "--ftz=false", "--prec-div=true", "--prec-sqrt=true"]
    if target == "attention":
        src = PACKAGE / "attention/csrc/sla_sm75_sparse.cu"
        out = PACKAGE / "attention/bin/linux_x86_64/star7_sla_sm75_v7.so"
        flags = ["--use_fast_math", "--cudart=static", "-Xlinker=--exclude-libs,ALL"]
    elif target == "swiglu":
        src = PACKAGE / "swiglu_convrot/joint.cu"
        out = src.with_name("libjoint.so")
        flags = ieee + ["-Xptxas=-v"]
    elif target == "headtau":
        src = PACKAGE / "experimental/headtau_routes.cu"
        out = src.with_name("libheadtau_v1.so")
        flags = ["--use_fast_math"]
    else:
        src = PACKAGE / "experimental/native_quant.cu"
        out = src.with_name("libh3_quant.so")
        flags = ["--use_fast_math"]
    out.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run(common + flags + ["-shared", "-Xcompiler=-fPIC", str(src), "-o", str(out)], check=True)
    if target == "swiglu":
        ptx = src.with_suffix(".ptx")
        subprocess.run(common + flags + ["-ptx", str(src), "-o", str(ptx)], check=True)
        text = ptx.read_text()
        if "fma.rn.f32" in text or any("div.approx.ftz.f32" not in line for line in text.splitlines() if ".ftz" in line):
            raise RuntimeError("Joint PTX failed the historical arithmetic contract")
    if target == "attention":
        manifest = {"linux_x86_64": {"file": "linux_x86_64/" + out.name, "sha256": sha(out)}}
        (out.parent.parent / "sm75_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps({"target": target, "source_sha256": sha(src), "binary_sha256": sha(out)}))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("targets", nargs="+", choices=["attention", "swiglu", "headtau", "quant"])
    args = parser.parse_args()
    nvcc = compiler()
    for target in args.targets:
        build(target, nvcc)
