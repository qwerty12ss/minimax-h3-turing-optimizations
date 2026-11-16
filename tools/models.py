"""Download public weights or link a verified existing model cache. No tokens required."""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path, PurePosixPath
import urllib.request

REPO = Path(__file__).resolve().parents[1]


def model_path(base, relative):
    path = PurePosixPath(relative)
    if path.is_absolute() or ".." in path.parts or "\\" in relative or ":" in relative:
        raise ValueError("Unsafe model manifest path")
    return base.joinpath(*path.parts)


def verify(path, entry):
    if not path.is_file() or path.stat().st_size != entry["bytes"]:
        raise RuntimeError(f"Missing model or wrong size: {entry['path']}")
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    if digest.hexdigest() != entry["sha256"]:
        raise RuntimeError(f"SHA256 mismatch: {entry['path']}")


def url(entry):
    return f"https://huggingface.co/{entry['repository']}/resolve/{entry['revision']}/{entry['source_path']}"


def download(target, entry):
    if target.exists() or target.is_symlink():
        verify(target, entry)
        return
    target.parent.mkdir(parents=True, exist_ok=True)
    partial = target.with_suffix(target.suffix + ".partial")
    # Restart interrupted transfers; a partial file is never accepted as a model.
    with urllib.request.urlopen(url(entry), timeout=120) as source, partial.open("wb") as output:
        while chunk := source.read(8 * 1024 * 1024):
            output.write(chunk)
    verify(partial, entry)
    partial.replace(target)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=["download", "verify", "link"])
    parser.add_argument("--root", type=Path, default=Path.home() / "h3-turing-runtime")
    parser.add_argument("--from-models", type=Path, help="Existing ComfyUI/models directory (link action)")
    args = parser.parse_args()
    root = args.root.expanduser().resolve()
    if not (root / "h3-stack.json").is_file():
        parser.error("Install this runtime with tools/setup_stack.py first")
    if args.action == "link" and args.from_models is None:
        parser.error("link requires --from-models")
    entries = json.loads((REPO / "configs/models.lock.json").read_text())["models"]
    for entry in entries:
        print(f"{args.action}: {entry['path']}", flush=True)
        target = model_path(root / "ComfyUI/models", entry["path"])
        if args.action == "download":
            download(target, entry)
        elif args.action == "verify":
            verify(target, entry)
        else:
            source = model_path(args.from_models.expanduser().resolve(), entry["path"])
            verify(source, entry)
            if target.exists() or target.is_symlink():
                verify(target, entry)
            else:
                target.parent.mkdir(parents=True, exist_ok=True)
                target.symlink_to(source)
    print("All five public models passed SHA256 verification.")


if __name__ == "__main__":
    main()
