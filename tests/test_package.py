import ast
import hashlib
import json
from pathlib import Path
import re
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]


def test_cpu_entrypoint_does_not_import_gpu_frameworks():
    result = subprocess.run([sys.executable, "-c",
        "import sys; sys.path.insert(0, 'src'); import h3_turing; assert 'torch' not in sys.modules; assert 'triton' not in sys.modules"],
        cwd=ROOT, capture_output=True, text=True)
    assert result.returncode == 0, result.stderr


def test_reviewed_source_bytes_and_python_syntax():
    data = json.loads((ROOT / "results/source_manifest.json").read_text())
    paths = []
    for record in data["sources"]:
        p = ROOT / record["path"]
        paths.append(record["path"])
        assert hashlib.sha256(p.read_bytes()).hexdigest() == record["published_sha256"], record["path"]
    assert len(paths) == len(set(paths))
    for p in itertools_python_files():
        ast.parse(p.read_text(encoding="utf-8"), filename=str(p))


def itertools_python_files():
    for folder in ("src", "tools", "benchmarks", "tests"):
        yield from (ROOT / folder).rglob("*.py")


def test_native_quoted_include_dependencies_are_present():
    for p in (ROOT / "src").rglob("*"):
        if p.suffix not in {".cu", ".cuh"}:
            continue
        for name in re.findall(r'^\s*#include\s+"([^"\n]+)"', p.read_text(), flags=re.M):
            assert (p.parent / name).is_file(), (p.relative_to(ROOT), name)
