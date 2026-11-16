import hashlib
import importlib.util
import json
from pathlib import Path
import sys

import pytest

ROOT = Path(__file__).resolve().parents[1]


def load_tool(name):
    spec = importlib.util.spec_from_file_location(name, ROOT / "tools" / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_models_reject_corruption_and_directory_escape(tmp_path):
    models = load_tool("models")
    blob = tmp_path / "model.safetensors"
    blob.write_bytes(b"small public test fixture")
    entry = dict(path="vae/model.safetensors", bytes=blob.stat().st_size,
                 sha256=hashlib.sha256(blob.read_bytes()).hexdigest())
    models.verify(blob, entry)
    blob.write_bytes(b"corrupted fixture content")
    with pytest.raises(RuntimeError):
        models.verify(blob, entry)
    with pytest.raises(RuntimeError):
        models.download(blob, entry)
    assert blob.read_bytes() == b"corrupted fixture content"
    for value in ("../secret", "/absolute", "vae/../../escape", "C:/secret", "vae\\secret"):
        with pytest.raises(ValueError):
            models.model_path(tmp_path, value)


def test_verified_download_publishes_only_matching_bytes(tmp_path, monkeypatch):
    import io
    models = load_tool("models")
    target = tmp_path / "vae/test.safetensors"
    data = b"a public synthetic fixture"
    entry = dict(path="vae/test.safetensors", bytes=len(data), sha256=hashlib.sha256(data).hexdigest(),
                 repository="example/model", revision="0" * 40, source_path="test.safetensors")
    monkeypatch.setattr(models.urllib.request, "urlopen", lambda *a, **k: io.BytesIO(b"x" * len(data)))
    with pytest.raises(RuntimeError):
        models.download(target, entry)
    assert not target.exists()
    monkeypatch.setattr(models.urllib.request, "urlopen", lambda *a, **k: io.BytesIO(data))
    models.download(target, entry)
    assert target.read_bytes() == data and not target.with_suffix(".safetensors.partial").exists()


def test_workflow_generation_and_connection_contracts():
    make = load_tool("make_workflows")
    for name, dimensions in [("h3_turing_smoke", (640, 352, 124, "serial")), ("h3_turing_8s", (960, 512, 192, "temporal_hybrid"))]:
        graph = json.loads((ROOT / f"workflows/{name}_api.json").read_text())
        assert graph == make.graph(*dimensions, name)
        assert graph["3"]["inputs"]["nfe"] == "8"
        assert graph["14"]["inputs"]["sigmas"] == ["3", 1]
        assert graph["9"]["inputs"]["first_frame"] == ["19", 0]
        assert graph["17"]["inputs"]["images"] == ["20", 0]
        for node in graph.values():
            for value in node["inputs"].values():
                if isinstance(value, list):
                    assert value[0] in graph and isinstance(value[1], int)


def test_public_model_manifest_is_complete_and_revision_pinned():
    models = load_tool("models")
    entries = json.loads((ROOT / "configs/models.lock.json").read_text())["models"]
    assert len(entries) == 5 and len({e["path"] for e in entries}) == 5
    for entry in entries:
        assert len(entry["sha256"]) == 64 and len(entry["revision"]) == 40
        assert entry["bytes"] > 0
        assert models.url(entry).startswith("https://huggingface.co/")
        models.model_path(Path("models"), entry["path"])


def test_physical_gpu_selection_does_not_assume_cuda_numeric_order(monkeypatch):
    launch = load_tool("launch")
    monkeypatch.setattr(launch.subprocess, "check_output", lambda *a, **k:
                        "0, GPU-synthetic-a\n1, GPU-synthetic-b\n2, GPU-synthetic-c\n")
    assert launch.physical_devices(["2", "1"]) == "GPU-synthetic-c,GPU-synthetic-b"
    with pytest.raises(ValueError):
        launch.physical_devices(["1", "3"])


def test_gui_conversion_preserves_optional_defaults_and_link_slots():
    make = load_tool("make_gui_workflows")
    schemas = {
        "Source": {"input": {}, "output": ["MODEL"], "output_name": ["model"]},
        "Target": {"input": {"required": {"model": ["MODEL"]}, "optional": {
            "partition": ["STRING", {"default": ""}], "enabled": ["BOOLEAN", {"default": True}],
            "other": ["MODEL"], "check": [["error", "warn"], {"default": "error"}]}},
            "output": []}}
    graph = {"1": dict(class_type="Source", inputs={}),
             "2": dict(class_type="Target", inputs={"model": ["1", 0], "check": "warn"})}
    gui = make.convert(graph, schemas)
    assert gui["nodes"][1]["widgets_values"] == ["", True, "warn"]
    assert gui["nodes"][1]["inputs"] == [dict(name="model", type="MODEL", link=1), dict(name="other", type="MODEL", link=None)]
    assert gui["links"] == [[1, 1, 0, 2, 0, "MODEL"]]


def test_original_tp2_and_prepare_transforms_keep_source_guards():
    from h3_turing.norm_mod.source_transform import transform_central
    from h3_turing.deployment.pdd.transform import audit
    central = ROOT / "src/h3_turing/h3_native_int8_tp2_central_sparse/central_block.py"
    _, proof = transform_central(central.read_bytes())
    assert proof
    # Changed arithmetic must fail the exact expression gate.
    with pytest.raises((AssertionError, RuntimeError)):
        transform_central(central.read_bytes().replace(b"sh['eps1']", b"sh['eps1'] * 2"))
    assert audit()["whole_inverse_equal"]


def test_scalar_validation_receipt_is_bound_to_current_job():
    from h3_turing.deployment.validation import record_optimizer, optimizer_reports
    record_optimizer("synthetic-job-a", {"blocks": 400})
    assert optimizer_reports("synthetic-job-a") == [{"blocks": 400}]
    assert optimizer_reports("synthetic-job-b") == []
    record_optimizer("synthetic-job-b", {"blocks": 800})
    assert optimizer_reports("synthetic-job-a") == []
