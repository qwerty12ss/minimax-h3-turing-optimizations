"""Public-client boundaries: fresh sampling dependency and preserved test settings."""
import importlib.util
import json
from pathlib import Path
import sys
import types
from unittest.mock import Mock

import pytest

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("recipe_client", ROOT / "tools/run_workflow.py")
client = importlib.util.module_from_spec(spec)
spec.loader.exec_module(client)


def test_repeated_recipe_jobs_invalidate_generation_without_changing_input():
    original = json.loads((ROOT / "workflows/h3_turing_8s_api.json").read_text())
    first = client.luna_graph(original, "qwen_fc2_cache", "first-job")
    second = client.luna_graph(original, "qwen_fc2_cache", "second-job")
    assert "30" not in original
    assert first["30"]["inputs"]["tag"] != second["30"]["inputs"]["tag"]
    assert first["9"]["inputs"]["clip"] == ["30", 1]
    assert first["11"]["inputs"]["model"] == ["30", 0]
    for key in original:
        if key not in ("9", "11", "18"):
            assert first[key] == original[key]
    assert {k: v for k, v in first["9"]["inputs"].items() if k != "clip"} == {
        k: v for k, v in original["9"]["inputs"].items() if k != "clip"}
    assert first["11"]["inputs"]["conditioning"] == original["11"]["inputs"]["conditioning"]


@pytest.mark.parametrize("mode", client.LUNA_MODES)
def test_all_recipe_modes_use_the_same_source_settings(mode):
    original = json.loads((ROOT / "workflows/h3_turing_8s_api.json").read_text())
    candidate = client.luna_graph(original, mode, "test-job")
    assert candidate["30"]["inputs"]["mode"] == mode
    for key in ("3", "7", "8", "10", "12", "13", "14"):
        assert candidate[key] == original[key]


def test_recipe_rejects_unqualified_geometry_and_node_structure():
    graph = json.loads((ROOT / "workflows/h3_turing_smoke_api.json").read_text())
    with pytest.raises(ValueError, match="960x512"):
        client.luna_graph(graph, "qwen_fc2_cache", "job")
    graph = json.loads((ROOT / "workflows/h3_turing_8s_api.json").read_text())
    graph["8"]["inputs"]["preset"] = "baseline"
    with pytest.raises(ValueError, match="optimized"):
        client.luna_graph(graph, "qwen_fc2_cache", "job")


@pytest.mark.parametrize("wrong_tag", [False, True])
def test_client_binds_scalar_stats_to_its_fresh_job(tmp_path, monkeypatch, wrong_tag):
    queued = {}
    graph_path = ROOT / "workflows/h3_turing_8s_api.json"
    schema = {v["class_type"]: {} for v in json.loads(graph_path.read_text()).values()}
    schema["LunaBenchConfigure"] = {}
    counters = dict(blocks=400, norm_calls=800, shared_counts={"phases": 800}, rope_calls=3200, joint_calls=3200)
    def request(base, path, body=None):
        if path == "/object_info":
            return schema
        if path == "/prompt":
            queued.update(body["prompt"])
            return {"prompt_id": "synthetic-job"}
        if path == "/history/synthetic-job":
            return {"synthetic-job": {"status": {"status_str": "success", "messages": [
                ("execution_start", {"timestamp": 1000}), ("execution_success", {"timestamp": 4000})]}}}
        if path == "/h3_turing/validation":
            return {"prompt_id": "synthetic-job", "validation": {
                "video_finite": True, "audio_finite": True, "optimizers": [counters]}}
        if path == "/h3_luna_recipe/stats":
            tag = "another-job" if wrong_tag else queued["30"]["inputs"]["tag"]
            return {"mode": "qwen_fc2_cache", "tag": tag, "stages": {"pack_hits": 50}}
        raise AssertionError(path)
    report = tmp_path / "receipt.json"
    monkeypatch.setattr(client, "request", request)
    monkeypatch.setattr(sys, "argv", ["run_workflow.py", str(graph_path), "--luna-mode", "qwen_fc2_cache", "--report", str(report)])
    if wrong_tag:
        with pytest.raises(RuntimeError, match="another job"):
            client.main()
        assert not report.exists()
    else:
        client.main()
        receipt = json.loads(report.read_text())
        assert receipt["execution_seconds"] == 3.0
        assert receipt["luna_recipe"]["stages"]["pack_hits"] == 50
        assert "outputs" not in receipt


@pytest.mark.parametrize("mode", ["baseline", "cache", "qwen_fc2_cache"])
def test_public_node_keeps_original_calls_and_selects_only_requested_extensions(tmp_path, monkeypatch, mode):
    torch = pytest.importorskip("torch")
    from h3_turing.deployment.pdd.source import runtime_pdd
    routes = []
    def register(path):
        routes.append(path)
        return lambda f: f
    monkeypatch.setenv("H3_LUNA_TRIAL_ROOT", str(tmp_path))
    monkeypatch.setitem(sys.modules, "server", types.SimpleNamespace(PromptServer=types.SimpleNamespace(
        instance=types.SimpleNamespace(routes=types.SimpleNamespace(get=register)))))
    monkeypatch.setitem(sys.modules, "aiohttp", types.SimpleNamespace(web=types.SimpleNamespace(json_response=lambda data: data)))
    name = "h3_public_recipe_test"
    spec = importlib.util.spec_from_file_location(name, ROOT / "recipes/luna/__init__.py")
    recipe = importlib.util.module_from_spec(spec)
    monkeypatch.setitem(sys.modules, name, recipe)
    spec.loader.exec_module(recipe)
    assert routes == ["/h3_luna_recipe/stats"]  # no raw media/tensor dump routes
    monkeypatch.setattr(recipe, "install_gemm", lambda: None)
    monkeypatch.setattr(recipe.ctypes, "CDLL", lambda path: types.SimpleNamespace(launch=Mock(return_value=0)))
    monkeypatch.setattr(torch.cuda, "synchronize", lambda device: None)
    cache_install, qwen_wrap = Mock(), Mock(side_effect=lambda clip, transformer, stats: clip.clone())
    monkeypatch.setitem(sys.modules, name + ".packed_shards", types.SimpleNamespace(install=cache_install))
    monkeypatch.setitem(sys.modules, name + ".awq_tp", types.SimpleNamespace(wrap=qwen_wrap))
    monkeypatch.setitem(sys.modules, "h3_turing.swiglu_convrot.joint", types.SimpleNamespace(
        mlp_function=lambda core, prepare, fc2: (fc2, {})))
    class Conditioning:
        def named_modules(self):
            return [("qwen", types.SimpleNamespace(model=types.SimpleNamespace(layers=[])))]
        def encode_token_weights(self, value):
            return value
    class Clip:
        def __init__(self, csm=None):
            self.cond_stage_model = csm or Conditioning()
        def clone(self):
            return Clip(self.cond_stage_model)
        def encode_from_tokens(self, value):
            return self.cond_stage_model.encode_token_weights(value)
    state = types.SimpleNamespace(_h3_public_optimizer=types.SimpleNamespace(
        joint_fc2=lambda x: x, core=object(), joint_prepare=object()), prepare=lambda x: x)
    model = types.SimpleNamespace(**{runtime_pdd.STATE_ATTRIBUTE: state})
    original_clip = Clip()
    result_model, result_clip = recipe.LunaBenchConfigure().apply(model, original_clip, mode, "job")
    sentinel = object()
    assert result_model is model and result_clip is not original_clip
    assert result_clip.encode_from_tokens(sentinel) is sentinel
    assert state.prepare(sentinel) is sentinel
    assert state._h3_public_optimizer.mlp(sentinel) is sentinel
    assert not getattr(recipe.LOCAL, "fc2", False)
    assert cache_install.call_count == int("cache" in mode)
    assert qwen_wrap.call_count == int("qwen" in mode)
