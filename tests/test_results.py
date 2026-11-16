import csv
import importlib.util
import json
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("summary", ROOT / "tools/summarize_results.py")
summary = importlib.util.module_from_spec(spec)
spec.loader.exec_module(summary)


def test_headline_metrics_recompute_from_public_rows():
    for name, actual in summary.published_summaries().items():
        recorded = json.loads((ROOT / f"results/{name}.json").read_text())
        assert actual["control_seconds"] == pytest.approx(recorded.get("control_mean_of_medians_seconds", recorded.get("control_hot_median_seconds")))
        assert actual["candidate_seconds"] == pytest.approx(recorded.get("candidate_hot_median_seconds", recorded.get("hot_medians_seconds", {}).get("candidate")))
        assert actual["reduction_percent"] == pytest.approx(recorded.get("speedup_percent", recorded.get("reduction_percent")))


def test_cold_and_diagnostic_observations_cannot_change_hot_summary():
    rows = [{"arm": arm, "phase": "hot1", "execution_seconds": str(t)}
            for arm, t in [("control-before", 10), ("control-after", 14), ("candidate", 9)]]
    rows = rows + [{**r, "phase": "hot2"} for r in rows]
    base = summary.summarize(rows)
    rows += [{"arm": "candidate", "phase": "cold", "execution_seconds": "10000"},
             {"arm": "candidate", "phase": "hot1", "measurement_class": "diagnostic_replay", "execution_seconds": "0.001"}]
    assert summary.summarize(rows) == base
    with pytest.raises(ValueError):
        summary.summarize([r for r in rows if r["arm"] != "control-before"])


def test_ledgers_have_recorded_counts_and_columns():
    manifest = json.loads((ROOT / "results/table_manifest.json").read_text())
    for name, expected in manifest.items():
        with (ROOT / f"results/{name}.csv").open(newline="") as f:
            reader = csv.DictReader(f)
            assert reader.fieldnames == expected["columns"]
            assert len(list(reader)) == expected["rows"]
    for name in ("shared_pipeline", "temporal_vae"):
        d = json.loads((ROOT / f"results/{name}.json").read_text())
        assert d["all_99_fingerprints_match"] is False
    r = json.loads((ROOT / "results/rtx3090.json").read_text())
    assert r["same_ck_sol_hybrid"]["execution_seconds"] is None
    assert r["historical_reference"]["phase"] == "cold"
    assert r["historical_reference"]["hot_execution_seconds"] is None
