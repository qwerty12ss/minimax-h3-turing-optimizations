"""Recompute published full-job summaries from the public measurement rows."""
from __future__ import annotations
import csv
import json
from pathlib import Path
from statistics import median

ROOT = Path(__file__).resolve().parents[1]


def summarize(rows, *, pooled_control=False):
    arms = {}
    for row in rows:
        if not row["phase"].startswith("hot"):
            continue
        if row.get("measurement_class", "ordinary") != "ordinary":
            continue
        arms.setdefault(row["arm"], []).append(float(row["execution_seconds"]))
    required = {"control-before", "candidate", "control-after"}
    if set(arms) != required or any(len(v) < 2 for v in arms.values()):
        raise ValueError("Expected two or more ordinary hot observations in each bracket arm")
    medians = {k: median(v) for k, v in arms.items()}
    control = (median(arms["control-before"] + arms["control-after"]) if pooled_control else
               (medians["control-before"] + medians["control-after"]) / 2)
    candidate = medians["candidate"]
    return {"hot_arm_medians": medians, "control_seconds": control, "candidate_seconds": candidate,
            "reduction_percent": 100 * (1 - candidate / control),
            "estimator": "pooled control hot median" if pooled_control else "mean of bracketing arm hot medians"}


def published_summaries():
    with (ROOT / "results/followups.csv").open(newline="") as f:
        followups = list(csv.DictReader(f))
    summaries = {}
    for label, series in [("shared_pipeline", "next-shared-confirm-20261101"),
                          ("temporal_vae", "next-selected-initialized-20261101")]:
        summaries[label] = summarize([r for r in followups if r["series"] == series])
    with (ROOT / "results/swiglu_full.csv").open(newline="") as f:
        summaries["swiglu_full"] = summarize(list(csv.DictReader(f)), pooled_control=True)
    return summaries


if __name__ == "__main__":
    print(json.dumps(published_summaries(), indent=2, allow_nan=False))
