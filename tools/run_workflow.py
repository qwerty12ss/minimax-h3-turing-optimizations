"""Submit a portable API workflow, wait for completion, save a small validation receipt."""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import time
import urllib.error
import urllib.request


def request(base, path, body=None):
    payload = None if body is None else json.dumps(body).encode()
    req = urllib.request.Request(base.rstrip("/") + path, data=payload, headers={"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=60) as response:
            return json.load(response)
    except urllib.error.HTTPError as error:
        # Display server validation errors locally, never write raw history into a public report.
        raise RuntimeError(error.read().decode()) from error


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("workflow", type=Path)
    parser.add_argument("--url", default="http://127.0.0.1:8188")
    parser.add_argument("--timeout", type=float, default=3600)
    parser.add_argument("--report", type=Path, help="Optional local receipt, with no paths, prompt or machine identity")
    args = parser.parse_args()
    graph = json.loads(args.workflow.read_text())
    schema = request(args.url, "/object_info")
    missing = sorted({n["class_type"] for n in graph.values()} - schema.keys())
    if missing:
        raise RuntimeError(f"Missing ComfyUI nodes: {missing}; inspect startup errors and pinned revisions")
    queued = request(args.url, "/prompt", {"prompt": graph})
    job = queued["prompt_id"]
    deadline = time.monotonic() + args.timeout
    print("Workflow accepted; waiting for video and audio.", flush=True)
    while time.monotonic() < deadline:
        history = request(args.url, "/history/" + job).get(job)
        if history:
            status = history["status"]
            if status["status_str"] != "success":
                errors = [(kind, data.get("exception_type"), data.get("exception_message"))
                          for kind, data in status.get("messages", []) if kind in ("execution_error", "execution_interrupted")]
                raise RuntimeError(f"Workflow failed: {errors}")
            response = request(args.url, "/h3_turing/validation")
            if response.get("prompt_id") != job:
                raise RuntimeError("Validation receipt belongs to another job; run with --cache-none and keep the test queue exclusive")
            validation = response["validation"]
            if not validation or not validation.get("video_finite") or not validation.get("audio_finite"):
                raise RuntimeError("No successful finite video/audio validation receipt")
            optimized = any(n["class_type"] == "H3TuringApply" and n["inputs"].get("preset") == "optimized" for n in graph.values())
            if optimized and not any(r.get("blocks", 0) >= 400 and r.get("norm_calls") == 2 * r["blocks"]
                                     and r.get("shared_counts", {}).get("phases") == 2 * r["blocks"]
                                     and r.get("rope_calls", 0) > 0 and r.get("joint_calls", 0) > 0
                                     for r in validation.get("optimizers", [])):
                raise RuntimeError("Missing complete optimized eight-step TP2 execution counters")
            receipt = {"workflow": args.workflow.name, "status": "success", "validation": validation}
            print(json.dumps(receipt, indent=2))
            if args.report:
                args.report.write_text(json.dumps(receipt, indent=2) + "\n")
            print("Video saved under this runtime's ComfyUI/output/video directory.")
            return
        time.sleep(5)
    raise TimeoutError("Workflow is still running; check ComfyUI before submitting another job")


if __name__ == "__main__":
    main()
