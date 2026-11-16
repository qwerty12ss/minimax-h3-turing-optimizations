"""Synthetic SM75 routing smoke test; this is not a historical video benchmark."""
import json
import torch
from h3_turing.attention import sol_sm75_backend as sol


def main():
    if not torch.cuda.is_available() or torch.cuda.get_device_capability() != (7, 5):
        raise RuntimeError("This benchmark is qualified for SM75 only")
    torch.manual_seed(1234)
    # Small deterministic inputs exercise routing without private scene captures.
    q, k, v = [torch.randn(1, 4, 512, 128, device="cuda", dtype=torch.float16) for _ in range(3)]
    result = sol.prepare(q, k, v, tau=0.1, quantize_qk=True)
    print(json.dumps({"availability": sol.availability(), "prepare_result_type": type(result).__name__,
                      "finite_inputs": all(torch.isfinite(t).all().item() for t in (q, k, v)),
                      "scope": "routing smoke only; not output accuracy or full-job latency"}))


if __name__ == "__main__":
    main()
