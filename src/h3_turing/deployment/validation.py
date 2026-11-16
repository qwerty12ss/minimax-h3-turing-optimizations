"""Bounded scalar-only receipts survive model unloading without retaining models."""
_LAST_OPTIMIZER = (None, None)


def record_optimizer(prompt_id, report):
    global _LAST_OPTIMIZER
    _LAST_OPTIMIZER = (prompt_id, report)


def optimizer_reports(prompt_id):
    owner, report = _LAST_OPTIMIZER
    return [report] if owner == prompt_id and report is not None else []
