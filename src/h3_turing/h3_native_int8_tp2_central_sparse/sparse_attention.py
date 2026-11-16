"""独立 TP2 hybrid 策略；运行时调用已安装的 Star7 后端，不复制其代码。"""

import importlib.util

import logging

import os

from pathlib import Path

import sys

import torch

MODES = ("ck_sol_all_int8_hybrid",)

CK_STEPS = (0, 1, 6, 7)

def resolve_step(options):
    # 每次 model forward 只解析一次真实 sigma，不能用 block 调用数猜步数。
    schedule, current = options.get("sample_sigmas"), options.get("sigmas")
    if not torch.is_tensor(schedule) or not torch.is_tensor(current):
        raise RuntimeError("TP2 hybrid requires sample_sigmas and sigmas sampler context")
    schedule, current = schedule.detach().flatten(), current.detach().flatten()
    if schedule.numel() != 9 or current.numel() < 1:
        raise RuntimeError("TP2 hybrid first profile requires exactly 8 sampling steps")
    if not bool(torch.isclose(current, current[0], rtol=1e-5, atol=1e-6).all().item()):
        raise RuntimeError("TP2 hybrid cannot use different sigmas in one model call")
    sigma = current[0].to(device=schedule.device, dtype=schedule.dtype)
    matches = torch.nonzero(torch.isclose(schedule[:-1], sigma, rtol=1e-4, atol=1e-6)).flatten()
    if matches.numel() != 1:
        raise RuntimeError("TP2 hybrid needs one unique matching sigma; refusing guessed schedule")
    index = int(matches[0].item())
    return index, ("CK" if index in CK_STEPS else "SPARSE")

def audio_ranges(segments):
    ranges = set()
    for start, end, row in segments:
        tag = int(row) % 3 if isinstance(row, int) else -1
        if torch.is_tensor(row) and row.numel() == 1:
            tag = int(row.item()) % 3
        if tag == 2:
            ranges.add((int(start), int(end)))
    if len(segments) >= 2:
        ranges.add((int(segments[-2][0]), int(segments[-2][1])))
    return tuple(sorted(ranges))

def _load_backend(kind):
    if kind != "sol":
        raise RuntimeError("The packaged TP2 profile supports SM75 Sol only")
    from h3_turing.attention import sol_backend
    return sol_backend

class SparsePolicy:
    def __init__(self, mode, tau=1.0, sla_sparsity=0.85):
        if mode not in MODES:
            raise ValueError(mode)
        self.mode, self.tau, self.sparsity = mode, float(tau), float(sla_sparsity)
        self.kind = "sol" if "_sol_" in mode else "sla"
        self.all_int8 = "_all_int8_" in mode
        self.backend = None
        self.logged = set()

    def prepare_runtime(self, devices):
        # 在 host 主线程完成模块和 ABI 预加载，避免双 rank 首次加载竞争。
        self.backend = _load_backend(self.kind)
        if self.kind == "sol":
            requested = self.backend.SOL_SM75_ALL_INT8_BACKEND_NAME if self.all_int8 else self.backend.SOL_SM75_BACKEND_NAME
        else:
            requested = self.backend.SM75_ALL_INT8_BACKEND_NAME if self.all_int8 else self.backend.SM75_BACKEND_NAME
        for device in devices:
            with torch.cuda.device(device):
                capability = self.backend.check_runtime_support(requested_backend=requested, device=device)
                if capability != (7, 5):
                    raise RuntimeError("This TP2 sparse profile is restricted to SM75")

    def run(self, owned_qkv, options, segments):
        step, selected = options["_h3_tp2_sparse_step"]
        if selected != "SPARSE":
            raise RuntimeError("SparsePolicy called for a CK step")
        if len(owned_qkv) != 3 or tuple(owned_qkv[0].shape[:2]) != (1, 28):
            raise RuntimeError("TP2 sparse expects [1,28,S,128] Q/K/V")
        if any(v.dtype != torch.float16 or v.shape[-1] != 128 for v in owned_qkv):
            raise RuntimeError("TP2 sparse requires FP16 head_dim128 tensors")
        device, sequence = owned_qkv[0].device, owned_qkv[0].shape[-2]
        ranges = audio_ranges(segments)
        if self.kind == "sol":
            sink_start, sink_tokens = None, 0
            if len(segments) >= 2:
                sink_start = int(segments[-2][0])
                sink_tokens = max(0, int(segments[-2][1]) - sink_start)
            result = self.backend.run_custom_consume(
                owned_qkv, all_int8=self.all_int8, tau=self.tau,
                topk_blocks=self.backend.DEFAULT_TOPK_BLOCKS,
                sink_tokens=sink_tokens, sink_start=sink_start)
            density = result.mean_density
        else:
            result = self.backend.sparse_attention_consume(
                owned_qkv, sparsity=self.sparsity, run_self_test=True,
                query_priority_ranges=ranges, all_int8=self.all_int8, debug=False)
            density = 1.0 - result.effective_sparsity
        key = (device.index, sequence, step)
        if key not in self.logged:
            self.logged.add(key)
            logging.info("[H3 INT8 TP2 Sparse] step=%d/8 rank_device=%s heads=28 mode=%s density=%.4f implementation=%s",
                         step, device, self.mode, density, result.implementation)
        return result.output.transpose(1, 2).reshape(sequence, 28*128)
