"""TP2 sharding + block forward for MiniMax-H3.

Derived from NorwinYu/minimax-h3-dual-2080ti (MIT), commit
047e06890fc599638b7ca07947f445a15f278713. Native INT8 adaptation has passed
CPU shard/ConvRot consistency checks only; GPU inference is not yet verified.

Layout facts:
  qkv_proj [21504,5376] = [q(7168)|k(7168)|v(7168)] -> head-parallel takes heads
    0..27 from EACH of q/k/v (three interleaved slices), concat on the output.
  out_proj [5376,7168]  -> row-parallel over heads, the two halves SUM (allreduce).
  fc1 [28672,5376] = [gate(14336)|up(14336)] -> each rank takes
    gate[0:7168] ++ up[0:7168] (interleaved), concat on the output.
  fc2 [5376,14336]      -> row-parallel over FFN, halves SUM (allreduce).
  FP16-Exact: out_proj scales input /64 then *64; MLP scales /256 then *256.
"""
import dataclasses
import os as _os
import time as _time
from concurrent.futures import ThreadPoolExecutor

import torch
import torch.nn.functional as F

LAYOUT = "AsymW4A8Int8Layout"
INT8_LAYOUT = "TensorWiseINT8Layout"
KO, KC2 = 64.0, 256.0

PH = {}
NCALL = [0]
_EXEC = None


def _run(fn, inference):
    """InferenceMode and grad mode are thread-local, so the worker must re-enter the
    caller's mode; otherwise in-place ops on the caller's inference tensors raise and the
    block's elementwise work builds autograd nodes."""
    if inference:
        with torch.inference_mode():
            return fn()
    return fn()


def _par(f0, f1):
    """Run the two ranks on two host threads.

    The W4A8 GEMM blocks the calling host thread until its kernel retires, so one thread
    cannot keep both ranks' GEMMs in flight at once (measured: 5% overlap). One thread per
    rank reaches ~90%, because each thread carries its own current device and stream.
    """
    global _EXEC
    if _EXEC is None:
        _EXEC = ThreadPoolExecutor(max_workers=2, thread_name_prefix="h3-tp2")
    inference = torch.is_inference_mode_enabled()
    a = _EXEC.submit(_run, f0, inference)
    b = _EXEC.submit(_run, f1, inference)
    return a.result(), b.result()


def _snap(name, t0):
    if _os.environ.get("H3_TP2_PHASE_TIMING") != "1":
        return
    for _i in range(torch.cuda.device_count()):
        torch.cuda.synchronize(_i)
    PH[name] = PH.get(name, 0.0) + (_time.perf_counter() - t0) * 1000.0


def phase_report():
    n = max(1, NCALL[0])
    return " ".join(f"{k}={v / n:.1f}" for k, v in PH.items())


def _fp(t):
    """fp8 scales cannot be indexed on CPU; fp8->fp32 is lossless."""
    return t.float() if isinstance(t, torch.Tensor) and t.dtype == torch.float8_e4m3fn else t


def _qt(qdata, params, layout=LAYOUT):
    from comfy_kitchen.tensor.base import QuantizedTensor
    return QuantizedTensor(qdata.contiguous(), layout, params)


def _int8_parts(w):
    # 只接收原生权重布局；不反量化，也不重新量化已有 PDD8 权重。
    assert w._layout_cls == INT8_LAYOUT, w._layout_cls
    p = w._params
    assert p.is_weight and not p.transposed, "Expected untransposed INT8 linear weight"
    assert tuple(w._qdata.shape) == tuple(w.shape), (w.shape, w._qdata.shape)
    assert w._qdata.dtype == torch.int8
    assert p.scale.numel() in (1, w.shape[0]), p.scale.shape
    if p.convrot:
        assert p.scale.numel() == w.shape[0]
        assert w.shape[1] % p.convrot_groupsize == 0
    return w._qdata, p


def _int8_shard_n(w, rows0, rows1, dev0, dev1):
    q, p = _int8_parts(w)
    result = []
    for rows, dev in ((list(rows0), dev0), (list(rows1), dev1)):
        scale = p.scale if p.scale.numel() == 1 else p.scale[rows]
        pr = dataclasses.replace(p, scale=scale.to(dev), orig_shape=(len(rows), w.shape[1]))
        result.append(_qt(q[rows].to(dev), pr, INT8_LAYOUT))
    return tuple(result)


def _int8_shard_k(w, mid, dev0, dev1):
    q, p = _int8_parts(w)
    assert 0 < mid < w.shape[1]
    # Hadamard/ConvRot 是组内变换：只允许在完整组边界切分输入。
    if p.convrot:
        assert mid % p.convrot_groupsize == 0, (mid, p.convrot_groupsize)
    result = []
    for a, b, dev in ((0, mid, dev0), (mid, w.shape[1], dev1)):
        pr = dataclasses.replace(p, scale=p.scale.to(dev), orig_shape=(w.shape[0], b-a))
        result.append(_qt(q[:, a:b].to(dev), pr, INT8_LAYOUT))
    return tuple(result)


def _shard_n(w, rows0, rows1, dev0, dev1):
    """Column-parallel: split output rows (rows may be an interleaved list)."""
    if getattr(w, "_layout_cls", None) == INT8_LAYOUT:
        return _int8_shard_n(w, rows0, rows1, dev0, dev1)
    assert getattr(w, "_layout_cls", None) == LAYOUT, "Unsupported quantized weight layout"
    from comfy_kitchen.tensor.w4a8_int8 import AsymW4A8Int8Layout as L
    qdata, scale, s_channel, correction, _cb = L.get_plain_tensors(w)
    p = w._params
    scale, s_channel = _fp(scale), _fp(s_channel)
    correction = _fp(correction)
    out = []
    for rows, dev in ((rows0, dev0), (rows1, dev1)):
        rows = list(rows)
        pr = dataclasses.replace(
            p, scale=scale[rows].to(dev), s_channel=s_channel[rows].to(dev),
            correction=correction[:, rows].to(dev) if correction is not None else None,
            orig_shape=(len(rows), w.shape[1]))
        out.append(_qt(qdata[rows].to(dev), pr))
    return out


def _shard_k(w, mid, dev0, dev1):
    """Row-parallel: split input columns at a convrot-group boundary."""
    if getattr(w, "_layout_cls", None) == INT8_LAYOUT:
        return _int8_shard_k(w, mid, dev0, dev1)
    assert getattr(w, "_layout_cls", None) == LAYOUT, "Unsupported quantized weight layout"
    from comfy_kitchen.tensor.w4a8_int8 import AsymW4A8Int8Layout as L
    qdata, scale, _sch, _corr, _cb = L.get_plain_tensors(w)
    p = w._params
    scale = _fp(scale)
    sc = mid // p.group_size
    p0 = dataclasses.replace(p, scale=scale[:, :sc].to(dev0), orig_shape=(w.shape[0], mid))
    p1 = dataclasses.replace(p, scale=scale[:, sc:].to(dev1), orig_shape=(w.shape[0], w.shape[1] - mid))
    return (_qt(qdata[:, :mid // 2].to(dev0), p0),
            _qt(qdata[:, mid // 2:].to(dev1), p1))


def _lora_shard(lora, rows0, rows1, mid, kind, dev0, dev1):
    """Slice a LoRA (up[out,rank], down[rank,in], scale) onto the TP ranks.

    Column-parallel layers slice up's output rows; row-parallel layers slice
    down's input columns. The LoRA is applied as a runtime bypass (base stays
    quantized), which is also what the turbo node recommends over merging.
    """
    if lora is None:
        return (None, None)
    up, down, scale = lora
    if kind == "out":
        r0, r1 = list(rows0), list(rows1)
        return ((up[r0].to(dev0), down.to(dev0), scale),
                (up[r1].to(dev1), down.to(dev1), scale))
    return ((up.to(dev0), down[:, :mid].to(dev0), scale),
            (up.to(dev1), down[:, mid:].to(dev1), scale))


def _lin(x, w, lora):
    """Linear + optional LoRA bypass on the (possibly sharded) weight."""
    y = F.linear(x, w)
    if lora is not None:
        up, down, scale = lora
        y = y + F.linear(F.linear(x, down), up) * scale
    return y


def shard_block(blk, dev0, dev1, lora=None):
    """Shard one DiT block's linear weights (+ its LoRA) onto dev0/dev1."""
    att, mlp = blk.attn, blk.mlp
    heads, hd = att.heads, att.head_dim
    assert heads % 2 == 0 and heads == 56 and hd == 128, (heads, hd)
    assert att.qkv_proj.bias is None and att.out_proj.bias is None
    assert mlp.fc1.bias is None and mlp.fc2.bias is None
    h2 = heads // 2
    full, qh = heads * hd, (heads // 2) * hd
    rows0 = list(range(0, qh)) + list(range(full, full + qh)) + list(range(2 * full, 2 * full + qh))
    rows1 = list(range(qh, full)) + list(range(full + qh, 2 * full)) + list(range(2 * full + qh, 3 * full))
    ffn_half = mlp.fc1.weight.shape[0] // 4
    g0 = list(range(0, ffn_half)) + list(range(2 * ffn_half, 3 * ffn_half))
    g1 = list(range(ffn_half, 2 * ffn_half)) + list(range(3 * ffn_half, 4 * ffn_half))
    lora = lora or {}
    return {
        "att": att, "adaln": blk.adaln_proj, "h2": h2, "hd": hd,
        "qkv": _shard_n(att.qkv_proj.weight, rows0, rows1, dev0, dev1),
        "op": _shard_k(att.out_proj.weight, h2 * hd, dev0, dev1),
        "fc1": _shard_n(mlp.fc1.weight, g0, g1, dev0, dev1),
        "fc2": _shard_k(mlp.fc2.weight, ffn_half, dev0, dev1),
        "lqkv": _lora_shard(lora.get("qkv"), rows0, rows1, None, "out", dev0, dev1),
        "lop": _lora_shard(lora.get("op"), None, None, h2 * hd, "in", dev0, dev1),
        "lfc1": _lora_shard(lora.get("fc1"), g0, g1, None, "out", dev0, dev1),
        "lfc2": _lora_shard(lora.get("fc2"), None, None, ffn_half, "in", dev0, dev1),
        "n1": tuple(blk.norm1.weight.detach().to(d) for d in (dev0, dev1)),
        "n2": tuple(blk.norm2.weight.detach().to(d) for d in (dev0, dev1)),
        "qn": tuple(att.q_norm.weight.detach().to(d) for d in (dev0, dev1)),
        "kn": tuple(att.k_norm.weight.detach().to(d) for d in (dev0, dev1)),
        "dev0": dev0, "dev1": dev1,
        "eps1": blk.norm1.eps, "eps2": blk.norm2.eps,
    }


def _rms(t, w, eps):
    return t * torch.rsqrt(t.float().pow(2).mean(-1, keepdim=True) + eps).to(t.dtype) * w


def _qkv_rope(sh, h, rf, rank):
    import comfy.model_management as mm
    import comfy.quant_ops as qo
    dev = sh["dev0"] if rank == 0 else sh["dev1"]
    att, heads, hd = sh["att"], sh["h2"], sh["hd"]
    assert h.device == dev and h.dtype == torch.float16
    sequence = h.shape[0]
    if rf is None or rf.ndim < 2 or rf.shape[1] != sequence:
        raise RuntimeError("Expected H3 RoPE [1,S,1,rot/2,2,2] for chunked QKV")
    rf = rf.to(dev)
    chunk = min(sequence, int(sh["qkv_chunk_tokens"]))
    # 完整 attention inputs 必须保留；chunk 只限制投影临时存储。
    output = [torch.empty((1, heads, sequence, hd), dtype=h.dtype, device=dev) for _ in range(3)]
    qw = mm.cast_to(sh["qn"][rank], device=dev)
    kw = mm.cast_to(sh["kn"][rank], device=dev)
    for start in range(0, sequence, chunk):
        end = min(sequence, start + chunk)
        qkv = _lin(h[start:end], sh["qkv"][rank], sh["lqkv"][rank])
        q, k, v = qkv.split(heads*hd, dim=-1)
        q = q.view(1, end-start, heads, hd)
        k = k.view(1, end-start, heads, hd)
        qo.ck.rms_rope_split_half_(q, k, rf[:, start:end, ...], qw, kw,
                                  epsilon=att.q_norm.eps, rot_dim=rf.shape[-3]*2)
        output[0][:, :, start:end].copy_(q.permute(0, 2, 1, 3))
        output[1][:, :, start:end].copy_(k.permute(0, 2, 1, 3))
        output[2][:, :, start:end].copy_(v.view(end-start, heads, hd).permute(1, 0, 2).unsqueeze(0))
        del qkv, q, k, v
    return output


def _attn(owned_qkv, sh, options, segments):
    """Attention core for one rank's heads. Uses Comfy Kitchen INT8 (the same
    backend the single-GPU path uses via the chunk node) so splitting the heads
    does not silently switch to a slower kernel."""
    from comfy.ldm.modules.attention import AttentionTensorContainer, attention_comfy_kitchen_int8
    if sh["sparse_policy"] is not None and options["_h3_tp2_sparse_step"][1] == "SPARSE":
        return sh["sparse_policy"].run(owned_qkv, options, segments)
    heads = owned_qkv[0].shape[1]
    qq, kk, vv = (AttentionTensorContainer(value) for value in owned_qkv)
    owned_qkv.clear()
    o = attention_comfy_kitchen_int8(qq, kk, vv, heads, mask=None, skip_reshape=True,
                                     transformer_options={})
    return o.squeeze(0)


def _mlp_chunked(sh, h, rank):
    # 每行保留 FP32 SwiGLU 和 FC2 前的 /256 -> FP16。
    sequence = h.shape[0]
    chunk = min(sequence, int(sh["mlp_chunk_tokens"]))
    output = torch.empty((sequence, sh["fc2"][rank].shape[0]), dtype=torch.float16, device=h.device)
    for start in range(0, sequence, chunk):
        end = min(sequence, start+chunk)
        g = _lin(h[start:end], sh["fc1"][rank], sh["lfc1"][rank])
        gate, up = g.chunk(2, dim=-1)
        act = F.silu(gate.to(torch.float32)) * up.to(torch.float32)
        del g, gate, up
        y = _lin((act / KC2).to(torch.float16), sh["fc2"][rank], sh["lfc2"][rank])
        output[start:end].copy_(y)
        del act, y
    return output


def _sum_scaled_fp32(a, b, scale):
    # 与 (a.float()+b.float())*scale 等价，避免两个全长 FP32 临时张量。
    result = a.to(torch.float32)
    result.add_(b)
    result.mul_(scale)
    return result


def tp2_block(sh, x, t_emb, mod_segments, rope_freqs,
              transformer_options=None, attention=None, **kwargs):
    """Interface-compatible with DiTBlock.forward; runs the block tensor-parallel."""
    from comfy.ldm.minimax.model import _mod_gate, _mod_scale_shift
    d0, d1 = sh["dev0"], sh["dev1"]
    assert x.device == d0, (x.device, d0)
    if attention is not None:
        raise RuntimeError("TP2 does not implement externally replaced attention")
    i0, i1 = d0.index, d1.index
    st0, st1 = torch.cuda.current_stream(i0), torch.cuda.current_stream(i1)
    adaln, segs = sh["adaln"], mod_segments
    NCALL[0] += 1
    _t = _time.perf_counter()

    m0 = adaln(t_emb)
    m1 = tuple(v.to(d1) for v in m0)

    def front(rank, i, st, xr, m):
        with torch.cuda.device(i), torch.cuda.stream(st):
            xr = xr.to(torch.float32)
            h = _mod_scale_shift(_rms(xr, sh["n1"][rank], sh["eps1"]), m[0], m[1], segs).to(torch.float16)
            owned = _qkv_rope(sh, h, rope_freqs, rank)
            del h
            O = _attn(owned, sh, transformer_options, segs)
            p = _lin((O / KO).to(torch.float16), sh["op"][rank], sh["lop"][rank])
            return xr, p

    with torch.cuda.device(i1):
        xr1 = x.to(d1, non_blocking=True)
    (x0, p0), (x1, p1) = _par(
        lambda: front(0, i0, st0, x, m0),
        lambda: front(1, i1, st1, xr1, m1),
    )
    _snap("qkv+attn", _t); _t = _time.perf_counter()
    # symmetric allreduce: each rank receives the other's partial and sums locally,
    # so no separate broadcast copy is needed afterwards
    r1 = torch.empty_like(p1, device=d0)
    r0 = torch.empty_like(p0, device=d1)
    st0.wait_stream(st1)
    st1.wait_stream(st0)
    with torch.cuda.device(i0):
        r1.copy_(p1, non_blocking=True)
    with torch.cuda.device(i1):
        r0.copy_(p0, non_blocking=True)

    def half(rank, i, st, xr, pr, other, m):
        with torch.cuda.device(i), torch.cuda.stream(st):
            a = _sum_scaled_fp32(pr, other, KO)
            xr = _mod_gate(xr, m[2], a, segs)
            del a
            h = _mod_scale_shift(_rms(xr, sh["n2"][rank], sh["eps2"]), m[3], m[4], segs).to(torch.float16)
            mm = _mlp_chunked(sh, h, rank)
            return xr, mm

    (x0, mm0), (x1, mm1) = _par(
        lambda: half(0, i0, st0, x0, p0, r1, m0),
        lambda: half(1, i1, st1, x1, p1, r0, m1),
    )
    del p0, p1, r0, r1, x1
    _snap("mlp", _t); _t = _time.perf_counter()
    r1b = torch.empty_like(mm1, device=d0)
    st0.wait_stream(st1)
    with torch.cuda.device(i0):
        r1b.copy_(mm1, non_blocking=True)
    with torch.cuda.device(i0):
        mo0 = _sum_scaled_fp32(mm0, r1b, KC2)
        out = _mod_gate(x0, m0[5], mo0, segs)
    _snap("final_allreduce", _t)
    return out
