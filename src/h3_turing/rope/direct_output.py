"""Original F.rms_norm unchanged; opt-in pointwise launch after its output."""
from .contract import CHUNKS, D, H, ROT, S, require

def identity(value):
    return value.untyped_storage()._cdata

def pointwise(torch, normal, freqs, owner, start):
    count = normal.shape[1]
    require(type(start) is int and start >= 0 and 1 <= count <= 8192 and tuple(normal.shape) == (1, count, H, D), 'Actual trunk RMS result required')
    require(normal.device.type == 'cuda' and normal.dtype in (torch.float16, torch.float32) and (normal.stride(-1) == 1) and all((x > 0 for x in normal.stride())) and (not normal.requires_grad), 'Original CUDA RMS output layout/dtype required')
    require(tuple(freqs.shape) == (1, count, 1, ROT // 2, 2, 2) and freqs.dtype in (torch.float16, torch.float32) and (freqs.stride()[-3:] == (4, 2, 1)) and all((x > 0 for x in freqs.stride())), 'Qualified96 frequency table required')
    require(owner.ndim == 4 and tuple(owner.shape[:2]) == (1,H) and owner.shape[-1] == D and start + count <= owner.shape[2] and owner.dtype is torch.float16 and owner.is_contiguous() and (owner.storage_offset() == 0) and (not owner.requires_grad), 'Full owned BHND FP16 target required')
    require(owner.device == normal.device == freqs.device and (not freqs.requires_grad), 'Same-device inference only')
    require(identity(owner) not in (identity(normal), identity(freqs)), 'Output aliases an input')
    require(start % 8192 == 0 and count == min(8192, owner.shape[2]-start), 'Original chunk plan required')
    from .kernel import launch
    with torch.cuda.device(normal.device):
        return launch(normal, freqs, owner, start)

def direct(torch, q, k, freqs, q_scale, k_scale, owners, start, epsilon, rot_dim):
    count = q.shape[1]
    require(type(start) is int and start >= 0 and 1 <= count <= 8192 and tuple(q.shape) == tuple(k.shape) == (1, count, H, D) and (q.dtype is k.dtype is torch.float16) and (q.device == k.device), 'Actual native packed Q/K required')
    require(q.stride() == k.stride() == (count * 3 * H * D, 3 * H * D, D, 1) and identity(q) == identity(k) and (k.storage_offset() - q.storage_offset() == H * D), 'Original packed Q/K strides changed')
    require(rot_dim == ROT and epsilon in (1e-06, 1e-05) and (len(owners) == 2) and (identity(owners[0]) != identity(owners[1])), 'Qualified rotation/epsilon/distinct owners only')
    require(q_scale.shape == k_scale.shape == (D,) and q_scale.dtype == k_scale.dtype and (q_scale.dtype in (torch.float16, torch.float32)) and (q_scale.stride() == k_scale.stride() == (1,)), 'Original scale dtype/layout changed')
    require(all((t.device == q.device for t in (freqs, q_scale, k_scale, *owners))) and all((identity(out) != identity(src) for out in owners for src in (q, k, freqs, q_scale, k_scale))), 'Same-device, nonaliasing Q/K output required')
    results = []
    with torch.cuda.device(q.device):
        for source, scale, out in ((q, q_scale, owners[0]), (k, k_scale, owners[1])):
            normal = torch.nn.functional.rms_norm(source, (source.shape[-1],), weight=scale, eps=epsilon)
            results.append(pointwise(torch, normal, freqs, out, start))
            del normal
    return results
