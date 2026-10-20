"""Use existing unsynchronized training gradients: no extra examples or forwards.

Call only after EVERY backward in this window used DDP no_sync. Input gradients
are local SUM-loss gradients scaled by world/global_valid_targets. This function
performs the one required DDP-equivalent gradient average in bounded chunks.
Never run it on gradients DDP already averaged; that erases the noise signal.
"""
from critical_batch import noise_moments


def synchronize_and_measure(named_parameters, local_targets, total_targets, chunk_elements=1048576):
    import torch
    import torch.distributed as dist
    params = list(named_parameters)
    world = dist.get_world_size()
    if world < 2 or local_targets <= 0:
        raise ValueError('Invalid distributed probe geometry')
    device = params[0][1].device
    counts = torch.tensor(local_targets, device=device, dtype=torch.int64)
    dist.all_reduce(counts)
    if counts.item() != total_targets:
        raise ValueError('Target counts differ across ranks')
    # named_parameters deduplicates the tied embedding/head; optimizer ownership
    # does NOT shard these DDP gradients. Measure the complete student once.
    groups = ['global', 'muon', 'adamw']
    def group(name, p):
        return 2 if p.ndim < 2 or 'embed_tokens' in name or 'lm_head' in name else 1
    local = torch.zeros(3, device=device, dtype=torch.float64)
    mean = torch.zeros_like(local)
    buffer = torch.empty(chunk_elements, device=device, dtype=torch.float32)
    for name, p in params:
        if p.grad is None or not p.grad.is_contiguous() or p.grad.dtype != torch.float32:
            raise ValueError('Probe requires dense contiguous FP32 gradients for every parameter')
        gi = group(name, p)
        flat = p.grad.view(-1)
        for offset in range(0, flat.numel(), chunk_elements):
            view = flat[offset:offset+chunk_elements]
            norm = torch.linalg.vector_norm(view, dtype=torch.float64).square()
            local[0] += norm/local_targets
            local[gi] += norm/local_targets
            scratch = buffer[:view.numel()]
            scratch.copy_(view)
            dist.all_reduce(scratch)
            scratch.div_(world)
            view.copy_(scratch)
            norm = torch.linalg.vector_norm(scratch, dtype=torch.float64).square()
            mean[0] += norm
            mean[gi] += norm
    dist.all_reduce(local)
    local.mul_(total_targets/(world*world))
    # mean gradients are replicas: NEVER sum their squared norms over ranks.
    values = torch.stack((local, mean), dim=1).cpu().tolist()
    result = {}
    for name, (weighted, averaged) in zip(groups, values):
        signal, noise = noise_moments(weighted, averaged, total_targets, world)
        result[name] = dict(signal=signal, noise=noise, weighted_local_norm2=weighted,
                            mean_norm2=averaged)
    return result
