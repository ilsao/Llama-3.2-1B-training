"""Full-vocabulary forward KL, T=1; first-order gradients computed in forward.

No teacher approximation or objective change. Compared with recomputation,
scaling after the GEMMs changes BF16 rounding, so this is not bitwise equivalent.
Keep this opt-in module separate from the sealed experiment source.
"""
import torch
from torch.nn import functional as F
from torch.autograd.function import once_differentiable


class PrecomputedKD(torch.autograd.Function):
    @staticmethod
    def forward(ctx, hidden, weight, th, tw, labels, lam, chunk):
        if not 0 <= lam <= 1 or not isinstance(chunk, int) or chunk <= 0:
            raise ValueError('Require 0 <= teacher_weight <= 1 and positive integer chunk')
        if hidden.shape != th.shape or weight.shape != tw.shape:
            raise ValueError('Teacher/student hidden and vocabulary dimensions must match')
        if labels.shape != hidden.shape[:1]:
            raise ValueError('Labels must match flattened prediction positions')
        ce = torch.zeros((), device=hidden.device, dtype=torch.float32)
        kl = torch.zeros_like(ce)
        dh, dw = torch.zeros_like(hidden), torch.zeros_like(weight)
        for start in range(0, hidden.shape[0], chunk):
            h = hidden[start:start+chunk]
            target = labels[start:start+chunk]
            valid = target != -100
            ls = F.log_softmax((h @ weight.T).float(), dim=-1)
            lt = F.log_softmax((th[start:start+chunk] @ tw.T).float(), dim=-1)
            pt = lt.exp()
            ce.add_(F.nll_loss(ls, target, ignore_index=-100, reduction='sum'))
            kl.add_(((pt * (lt-ls)).sum(-1) * valid).sum())
            # d[(1-lambda) CE + lambda KL(t || s)]/d logits_s
            dl = ls.exp().sub_(pt, alpha=lam)
            dl[torch.arange(len(target), device=target.device), target.clamp_min(0)] -= 1-lam
            dl.mul_(valid[:, None])
            dh[start:start+chunk] = dl @ weight
            dw.add_((dl.T @ h).to(dw.dtype))
        ctx.save_for_backward(dh, dw)
        ctx.mark_non_differentiable(ce, kl)
        return (1-lam)*ce+lam*kl, ce, kl

    @staticmethod
    @once_differentiable
    def backward(ctx, grad, _ce, _kl):
        dh, dw = ctx.saved_tensors
        return dh * grad, dw * grad, None, None, None, None, None


def kd_loss(hidden, weight, teacher_hidden, teacher_weight, labels, lam=.25, chunk=1024):
    return PrecomputedKD.apply(hidden[:, :-1].reshape(-1, hidden.shape[-1]), weight,
        teacher_hidden[:, :-1].reshape(-1, teacher_hidden.shape[-1]), teacher_weight,
        labels[:, 1:].reshape(-1), lam, chunk)


def configured_loss(method):
    """Explicitly honor configuration; reject unsupported temperature silently changing math."""
    if method['name'] != 'kd' or method['temperature'] != 1.0:
        raise ValueError('This implementation supports KD at T=1 only')
    from functools import partial
    return partial(kd_loss, lam=method['teacher_weight'], chunk=method['logit_chunk'])
