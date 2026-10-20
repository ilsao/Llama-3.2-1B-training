"""Experimental full-vocabulary CE + forward KL, T=1. GPU validation required.

All vocabulary entries participate; BF16 projection GEMMs, FP32 reductions.
The student-logit buffer is overwritten by its BF16 gradient only after both
normalizers are known. No target clipping, top-k, or gradient filtering.
Keep projection/weight-gradient accumulation identical to sealed fast_kd.py.
"""
import torch
import triton
import triton.language as tl
from torch.autograd.function import once_differentiable


@triton.jit
def _ce_kl(S, T, Y, CE, KL, V: tl.constexpr, LAM: tl.constexpr,
           BLOCK: tl.constexpr):
    row = tl.program_id(0)
    y = tl.load(Y + row)
    lane = tl.arange(0, BLOCK)
    if y == -100:
        for base in range(0, V, BLOCK):
            col = base + lane
            tl.store(S + row * V + col, 0., col < V)
        tl.store(CE + row, 0.)
        tl.store(KL + row, 0.)
        return
    ms = float('-inf')
    mt = float('-inf')
    for base in range(0, V, BLOCK):
        col = base + lane
        s = tl.load(S + row * V + col, col < V, other=float('-inf')).to(tl.float32)
        t = tl.load(T + row * V + col, col < V, other=float('-inf')).to(tl.float32)
        ms = tl.maximum(ms, tl.max(s, 0))
        mt = tl.maximum(mt, tl.max(t, 0))
    zs = 0.
    zt = 0.
    for base in range(0, V, BLOCK):
        col = base + lane
        s = tl.load(S + row * V + col, col < V, other=float('-inf')).to(tl.float32)
        t = tl.load(T + row * V + col, col < V, other=float('-inf')).to(tl.float32)
        zs += tl.sum(tl.exp(s - ms), 0)
        zt += tl.sum(tl.exp(t - mt), 0)
    logzs = tl.log(zs)
    logzt = tl.log(zt)
    sy = tl.load(S + row * V + y).to(tl.float32)
    tl.store(CE + row, (ms - sy) + logzs)
    kl = 0.
    for base in range(0, V, BLOCK):
        col = base + lane
        # Finite masked values avoid inf-inf and 0*inf in the KL reduction.
        s = tl.load(S + row * V + col, col < V, other=0.).to(tl.float32)
        t = tl.load(T + row * V + col, col < V, other=0.).to(tl.float32)
        ls = (s - ms) - logzs
        lt = (t - mt) - logzt
        pt = tl.exp(lt)
        kl += tl.sum(tl.where(col < V, pt * (lt - ls), 0.), 0)
        grad = tl.exp(ls) - LAM * pt - tl.where(col == y, 1. - LAM, 0.)
        tl.store(S + row * V + col, grad, col < V)
    tl.store(KL + row, kl)


class FusedKD(torch.autograd.Function):
    @staticmethod
    def forward(ctx, h, w, th, tw, labels, lam, chunk):
        if not 0 <= lam <= 1 or not isinstance(chunk, int) or chunk < 1:
            raise ValueError('Invalid mixing weight/chunk')
        if (h.ndim != 2 or w.ndim != 2 or th.ndim != 2 or tw.ndim != 2
                or h.shape[0] != th.shape[0] or w.shape[0] != tw.shape[0]
                or h.shape[1] != w.shape[1] or th.shape[1] != tw.shape[1]
                or labels.shape != h.shape[:1] or not h.is_cuda):
            raise ValueError('Invalid CUDA hidden/head/label geometry')
        ce_rows = torch.empty(h.shape[0], device=h.device, dtype=torch.float32)
        kl_rows = torch.empty_like(ce_rows)
        dh, dw = torch.zeros_like(h), torch.zeros_like(w)
        for start in range(0, h.shape[0], chunk):
            x = h[start:start + chunk]
            s = (x @ w.T).contiguous()
            t = (th[start:start + chunk] @ tw.T).contiguous()
            _ce_kl[(len(x),)](s, t, labels[start:start + chunk].contiguous(),
                ce_rows[start:start + chunk], kl_rows[start:start + chunk],
                w.shape[0], lam, 8192, num_warps=8, enable_fp_fusion=False)
            dh[start:start + chunk] = s @ w
            dw.add_((s.T @ x).to(dw.dtype))
        ce, kl = ce_rows.sum(), kl_rows.sum()
        ctx.save_for_backward(dh, dw)
        ctx.mark_non_differentiable(ce, kl)
        return (1 - lam) * ce + lam * kl, ce, kl

    @staticmethod
    @once_differentiable
    def backward(ctx, grad, _ce, _kl):
        dh, dw = ctx.saved_tensors
        return dh * grad, dw * grad, None, None, None, None, None
