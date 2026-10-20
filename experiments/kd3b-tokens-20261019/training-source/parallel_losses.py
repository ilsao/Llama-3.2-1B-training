"""Allocated-node-only exact-vocabulary CCE and bounded-memory KD losses."""
import contextlib
import types
import torch
from torch.nn import functional as F

_STATS = None


class ChunkedKD(torch.autograd.Function):
    """Recompute vocabulary chunks in backward; full forward KL at T=1."""
    @staticmethod
    def forward(ctx, hidden, weight, teacher_hidden, teacher_weight, labels, lam, chunk):
        ctx.save_for_backward(hidden, weight, teacher_hidden, teacher_weight, labels)
        ctx.lam, ctx.chunk = lam, chunk
        ctx.amp = torch.is_autocast_enabled('cuda')
        ctx.amp_dtype = torch.get_autocast_dtype('cuda')
        ce = torch.zeros((), device=hidden.device, dtype=torch.float32)
        kl = torch.zeros_like(ce)
        for start in range(0, hidden.shape[0], chunk):
            stop = start + chunk
            ls = F.log_softmax((hidden[start:stop] @ weight.T).float(), dim=-1)
            lt = F.log_softmax((teacher_hidden[start:stop] @ teacher_weight.T).float(), dim=-1)
            valid = labels[start:stop] != -100
            ce += F.nll_loss(ls, labels[start:stop], ignore_index=-100, reduction='sum')
            kl += ((lt.exp() * (lt-ls)).sum(-1) * valid).sum()
        ctx.mark_non_differentiable(ce, kl)
        return (1-lam)*ce+lam*kl, ce, kl

    @staticmethod
    def backward(ctx, grad, _ce, _kl):
        hidden, weight, th, tw, labels = ctx.saved_tensors
        dh = torch.zeros_like(hidden)
        dw = torch.zeros_like(weight)
        # Autocast is thread-local and normally disabled during backward.
        with torch.autocast('cuda', dtype=ctx.amp_dtype, enabled=ctx.amp):
            for start in range(0, hidden.shape[0], ctx.chunk):
                stop = start+ctx.chunk
                h = hidden[start:stop]
                ps = F.softmax((h @ weight.T).float(), dim=-1)
                pt = F.softmax((th[start:stop] @ tw.T).float(), dim=-1)
                valid = labels[start:stop] != -100
                dl = ps - ctx.lam*pt
                rows = torch.arange(dl.shape[0], device=dl.device)
                dl[rows, labels[start:stop].clamp_min(0)] -= 1-ctx.lam
                dl *= valid[:,None]
                dl *= grad.float()
                dh[start:stop] = dl @ weight
                dw.add_((dl.T @ h).to(dw.dtype))
        return dh, dw, None, None, None, None, None


def kd_loss(hidden, weight, teacher_hidden, teacher_weight, labels, lam=.25, chunk=128):
    return ChunkedKD.apply(hidden[:,:-1].reshape(-1,hidden.shape[-1]), weight,
                           teacher_hidden[:,:-1].reshape(-1,hidden.shape[-1]), teacher_weight,
                           labels[:,1:].reshape(-1), lam, chunk)


def attach_loss(model, cfg, accelerator):
    from transformers.modeling_outputs import CausalLMOutputWithPast
    method = cfg['parallel_method']['name']
    original = model.forward
    teacher = None
    if method == 'kd':
        from pathlib import Path
        from training_state import read_json, file_digest
        from transformers import AutoModelForCausalLM
        from attention_backend import configure_attention
        spec = cfg['parallel_method']['teacher']
        root = Path(spec['export'])
        # The small immutable export manifest pins every weight shard. Full-file
        # hashes are checked once, by rank zero, before any teacher load.
        if file_digest(Path(__file__).parent/'teacher-manifest.json') != spec['manifest_sha256']:
            raise ValueError('Teacher export manifest changed')
        from train import on_main
        def verify_teacher():
            for name,item in read_json(Path(__file__).parent/'teacher-manifest.json')['files'].items():
                path = root/name
                if path.stat().st_size != item['size'] or file_digest(path) != item['sha256']:
                    raise ValueError('Teacher export file changed: '+name)
            return True
        on_main(accelerator, verify_teacher)
        teacher = AutoModelForCausalLM.from_pretrained(root, local_files_only=True,
                  torch_dtype=torch.bfloat16, attn_implementation='sdpa').to(accelerator.device)
        teacher.requires_grad_(False); teacher.eval(); teacher.config.use_cache=False
        configure_attention(teacher, cfg['attention_backend'])
        for key in ('vocab_size','hidden_size','num_hidden_layers','rope_theta','rope_scaling'):
            if getattr(teacher.config,key) != getattr(model.config,key):
                raise ValueError('Teacher architecture/tokenizer contract differs: '+key)
    else:
        from cut_cross_entropy import linear_cross_entropy

    def forward(self, input_ids=None, attention_mask=None, labels=None, **kwargs):
        global _STATS
        if not self.training or labels is None:
            return original(input_ids=input_ids,attention_mask=attention_mask,labels=labels,**kwargs)
        # Remain inside the complete DDP forward; do not bypass the wrapper.
        kwargs.pop('num_items_in_batch',None); kwargs.pop('use_cache',None)
        kwargs.pop('logits_to_keep',None); kwargs.pop('num_logits_to_keep',None)
        hidden = self.model(input_ids=input_ids,attention_mask=attention_mask,use_cache=False,
                            return_dict=True,**kwargs).last_hidden_state
        weight = self.get_output_embeddings().weight
        if method == 'kd':
            with torch.no_grad():
                th = teacher.model(input_ids=input_ids,attention_mask=attention_mask,
                                   use_cache=False,return_dict=True,**kwargs).last_hidden_state
            loss, ce, kl = kd_loss(hidden,weight,th,teacher.get_output_embeddings().weight,labels)
            values = torch.stack((ce.double(),kl.double(),(labels[:,1:]!=-100).sum().double()))
            _STATS = values if _STATS is None else _STATS+values
        else:
            # Vendor preset cce_exact disables gradient filtering and enables Kahan.
            loss = linear_cross_entropy(hidden,weight,labels,shift=True,reduction='sum',impl='cce_exact')
        return CausalLMOutputWithPast(loss=loss,logits=None)
    model.forward = types.MethodType(forward,model)


def consume_stats(accelerator):
    global _STATS
    if _STATS is None:return {}
    values = accelerator.reduce(_STATS,reduction='sum').cpu().tolist(); _STATS=None
    ce,kl,count=values
    return {'train/hard_ce':ce/count,'train/teacher_kl':kl/count,
            'train/distillation_objective':(.75*ce+.25*kl)/count}
