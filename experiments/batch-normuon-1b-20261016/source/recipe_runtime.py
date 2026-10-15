"""Opt-in recipe experiments. Tensor operations are only called inside Slurm."""
import contextlib
import hashlib
import math

INITIALIZERS = ('gaussian', 'residual_half', 'residual_depth', 'fan_in', 'orthogonal', 'embedding_half')


def validate(cfg):
    if cfg.get('initialization', 'gaussian') not in INITIALIZERS:
        raise ValueError('Unknown initialization')
    if cfg.get('clipping', 'global') not in ('global', 'late_one', 'per_optimizer'):
        raise ValueError('Unknown clipping')
    ema = cfg.get('ema', [])
    if not isinstance(ema, list) or len(set(ema)) != len(ema) or any(type(b) not in (float,int) or not 0 < b < 1 for b in ema):
        raise ValueError('EMA betas must be unique and between zero and one')
    if len(ema) > 3: raise ValueError('At most three EMA shadows')
    transfer=cfg.get('lr_transfer')
    if transfer is not None:
        if set(transfer)!={'study','kind','multiplier','ema_stride_steps'} or transfer['study']!='lr-transfer-20261015' or transfer['kind'] not in ('validation','screen','proxy','decay'):
            raise ValueError('Invalid LR transfer configuration')
        if type(transfer['ema_stride_steps']) is not int or transfer['ema_stride_steps']<1:
            raise ValueError('Invalid token-spaced EMA cadence')
        proxy=transfer['kind']!='decay'
        if (cfg['ema']!=([.6] if proxy else []) or cfg['lr_scheduler']!=('warmup_constant' if proxy else 'wsd_hold_zero') or 'phase_schedule' in cfg):
            raise ValueError('LR transfer must use a stable proxy or single WSD target')


def initialize(model, cfg, device):
    import torch
    mode = cfg.get('initialization', 'gaussian')
    statistics = {}
    with torch.no_grad(), torch.random.fork_rng(devices=[device.index]):
        for name, p in model.named_parameters():
            hidden = p.ndim == 2 and name.startswith('model.layers.')
            residual = name.endswith(('o_proj.weight', 'down_proj.weight'))
            scale = 1.
            if mode == 'residual_half' and residual: scale = .5
            if mode == 'residual_depth' and residual: scale = 1/math.sqrt(2*model.config.num_hidden_layers)
            if mode == 'fan_in' and hidden: scale = math.sqrt(2048/p.shape[1])
            if mode == 'embedding_half' and p is model.get_input_embeddings().weight: scale = .5
            if mode == 'orthogonal' and hidden:
                norm = p.norm().to(device)
                seed = int.from_bytes(hashlib.sha256(f'{cfg["seed"]}:{name}'.encode()).digest()[:8], 'little') % (2**63-1)
                generator = torch.Generator(device=device).manual_seed(seed)
                shape = tuple(p.shape)
                work = torch.randn(shape if shape[0] >= shape[1] else shape[::-1], device=device, generator=generator)
                q,r = torch.linalg.qr(work, mode='reduced')
                q.mul_(torch.where(r.diagonal() >= 0, 1., -1.))
                if shape[0] < shape[1]: q = q.T
                q.mul_(norm/q.norm()); p.copy_(q.cpu())
                del work,q,r
            elif scale != 1: p.mul_(scale)
            if residual or p is model.get_input_embeddings().weight:
                statistics[name] = {'std':float(p.std()), 'norm':float(p.norm()), 'shape':list(p.shape)}
    return {'variant':mode, 'statistics':statistics, 'rng_preserved':True, 'unique_parameters':len(list(model.parameters()))}


def clip(model, optimizers, cfg, tokens, accelerator):
    import torch
    accelerator.unscale_gradients()
    groups = {g['name']:g['params'] for o in optimizers for g in o.param_groups}
    norms = {name:torch.linalg.vector_norm(torch.stack([p.grad.detach().norm() for p in ps if p.grad is not None]))
             for name,ps in groups.items() if any(p.grad is not None for p in ps)}
    total = torch.linalg.vector_norm(torch.stack(list(norms.values())))
    mode = cfg.get('clipping','global')
    limit = 1. if mode == 'late_one' and tokens >= .8*cfg['target_training_tokens'] else cfg['max_grad_norm']
    if mode == 'per_optimizer':
        for o in optimizers:
            torch.nn.utils.clip_grad_norm_([p for g in o.param_groups for p in g['params']], .5)
    elif limit > 0: torch.nn.utils.clip_grad_norm_(model.parameters(),limit)
    metrics = {f'train/grad_norm_{k}':float(v) for k,v in norms.items()}
    metrics['train/clip_limit'] = .5 if mode == 'per_optimizer' else limit
    return total,metrics


def diagnostic_snapshot(model, step):
    if step % 100: return {},{}
    samples, metrics = {},{}
    for name,p in model.named_parameters():
        if p.grad is None:continue
        samples[name] = p.detach().flatten()[:256].clone()
        metrics['train/layer_grad_norm/'+name] = float(p.grad.norm())
    return samples,metrics


def diagnostic_update(model, samples):
    return {'train/sampled_update_weight_ratio/'+name:float((p.detach().flatten()[:256]-samples[name]).norm()/(samples[name].norm()+1e-12))
            for name,p in model.named_parameters() if name in samples}


class EMA:
    def __init__(self, model, cfg):
        self.model=model;self.betas=cfg.get('ema',[]);self.target=cfg['target_training_tokens'];self.shadows={};self.updates=0;self.transfer=cfg.get('lr_transfer');self.batch=cfg['world_size']*cfg['microbatch_size']*cfg['gradient_accumulation_steps']*cfg['sequence_length'] if self.transfer else 0;self.last_tokens=0
    def update(self, tokens):
        import torch
        if not self.betas:return
        if self.transfer:
            if tokens<=self.last_tokens or tokens%self.batch:raise ValueError('EMA token cursor must advance in whole updates')
            self.last_tokens=tokens
            if (tokens//self.batch)%self.transfer['ema_stride_steps']:return
        elif tokens < .8*self.target:return
        with torch.no_grad():
            if not self.shadows:
                self.shadows={str(b):{n:p.detach().clone() for n,p in self.model.named_parameters()} for b in self.betas}
            else:
                for b in self.betas:
                    for n,p in self.model.named_parameters():self.shadows[str(b)][n].lerp_(p.detach(),1-b)
        self.updates+=1
    def state_dict(self):return {'shadows':self.shadows,'updates':self.updates,'betas':self.betas,'last_tokens':self.last_tokens,'transfer':self.transfer}
    def load_state_dict(self,state):
        if state['betas'] != self.betas:raise ValueError('EMA settings changed')
        if state.get('transfer')!=self.transfer:raise ValueError('EMA cadence changed')
        self.shadows=state['shadows'];self.updates=state['updates'];self.last_tokens=state.get('last_tokens',0)
    @contextlib.contextmanager
    def weights(self,beta):
        import torch
        with torch.no_grad():
            saved={n:p.detach().clone() for n,p in self.model.named_parameters()}
            try:
                for n,p in self.model.named_parameters():p.copy_(self.shadows[str(beta)][n])
                yield
            finally:
                for n,p in self.model.named_parameters():p.copy_(saved[n])
