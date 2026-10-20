"""Isolated fixed-recipe adapters. Import only on allocated compute nodes."""
import contextlib
import math
from pathlib import Path
import sys
import types
import torch

SOURCE=Path(__file__).resolve().parent
for package in ('dion','SOAP','ml-cross-entropy'):
    sys.path.insert(0,str(SOURCE/'vendor'/package))
ACTIVE=[]
CFG=None
LOSS_STATS=None


def ns5(g,epsilon=1e-7):
    x=g.bfloat16()
    transposed=x.shape[-2]>x.shape[-1]
    if transposed:x=x.mT
    x=x/(x.norm(dim=(-2,-1),keepdim=True)+epsilon)
    for _ in range(5):
        a=x@x.mT
        x=3.4445*x+(-4.775*a+2.0315*(a@a))@x
    return x.mT if transposed else x


class AdaptiveMatrix(torch.optim.Optimizer):
    """Replicated-DDP AdaMuon v3 direction; no extra optimizer collectives."""
    def __init__(self,groups):super().__init__(groups,dict(lr=.001,weight_decay=0.))
    @torch.no_grad()
    def step(self,closure=None):
        if closure is not None:raise ValueError('No closure')
        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None:continue
                s=self.state[p]
                if not s:s.update(m=torch.zeros_like(p),v=torch.zeros_like(p))
                s['m'].mul_(.95).add_(p.grad)
                direction=p.grad.add(s['m'],alpha=.95)
                u=ns5(direction.sign()).float()
                s['v'].lerp_(u.square(),.05)
                u=u/(s['v'].sqrt()+1e-8)
                u.mul_(.2*math.sqrt(p.numel())/u.norm().clamp_min(1e-8))
                p.add_(u,alpha=-float(group['lr']))


class FeedbackOptimizer(torch.optim.Optimizer):
    """SF-NorMuon + SF-AdamW, or a constant-LR observer-only bridge."""
    def __init__(self,groups,feedback):
        super().__init__(groups,dict(lr=.001,weight_decay=0.,step=0,weight_sum=0.))
        self.feedback=feedback
    @torch.no_grad()
    def step(self,closure=None):
        if closure is not None:raise ValueError('No closure')
        beta=.9
        for g in self.param_groups:
            lr=float(g['lr']);g['step']+=1
            if not g['name'].startswith('muon'):lr*=math.sqrt(1-.95**g['step'])
            g['last_effective_lr']=lr
            weight=lr*lr;g['weight_sum']+=weight
            c=weight/max(g['weight_sum'],1e-30)
            for p in g['params']:
                if p.grad is None:continue
                s=self.state[p]
                if not s:
                    s.update(z=p.detach().clone(),x=p.detach().clone(),
                             v=torch.zeros((p.shape[0],1),device=p.device) if g['name'].startswith('muon') else torch.zeros_like(p))
                    if g['name'].startswith('muon'):s['m']=torch.zeros_like(p)
                if g['name'].startswith('muon'):
                    s['m'].lerp_(p.grad,.05)
                    u=ns5(s['m']).float();s['v'].lerp_(u.square().mean(-1,keepdim=True),.05)
                    u=u/(s['v'].sqrt()+1e-8)
                    u.mul_(.2*math.sqrt(p.numel())/u.norm().clamp_min(1e-8))
                else:
                    # SF-AdamW has interpolation momentum, not an additional Adam first moment.
                    s['v'].lerp_(p.grad.square(),.05)
                    u=p.grad/(s['v'].sqrt()+1e-8)
                z=s['z'] if self.feedback else p
                if g['name'].startswith('muon'):
                    g['last_decay_factor']=1-g['decay_lr']*.1
                    z.mul_(g['last_decay_factor'])
                z.add_(u,alpha=-lr)
                s['x'].lerp_(z,c)
                if self.feedback:p.copy_(s['x']*beta+z*(1-beta))
                else:s['z'].copy_(p)
    @contextlib.contextmanager
    def evaluation_weights(self):
        # Retain exact y/fast weights; eval/train conversion cannot accumulate rounding drift.
        saved=[]
        with torch.no_grad():
            for g in self.param_groups:
                for p in g['params']:
                    if 'x' in self.state[p]:
                        saved.append((p,p.detach().clone()));p.copy_(self.state[p]['x'])
        try:yield
        finally:
            with torch.no_grad():
                for p,value in saved:p.copy_(value)


def _matrix_groups(group,method,cfg):
    buckets={}
    for p in group['params']:buckets.setdefault(p.shape[1],[]).append(p)
    result=[]
    for fan_in,params in sorted(buckets.items()):
        # Nominal original-Muon update RMS = base LR / sqrt(fan_in).
        denom=math.sqrt(fan_in)*(1. if method=='soap' else .2)
        lr=cfg['muon']['learning_rate']/denom
        if method=='dion025':lr*=2
        result.append(dict(params=params,name=f'muon_in{fan_in}',lr=lr,weight_decay=0.,
                           reference_peak=cfg['muon']['learning_rate'],decay_lr=0.))
    return result


def build_optimizers(model,cfg):
    global ACTIVE,CFG
    CFG=cfg
    from optimizers import muon_groups
    from ppl_optimizer import StudyMuon
    groups,metadata=muon_groups(model,cfg)
    method=cfg['parallel_method']['name']
    if method in ('sf','observer'):
        aux=[g for g in groups[:2] if g['params']]
        matrix=_matrix_groups(groups[2],method,cfg)
        opts=[FeedbackOptimizer(aux,feedback=method=='sf'), FeedbackOptimizer(matrix,feedback=method=='sf')]
    else:
        adam=torch.optim.AdamW(groups[:2],lr=cfg['learning_rate'],betas=tuple(cfg['betas']),eps=cfg['epsilon'],fused=True)
        if method in ('native','kd','cce'):
            groups[2]['weight_decay']=0.
            matrix=StudyMuon(groups[2:],'nor')
        elif method in ('dion1','dion025'):
            from dion import Dion3
            matrix=Dion3(_matrix_groups(groups[2],method,cfg),distributed_mesh=None,
                         fraction=1. if method=='dion1' else .25,mu=.95,muon_beta2=.95,
                         weight_decay=0.,adjust_lr='rms_norm',use_triton=False,use_polar_express=False,
                         use_gram_newton_schulz=False,newton_schulz_func=ns5)
        elif method=='adamuon':matrix=AdaptiveMatrix(_matrix_groups(groups[2],method,cfg))
        elif method=='soap':
            from soap import SOAP
            matrix=SOAP(_matrix_groups(groups[2],method,cfg),betas=(.95,.95),shampoo_beta=.95,
                        weight_decay=0.,precondition_frequency=10,max_precond_dim=10000,
                        merge_dims=False,precondition_1d=False,normalize_grads=False,correct_bias=True)
        else:raise ValueError(method)
        def apply_decay(optimizer,args,kwargs):
            with torch.no_grad():
                for group in optimizer.param_groups:
                    group['last_decay_factor']=1-group['decay_lr']*.1
                    for p in group['params']:
                        if p.grad is not None:p.mul_(group['last_decay_factor'])
        matrix.register_step_pre_hook(apply_decay)
        opts=[adam,matrix]
    names={id(p):name for name,p in model.named_parameters()}
    all_groups=[g for o in opts for g in o.param_groups]
    assigned=[id(p) for g in all_groups for p in g['params']]
    assert len(assigned)==len(set(assigned))==len(list(model.parameters()))
    assert len({g['name'] for g in all_groups})==len(all_groups)
    metadata.update(implementation='parallel fixed-recipe '+method,
        parameter_names={g['name']:[names[id(p)] for p in g['params']] for g in all_groups},
        parameter_counts={g['name']:sum(p.numel() for p in g['params']) for g in all_groups},
        lr_mapping='nominal RMS matching by input dimension; fixed, not tuned',
        weight_decay='external pre-update reference-LR decay; SF decay on z',
        optimizer_parallelism='replicated DDP; Dion distributed_mesh=None; no communication-speedup claim',
        export_weights='average x' if method in ('sf','observer') else 'live',
        scheduler='single token warmup then constant' if method in ('sf','observer') else 'original two-phase token-clock WSD',
        resolved_groups=[dict(name=g['name'],base_learning_rate=float(g['lr']),internal_weight_decay=float(g['weight_decay']),
                             optimizer=type(o).__module__+'.'+type(o).__name__) for o in opts for g in o.param_groups])
    ACTIVE=opts
    return opts,metadata


from parallel_contracts import Scheduler


@contextlib.contextmanager
def eval_weights():
    with contextlib.ExitStack() as stack:
        for opt in ACTIVE:
            if hasattr(opt,'evaluation_weights'):stack.enter_context(opt.evaluation_weights())
        yield


def install():
    import train
    original_build=train.build_model
    original_evaluate=train.evaluate
    original_export=train.export_model
    def build(cfg,run,provenance,accelerator):
        model,metadata=original_build(cfg,run,provenance,accelerator)
        method=cfg['parallel_method']['name']
        if method in ('kd','cce'):
            from parallel_losses import attach_loss
            attach_loss(model,cfg,accelerator)
        return model,metadata
    def evaluate(*args,**kwargs):
        with eval_weights():return original_evaluate(*args,**kwargs)
    def export(*args,**kwargs):
        with eval_weights():return original_export(*args,**kwargs)
    train.build_model=build;train.build_optimizers=build_optimizers;train.TokenScheduler=Scheduler
    train.configure_update=lambda optimizers,cfg,width,tokens: .95
    train.evaluate=evaluate;train.export_model=export


def extra_metrics(accelerator,inputs,seen):
    result={}
    from train import reduce_max
    peak=torch.tensor([torch.cuda.max_memory_allocated(),torch.cuda.max_memory_reserved()],device=accelerator.device,dtype=torch.float64)
    peak=reduce_max(peak,accelerator).cpu().tolist()
    result.update({'train/peak_allocated_bytes':peak[0],'train/peak_reserved_bytes':peak[1]})
    if CFG and CFG['parallel_method']['name']=='kd':
        from online_pipeline import consume_stats
        values=consume_stats(accelerator)
        result.update(values)
        result['train/teacher_inputs_consumed']=seen
    for opt in ACTIVE:
        for g in opt.param_groups:
            if g['name'].startswith('muon'):result['train/actual_decay_factor_'+g['name']]=g.get('last_decay_factor',1.)
            if 'last_effective_lr' in g:result['train/effective_step_lr_'+g['name']]=g['last_effective_lr']
    return result
