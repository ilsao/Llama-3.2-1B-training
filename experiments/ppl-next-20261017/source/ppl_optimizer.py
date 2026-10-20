"""Native Muon outer update with isolated PE / neuron normalization switches."""
import torch
from torch.optim._muon import _zeropower_via_newtonschulz, _adjust_lr

PE_COEFFICIENTS=(
 (8.156554524902461,-22.48329292557795,15.878769915207462),
 (4.042929935166739,-2.808917465908714,.5000178451051316),
 (3.8916678022926607,-2.772484153217685,.5060648178503393),
 (3.285753657755655,-2.3681294933425376,.46449024233003106),
 (2.3465413258596377,-1.7097828382687081,.42323551169305323))

def polar_express(g):
    x=g.bfloat16()
    transpose=x.shape[0]>x.shape[1]
    if transpose:x=x.T
    x=x/(1.02*x.norm()+1e-6)
    for a,b,c in PE_COEFFICIENTS:
        gram=x@x.T
        x=a*x+(b*gram+c*(gram@gram))@x
    return x.T if transpose else x

def neuron_normalize(u,v):
    u=u.float()
    norm=u.norm()
    v.lerp_(u.square().mean(dim=-1,keepdim=True),.05)
    y=u/(v.sqrt()+1e-8)
    return y*(norm/y.norm().clamp_min(1e-8))

class StudyMuon(torch.optim.Muon):
    def __init__(self,groups,method,cautious_decay=False):
        super().__init__(groups)
        if method not in ('ns','pe','nor'):raise ValueError(method)
        self.method=method
        self.cautious_decay=cautious_decay
        self.collect_diagnostics=False
        self.mask_counts=[]

    @torch.no_grad()
    def step(self,closure=None):
        if closure is not None:raise ValueError('Closures unsupported')
        self.mask_counts=[]
        for group in self.param_groups:
            lr=float(group['lr'])
            for p in group['params']:
                if p.grad is None:continue
                g=p.grad
                if g.is_sparse or p.ndim!=2:raise ValueError('Dense matrices only')
                state=self.state[p]
                if 'momentum_buffer' not in state:state['momentum_buffer']=torch.zeros_like(p)
                buf=state['momentum_buffer']
                buf.lerp_(g,1-group['momentum'])
                u=g.lerp(buf,group['momentum']) if group['nesterov'] else buf
                if self.method=='pe':u=polar_express(u)
                else:u=_zeropower_via_newtonschulz(u,group['ns_coefficients'],group['ns_steps'],group['eps'])
                if self.method=='nor':
                    if 'variance_neuron' not in state:
                        state['variance_neuron']=torch.zeros((p.shape[0],1),device=p.device,dtype=torch.float32)
                    u=neuron_normalize(u,state['variance_neuron'])
                if self.cautious_decay:
                    # Select on pre-update W and post-NorMuon descent direction.
                    mask=(p*u>=0).to(p.dtype)
                    if self.collect_diagnostics:self.mask_counts.append((mask.sum(),p.numel()))
                    p.addcmul_(p,mask,value=-lr*group['weight_decay'])
                else:
                    p.mul_(1-lr*group['weight_decay'])
                p.add_(u,alpha=-_adjust_lr(lr,group['adjust_lr_fn'],p.shape))

