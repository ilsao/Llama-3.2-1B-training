"""Token-local observers, no RNG calls and no feedback into training."""
import contextlib
import torch
from batch3b_contracts import TARGET,observer_knots

class Observers:
    def __init__(self,model,cfg):
        self.model=model;self.betas=['token_ema','finite_window_mean']
        self.knots=observer_knots(cfg);self.shadows={};self.last_tokens=0;self.mean_count=0
        self.updates=0;self.transfer=None

    @torch.no_grad()
    def update(self,tokens):
        if tokens<=self.last_tokens:raise ValueError('Observer token clock must advance')
        params=dict(self.model.named_parameters())
        if tokens>=self.knots['ema_start']:
            key='token_ema'
            if key not in self.shadows:self.shadows[key]={n:p.detach().clone() for n,p in params.items()}
            else:
                beta=2**(-(tokens-self.last_tokens)/(.01*TARGET))
                for n,p in params.items():self.shadows[key][n].lerp_(p,1-beta)
        if tokens in self.knots['mean']:
            key='finite_window_mean';self.mean_count+=1
            if key not in self.shadows:self.shadows[key]={n:p.detach().clone() for n,p in params.items()}
            else:
                for n,p in params.items():self.shadows[key][n].lerp_(p,1/self.mean_count)
        self.last_tokens=tokens;self.updates+=1

    def state_dict(self):
        return dict(knots=self.knots,shadows=self.shadows,last_tokens=self.last_tokens,
                    mean_count=self.mean_count,updates=self.updates)

    def load_state_dict(self,state):
        if state['knots']!=self.knots:raise ValueError('Observer plan changed')
        self.shadows=state['shadows'];self.last_tokens=state['last_tokens']
        self.mean_count=state['mean_count'];self.updates=state['updates']

    @contextlib.contextmanager
    def weights(self,key):
        with torch.no_grad():
            params=dict(self.model.named_parameters())
            saved={n:p.detach().clone() for n,p in params.items()}
            try:
                for n,p in params.items():p.copy_(self.shadows[key][n])
                yield
            finally:
                for n,p in params.items():p.copy_(saved[n])
