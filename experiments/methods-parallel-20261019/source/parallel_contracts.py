"""Dependency-free fixed token-clock scheduler for independent methods."""
class Scheduler:
    def __init__(self,optimizer,cfg):
        self.optimizer,self.cfg=optimizer,cfg
        self.base_lrs=[float(g['lr']) for g in optimizer.param_groups]
        self.set_tokens(0)
    def set_tokens(self,tokens):
        from training_state import schedule_multiplier
        self.tokens=tokens
        if self.cfg['parallel_method']['name'] in ('sf','observer'):
            multiplier=min(1.,(tokens/131072+1)/170)
        else:multiplier=schedule_multiplier(min(tokens//131072,7629),self.cfg)
        for g,lr in zip(self.optimizer.param_groups,self.base_lrs):
            g['initial_lr']=lr;g['lr']=lr*multiplier
            g['decay_lr']=self.cfg['muon']['learning_rate']*multiplier if g['name'].startswith('muon') else 0.
    def state_dict(self):return dict(tokens=self.tokens,base_lrs=self.base_lrs)
    def load_state_dict(self,state):
        if state['base_lrs']!=self.base_lrs:raise ValueError('LR mapping changed')
        self.set_tokens(state['tokens'])

