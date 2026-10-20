"""Token-bound piecewise LR; independent of W&B and optimizer-update counters."""
Q=131072

def multiplier(tokens,cfg):
    if type(tokens) is not int or tokens<0 or tokens%Q:
        raise ValueError('Schedule requires exact nonpadding input-token quanta')
    if tokens>cfg['max_training_tokens']:raise ValueError('Token endpoint exceeded')
    if tokens==cfg['max_training_tokens']:return 0.0
    p=next(p for p in cfg['token_phases'] if p['start_tokens']<=tokens<p['end_tokens'])
    a,w,d,h,f,e=[p[k] for k in ('start_tokens','warmup_end_tokens','decay_start_tokens','plateau_start_tokens','final_decay_start_tokens','end_tokens')]
    if not a<w<d<h<f<e:raise ValueError('Overlapping token schedule segments')
    peak=p['peak_ratio'];plateau=p['plateau_ratio']
    if tokens<w:return peak*(tokens-a+Q)/(w-a)
    if tokens<d:return peak
    if tokens<h:return peak*(1+(plateau-1)*(tokens-d)/(h-d-Q))
    if tokens<f:return peak*plateau
    return peak*plateau*max(0.,1-(tokens-f)/(e-f-Q))

class Scheduler:
    def __init__(self,optimizer,cfg):
        self.optimizer,self.cfg=optimizer,cfg
        self.base_lrs=[float(g['lr']) for g in optimizer.param_groups]
        self.set_tokens(0)
    def set_tokens(self,tokens):
        self.tokens=tokens;m=multiplier(tokens,self.cfg)
        for g,lr in zip(self.optimizer.param_groups,self.base_lrs):
            g['initial_lr']=lr;g['lr']=lr*m
            g['decay_lr']=self.cfg['muon']['learning_rate']*m if g['name'].startswith('muon') else 0.
    def state_dict(self):
        return dict(tokens=self.tokens,base_lrs=self.base_lrs,clock='nonpadding_input_tokens',token_phases=self.cfg['token_phases'])
    def load_state_dict(self,state):
        if state['base_lrs']!=self.base_lrs or state['clock']!='nonpadding_input_tokens' or state['token_phases']!=self.cfg['token_phases']:
            raise ValueError('Scheduler provenance changed')
        self.set_tokens(state['tokens'])
