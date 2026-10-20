"""Validated token scheduler and native Muon/AdamW updates for fresh3B."""
import math


class StepScheduler:
    def __init__(self, optimizer, cfg):
        self.optimizer, self.cfg = optimizer, cfg
        self.base_lrs = [g.get('initial_lr', float(g['lr'])) for g in optimizer.param_groups]
        self.set_step(0, 0)

    def set_step(self, completed_updates, tokens):
        from training_state import schedule_multiplier
        from batch3b_contracts import token_prefix
        PLANNED_UPDATES = 19080
        if (type(completed_updates) is not int or not 0 <= completed_updates <= PLANNED_UPDATES
                or token_prefix()[completed_updates] != tokens):
            raise ValueError('Actual optimizer-step and token/data cursors disagree')
        self.completed_updates = completed_updates
        self.tokens = tokens
        multiplier = schedule_multiplier(completed_updates, self.cfg)
        for group, peak in zip(self.optimizer.param_groups, self.base_lrs):
            group['initial_lr'] = peak
            if hasattr(group['lr'], 'fill_'):
                group['lr'].fill_(peak * multiplier)
            else:
                group['lr'] = peak * multiplier

    def state_dict(self):
        return dict(tokens=self.tokens, completed_updates=self.completed_updates,
                    clock='optimizer_updates', warmup_clock=self.cfg['warmup_clock'],base_lrs=self.base_lrs,
                    names=[g['name'] for g in self.optimizer.param_groups])

    def load_state_dict(self, state):
        if state['base_lrs'] != self.base_lrs or state['names'] != [g['name'] for g in self.optimizer.param_groups]:
            raise ValueError('Checkpoint scheduler group/peak mismatch')
        if state['clock'] != 'optimizer_updates' or state['warmup_clock'] != self.cfg['warmup_clock']:
            raise ValueError('Checkpoint scheduler clock changed')
        self.set_step(state['completed_updates'], state['tokens'])


def adamw_class():
    import torch

    class VariableBetaAdamW(torch.optim.AdamW):
        """Native fixed-beta updates; exact historical-product correction after switch."""
        @torch.no_grad()
        def step(self, closure=None):
            if closure is not None:
                raise ValueError('Closures are not part of this study')
            if not any(g.get('variable_beta2', False) for g in self.param_groups):
                old = {p:self.state[p].get('beta2_product') for g in self.param_groups
                       for p in g['params'] if p.grad is not None}
                result = super().step()
                for group in self.param_groups:
                    for p in group['params']:
                        if p.grad is None:
                            continue
                        state = self.state[p]
                        previous = old[p]
                        state['beta2_product'] = (group['betas'][1] ** int(state['step'].item())
                                                 if previous is None else previous * group['betas'][1])
                return result
            for group in self.param_groups:
                b1, b2 = group['betas']
                lr, eps, wd = group['lr'], group['eps'], group['weight_decay']
                for p in group['params']:
                    if p.grad is None:
                        continue
                    grad, state = p.grad, self.state[p]
                    if grad.is_sparse:
                        raise ValueError('Sparse gradients are unsupported')
                    if not state:
                        state.update(step=torch.zeros((), device=p.device),
                                     exp_avg=torch.zeros_like(p), exp_avg_sq=torch.zeros_like(p),
                                     beta2_product=1.)
                    state['step'].add_(1)
                    state['beta2_product'] *= b2
                    step = int(state['step'].item())
                    p.mul_(1 - lr * wd)
                    state['exp_avg'].lerp_(grad, 1 - b1)
                    state['exp_avg_sq'].mul_(b2).addcmul_(grad, grad, value=1 - b2)
                    denominator = state['exp_avg_sq'].sqrt().div_(math.sqrt(1 - state['beta2_product'])).add_(eps)
                    p.addcdiv_(state['exp_avg'], denominator, value=-lr / (1 - b1 ** step))
            return None
    return VariableBetaAdamW


def build_optimizers(model, cfg):
    import torch
    from optimizers import muon_groups
    groups, metadata = muon_groups(model, cfg)
    adamw = adamw_class()(groups[:2], lr=cfg['learning_rate'],
                          betas=tuple(cfg['betas']), eps=cfg['epsilon'], fused=True)
    from ppl_optimizer import StudyMuon
    method=cfg['ppl']['matrix']
    matrix = torch.optim.Muon(groups[2:]) if method=='ns' else StudyMuon(groups[2:],method)
    metadata.update(adamw_beta2_policy=cfg['adamw_beta2_policy'],
                    beta2_bias_correction='1 minus historical product; state retained across switches',
                    scheduler='common nonpadding-input-token clock',
                    matrix_method=method,attention_mask=cfg['ppl']['mask'],
                    implementation='native Muon' if method=='ns' else 'native-compatible isolated '+method,
                    normuon_beta2=.95 if method=='nor' else None)
    return [adamw, matrix], metadata


def configure_update(optimizers, cfg, width, tokens):
    from batch3b_contracts import beta2_for
    beta2 = beta2_for(cfg, width, tokens)
    for group in optimizers[0].param_groups:
        group['betas'] = (cfg['betas'][0], beta2)
        group['variable_beta2'] = cfg['adamw_beta2_policy'] == 'token_scaled' and tokens >= cfg['batch_switch_tokens']
    return beta2
