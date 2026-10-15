"""Slurm-only optimizer, token scheduler and immutable checkpoint branching."""
import copy
import math
from pathlib import Path


def verify_checkpoint_files(checkpoint, complete):
    from training_state import file_digest
    checkpoint=Path(checkpoint).resolve()
    required={'model.safetensors','optimizer.bin','optimizer_1.bin','training-state.json',
              *(f'custom_checkpoint_{i}.pkl' for i in range(3)),
              *(f'random_states_{i}.pkl' for i in range(8))}
    names=[item['name'] for item in complete['files']]
    if len(set(names))!=len(names) or not required <= set(names):
        raise ValueError('Missing model, optimizer, scheduler, EMA or per-rank RNG checkpoint file')
    for item in complete['files']:
        path=(checkpoint/item['name']).resolve()
        if (not path.is_relative_to(checkpoint) or item['size']<=0 or
            path.stat().st_size!=item['size'] or file_digest(path)!=item['sha256']):
            raise ValueError('Checkpoint checksum mismatch')


class TokenScheduler:
    def __init__(self, optimizer, cfg):
        self.optimizer, self.cfg = optimizer, cfg
        self.base_lrs = [g.get('initial_lr', float(g['lr'])) for g in optimizer.param_groups]
        self.set_tokens(0)

    def set_tokens(self, tokens):
        from training_state import schedule_multiplier
        from batch1b_contracts import Q, END
        self.tokens = tokens
        multiplier = schedule_multiplier(min(tokens // Q, END - 1), self.cfg)
        for group, peak in zip(self.optimizer.param_groups, self.base_lrs):
            group['initial_lr'] = peak
            if hasattr(group['lr'], 'fill_'):
                group['lr'].fill_(peak * multiplier)
            else:
                group['lr'] = peak * multiplier

    def state_dict(self):
        return dict(tokens=self.tokens, base_lrs=self.base_lrs,
                    names=[g['name'] for g in self.optimizer.param_groups])

    def load_state_dict(self, state):
        if state['base_lrs'] != self.base_lrs or state['names'] != [g['name'] for g in self.optimizer.param_groups]:
            raise ValueError('Checkpoint scheduler group/peak mismatch')
        self.set_tokens(state['tokens'])


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
    variant = cfg.get('matrix_optimizer')
    if variant:
        import os
        if os.environ.get('TORCH_COMPILE_DISABLE') != '1':
            raise ValueError('This recovery requires explicit eager NorMuon execution')
        import importlib.metadata
        from muonium import Muon
        if importlib.metadata.version('muonium') != '0.2.0':
            raise ValueError('Require muonium==0.2.0')
        old = groups[2]
        matrix = Muon([dict(params=old['params'], name='muon', algorithm='normuon',
                           muon_beta2=.95, lr=old['lr'], wd=old['weight_decay'],
                           weight_decay=old['weight_decay'])], lr=old['lr'], wd=old['weight_decay'],
                      momentum=.95, nesterov=True, ns_steps=5,
                      orthogonalization_strategy='polar_express', use_cautious_wd=False, foreach=True)
        # muonium intentionally constructs fresh group dictionaries and strips
        # caller metadata. Restore our logging/scheduler contract explicitly.
        for group in matrix.param_groups:
            group.update(name='muon', weight_decay=old['weight_decay'], initial_lr=old['lr'])
        metadata.update(implementation='muonium==0.2.0 NorMuon + torch AdamW',
            matrix_variant=variant,
            matrix_weight_decay='muonium: decoupled term scaled by shape-adjusted LR; cautious disabled',
            matrix_lr_adjustment='sqrt(max(1, rows/columns))',
            matrix_driver='explicit foreach=True; same dispatch in validation and CPU-first training construction',
            matrix_execution='eager torch operations; TORCH_COMPILE_DISABLE=1; unchanged NorMuon and Polar Express math',
            normalization_epsilon='NorMuon1e-8; Polar Express1e-6, pinned package implementation')
    else:
        matrix = torch.optim.Muon(groups[2:])
    metadata.update(adamw_beta2_policy=cfg['adamw_beta2_policy'],
                    beta2_bias_correction='1 minus historical product; state retained across switches',
                    scheduler='common nonpadding-input-token clock')
    return [adamw, matrix], metadata


def configure_update(optimizers, cfg, width, tokens):
    from batch1b_contracts import beta2_for
    beta2 = beta2_for(cfg, width, tokens)
    for group in optimizers[0].param_groups:
        group['betas'] = (cfg['betas'][0], beta2)
        group['variable_beta2'] = cfg['adamw_beta2_policy'] == 'token_scaled' and tokens >= cfg['batch_switch_tokens']
    return beta2


def publish_snapshot(run_path, checkpoint, cfg, state, initialization, provenance, job_start):
    import os
    import time
    from training_state import atomic_json, digest, read_json
    from batch1b_contracts import Q
    tokens = state['confirmed_tokens_at_checkpoint']
    if state['cursor'] * 8192 != tokens or state['global_step'] != tokens // Q:
        raise ValueError('Only exact, successful control updates can publish snapshots')
    checkpoint = Path(checkpoint)
    manifest = dict(checkpoint=str(checkpoint), tokens=tokens,
                    global_step=state['global_step'], cursor=state['cursor'],
                    config_digest=digest(cfg), code_digest=state['code_digest'],
                    environment_digest=state['environment_digest'],
                    provenance_digest=digest(provenance), initialization_signature=initialization['sample_signature'],
                    complete=read_json(checkpoint/'complete.json'), job_id=os.environ['SLURM_JOB_ID'],
                    inherited_h200_hours=8 * (time.time() - job_start) / 3600,
                    published_at=time.time())
    path = run_path/'snapshots'/f'{tokens}.json'
    if path.exists():
        raise ValueError('Never replace a published snapshot')
    atomic_json(path, manifest)


def fork_checkpoint(args, cfg, state, run_path, provenance, initialization, accelerator, schedulers, optimizers):
    from batch1b_contracts import STUDY, Q, check_fork
    from training_state import read_json, digest, file_digest, atomic_json
    from train import on_main, restore_rng
    if args.resume or args.fork_from_run != STUDY + '-control':
        raise ValueError('Only approved study control forks are permitted')
    parent = Path(args.root)/'runs/experiments'/args.fork_from_run

    def verify():
        tokens = cfg['fork_snapshot_tokens']
        manifest = read_json(parent/'snapshots'/f'{tokens}.json')
        cp = Path(manifest['checkpoint'])
        if cp.parent != parent/'checkpoints' or cp.name.endswith('.partial'):
            raise ValueError('Untrusted checkpoint path')
        original = read_json(parent/'resolved-config.json')
        check_fork(original, cfg, tokens)
        saved, complete = read_json(cp/'training-state.json'), read_json(cp/'complete.json')
        if (complete != manifest['complete'] or complete['state_digest'] != digest(saved) or
            manifest['config_digest'] != digest(original) or saved['config_digest'] != digest(original) or
            manifest['code_digest'] != state['code_digest'] or saved['code_digest'] != state['code_digest'] or
            manifest['environment_digest'] != state['environment_digest'] or
            saved['environment_digest'] != state['environment_digest'] or
            manifest['provenance_digest'] != digest(provenance) or
            saved['preparation_fingerprint'] != state['preparation_fingerprint'] or
            manifest['initialization_signature'] != initialization['sample_signature'] or
            saved['confirmed_tokens_at_checkpoint'] != tokens or saved['charged_tokens_at_checkpoint'] != tokens or
            saved['global_step'] != tokens//Q or saved['attempted_updates'] != tokens//Q or
            saved['cursor'] != tokens//8192):
            raise ValueError('Checkpoint identity, state, split or budget mismatch')
        verify_checkpoint_files(cp, complete)
        receipt = read_json(parent/'raw-prefix-receipt.json')
        if receipt['cursor'] < saved['cursor']:
            raise ValueError('Missing consumed raw prefix')
        atomic_json(run_path/'raw-prefix-receipt.json', receipt)
        return str(cp), saved, manifest

    checkpoint, saved, manifest = on_main(accelerator, verify)
    accelerator.load_state(checkpoint)
    restore_rng(checkpoint, accelerator)
    for scheduler in schedulers:
        if scheduler.tokens != manifest['tokens']:
            raise ValueError('Checkpoint scheduler cursor differs from data/token cursor')
    child = copy.deepcopy(saved)
    for key in ('config_digest', 'wandb_id', 'code_digest', 'environment_digest', 'preparation_fingerprint'):
        child[key] = state[key]
    child.update(planned_steps=cfg['max_steps'], interval_nll=0., interval_predictions=0,
                 last_eval_step=saved['global_step'], fork_start_step=saved['global_step'])
    child.pop('last_evaluation', None)
    child.pop('new_best', None)
    child['lineage'] = dict(parent_run=args.fork_from_run, parent_checkpoint=checkpoint,
                            inherited_tokens=manifest['tokens'], inherited_h200_hours=manifest['inherited_h200_hours'],
                            parent_wandb_id=saved['wandb_id'], optimizer_state_preserved=True,
                            scheduler_rebased=False, token_clock_preserved=True)
    on_main(accelerator, lambda: atomic_json(run_path/'fork-lineage.json', child['lineage']))
    return checkpoint, child
