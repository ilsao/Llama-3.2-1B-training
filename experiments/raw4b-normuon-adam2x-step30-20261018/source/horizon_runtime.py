"""Slurm-only checkpoint branching; metadata is checked before tensor loading."""
import copy
import json
from pathlib import Path


def unclipped_gradient_norm(model, accelerator):
    import torch
    accelerator.unscale_gradients()
    # Gradients are synchronized replicated DDP tensors, so count each unique
    # parameter once on one rank; MAX agrees across ranks without world-size sum.
    from train import reduce_max
    squared = torch.zeros((), dtype=torch.float64, device=accelerator.device)
    for parameter in model.parameters():
        if parameter.grad is not None:
            squared += parameter.grad.detach().double().square().sum()
    return reduce_max(squared.sqrt(), accelerator)


def seed_journal(run_path, cfg, state):
    from training_state import atomic_json
    path = run_path / 'token-journal.json'
    if path.exists():
        raise ValueError('Fork journal already exists; no implicit retry')
    confirmed = state['confirmed_tokens_at_checkpoint']
    charged = state['charged_tokens_at_checkpoint']
    if confirmed != charged:
        raise ValueError('Cannot branch from uncertain token accounting')
    atomic_json(path, dict(limit=cfg['max_training_tokens'], confirmed_tokens=confirmed,
                          charged_tokens=charged, next_transaction=0, pending={}))


def fork_checkpoint(args, cfg, state, run_path, provenance, initialization, accelerator, schedulers, optimizers):
    from train import checkpoint_state, on_main
    from training_state import read_json, digest, atomic_json, identifier
    from horizon_contracts import CAMPAIGN, PREFIX_STEP, BATCH_TOKENS, check_fork_configs
    parent = Path(args.root) / 'runs/experiments' / identifier(args.fork_from_run)
    if args.fork_from_run != CAMPAIGN + '-stable-prefix' or not args.run_id.startswith(CAMPAIGN + '-decay-'):
        raise ValueError('Checkpoint fork restricted to the requested five decay branches')

    def verify():
        checkpoint, saved = checkpoint_state(parent)
        original = read_json(parent / 'resolved-config.json')
        check_fork_configs(original, cfg)
        summary = read_json(parent / 'summary.json')
        accounting = read_json(parent / 'resource-accounting.json')
        journal = read_json(parent / 'token-journal.json')
        if (summary['stop_reason'] != 'validation_pause' or saved['global_step'] != PREFIX_STEP or
                saved['attempted_updates'] != PREFIX_STEP or saved['cursor'] != PREFIX_STEP * 16 or
                saved['confirmed_tokens_at_checkpoint'] != PREFIX_STEP * BATCH_TOKENS or
                not accounting['all_jobs_reconciled'] or accounting['budget_exceeded'] or
                read_json(parent/'data-provenance.json') != provenance or
                saved['config_digest'] != digest(original) or
                saved['code_digest'] != state['code_digest'] or
                saved['environment_digest'] != state['environment_digest'] or
                saved['preparation_fingerprint'] != state['preparation_fingerprint'] or
                journal['pending'] or journal['confirmed_tokens'] != saved['confirmed_tokens_at_checkpoint'] or
                journal['charged_tokens'] != saved['charged_tokens_at_checkpoint'] or
                summary['initialization']['sample_signature'] != initialization['sample_signature']):
            raise ValueError('Parent checkpoint, lineage, prefix, or high-water budget mismatch')
        receipt = parent/'raw-prefix-receipt.json'
        if receipt.exists():
            atomic_json(run_path/'raw-prefix-receipt.json', read_json(receipt))
        return checkpoint, saved

    checkpoint, saved = on_main(accelerator, verify)
    accelerator.load_state(checkpoint)
    # Restore model, momentum and per-rank RNG. Rebase only authorized schedule
    # shape; LambdaLR state otherwise retains the parent's future LR curve.
    peaks = [cfg['learning_rate'], cfg['muon']['learning_rate']]
    if len(schedulers) != len(peaks):
        raise ValueError('Unexpected mixed optimizer layout')
    from training_state import schedule_multiplier
    for scheduler, optimizer, peak in zip(schedulers, optimizers, peaks):
        scheduler.base_lrs = [peak] * len(optimizer.param_groups)
        scheduler.last_epoch = saved['global_step']
        scheduler._last_lr = [peak * schedule_multiplier(saved['global_step'], cfg)] * len(optimizer.param_groups)
        for group, value in zip(optimizer.param_groups, scheduler._last_lr):
            group['initial_lr'] = peak
            group['lr'] = value
    child = copy.deepcopy(saved)
    for key in ('config_digest','wandb_id','code_digest','environment_digest','preparation_fingerprint'):
        child[key] = state[key]
    child['planned_steps'] = cfg['max_steps']
    child['lineage'] = dict(parent_run=args.fork_from_run, parent_checkpoint=checkpoint,
                           inherited_tokens=saved['confirmed_tokens_at_checkpoint'],
                           parent_wandb_id=saved['wandb_id'], scheduler_rebased=True)
    # Best metrics/timing include prefix for logical lineage; branch selection
    # uses final endpoint only. W&B starts a fresh identity at the inherited step.
    on_main(accelerator, lambda: atomic_json(run_path/'fork-lineage.json', child['lineage']))
    return checkpoint, child
