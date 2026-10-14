"""Explicit frozen phase1 checkpoint forks for the user-requested LR sweep."""
import copy
from pathlib import Path

REPO=Path(__file__).resolve().parents[1]
DIRECTORY=REPO/'experiments/horizon-wsd-20261014'
PARENT='horizon-wsd-20261014-final-resplit'
SCALES={'tail-lr100':1.,'tail-lr070':.7,'tail-lr050':.5,'tail-lr030':.3}
PREFIX_STEP=12715
BATCH_TOKENS=131072


def validate_configs(parent, child, run_id):
    prefix='horizon-wsd-20261014-'
    key=run_id.removeprefix(prefix)
    if not run_id.startswith(prefix) or key not in SCALES:
        raise ValueError('Fork restricted to four requested resplit LR branches')
    allowed={'category','parent_runs','eval_steps','sweep_purpose','sweep_stage','phase_schedule'}
    if {k:v for k,v in parent.items() if k not in allowed}!={k:v for k,v in child.items() if k not in allowed}:
        raise ValueError('LR fork changed numerical/data settings outside requested phase2 scale')
    if child['category']!='pilot' or child['parent_runs']!=[PARENT] or child['eval_steps']!=254:
        raise ValueError('Incorrect branch identity/evaluation contract')
    if parent['phase_schedule'][0]!=child['phase_schedule'][0]:
        raise ValueError('First phase schedule changed')
    phase=copy.deepcopy(parent['phase_schedule'][1]);phase['peak_ratio']=SCALES[key]
    if child['phase_schedule'][1]!=phase or child['phase_schedule'][0]['end']!=PREFIX_STEP:
        raise ValueError('Incorrect second phase scale or frozen checkpoint boundary')
    return SCALES[key]


def fork_checkpoint(args,cfg,state,run_path,provenance,initialization,accelerator,schedulers,optimizers):
    from train import on_main
    from training_state import read_json,digest,atomic_json,file_digest,schedule_multiplier
    plan=read_json(DIRECTORY/'resplit-lr-plan.json')
    if args.fork_from_run!=PARENT:
        raise ValueError('Only the exact resplit phase1 parent is authorized')
    parent=Path(args.root)/'runs/experiments'/PARENT
    checkpoint=Path(plan['checkpoint'])
    if checkpoint.parent!=parent/'checkpoints':
        raise ValueError('Frozen checkpoint path differs from the parent')

    def verify():
        original=read_json(parent/'resolved-config.json')
        scale=validate_configs(original,cfg,args.run_id)
        if file_digest(checkpoint/'training-state.json')!=plan['checkpoint_state_sha256'] or file_digest(checkpoint/'complete.json')!=plan['checkpoint_complete_sha256']:
            raise ValueError('Frozen checkpoint metadata changed')
        saved=read_json(checkpoint/'training-state.json');complete=read_json(checkpoint/'complete.json')
        if digest(saved)!=complete['state_digest']:
            raise ValueError('Checkpoint state digest differs from completion seal')
        for item in complete['files']:
            p=checkpoint/item['name']
            if not p.is_file() or p.stat().st_size!=item['size']:
                raise ValueError('Incomplete frozen checkpoint')
        accounting=read_json(parent/'resource-accounting.json');journal=read_json(parent/'token-journal.json')
        summary=read_json(parent/'summary.json')
        if (saved['global_step']!=PREFIX_STEP or saved['attempted_updates']!=PREFIX_STEP or saved['cursor']!=PREFIX_STEP*16 or
                saved['confirmed_tokens_at_checkpoint']!=PREFIX_STEP*BATCH_TOKENS or saved['charged_tokens_at_checkpoint']!=PREFIX_STEP*BATCH_TOKENS or
                saved['config_digest']!=digest(original) or saved['code_digest']!=state['code_digest'] or
                saved['environment_digest']!=state['environment_digest'] or saved['preparation_fingerprint']!=state['preparation_fingerprint'] or
                read_json(parent/'data-provenance.json')!=provenance or
                summary['initialization']['sample_signature']!=initialization['sample_signature'] or
                not accounting['all_jobs_reconciled'] or accounting['budget_exceeded'] or journal['pending'] or
                journal['confirmed_tokens']<saved['confirmed_tokens_at_checkpoint'] or journal['charged_tokens']!=journal['confirmed_tokens']):
            raise ValueError('Parent lineage, accounting, data or frozen prefix mismatch')
        receipt=parent/'raw-prefix-receipt.json'
        if receipt.exists():atomic_json(run_path/'raw-prefix-receipt.json',read_json(receipt))
        atomic_json(run_path/'execution-overlay.json',dict(kind='user-authorized resplit phase2 LR fork',
            source_hashes=plan['overlay_hashes'],parent_current_journal_high_water=journal['charged_tokens'],
            parent_prior_processing_retained_in_shared_ledger=True,scale_relative_to_first_phase_peak=scale))
        return saved,scale

    saved,scale=on_main(accelerator,verify)
    accelerator.load_state(str(checkpoint))
    peaks=[cfg['learning_rate'],cfg['muon']['learning_rate']]
    if len(schedulers)!=len(peaks):raise ValueError('Unexpected optimizer layout')
    for scheduler,optimizer,peak in zip(schedulers,optimizers,peaks):
        scheduler.base_lrs=[peak]*len(optimizer.param_groups)
        scheduler.last_epoch=PREFIX_STEP
        scheduler._last_lr=[peak*schedule_multiplier(PREFIX_STEP,cfg)]*len(optimizer.param_groups)
        for group,value in zip(optimizer.param_groups,scheduler._last_lr):
            group['initial_lr']=peak;group['lr']=value
    child=copy.deepcopy(saved)
    for key in ('config_digest','wandb_id','code_digest','environment_digest','preparation_fingerprint'):child[key]=state[key]
    child['planned_steps']=cfg['max_steps']
    child['lineage']=dict(parent_run=PARENT,parent_checkpoint=str(checkpoint),inherited_tokens=PREFIX_STEP*BATCH_TOKENS,
        parent_wandb_id=saved['wandb_id'],scheduler_rebased=True,lr_scale=scale,lr_reference='first phase peak')
    on_main(accelerator,lambda:atomic_json(run_path/'fork-lineage.json',child['lineage']))
    return str(checkpoint),child
