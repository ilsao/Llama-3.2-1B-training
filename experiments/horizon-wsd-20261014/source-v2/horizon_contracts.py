"""Pure contracts for the explicitly requested 0.7B/2B campaign."""
import math

CAMPAIGN = 'horizon-wsd-20261014'
BATCH_TOKENS = 131072
SWEEP_STEPS = 5341
FINAL_STEPS = 15259
PREFIX_STEP = SWEEP_STEPS - round(.6 * SWEEP_STEPS)
FINAL_SPLIT = FINAL_STEPS * 5 // 6


def scaled_lr(lr, source_tokens, target_tokens, ratio=1.0):
    if min(lr, source_tokens, target_tokens, ratio) <= 0:
        raise ValueError('LR and horizons must be positive')
    return lr * (target_tokens / source_tokens) ** (-.32) * ratio


def tail_settings(tail):
    # Preserve BE60P20's first-ramp : hold : last-ramp = 4:1:1.
    return dict(first_decay_ratio=tail * 2 / 3, hold_ratio=tail / 6,
                final_decay_ratio=tail / 6, plateau_lr_ratio=.2)


def validate_phases(cfg):
    from training_state import hold_schedule_boundaries
    phases = cfg['phase_schedule']
    if len(phases) != 2 or cfg['max_steps'] != FINAL_STEPS:
        raise ValueError('Two final stages required')
    cursor = 0
    for phase in phases:
        if set(phase) != {'start', 'end', 'peak_ratio', 'warmup_steps', 'hold_schedule'}:
            raise ValueError('Incomplete phase')
        if phase['start'] != cursor or phase['end'] <= cursor:
            raise ValueError('Phases must be contiguous')
        if phase['warmup_steps'] != 170 or not math.isfinite(phase['peak_ratio']) or phase['peak_ratio'] <= 0:
            raise ValueError('Invalid phase peak/warmup')
        hold_schedule_boundaries(phase['end'] - cursor, phase['warmup_steps'], phase['hold_schedule'])
        cursor = phase['end']
    if phases[0]['end'] != FINAL_SPLIT or cursor != cfg['max_steps']:
        raise ValueError('Incorrect 5/6 boundary or final horizon')


def phase_multiplier(index, cfg):
    from training_state import lr_multiplier
    for phase in cfg['phase_schedule']:
        if index < phase['end'] or phase is cfg['phase_schedule'][-1]:
            return phase['peak_ratio'] * lr_multiplier(
                index - phase['start'], phase['end'] - phase['start'],
                phase['warmup_steps'], 0, 'wsd_hold_zero',
                sum(phase['hold_schedule'][k] for k in ('first_decay_ratio', 'hold_ratio', 'final_decay_ratio')),
                phase['hold_schedule'])
    raise ValueError('Missing phase')


def campaign_allocation_limits(run_id, cfg):
    # Only the two named final runs. No global relaxation of historical gates.
    names = {CAMPAIGN + '-final-standard', CAMPAIGN + '-final-resplit'}
    if run_id not in names:
        return None
    if (cfg.get('sweep_id') != CAMPAIGN or cfg.get('category') != 'full' or
            cfg.get('world_size') != 8 or cfg.get('max_steps') != FINAL_STEPS or
            cfg.get('max_training_tokens') != FINAL_STEPS * BATCH_TOKENS or
            cfg.get('max_h200_hours') != 32 or cfg.get('sequence_length') != 8192):
        raise ValueError('Final exception must match this campaign')
    return 10800, 24


def check_fork_configs(parent, child):
    allowed = {'parent_runs', 'sweep_purpose', 'max_h200_hours', 'hold_schedule', 'wsd_decay_ratio'}
    if {k:v for k,v in parent.items() if k not in allowed} != {k:v for k,v in child.items() if k not in allowed}:
        raise ValueError('Decay fork may change only cooldown timing and lineage/budget metadata')
    from training_state import hold_schedule_boundaries, schedule_multiplier
    for cfg in (parent, child):
        if cfg['max_steps'] != SWEEP_STEPS or cfg['category'] != 'pilot':
            raise ValueError('Fork restricted to sweep horizon')
        start = hold_schedule_boundaries(cfg['max_steps'], cfg['warmup_steps'], cfg['hold_schedule'])[0]
        if start < PREFIX_STEP:
            raise ValueError('Fork would begin after decay')
    if any(schedule_multiplier(i,parent) != schedule_multiplier(i,child) for i in range(PREFIX_STEP)):
        raise ValueError('Shared prefix LR histories differ')
