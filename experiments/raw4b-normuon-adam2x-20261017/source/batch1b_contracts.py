"""Dependency-free, token-clock contracts for the approved nine-endpoint study."""
import copy
import math

STUDY = 'batch-normuon-1b-20261016'
Q = 131072
END = 7630
TARGET = END * Q
SPLIT = 6358
SWITCHES = {'50': 3815, 'predecay': 5086, '70': 5341}
MUON_LR = 0.008614891542046706
ADAM_LR = 0.00025844674626140116
PREP = 'dolma-raw-baseline-20261008'


def boundaries():
    return (0, 170, 5086, 5934, 6146, 6357, 6358,
            6528, 7376, 7546, 7588, 7629, END)


def events(cfg):
    extra = (cfg.get('snapshot_tokens', []) +
             [cfg.get('batch_switch_tokens', TARGET)])
    return sorted(set(boundaries()) | set(range(763, END, 763)) |
                  {int(t // Q) for t in extra} | {END})


def window(cfg, tokens):
    if tokens % Q or not 0 <= tokens <= TARGET:
        raise ValueError('Token clock must advance in complete BS16 units')
    if tokens == TARGET:
        return 0, 0
    unit = tokens // Q
    nominal = 64 if tokens >= cfg.get('batch_switch_tokens', TARGET) else 16
    next_event = next(x for x in events(cfg) if x > unit)
    units = min(nominal // 16, next_event - unit)
    return units * 16, units * 2


def count_updates(cfg, start_tokens=0):
    tokens, updates = start_tokens, 0
    while tokens < TARGET:
        width, _ = window(cfg, tokens)
        tokens += width * 8192
        updates += 1
    return updates


def beta2_for(cfg, width, tokens):
    if cfg.get('adamw_beta2_policy', 'fixed') == 'token_scaled' and tokens >= cfg['batch_switch_tokens']:
        return cfg['betas'][1] ** (width / 16)
    return cfg['betas'][1]


def build_configs(parent):
    base = copy.deepcopy(parent)
    for key in ('sweep_environment_digest', 'sweep_calibration_digest', 'phase_schedule'):
        base.pop(key, None)
    tail = dict(first_decay_ratio=.2 * 2 / 3, hold_ratio=.2 / 6,
                final_decay_ratio=.2 / 6, plateau_lr_ratio=.2)
    base.update(category='pilot', sweep_id=STUDY, sweep_stage='matched-1b',
                sweep_purpose='1B BS16-to64 timing, NorMuon+PE, LR and AdamW beta2 comparison',
                parent_runs=[], learning_rate=ADAM_LR, max_steps=END,
                scheduler_steps=END, target_training_tokens=TARGET,
                max_training_tokens=TARGET, max_h200_hours=16,
                eval_steps=763, eval_documents=1000, logging_steps=10,
                stop_margin_seconds=300, final_evaluation_reserve_seconds=120,
                checkpoint_policy='final_only', retain_all_checkpoints=True,
                export_on_completion=True, wsd_decay_ratio=.2,
                hold_schedule=tail, schedule_clock='nonpadding_tokens',
                reference_batch_tokens=Q, adamw_beta2_policy='fixed',
                snapshot_tokens=[], batch_switch_tokens=TARGET)
    base['muon']['learning_rate'] = MUON_LR
    base['phase_schedule'] = [
        dict(start=0, end=SPLIT, peak_ratio=1., warmup_steps=170, hold_schedule=copy.deepcopy(tail)),
        dict(start=SPLIT, end=END, peak_ratio=.1, warmup_steps=170, hold_schedule=copy.deepcopy(tail))]
    configs = {}
    for name in ('control', 'normuon', 'lr001'):
        cfg = copy.deepcopy(base)
        if name == 'control':
            cfg['snapshot_tokens'] = sorted(v * Q for v in SWITCHES.values())
        elif name == 'normuon':
            cfg['matrix_optimizer'] = dict(algorithm='normuon', muon_beta2=.95,
                orthogonalization_strategy='polar_express', use_cautious_wd=False,
                package='muonium', version='0.2.0')
        else:
            cfg['learning_rate'] = .0003
            cfg['muon']['learning_rate'] = .01
        configs[name] = cfg
    for name, unit in SWITCHES.items():
        for policy in ('fixed', 'token_scaled'):
            cfg = copy.deepcopy(base)
            cfg.update(batch_switch_tokens=unit * Q, adamw_beta2_policy=policy,
                       fork_snapshot_tokens=unit * Q, max_h200_hours=32,
                       parent_runs=[STUDY + '-control'])
            cfg['eval_steps'] = max(1, count_updates(cfg, unit * Q) // 10)
            configs['bs-' + name + ('-scaled' if policy == 'token_scaled' else '-fixed')] = cfg
    return configs


def validate(cfg):
    if cfg.get('sweep_id') != STUDY:
        raise ValueError('Wrong study identity')
    if (cfg['world_size'], cfg['microbatch_size'], cfg['gradient_accumulation_steps'],
        cfg['sequence_length']) != (8, 1, 2, 8192):
        raise ValueError('Keep eight-rank BS16 geometry; switch through accumulation only')
    if cfg['schedule_clock'] != 'nonpadding_tokens' or cfg['reference_batch_tokens'] != Q:
        raise ValueError('Require common token schedule')
    if cfg['target_training_tokens'] != TARGET or cfg['max_training_tokens'] != TARGET:
        raise ValueError('All nine endpoints must match')
    if cfg['batch_switch_tokens'] not in {TARGET, *(v * Q for v in SWITCHES.values())}:
        raise ValueError('Unapproved switch')
    if cfg['adamw_beta2_policy'] not in ('fixed', 'token_scaled'):
        raise ValueError('Unapproved beta2 policy')
    if cfg['snapshot_tokens'] and cfg['snapshot_tokens'] != sorted(v * Q for v in SWITCHES.values()):
        raise ValueError('Wrong snapshot milestones')
    if cfg.get('matrix_optimizer') and cfg['matrix_optimizer'] != dict(
        algorithm='normuon', muon_beta2=.95, orthogonalization_strategy='polar_express',
        use_cautious_wd=False, package='muonium', version='0.2.0'):
        raise ValueError('NorMuon must match the pinned approved bundle')


def check_fork(parent, child, tokens):
    validate(parent); validate(child)
    allowed = {'batch_switch_tokens', 'fork_snapshot_tokens', 'adamw_beta2_policy',
               'parent_runs', 'max_h200_hours', 'eval_steps', 'snapshot_tokens'}
    if {k:v for k,v in parent.items() if k not in allowed} != {k:v for k,v in child.items() if k not in allowed}:
        raise ValueError('Fork may change only batch/beta2 policy and lineage metadata')
    if tokens != child['fork_snapshot_tokens'] or tokens != child['batch_switch_tokens']:
        raise ValueError('Fork must begin at its exact batch switch')
    if tokens not in parent['snapshot_tokens']:
        raise ValueError('Snapshot was not planned on control')
