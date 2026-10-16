"""Actual-update decay clock,with batch windows matched to formal12."""
from functools import lru_cache
STUDY = 'raw3b-bs64-stepwsd40-20261016'
PREP = 'dolma-raw-baseline-20261008'
Q = 131072
END = 22889
SPLIT = 19074
SWITCH = 11444
TARGET = END * Q
MUON_LR = .005593716406714465
ADAM_LR = .00016781149220143396
PLANNED_UPDATES = 14312
STEP_SPLIT = 13355


def evaluation_units():
    return tuple(k * END // 10 for k in range(1, 10))


def boundaries():
    # Preserve formal12's complete batch trajectory,including small boundary
    # batches. These are data-window boundaries,NOT the new decay boundaries.
    return (0, 170, SWITCH, 16530, 17802, SPLIT - 1, SPLIT,
            SPLIT + 170, 21364, 22381, 22635, END - 1, END)


def events(cfg):
    return sorted(set(boundaries()) | set(evaluation_units()))


def window(cfg, tokens):
    if type(tokens) is not int or tokens % Q or not 0 <= tokens <= TARGET:
        raise ValueError('Require a complete BS16-equivalent token cursor')
    if tokens == TARGET:
        return 0, 0
    unit = tokens // Q
    nominal = 16 if unit < SWITCH else 64
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


@lru_cache(maxsize=1)
def token_prefix():
    tokens = [0]
    while tokens[-1] < TARGET:
        width, _ = window(None, tokens[-1])
        tokens.append(tokens[-1] + width * 8192)
    return tuple(tokens)


def multiplier_for_step(completed_updates, cfg):
    """LR for the next update;both warmup and decay use actual steps."""
    from horizon_contracts import phase_multiplier
    if type(completed_updates) is not int or not 0 <= completed_updates <= PLANNED_UPDATES:
        raise ValueError('Invalid actual optimizer-update cursor')
    index = min(completed_updates, PLANNED_UPDATES - 1)
    return phase_multiplier(index, cfg)


def beta2_for(cfg, width, tokens):
    return .95


def validate(cfg):
    tail = dict(first_decay_ratio=.4 * 2 / 3, hold_ratio=.4 / 6,
                final_decay_ratio=.4 / 6, plateau_lr_ratio=.2)
    phases = [dict(start=0, end=STEP_SPLIT, peak_ratio=1., warmup_steps=170, hold_schedule=tail),
              dict(start=STEP_SPLIT, end=PLANNED_UPDATES, peak_ratio=.1, warmup_steps=170, hold_schedule=tail)]
    expected = dict(sweep_id=STUDY, category='full', world_size=8, microbatch_size=1,
                    gradient_accumulation_steps=2, sequence_length=8192,
                    target_training_tokens=TARGET, max_training_tokens=TARGET,
                    max_steps=PLANNED_UPDATES, scheduler_steps=PLANNED_UPDATES, max_h200_hours=48,
                    batch_switch_tokens=SWITCH * Q, schedule_clock='optimizer_updates',
                    warmup_clock='optimizer_updates',
                    reference_batch_tokens=Q, adamw_beta2_policy='fixed', betas=[.9, .95],
                    learning_rate=ADAM_LR, warmup_steps=170, wsd_decay_ratio=.4,
                    hold_schedule=tail, phase_schedule=phases, snapshot_tokens=[],
                    eval_documents=1000, eval_steps=1431, initialization='residual_half',
                    seed=42, ema=[], clipping='global', max_grad_norm=.5,
                    stop_margin_seconds=300, final_evaluation_reserve_seconds=900,
                    checkpoint_policy='final_only', fixed_horizon=True,
                    export_on_completion=True, wandb_mode='online')
    if any(cfg.get(key) != value for key, value in expected.items()):
        raise ValueError('Configuration differs from the approved fresh3B recipe')
    if cfg.get('matrix_optimizer') or cfg.get('fork_snapshot_tokens'):
        raise ValueError('Native Muon and fresh random weights only')
    if cfg['muon']['learning_rate'] != MUON_LR or count_updates(cfg) != PLANNED_UPDATES:
        raise ValueError('Incorrect table LR or dynamic update count')
    if token_prefix()[STEP_SPLIT] != SPLIT * Q:
        raise ValueError('Actual-step phase split differs from matched data cursor')


def check_fork(*_):
    raise ValueError('This single fresh run never loads a training checkpoint')
