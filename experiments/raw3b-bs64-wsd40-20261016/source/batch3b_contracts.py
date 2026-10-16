"""Pure token-clock contracts for the user-approved fresh formal12."""
STUDY = 'raw3b-bs64-wsd40-20261016'
PREP = 'dolma-raw-baseline-20261008'
Q = 131072
END = 22889
SPLIT = 19074
SWITCH = 11444
TARGET = END * Q
MUON_LR = .005593716406714465
ADAM_LR = .00016781149220143396
PLANNED_UPDATES = 14312


def evaluation_units():
    return tuple(k * END // 10 for k in range(1, 10))


def boundaries():
    # Include warmup ends and the final zero-LR BS16 update of each phase.
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


def beta2_for(cfg, width, tokens):
    return .95


def validate(cfg):
    tail = dict(first_decay_ratio=.4 * 2 / 3, hold_ratio=.4 / 6,
                final_decay_ratio=.4 / 6, plateau_lr_ratio=.2)
    phases = [dict(start=0, end=SPLIT, peak_ratio=1., warmup_steps=170, hold_schedule=tail),
              dict(start=SPLIT, end=END, peak_ratio=.1, warmup_steps=170, hold_schedule=tail)]
    expected = dict(sweep_id=STUDY, category='full', world_size=8, microbatch_size=1,
                    gradient_accumulation_steps=2, sequence_length=8192,
                    target_training_tokens=TARGET, max_training_tokens=TARGET,
                    max_steps=END, scheduler_steps=END, max_h200_hours=48,
                    batch_switch_tokens=SWITCH * Q, schedule_clock='nonpadding_tokens',
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


def check_fork(*_):
    raise ValueError('This single fresh run never loads a training checkpoint')
