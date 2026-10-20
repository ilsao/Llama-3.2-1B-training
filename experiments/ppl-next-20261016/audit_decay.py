"""Reconstruct decay-only factors from the sealed, dependency-free contracts.

No tensor imports or model work. The product is NOT a model norm prediction.
"""
import hashlib
import json
import math
from pathlib import Path
import sys

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
SOURCE = REPO / 'experiments/raw3b-bs64-wsd40-20261016/source'
sys.path.insert(0, str(SOURCE))
from training_state import schedule_multiplier
from batch3b_contracts import window, Q, TARGET

cfg_path = REPO / 'experiments/raw3b-bs64-wsd40-20261016/config.json'
cfg = json.loads(cfg_path.read_text())
tokens, updates, log_decay = 0, 0, 0.0
lr_by_step = {}
while tokens < TARGET:
    lr = cfg['muon']['learning_rate'] * schedule_multiplier(tokens // Q, cfg)
    log_decay += math.log1p(-lr * cfg['muon']['weight_decay'])
    width, _ = window(cfg, tokens)
    tokens += width * 8192
    updates += 1
    lr_by_step[updates] = lr
assert updates == 14312 and tokens == 3000107008
metrics = Path('/work/asciibase64/Llama-3.2-1B-training/runs/experiments') / cfg['sweep_id'] / 'metrics.jsonl'
checked = 0
with metrics.open() as stream:
    for line in stream:
        record = json.loads(line)
        if 'train/lr_muon' in record:
            assert math.isclose(record['train/lr_muon'], lr_by_step[record['train/global_step']], rel_tol=1e-6, abs_tol=1e-10)
            checked += 1
virtual = sum(math.log1p(-cfg['muon']['weight_decay'] * cfg['muon']['learning_rate'] *
                        schedule_multiplier(i, cfg)) for i in range(cfg['scheduler_steps']))
sources = [cfg_path, SOURCE / 'training_state.py', SOURCE / 'horizon_contracts.py', SOURCE / 'batch3b_contracts.py']
result = dict(actual_updates=updates, input_tokens=tokens, logged_lr_checks=checked,
              actual_decay_only_multiplier=math.exp(log_decay),
              same_token_schedule_bs16_decay_only_multiplier=math.exp(virtual),
              cumulative_log_decay_ratio=log_decay/virtual,
              interpretation='Scheduled decoupled decay component only, not the actual model norm or a quality measurement',
              sources={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in sources})
(HERE / 'decay-audit.json').write_text(json.dumps(result, indent=2) + '\n')
print(json.dumps(result, indent=2))
