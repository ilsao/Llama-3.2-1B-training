"""Observe both accumulation geometries without changing their training code."""
from pathlib import Path
import hashlib
import json
import os
import runpy
import sys
import time

expected = os.environ['PERFORMANCE_LAUNCHER_SHA256']
if hashlib.sha256(Path(__file__).read_bytes()).hexdigest() != expected:
    raise RuntimeError('Measurement launcher changed after submission')
cfg = json.loads(Path(os.environ['ACCUMULATION_CONFIGURATION']).read_text())
accumulation = cfg['gradient_accumulation_steps']
rank = os.environ.get('LOCAL_RANK', '0')
cache = (Path(os.environ['DATA_ROOT']) / 'caches/compile' /
         os.environ['PERFORMANCE_RUN_ID'] / ('rank-' + rank))
os.environ['TORCHINDUCTOR_CACHE_DIR'] = str(cache / 'inductor')
os.environ['TRITON_CACHE_DIR'] = str(cache / 'triton')
os.environ['TORCHINDUCTOR_COMPILE_THREADS'] = '1'
os.environ['TORCH_LOGS'] = 'graph_breaks,recompiles,perf_hints'
target = sys.argv.pop(1)
sys.path.insert(0, os.environ['TRAIN_SOURCE_ROOT'])
import train
import torch
from torch._dynamo.utils import counters, compilation_time_metrics
from training_state import atomic_json

original_loss = train.causal_loss_sum
micro_calls = 0
observations = []


def observe_loss(model, batch):
    global micro_calls
    result = original_loss(model, batch)
    if model.training:
        micro_calls += 1
        if micro_calls % accumulation == 0:
            observations.append(dict(
                step=micro_calls // accumulation,
                microbatch_size=int(batch['input_ids'].shape[0]),
                unique_graphs=counters['stats']['unique_graphs'],
                peak_allocated_bytes=torch.cuda.max_memory_allocated(),
                peak_reserved_bytes=torch.cuda.max_memory_reserved(),
                recorded_monotonic=time.monotonic()))
    return result


train.causal_loss_sum = observe_loss
try:
    runpy.run_path(target, run_name='__main__')
finally:
    if torch.cuda.is_initialized():
        atomic_json(Path(os.environ['JOB_RECORD']) /
                    ('accumulation-observer-rank-' + rank + '.json'), dict(
            gradient_accumulation_steps=accumulation,
            peak_allocated_bytes=torch.cuda.max_memory_allocated(),
            peak_reserved_bytes=torch.cuda.max_memory_reserved(),
            compiler_time_seconds_by_category={
                k: sum(v) for k, v in compilation_time_metrics.items()},
            training_graph_observations=observations))
