"""Per-rank compiler caches, set before importing torch; no login execution."""
from pathlib import Path
import os
import runpy
import sys
import time
import hashlib

expected_launcher = os.environ.get('PERFORMANCE_LAUNCHER_SHA256')
if expected_launcher and hashlib.sha256(Path(__file__).read_bytes()).hexdigest() != expected_launcher:
    raise RuntimeError('Pinned measurement launcher changed after submission')

root = Path(os.environ['DATA_ROOT'])
rank = os.environ.get('LOCAL_RANK', '0')
cache = root / 'caches/compile' / os.environ['PERFORMANCE_RUN_ID'] / ('rank-' + rank)
os.environ['TORCHINDUCTOR_CACHE_DIR'] = str(cache / 'inductor')
os.environ['TRITON_CACHE_DIR'] = str(cache / 'triton')
os.environ['TORCHINDUCTOR_COMPILE_THREADS'] = '1'
os.environ['TORCH_LOGS'] = 'graph_breaks,recompiles,perf_hints'
target = sys.argv.pop(1)
sys.path.insert(0, str(Path(target).resolve().parents[1]))
sys.path.insert(0, str(Path(os.environ['TRAIN_SOURCE_ROOT'])))
observations = []
if os.environ.get('PERFORMANCE_STAGE') in ('A', 'B'):
    import train
    original_loss = train.causal_loss_sum
    micro_calls = 0
    def observe_loss(model, batch):
        global micro_calls
        result = original_loss(model, batch)
        if getattr(model, 'training', False):
            micro_calls += 1
            if micro_calls % 2 == 0:
                from torch._dynamo.utils import counters
                observations.append(dict(step=micro_calls//2,
                    unique_graphs=counters['stats']['unique_graphs'],
                    recorded_monotonic=time.monotonic()))
        return result
    train.causal_loss_sum = observe_loss
try:
    runpy.run_path(target, run_name='__main__')
finally:
    if 'torch' in sys.modules:
        import torch
        from torch._dynamo.utils import compilation_time_metrics
        from training_state import atomic_json
        if torch.cuda.is_initialized():
            atomic_json(Path(os.environ['JOB_RECORD'])/('performance-observer-rank-'+rank+'.json'),dict(
                peak_allocated_bytes=torch.cuda.max_memory_allocated(),
                peak_reserved_bytes=torch.cuda.max_memory_reserved(),
                compiler_time_seconds_by_category={k:sum(v) for k,v in compilation_time_metrics.items()},
                training_graph_observations=observations))
