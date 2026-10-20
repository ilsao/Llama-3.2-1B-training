"""Per-rank compiler caches, set before importing torch; no login execution."""
from pathlib import Path
import os
import runpy
import sys

root = Path(os.environ['DATA_ROOT'])
rank = os.environ.get('LOCAL_RANK', '0')
cache = root / 'caches/compile' / os.environ['PERFORMANCE_RUN_ID'] / ('rank-' + rank)
os.environ['TORCHINDUCTOR_CACHE_DIR'] = str(cache / 'inductor')
os.environ['TRITON_CACHE_DIR'] = str(cache / 'triton')
os.environ['TORCHINDUCTOR_COMPILE_THREADS'] = '1'
os.environ['TORCH_LOGS'] = 'graph_breaks,recompiles'
target = sys.argv.pop(1)
sys.path.insert(0, str(Path(target).resolve().parents[1]))
sys.path.insert(0, str(Path(os.environ['TRAIN_SOURCE_ROOT'])))
runpy.run_path(target, run_name='__main__')
