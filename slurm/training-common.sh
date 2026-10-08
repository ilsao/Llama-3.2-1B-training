#!/bin/bash
# Sourced only inside Slurm. Keep the established data environment untouched.
set -euo pipefail
SKIP_DATA_ENV=1
source "${TRAIN_SOURCE_ROOT:?Submit from the intended training worktree}/slurm/common.sh"
export TRAIN_ENV_PREFIX=${TRAIN_ENV_PREFIX:-"$DATA_ROOT/environments/training-muon-torch291-fa3-94e22c9"}
RUN_ID=${1:?Supply a run ID through slurm/submit_training.py}
[[ "$RUN_ID" =~ ^[A-Za-z0-9][A-Za-z0-9_.-]{0,95}$ ]] || exit 1
printf '%s\n' "$RUN_ID" > "$JOB_RECORD/run-id.txt"
# Requeues must be separate registered submissions so allocation costs persist.
RESTARTS=$(sed -n 's/.* Restarts=\([^ ]*\).*/\1/p' "$JOB_RECORD/allocation.txt")
[[ ${RESTARTS:-0} == 0 ]] || { echo 'Automatic requeue is unsupported; reconcile and submit resume' >&2; exit 1; }
JOB_START=$(sed -n 's/.* StartTime=\([^ ]*\).*/\1/p' "$JOB_RECORD/allocation.txt")
export TRAIN_START_EPOCH=$(date -d "$JOB_START" +%s)
export TRAIN_END_EPOCH=$(date -d "$JOB_END" +%s)
mkdir -p "$DATA_ROOT"/{caches/pip,caches/training-verification,runs/experiments}
export WANDB_DIR="$DATA_ROOT/runs/wandb"
# All setup/validation/pilot allocations require a shared-ledger reservation.
python3 - "$REPO" "$DATA_ROOT" "$SLURM_JOB_ID" "$RUN_ID" <<'PY'
import sys, time
sys.path.insert(0, sys.argv[1])
from training_state import ExperimentLedger
for attempt in range(20):
    try:
        allocation = ExperimentLedger(sys.argv[2]).allocation(sys.argv[3], sys.argv[4])
        break
    except KeyError:
        time.sleep(0.5)  # sbatch can start before the wrapper binds its job ID.
else:
    raise SystemExit('Submit through slurm/submit_training.py to reserve compute budget')
import os
duration = int(os.environ['TRAIN_END_EPOCH']) - int(os.environ['TRAIN_START_EPOCH'])
if duration > allocation['requested_seconds']:
    raise SystemExit('Actual allocation exceeds reservation')
from training_state import atomic_json
from pathlib import Path
record = Path(os.environ['JOB_RECORD']) if 'JOB_RECORD' in os.environ else Path(sys.argv[2]) / 'logs' / ('job-' + sys.argv[3])
atomic_json(record / 'training-reservation.json', allocation)
(record / 'reserved-gpu-count.txt').write_text(str(allocation['gpu_count']) + '\n')
PY
# Record GPU model/count without loading a model; fail if allocation is not H200.
# common.sh already recorded the allocation hardware.
if [[ $(wc -l < "$JOB_RECORD/gpus.txt") -gt 8 ]] || ! awk 'index($0,"H200")==0 {bad=1} END {exit bad}' "$JOB_RECORD/gpus.txt"; then
    echo 'Require at most eight allocated H200s' >&2
    exit 1
fi
[[ $(wc -l < "$JOB_RECORD/gpus.txt") == $(cat "$JOB_RECORD/reserved-gpu-count.txt") ]] || { echo 'Visible GPU count differs from reservation' >&2; exit 1; }
# training code uses the actual end timestamp; no login-node timers are used.
if [[ ${SETUP_TRAINING_ENV:-0} != 1 ]]; then
    [[ -x "$TRAIN_ENV_PREFIX/bin/python" ]] || { echo 'Submit training setup first' >&2; exit 1; }
    source /work/envstack/apps/miniconda3/26.1.1/etc/profile.d/conda.sh
    conda activate "$TRAIN_ENV_PREFIX"
fi
