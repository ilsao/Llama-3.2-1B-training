#!/bin/bash
#SBATCH --job-name=accumulation-ab
#SBATCH --account=ACD115198
#SBATCH --partition=8gpus
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=200G
#SBATCH --gpus-per-node=8
#SBATCH --time=00:15:00
#SBATCH --signal=B:USR1@180
#SBATCH --no-requeue
#SBATCH --exclude=25a-hgpn001,25a-hgpn062
#SBATCH --output=/work/asciibase64/Llama-3.2-1B-training/logs/%x-%j.out
#SBATCH --error=/work/asciibase64/Llama-3.2-1B-training/logs/%x-%j.err
set -euo pipefail
source "${TRAIN_SOURCE_ROOT:?}/slurm/training-common.sh"
export DATA_ROOT TRAIN_ENV_PREFIX JOB_RECORD
export PERFORMANCE_RUN_ID="$RUN_ID"
CONFIGURATION=${2:?configuration}
export ACCUMULATION_CONFIGURATION="$CONFIGURATION"
python3 "${ACCUMULATION_CONTROLLER:?}" check
# afterok guarantees terminal success; require complete matched-token training too.
python3 - "$ACCUMULATION_PROJECT" "$DATA_ROOT" "$ACCUMULATION_PARENT_RUN" "$ACCUMULATION_PARENT_JOB" <<'PREFLIGHT'
import sys
from pathlib import Path
sys.path.insert(0, sys.argv[1])
from training_state import ExperimentLedger, read_json
from slurm.accumulation_ab import measurements, read_rows
run = Path(sys.argv[2]) / 'runs/experiments' / sys.argv[3]
measurements(read_json(run / 'summary.json'), read_rows(run / 'timing-steps.jsonl'))
accounting = ExperimentLedger(sys.argv[2]).reconcile(sys.argv[4])
if accounting['state'] != 'COMPLETED':
    raise SystemExit('Parent allocation did not complete successfully')
print('Parent completed 120 matched-token updates; actual allocation reconciled')
PREFLIGHT
srun --ntasks=1 --kill-on-bad-exit=1 accelerate launch --multi_gpu \
    --num_processes 8 --num_machines 1 --mixed_precision bf16 \
    --dynamo_backend no --num_cpu_threads_per_process 1 \
    "${PERFORMANCE_LAUNCHER:?}" "$REPO/slurm/training_entry.py" \
    --config "$CONFIGURATION" --prep-id dolma-raw-baseline-20261008 --run-id "$RUN_ID"
