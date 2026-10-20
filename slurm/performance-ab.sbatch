#!/bin/bash
#SBATCH --job-name=performance-ab
#SBATCH --account=ACD115198
#SBATCH --partition=8gpus
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=200G
#SBATCH --gpus-per-node=8
#SBATCH --time=00:45:00
#SBATCH --signal=B:USR1@180
#SBATCH --no-requeue
#SBATCH --exclude=25a-hgpn001,25a-hgpn062
#SBATCH --output=/work/asciibase64/Llama-3.2-1B-training/logs/%x-%j.out
#SBATCH --error=/work/asciibase64/Llama-3.2-1B-training/logs/%x-%j.err
set -euo pipefail
source "${TRAIN_SOURCE_ROOT:?}/slurm/training-common.sh"
export DATA_ROOT TRAIN_ENV_PREFIX JOB_RECORD
export PERFORMANCE_RUN_ID="$RUN_ID"
STAGE=${2:?runtime/A/B}
CONFIGURATION=${3:?}
launch() {
    srun --ntasks=1 --kill-on-bad-exit=1 accelerate launch --multi_gpu \
        --num_processes 8 --num_machines 1 --mixed_precision bf16 \
        --dynamo_backend no --num_cpu_threads_per_process 1 \
        "$PERFORMANCE_LAUNCHER" "$@"
}
if [[ "$STAGE" == runtime ]]; then
    launch "$REPO/tests/training_runtime.py" --run-id "$RUN_ID" --optimizer muon --scheduler wsd
    if launch "$REPO/tests/performance_runtime.py" --config "$CONFIGURATION" --run-id "$RUN_ID" \
        2>&1 | tee "$JOB_RECORD/compile-cg.log"; then
        :
    else
        # A CUDA-graph-specific compatibility failure may select the predefined
        # no-graph configuration inside this same bounded validation allocation.
        python - "$JOB_RECORD/compile-cg.log" <<'PY'
import sys
from pathlib import Path
text = Path(sys.argv[1]).read_text(errors='replace')
markers = ('overwritten by a subsequent run', 'operation not permitted when stream is capturing',
           'RuntimeError: CUDA graph', 'RuntimeError: CUDAGraph', 'cudagraphs does not support')
if not any(marker in text for marker in markers):
    raise SystemExit('Correctness failed outside recognized CUDA graph compatibility; stop.')
PY
        launch "$REPO/tests/performance_runtime.py" \
            --config "${CONFIGURATION%.json}-nocg.json" --run-id "$RUN_ID"
    fi
else
    [[ "$STAGE" == A || "$STAGE" == B ]] || exit 1
    launch "$REPO/slurm/training_entry.py" --config "$CONFIGURATION" \
        --prep-id dolma-raw-baseline-20261008 --run-id "$RUN_ID" --validate-export
fi
