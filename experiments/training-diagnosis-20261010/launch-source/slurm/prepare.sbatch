#!/bin/bash
#SBATCH --job-name=dolma-prepare
#SBATCH --account=ACD115198
#SBATCH --partition=8gpus
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=200G
#SBATCH --gpus-per-node=1
#SBATCH --time=16:00:00
#SBATCH --signal=B:USR1@900
#SBATCH --output=/work/asciibase64/Llama-3.2-1B-training/logs/%x-%j.out
#SBATCH --error=/work/asciibase64/Llama-3.2-1B-training/logs/%x-%j.err
set -euo pipefail
STAGE=${1:?Supply download, split, clean, select, pack, verify, smoke, or parallel-check}
PREPARATION_ID=${2:?Supply a unique preparation ID; reuse it only for identical continuations}
export FILTERING_TRAIN_WORKERS=7
export FILTERING_EVAL_WORKERS=7
arguments=("${@:3}")
for ((i=0; i<${#arguments[@]}; i++)); do
    if [[ ${arguments[i]} == --training-workers ]]; then
        [[ "$STAGE" == select && "$PREPARATION_ID" == dolma-dclm10-005 ]] || exit 1
        export FILTERING_TRAIN_WORKERS=${arguments[i+1]:?Supply training worker count}
        [[ "$FILTERING_TRAIN_WORKERS" == 7 || "$FILTERING_TRAIN_WORKERS" == 11 || "$FILTERING_TRAIN_WORKERS" == 15 ]] || exit 1
    fi
    if [[ ${arguments[i]} == --evaluation-workers ]]; then
        [[ "$STAGE" == select && "$PREPARATION_ID" == dolma-dclm10-005 ]] || exit 1
        export FILTERING_EVAL_WORKERS=${arguments[i+1]:?Supply evaluation worker count}
        [[ "$FILTERING_EVAL_WORKERS" == 7 || "$FILTERING_EVAL_WORKERS" == 11 ]] || exit 1
    fi
done
source /home/asciibase64/Llama-3.2-1B-training/slurm/common.sh
case "$STAGE" in download|split|clean|select|pack|verify|smoke|parallel-check|cache-audit|cache-repair) ;; *) echo 'Use score.sbatch for score' >&2; exit 1;; esac
printf '%s\n' "$PREPARATION_ID" > "$JOB_RECORD/preparation-id.txt"
printf '%s\n' "$STAGE" > "$JOB_RECORD/stage.txt"
if [[ "$STAGE" == cache-repair ]]; then
    timeout --signal=TERM --kill-after=30s 600s srun --ntasks=1 \
        python "$REPO/slurm/cache_repair.py" --run-id "$PREPARATION_ID" --root "$DATA_ROOT" "${@:3}"
    exit
fi
if [[ "$STAGE" == cache-audit ]]; then
    timeout --signal=TERM --kill-after=30s 600s srun --ntasks=1 \
        python "$REPO/slurm/cache_audit.py" --run-id "$PREPARATION_ID" --root "$DATA_ROOT" "${@:3}"
    exit
fi
if [[ "$STAGE" == parallel-check ]]; then
    timeout --signal=TERM --kill-after=30s 600s srun --ntasks=1 \
        python "$REPO/tests/preparation_runtime.py" --run-id "$PREPARATION_ID" --root "$DATA_ROOT"
    exit
fi
RUNTIME_LIMIT=()
for argument in "${@:3}"; do
    if [[ "$STAGE" == download && "$argument" == --repair-download-from ]]; then
        RUNTIME_LIMIT=(timeout --signal=TERM --kill-after=30s 600s)
    fi
done
ENTRY="$REPO/dataset.py"
if [[ "$STAGE" == select || "$STAGE" == verify ]]; then ENTRY="$REPO/slurm/filtering_entry.py"; fi
if [[ "$PREPARATION_ID" == dolma-dclm10-005 && ( "$STAGE" == pack || "$STAGE" == verify ) ]]; then
    ENTRY="$REPO/slurm/packing_entry.py"
fi
"${RUNTIME_LIMIT[@]}" srun --ntasks=1 python "$ENTRY" "$STAGE" --prep-id "$PREPARATION_ID" --root "$DATA_ROOT" \
    --config "${DATA_CONFIG:-$REPO/configs/dataset.json}" "${@:3}"
