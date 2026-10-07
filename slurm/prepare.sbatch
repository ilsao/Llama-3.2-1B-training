#!/bin/bash
#SBATCH --job-name=dolma-prepare
#SBATCH --account=ACD115198
#SBATCH --partition=8gpus
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=12
#SBATCH --mem=200G
#SBATCH --gpus-per-node=1
#SBATCH --time=24:00:00
#SBATCH --signal=B:USR1@900
#SBATCH --output=/work/asciibase64/Llama-3.2-1B-training/logs/%x-%j.out
#SBATCH --error=/work/asciibase64/Llama-3.2-1B-training/logs/%x-%j.err
set -euo pipefail
source /home/asciibase64/Llama-3.2-1B-training/slurm/common.sh
STAGE=${1:?Supply download, split, clean, select, pack, verify, or smoke}
PREPARATION_ID=${2:?Supply a unique preparation ID; reuse it only for identical continuations}
case "$STAGE" in download|split|clean|select|pack|verify|smoke) ;; *) echo 'Use score.sbatch for score' >&2; exit 1;; esac
printf '%s\n' "$PREPARATION_ID" > "$JOB_RECORD/preparation-id.txt"
printf '%s\n' "$STAGE" > "$JOB_RECORD/stage.txt"
srun --ntasks=1 python "$REPO/dataset.py" "$STAGE" --prep-id "$PREPARATION_ID" --root "$DATA_ROOT" \
    --config "${DATA_CONFIG:-$REPO/configs/dataset.json}"
