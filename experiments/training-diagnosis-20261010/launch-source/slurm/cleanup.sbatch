#!/bin/bash
#SBATCH --job-name=dolma-cleanup
#SBATCH --account=ACD115198
#SBATCH --partition=8gpus
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=200G
#SBATCH --gpus-per-node=1
#SBATCH --time=02:00:00
#SBATCH --signal=B:USR1@900
#SBATCH --output=/work/asciibase64/Llama-3.2-1B-training/logs/%x-%j.out
#SBATCH --error=/work/asciibase64/Llama-3.2-1B-training/logs/%x-%j.err
set -euo pipefail
RETENTION_EXCLUSIVE=1
source /home/asciibase64/Llama-3.2-1B-training/slurm/common.sh
PREPARATION_ID=${1:?Supply verified preparation ID}
printf '%s\n' "$PREPARATION_ID" > "$JOB_RECORD/preparation-id.txt"
printf '%s\n' cleanup > "$JOB_RECORD/stage.txt"
srun --ntasks=1 python "$REPO/slurm/cleanup.py" --prep-id "$PREPARATION_ID" "${@:2}"
