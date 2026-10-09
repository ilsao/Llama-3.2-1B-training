#!/bin/bash
#SBATCH --job-name=raw-buffer-once
#SBATCH --account=ACD115198
#SBATCH --partition=8gpus
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=200G
#SBATCH --gpus-per-node=1
#SBATCH --time=01:30:00
#SBATCH --signal=B:USR1@180
#SBATCH --no-requeue
#SBATCH --output=/work/asciibase64/Llama-3.2-1B-training/logs/%x-%j.out
#SBATCH --error=/work/asciibase64/Llama-3.2-1B-training/logs/%x-%j.err
set -euo pipefail
source /home/asciibase64/Llama-3.2-1B-training/slurm/common.sh
export DATA_ROOT JOB_RECORD
printf '%s\n' dolma-raw-baseline-20261008 > "$JOB_RECORD/preparation-id.txt"
printf '%s\n' raw-unfiltered-buffer > "$JOB_RECORD/stage.txt"
srun --ntasks=1 python "$REPO/slurm/raw_continue_once.py" "${1:?Supply authorization receipt}"
