#!/bin/bash
#SBATCH --job-name=hf-raw6b-upload
#SBATCH --account=ACD115198
#SBATCH --partition=8gpus
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=200G
#SBATCH --gpus-per-node=1
#SBATCH --constraint=H200
#SBATCH --time=00:30:00
#SBATCH --signal=B:USR1@180
#SBATCH --no-requeue
#SBATCH --exclude=25a-hgpn001,25a-hgpn062
#SBATCH --output=/work/asciibase64/Llama-3.2-1B-training/logs/%x-%j.out
#SBATCH --error=/work/asciibase64/Llama-3.2-1B-training/logs/%x-%j.err
set -euo pipefail
source "${TRAIN_SOURCE_ROOT:?}/slurm/training-common.sh"
export DATA_ROOT TRAIN_ENV_PREFIX JOB_RECORD
SECONDS_LEFT=$((TRAIN_END_EPOCH - $(date +%s) - 180))
[[ "$SECONDS_LEFT" -gt 0 ]] || exit 124
timeout --signal=TERM --kill-after=15s "$SECONDS_LEFT" \
    srun --ntasks=1 python -u "$REPO/slurm/hf_raw6b_upload.py" worker
