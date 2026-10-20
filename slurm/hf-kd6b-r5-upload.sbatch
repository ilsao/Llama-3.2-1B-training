#!/bin/bash
#SBATCH --job-name=hf-kd6b-r5-upload
#SBATCH --account=ACD115198
#SBATCH --partition=8gpus
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --gpus-per-node=1
#SBATCH --constraint=H200
#SBATCH --time=00:30:00
#SBATCH --no-requeue
#SBATCH --exclude=25a-hgpn001,25a-hgpn062
#SBATCH --output=/work/asciibase64/Llama-3.2-1B-training/logs/%x-%j.out
#SBATCH --error=/work/asciibase64/Llama-3.2-1B-training/logs/%x-%j.err
set -euo pipefail
cd /home/asciibase64/Llama-3.2-1B-training
source slurm/training-common.sh
export DATA_ROOT TRAIN_ENV_PREFIX JOB_RECORD
UPLOAD_SECONDS_LEFT=$((TRAIN_END_EPOCH - $(date +%s) - 180))
[[ "$UPLOAD_SECONDS_LEFT" -gt 0 ]] || exit 124
timeout --signal=TERM --kill-after=15s "$UPLOAD_SECONDS_LEFT" \
  srun --ntasks=1 python -u experiments/hf-kd6b-r5-20261020/publish.py worker
