#!/bin/bash
#SBATCH --job-name=hf-be60p20-upload
#SBATCH --account=ACD115198
#SBATCH --partition=8gpus
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=200G
#SBATCH --gpus-per-node=1
#SBATCH --time=00:30:00
#SBATCH --signal=B:USR1@180
#SBATCH --no-requeue
#SBATCH --exclude=25a-hgpn001,25a-hgpn062
#SBATCH --output=/work/asciibase64/Llama-3.2-1B-training/logs/%x-%j.out
#SBATCH --error=/work/asciibase64/Llama-3.2-1B-training/logs/%x-%j.err
set -euo pipefail
export TRAIN_ENV_PREFIX=/work/asciibase64/Llama-3.2-1B-training/environments/training-muon-torch291-fa3-94e22c9
source "${TRAIN_SOURCE_ROOT:?Submit through the publication wrapper}/slurm/training-common.sh"
export DATA_ROOT TRAIN_ENV_PREFIX JOB_RECORD
[[ "$RUN_ID" == hf-be60p20-2b-20261010 ]] || exit 1
SECONDS_LEFT=$((TRAIN_END_EPOCH - $(date +%s) - 180))
[[ "$SECONDS_LEFT" -gt 0 ]] || exit 124
timeout --signal=TERM --kill-after=15s "$SECONDS_LEFT" \
    srun --ntasks=1 python -u "$REPO/slurm/hf_be60p20_upload.py" worker
