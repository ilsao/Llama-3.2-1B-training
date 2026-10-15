#!/bin/bash
#SBATCH --job-name=batch1b-setup
#SBATCH --account=ACD115198
#SBATCH --partition=8gpus
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --gpus-per-node=1
#SBATCH --constraint=H200
#SBATCH --time=00:30:00
#SBATCH --no-requeue
#SBATCH --output=/work/asciibase64/Llama-3.2-1B-training/logs/%x-%j.out
#SBATCH --error=/work/asciibase64/Llama-3.2-1B-training/logs/%x-%j.err
set -euo pipefail
source "${TRAIN_SOURCE_ROOT:?}/slurm/training-common.sh"
export DATA_ROOT TRAIN_ENV_PREFIX JOB_RECORD
export PIP_CACHE_DIR="$DATA_ROOT/caches/pip"
python "${BATCH1B_CONTROLLER:?}" check
test ! -e "${MUONIUM_TARGET:?}"
srun --ntasks=1 --kill-on-bad-exit=1 python -m pip install --no-deps --disable-pip-version-check --target "$MUONIUM_TARGET" --require-hashes -r "$REPO/requirements-muonium.txt"
python "$BATCH1B_CONTROLLER" setup-receipt
