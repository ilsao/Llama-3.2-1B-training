#!/bin/bash
#SBATCH --job-name=llama-setup
#SBATCH --account=ACD115198
#SBATCH --partition=8gpus
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=200G
#SBATCH --gpus-per-node=1
#SBATCH --time=01:00:00
#SBATCH --no-requeue
#SBATCH --output=/work/asciibase64/Llama-3.2-1B-training/logs/%x-%j.out
#SBATCH --error=/work/asciibase64/Llama-3.2-1B-training/logs/%x-%j.err
set -euo pipefail
SETUP_TRAINING_ENV=1
export TRAIN_ENV_PREFIX=/work/asciibase64/Llama-3.2-1B-training/environments/training-muon-torch291-fa3-94e22c9
source "${TRAIN_SOURCE_ROOT:?Submit from the intended training worktree}/slurm/training-common.sh"
export DATA_ROOT TRAIN_ENV_PREFIX JOB_RECORD
[[ $(type -t module || true) == function ]] || source /etc/profile.d/lmod.sh
module load cuda/12.6
# Dependency installation only: no CUDA kernel/model/optimizer runtime checks.
srun --ntasks=1 "$DATA_ROOT/environments/training-py311/bin/python" "$REPO/slurm/fa3_setup.py"
