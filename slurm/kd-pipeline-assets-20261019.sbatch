#!/bin/bash
#SBATCH --job-name=kd-pipeline-assets
#SBATCH --account=ACD115198
#SBATCH --partition=8gpus
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --gres=gpu:1
#SBATCH --constraint=H200
#SBATCH --exclude=25a-hgpn001
#SBATCH --time=00:10:00
#SBATCH --no-requeue
#SBATCH --output=/work/asciibase64/Llama-3.2-1B-training/logs/%x-%j.out
#SBATCH --error=/work/asciibase64/Llama-3.2-1B-training/logs/%x-%j.err
set -euo pipefail
cd /home/asciibase64/Llama-3.2-1B-training
PIPELINE_ROOT=$PWD/experiments/methods-parallel-20261019/kd-acceleration/pipeline
export TRAIN_SOURCE_ROOT=$PWD/experiments/methods-parallel-20261019/source
source "$TRAIN_SOURCE_ROOT/slurm/training-common.sh" kd-pipeline-assets-20261019
export HF_HUB_DISABLE_XET=1 HF_HUB_DOWNLOAD_TIMEOUT=60
srun --ntasks=1 --kill-on-bad-exit=1 python "$PIPELINE_ROOT/prepare_teacher.py" \
    "$DATA_ROOT/caches/kd-teachers/llama32-3b-13afe512-pipeline-20261019"
