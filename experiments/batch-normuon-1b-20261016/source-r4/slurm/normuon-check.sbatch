#!/bin/bash
#SBATCH --job-name=b1b-normuon-check
#SBATCH --account=ACD115198
#SBATCH --partition=8gpus
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=100G
#SBATCH --gpus-per-node=1
#SBATCH --constraint=H200
#SBATCH --time=00:15:00
#SBATCH --signal=B:USR1@180
#SBATCH --no-requeue
#SBATCH --output=/work/asciibase64/Llama-3.2-1B-training/logs/%x-%j.out
#SBATCH --error=/work/asciibase64/Llama-3.2-1B-training/logs/%x-%j.err
set -euo pipefail
export NCCL_SOCKET_IFNAME='=lo'
export NCCL_SOCKET_FAMILY=AF_INET
source "${TRAIN_SOURCE_ROOT:?}/slurm/training-common.sh"
export DATA_ROOT TRAIN_ENV_PREFIX JOB_RECORD
export TORCH_COMPILE_DISABLE=1
export TORCHINDUCTOR_CACHE_DIR="$DATA_ROOT/caches/inductor/batch1b"
export TRITON_CACHE_DIR="$DATA_ROOT/caches/triton/batch1b"
export TORCHINDUCTOR_COMPILE_THREADS=1
export MAX_JOBS=1
export PYTHONPATH="${MUONIUM_TARGET:?}${PYTHONPATH:+:$PYTHONPATH}"
python "${BATCH1B_CONTROLLER:?}" check
srun --ntasks=1 --kill-on-bad-exit=1 python "$REPO/normuon_check.py" --config "${2:?configuration}"
