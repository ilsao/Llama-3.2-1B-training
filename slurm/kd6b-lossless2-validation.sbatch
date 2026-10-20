#!/bin/bash
#SBATCH --job-name=kd6b-lossless2-check
#SBATCH --account=ACD115198
#SBATCH --partition=8gpus
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=200G
#SBATCH --gres=gpu:8
#SBATCH --constraint=H200
#SBATCH --exclude=25a-hgpn001
#SBATCH --time=00:30:00
#SBATCH --no-requeue
#SBATCH --output=/work/asciibase64/Llama-3.2-1B-training/logs/%x-%j.out
#SBATCH --error=/work/asciibase64/Llama-3.2-1B-training/logs/%x-%j.err
set -euo pipefail
cd /home/asciibase64/Llama-3.2-1B-training
KD_CHECK_ROOT=$PWD/experiments/kd6b-lossless2-20261020
export TRAIN_SOURCE_ROOT=$PWD/experiments/kd3b-hidden8-20261019/source
python3 "$KD_CHECK_ROOT/launch.py" check
source "$TRAIN_SOURCE_ROOT/slurm/training-common.sh" kd6b-lossless2-validation-20261020
export HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1
export TORCHINDUCTOR_COMPILE_THREADS=1
export NCCL_SOCKET_IFNAME='=lo' NCCL_SOCKET_FAMILY=AF_INET
export TORCHINDUCTOR_CACHE_DIR="$DATA_ROOT/caches/torchinductor-kd6b-check2-$SLURM_JOB_ID"
export TRITON_CACHE_DIR="$DATA_ROOT/caches/triton-kd6b-check2-$SLURM_JOB_ID"
export TMPDIR="$DATA_ROOT/tmp/kd6b-check2-$SLURM_JOB_ID"
KD_CHECK_OUT="$DATA_ROOT/runs/experiments/kd6b-lossless2-validation-20261020"
mkdir -p "$TMPDIR" "$TORCHINDUCTOR_CACHE_DIR" "$TRITON_CACHE_DIR" "$KD_CHECK_OUT"
cp "$KD_CHECK_ROOT/execution-seal.json" "$KD_CHECK_OUT/source-seal.json"
srun --ntasks=1 --kill-on-bad-exit=1 accelerate launch --multi_gpu \
  --main_process_port "$((15000 + SLURM_JOB_ID % 40000))" --num_processes 8 \
  --num_machines 1 --mixed_precision bf16 --dynamo_backend no \
  --num_cpu_threads_per_process 1 "$KD_CHECK_ROOT/benchmark.py" "$KD_CHECK_OUT"
