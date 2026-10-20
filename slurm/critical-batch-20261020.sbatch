#!/bin/bash
#SBATCH --job-name=critical-batch
#SBATCH --account=ACD115198
#SBATCH --partition=8gpus
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=200G
#SBATCH --gres=gpu:8
#SBATCH --constraint=H200
#SBATCH --exclude=25a-hgpn001
#SBATCH --time=01:00:00
#SBATCH --no-requeue
#SBATCH --output=/work/asciibase64/Llama-3.2-1B-training/logs/%x-%j.out
#SBATCH --error=/work/asciibase64/Llama-3.2-1B-training/logs/%x-%j.err
set -euo pipefail
cd /home/asciibase64/Llama-3.2-1B-training
CBS_ROOT=$PWD/experiments/critical-batch-20261020
export TRAIN_SOURCE_ROOT=$PWD/experiments/kd3b-hidden8-20261019/source
python3 "$CBS_ROOT/launch.py" check
source "$TRAIN_SOURCE_ROOT/slurm/training-common.sh" "${1:?run ID required}"
export HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 TORCHINDUCTOR_COMPILE_THREADS=1
export NCCL_SOCKET_IFNAME='=lo' NCCL_SOCKET_FAMILY=AF_INET
export TORCHINDUCTOR_CACHE_DIR="$DATA_ROOT/caches/torchinductor-cbs-$SLURM_JOB_ID"
export TRITON_CACHE_DIR="$DATA_ROOT/caches/triton-cbs-$SLURM_JOB_ID"
export TMPDIR="$DATA_ROOT/tmp/cbs-$SLURM_JOB_ID"
mkdir -p "$TMPDIR" "$TORCHINDUCTOR_CACHE_DIR" "$TRITON_CACHE_DIR"
CBS_ENTRY="$CBS_ROOT/runner.py"
if [[ ${CBS_VALIDATION_ONLY:-0} == 1 ]]; then CBS_ENTRY="$CBS_ROOT/gpu_gate.py"; fi
srun --ntasks=1 --kill-on-bad-exit=1 accelerate launch --multi_gpu \
  --main_process_port "$((15000 + SLURM_JOB_ID % 40000))" --num_processes 8 \
  --num_machines 1 --mixed_precision bf16 --dynamo_backend no \
  --num_cpu_threads_per_process 1 "$CBS_ENTRY"
