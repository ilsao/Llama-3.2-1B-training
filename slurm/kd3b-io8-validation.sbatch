#!/bin/bash
#SBATCH --job-name=kd3b-io8-check
#SBATCH --account=ACD115198
#SBATCH --partition=8gpus
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=200G
#SBATCH --gres=gpu:8
#SBATCH --constraint=H200
#SBATCH --exclude=25a-hgpn001
#SBATCH --time=00:20:00
#SBATCH --no-requeue
#SBATCH --output=/work/asciibase64/Llama-3.2-1B-training/logs/%x-%j.out
#SBATCH --error=/work/asciibase64/Llama-3.2-1B-training/logs/%x-%j.err
set -euo pipefail
cd /home/asciibase64/Llama-3.2-1B-training
IO_ROOT=$PWD/experiments/kd3b-io8-validation-20261019
IO_ROLE=${1:?producer or consumer}
[[ "$IO_ROLE" == producer || "$IO_ROLE" == consumer ]] || exit 1
export TRAIN_SOURCE_ROOT=$PWD/experiments/kd3b-online-pipeline-20261019/source
python3 "$IO_ROOT/launch_validation.py" check
source "$TRAIN_SOURCE_ROOT/slurm/training-common.sh" "kd3b-io8-validation-20261019-$IO_ROLE"
export HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1
export NCCL_SOCKET_IFNAME='=lo' NCCL_SOCKET_FAMILY=AF_INET
export TORCHINDUCTOR_CACHE_DIR="$DATA_ROOT/caches/torchinductor-kd3b-io8-$SLURM_JOB_ID"
export TRITON_CACHE_DIR="$DATA_ROOT/caches/triton-kd3b-io8-$SLURM_JOB_ID"
export TMPDIR="$DATA_ROOT/tmp/kd3b-io8-$SLURM_JOB_ID"
IO_OUT="$DATA_ROOT/runs/experiments/kd3b-io8-validation-20261019"
mkdir -p "$TMPDIR" "$TORCHINDUCTOR_CACHE_DIR" "$TRITON_CACHE_DIR" "$IO_OUT/$IO_ROLE"
cp "$IO_ROOT/validation-config.json" "$IO_OUT/$IO_ROLE/resolved-config.json"
cp "$IO_ROOT/execution-seal.json" "$IO_OUT/$IO_ROLE/source-seal.json"
srun --ntasks=1 --kill-on-bad-exit=1 accelerate launch --multi_gpu \
    --main_process_port "$((15000 + SLURM_JOB_ID % 40000))" --num_processes 8 \
    --num_machines 1 --mixed_precision bf16 --dynamo_backend no \
    --num_cpu_threads_per_process 1 "$IO_ROOT/benchmark_io.py" "$IO_ROLE" "$IO_OUT"
