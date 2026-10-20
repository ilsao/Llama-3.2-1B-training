#!/bin/bash
#SBATCH --job-name=kd-pipeline-check
#SBATCH --account=ACD115198
#SBATCH --partition=8gpus
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=200G
#SBATCH --gres=gpu:8
#SBATCH --constraint=H200
#SBATCH --exclude=25a-hgpn001
#SBATCH --time=00:12:00
#SBATCH --no-requeue
#SBATCH --output=/work/asciibase64/Llama-3.2-1B-training/logs/%x-%j.out
#SBATCH --error=/work/asciibase64/Llama-3.2-1B-training/logs/%x-%j.err
set -euo pipefail
cd /home/asciibase64/Llama-3.2-1B-training
PIPELINE_ROOT=$PWD/experiments/methods-parallel-20261019/kd-acceleration/pipeline
export TRAIN_SOURCE_ROOT=$PWD/experiments/methods-parallel-20261019/source
python3 - "$PIPELINE_ROOT/validation-seal.json" <<'PY'
import sys,json,hashlib
from pathlib import Path
for name,sha in json.load(open(sys.argv[1]))['files'].items():
    if hashlib.sha256(Path(name).read_bytes()).hexdigest()!=sha:raise SystemExit('Source seal changed: '+name)
PY
source "$TRAIN_SOURCE_ROOT/slurm/training-common.sh" kd-pipeline-validation-20261019
export HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1
export NCCL_SOCKET_IFNAME='=lo' NCCL_SOCKET_FAMILY=AF_INET
export TORCHINDUCTOR_CACHE_DIR="$DATA_ROOT/caches/torchinductor-kd-pipeline-$SLURM_JOB_ID"
export TRITON_CACHE_DIR="$DATA_ROOT/caches/triton-kd-pipeline-$SLURM_JOB_ID"
export TMPDIR="$DATA_ROOT/tmp/kd-pipeline-$SLURM_JOB_ID"
mkdir -p "$TMPDIR" "$TORCHINDUCTOR_CACHE_DIR" "$TRITON_CACHE_DIR"
OUT="$DATA_ROOT/runs/experiments/kd-pipeline-validation-20261019"
mkdir -p "$OUT"
cp "$PIPELINE_ROOT/validation-config.json" "$OUT/config.json"
cp "$PIPELINE_ROOT/validation-seal.json" "$OUT/source-seal.json"
srun --ntasks=1 --kill-on-bad-exit=1 accelerate launch --multi_gpu \
    --main_process_port "$((15000 + SLURM_JOB_ID % 40000))" --num_processes 8 \
    --num_machines 1 --mixed_precision bf16 --dynamo_backend no \
    --num_cpu_threads_per_process 1 "$PIPELINE_ROOT/benchmark_pipeline.py" "$OUT"
