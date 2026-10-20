#!/bin/bash
#SBATCH --job-name=methods-parallel
#SBATCH --account=ACD115198
#SBATCH --partition=8gpus
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=200G
#SBATCH --gpus-per-node=8
#SBATCH --constraint=H200
#SBATCH --time=02:00:00
#SBATCH --signal=B:USR1@300
#SBATCH --no-requeue
#SBATCH --output=/work/asciibase64/Llama-3.2-1B-training/logs/%x-%j.out
#SBATCH --error=/work/asciibase64/Llama-3.2-1B-training/logs/%x-%j.err
set -euo pipefail
PROJECT_DIR=/home/asciibase64/Llama-3.2-1B-training
STUDY_DIR="$PROJECT_DIR/experiments/methods-parallel-20261019"
export TRAIN_SOURCE_ROOT="$STUDY_DIR/source"
KEY=${2:?Supply locked experiment key through methods_parallel.py}
[[ "$KEY" =~ ^[A-Z0-9]+-seed42$ ]] || exit 1
source "$TRAIN_SOURCE_ROOT/slurm/training-common.sh" "${1:?Supply run ID}"
export DATA_ROOT TRAIN_ENV_PREFIX JOB_RECORD
python "$PROJECT_DIR/slurm/methods_parallel.py" check --key "$KEY" > "$JOB_RECORD/execution-check.json"
CONFIG="$STUDY_DIR/configs/$KEY.json"
python - "$STUDY_DIR/matrix.json" "$KEY" "$RUN_ID" <<'PY'
import json,sys
rows=[r for r in json.load(open(sys.argv[1]))['jobs'] if r['key']==sys.argv[2]]
assert len(rows)==1 and rows[0]['run_id']==sys.argv[3]
PY
# Data and model/tokenizer assets are pre-existing; no dataset preparation here.
export HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1
export PYTHONPATH="$TRAIN_SOURCE_ROOT${PYTHONPATH:+:$PYTHONPATH}"
export NCCL_SOCKET_IFNAME='=lo' NCCL_SOCKET_FAMILY=AF_INET
JOB_CACHE="$DATA_ROOT/caches/methods-parallel/$SLURM_JOB_ID"
mkdir -p "$JOB_CACHE"/{triton,inductor,cuda,tmp,wandb}
export TRITON_CACHE_DIR="$JOB_CACHE/triton" TORCHINDUCTOR_CACHE_DIR="$JOB_CACHE/inductor"
export CUDA_CACHE_PATH="$JOB_CACHE/cuda" TMPDIR="$JOB_CACHE/tmp" WANDB_CACHE_DIR="$JOB_CACHE/wandb"
export EVAL_DOCUMENT_OUTPUT="$DATA_ROOT/runs/experiments/$RUN_ID/selection-documents.json"
LAUNCH=(srun --ntasks=1 --kill-on-bad-exit=1 accelerate launch --multi_gpu
  --main_process_port "$((15000+SLURM_JOB_ID%40000))" --num_processes 8
  --num_machines 1 --mixed_precision bf16 --dynamo_backend no --num_cpu_threads_per_process 1)
# Failure exits this job before full-model initialization; no dependencies/retries.
timeout --signal=TERM --kill-after=20s 600 "${LAUNCH[@]}" "$TRAIN_SOURCE_ROOT/parallel_gate.py" "$CONFIG"
SECONDS_LEFT=$((TRAIN_END_EPOCH - $(date +%s) - 900))
[[ "$SECONDS_LEFT" -gt 0 ]] || exit 124
timeout --signal=TERM --kill-after=20s "$SECONDS_LEFT" "${LAUNCH[@]}" \
  "$TRAIN_SOURCE_ROOT/parallel_entry.py" --config "$CONFIG" --run-id "$RUN_ID" \
  --prep-id dolma-raw-baseline-20261008 --validate-export
# Entry preserves incomplete exit code 75. Only completed endpoints reach here.
SECONDS_LEFT=$((TRAIN_END_EPOCH - $(date +%s) - 120))
[[ "$SECONDS_LEFT" -gt 0 ]] || exit 124
timeout --signal=TERM --kill-after=15s "$SECONDS_LEFT" "${LAUNCH[@]}" \
  "$TRAIN_SOURCE_ROOT/slurm/full_evaluation.py" "$CONFIG" "$RUN_ID" dolma-raw-baseline-20261008
