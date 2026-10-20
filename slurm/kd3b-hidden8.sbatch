#!/bin/bash
#SBATCH --job-name=kd3bh8
#SBATCH --account=ACD115198
#SBATCH --partition=8gpus
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=200G
#SBATCH --gpus-per-node=8
#SBATCH --constraint=H200
#SBATCH --exclude=25a-hgpn001
#SBATCH --time=06:00:00
#SBATCH --signal=B:USR1@300
#SBATCH --no-requeue
#SBATCH --output=/work/asciibase64/Llama-3.2-1B-training/logs/%x-%j.out
#SBATCH --error=/work/asciibase64/Llama-3.2-1B-training/logs/%x-%j.err
set -euo pipefail
PROJECT_DIR=/home/asciibase64/Llama-3.2-1B-training
STUDY_DIR="$PROJECT_DIR/experiments/kd3b-hidden8-20261019"
export TRAIN_SOURCE_ROOT="$STUDY_DIR/source"
source "$TRAIN_SOURCE_ROOT/slurm/training-common.sh" "${1:?Submit through kd3b_hidden8.py}"
ROLE=${2:?producer or consumer}
[[ "$ROLE" == producer || "$ROLE" == consumer ]] || exit 2
export DATA_ROOT TRAIN_ENV_PREFIX JOB_RECORD
export ROLE RUN_ID
hidden_finish() {
    local job_status=$?
    trap - EXIT
    date -u +%FT%TZ > "$JOB_RECORD/end.txt"
    printf '%s\n' "$job_status" > "$JOB_RECORD/exit-code.txt"
    if [[ "$job_status" != 0 ]]; then
        python3 - "$job_status" <<'PY'
import os,sys,time
from pathlib import Path
from training_state import atomic_json
root=Path(os.environ['DATA_ROOT'])/'runs/experiments'/os.environ['RUN_ID']/'transport'
root.mkdir(parents=True,exist_ok=True)
atomic_json(root/('abort-'+os.environ['ROLE']+'-job.json'),dict(exit_code=int(sys.argv[1]),job_id=os.environ['SLURM_JOB_ID'],timestamp=time.time()))
PY
    fi
    timeout 15s sacct -j "$SLURM_JOB_ID" -X --noheader --parsable2 \
      --format=JobIDRaw,State,ElapsedRaw,AllocTRES%500,Start,End,NodeList%128 \
      > "$JOB_RECORD/accounting-at-exit.txt" || true
    exit "$job_status"
}
trap hidden_finish EXIT
CONFIG="$STUDY_DIR/config.json"
python "$PROJECT_DIR/slurm/kd3b_hidden8.py" check > "$JOB_RECORD/execution-check.json"
export HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1
export PYTHONPATH="$TRAIN_SOURCE_ROOT${PYTHONPATH:+:$PYTHONPATH}"
export NCCL_SOCKET_IFNAME='=lo' NCCL_SOCKET_FAMILY=AF_INET
JOB_CACHE="$DATA_ROOT/caches/kd3b-hidden8/$SLURM_JOB_ID"
mkdir -p "$JOB_CACHE"/{triton,inductor,cuda,tmp,wandb}
export TRITON_CACHE_DIR="$JOB_CACHE/triton" TORCHINDUCTOR_CACHE_DIR="$JOB_CACHE/inductor"
export CUDA_CACHE_PATH="$JOB_CACHE/cuda" TMPDIR="$JOB_CACHE/tmp" WANDB_CACHE_DIR="$JOB_CACHE/wandb"
export EVAL_DOCUMENT_OUTPUT="$DATA_ROOT/runs/experiments/$RUN_ID/selection-documents.json"
export WANDB_INIT_TIMEOUT=120
LAUNCH=(srun --ntasks=1 --kill-on-bad-exit=1 accelerate launch --multi_gpu
  --main_process_port "$((15000+SLURM_JOB_ID%40000))" --num_processes 8
  --num_machines 1 --mixed_precision bf16 --dynamo_backend no --num_cpu_threads_per_process 1)
if [[ "$ROLE" == producer ]]; then
    SECONDS_LEFT=$((TRAIN_END_EPOCH - $(date +%s) - 180))
    [[ "$SECONDS_LEFT" -gt 0 ]] || exit 124
    timeout --signal=TERM --kill-after=20s "$SECONDS_LEFT" "${LAUNCH[@]}" \
      "$TRAIN_SOURCE_ROOT/hidden_producer.py" "$CONFIG"
else
    SECONDS_LEFT=$((TRAIN_END_EPOCH - $(date +%s) - 900))
    [[ "$SECONDS_LEFT" -gt 0 ]] || exit 124
    timeout --signal=TERM --kill-after=20s "$SECONDS_LEFT" "${LAUNCH[@]}" \
      "$TRAIN_SOURCE_ROOT/parallel_entry.py" --config "$CONFIG" --run-id "$RUN_ID" \
      --prep-id dolma-raw-baseline-20261008 --validate-export
    SECONDS_LEFT=$((TRAIN_END_EPOCH - $(date +%s) - 120))
    [[ "$SECONDS_LEFT" -gt 0 ]] || exit 124
    timeout --signal=TERM --kill-after=15s "$SECONDS_LEFT" "${LAUNCH[@]}" \
      "$TRAIN_SOURCE_ROOT/slurm/full_evaluation.py" "$CONFIG" "$RUN_ID" dolma-raw-baseline-20261008
fi
