#!/bin/bash
# Sourced only by allocated jobs. The user authorized one H200 for CPU stages
# because ACD115198 has no verified CPU-only partition access.
set -euo pipefail
REPO=${TRAIN_SOURCE_ROOT:-/home/asciibase64/Llama-3.2-1B-training}
DATA_ROOT=${DATA_ROOT:-/work/asciibase64/Llama-3.2-1B-training}
ENV_PREFIX="$DATA_ROOT/environments/data-py311"
[[ -n ${SLURM_JOB_ID:-} && -n ${SLURMD_NODENAME:-} ]] || { echo 'Use sbatch, not bash job.sbatch' >&2; exit 1; }
[[ "$DATA_ROOT" == /work/asciibase64/* ]] || { echo 'DATA_ROOT must be under /work/asciibase64/' >&2; exit 1; }
cd "$REPO"
mkdir -p "$DATA_ROOT"/{datasets,caches/hub,caches/datasets,caches/torch,caches/xdg,caches/wandb,tmp,logs,runs/wandb,environments,preprocessed}
# Preserve the existing HF login credential path while relocating large caches.
export HF_TOKEN_PATH=${HF_TOKEN_PATH:-${HF_HOME:-$HOME/.cache/huggingface}/token}
export HF_HOME="$DATA_ROOT/caches"
export HF_HUB_CACHE="$DATA_ROOT/caches/hub"
export HF_DATASETS_CACHE="$DATA_ROOT/caches/datasets"
export TORCH_HOME="$DATA_ROOT/caches/torch"
export XDG_CACHE_HOME="$DATA_ROOT/caches/xdg"
export WANDB_DIR="$DATA_ROOT/runs/wandb"
export WANDB_CACHE_DIR="$DATA_ROOT/caches/wandb"
export TMPDIR="$DATA_ROOT/tmp"
export SQLITE_TMPDIR="$TMPDIR"
export TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export VECLIB_MAXIMUM_THREADS=1
export RAYON_NUM_THREADS=1
export HF_ENABLE_PARALLEL_LOADING=false
if [[ ${RETENTION_EXCLUSIVE:-0} != 1 ]]; then
    exec 9>"$DATA_ROOT/.retention.lock"
    flock -s -n 9 || { echo 'Project cleanup is active; defer data job' >&2; exit 1; }
fi
JOB_RECORD="$DATA_ROOT/logs/job-${SLURM_JOB_ID}"
mkdir -p "$JOB_RECORD"
date -u +%FT%TZ > "$JOB_RECORD/start.txt"
finish_job() {
    local job_status=$?
    trap - EXIT
    date -u +%FT%TZ > "$JOB_RECORD/end.txt"
    printf '%s\n' "$job_status" > "$JOB_RECORD/exit-code.txt"
    sacct -j "$SLURM_JOB_ID" -X --noheader --parsable2 \
        --format=JobIDRaw,State,ElapsedRaw,AllocTRES%500,Start,End,NodeList%128 \
        > "$JOB_RECORD/accounting-at-exit.txt" || true
    exit "$job_status"
}
trap finish_job EXIT
scontrol show job -o "$SLURM_JOB_ID" > "$JOB_RECORD/allocation.txt"
scontrol show node "$SLURMD_NODENAME" > "$JOB_RECORD/node.txt"
ALLOCATED_CPUS=$(sed -n 's/.* NumCPUs=\([0-9]*\).*/\1/p' "$JOB_RECORD/allocation.txt")
[[ ${SLURM_JOB_NUM_NODES:-1} == 1 && ${SLURM_NTASKS:-1} == 1 ]] || { echo 'Use one node and one Slurm task' >&2; exit 1; }
CPU_LIMIT=8
# Explicit user override: training tokenization only, unchanged evaluation workers.
if [[ ${STAGE:-} == select && ${PREPARATION_ID:-} == dolma-dclm10-005 && ${FILTERING_TRAIN_WORKERS:-7} =~ ^(11|15)$ ]]; then
    CPU_LIMIT=$((FILTERING_TRAIN_WORKERS+1))
    [[ "$ALLOCATED_CPUS" == "$CPU_LIMIT" && ${SLURM_CPUS_PER_TASK:-0} == "$CPU_LIMIT" ]] || exit 1
fi
if [[ ${STAGE:-} == select && ${PREPARATION_ID:-} == dolma-dclm10-005 && ${FILTERING_EVAL_WORKERS:-7} == 11 ]]; then
    CPU_LIMIT=12
    [[ "$ALLOCATED_CPUS" == 12 && ${SLURM_CPUS_PER_TASK:-0} == 12 ]] || exit 1
fi
[[ "$ALLOCATED_CPUS" =~ ^[0-9]+$ && "$ALLOCATED_CPUS" -ge 1 && "$ALLOCATED_CPUS" -le "$CPU_LIMIT" ]] || { echo "Allocate at most $CPU_LIMIT CPUs for this job" >&2; exit 1; }
[[ ${SLURM_CPUS_PER_TASK:-0} -ge 1 && ${SLURM_CPUS_PER_TASK:-0} -le "$CPU_LIMIT" ]] || exit 1
printf '%s\n' "$ALLOCATED_CPUS" > "$JOB_RECORD/cpus.txt"
export MAX_JOBS="$ALLOCATED_CPUS"
export CMAKE_BUILD_PARALLEL_LEVEL="$ALLOCATED_CPUS"
# Capture actual allocated hardware for CPU-stage GPU accounting too.
srun --ntasks=1 bash -c 'nvidia-smi -i "${CUDA_VISIBLE_DEVICES:?Slurm must expose allocated GPUs}" --query-gpu=name,uuid --format=csv,noheader' > "$JOB_RECORD/gpus.txt"
GPU_COUNT=$(wc -l < "$JOB_RECORD/gpus.txt")
[[ "$GPU_COUNT" -ge 1 && "$GPU_COUNT" -le 8 ]] || { echo 'Allocate 1..8 H200s' >&2; exit 1; }
awk 'index($0,"H200")==0 {bad=1} END {exit bad}' "$JOB_RECORD/gpus.txt" || { echo 'Require H200 GPUs' >&2; exit 1; }
# Respect the actual allocation end time, including --time overrides.
JOB_END=$(sed -n 's/.* EndTime=\([^ ]*\).*/\1/p' "$JOB_RECORD/allocation.txt")
[[ -n "$JOB_END" && "$JOB_END" != Unknown ]] || { echo 'Cannot determine allocation end time' >&2; exit 1; }
PREP_STOP_EPOCH=$(( $(date -d "$JOB_END" +%s) - 900 ))
export PREP_STOP_EPOCH
trap ':' USR1
if [[ ${SKIP_DATA_ENV:-0} != 1 ]]; then
    [[ -x "$ENV_PREFIX/bin/python" ]] || { echo 'Submit slurm/setup.sbatch first' >&2; exit 1; }
    source /work/envstack/apps/miniconda3/26.1.1/etc/profile.d/conda.sh
    conda activate "$ENV_PREFIX"
fi
