#!/bin/bash
#SBATCH --job-name=batch1b
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
export NCCL_SOCKET_IFNAME='=lo'
export NCCL_SOCKET_FAMILY=AF_INET
source "${TRAIN_SOURCE_ROOT:?}/slurm/training-common.sh"
export DATA_ROOT TRAIN_ENV_PREFIX JOB_RECORD
export TORCHINDUCTOR_CACHE_DIR="$DATA_ROOT/caches/inductor/batch1b"
export TRITON_CACHE_DIR="$DATA_ROOT/caches/triton/batch1b"
export TORCHINDUCTOR_COMPILE_THREADS=1
export MAX_JOBS=1
CONFIGURATION=${2:?config}
KIND=${3:?kind}
python "${BATCH1B_CONTROLLER:?}" check
LAUNCH=(srun --ntasks=1 --kill-on-bad-exit=1 accelerate launch --multi_gpu --main_process_port "$((15000 + SLURM_JOB_ID % 40000))" --num_processes 8 --num_machines 1 --mixed_precision bf16 --dynamo_backend no --num_cpu_threads_per_process 1)
if [[ "$KIND" == validation ]]; then
    export PYTHONPATH="${MUONIUM_TARGET:?}${PYTHONPATH:+:$PYTHONPATH}"
    "${LAUNCH[@]}" "$REPO/batch1b_gpu_checks.py" --config "$CONFIGURATION"
else
    EXTRA=()
    if [[ "$KIND" == normuon ]]; then
        export TORCH_COMPILE_DISABLE=1
        export PYTHONPATH="${MUONIUM_TARGET:?}${PYTHONPATH:+:$PYTHONPATH}"
    elif [[ "$KIND" == fork ]]; then
        EXTRA+=(--fork-from-run batch-normuon-1b-20261016-control)
    fi
    "${LAUNCH[@]}" "$REPO/slurm/training_entry.py" --config "$CONFIGURATION" --prep-id dolma-raw-baseline-20261008 --run-id "$RUN_ID" --validate-export "${EXTRA[@]}"
    python "${BATCH1B_CONTROLLER:?}" receipt "$RUN_ID" "$CONFIGURATION"
fi
