#!/bin/bash
#SBATCH --job-name=raw4b-normuon-adam2x
#SBATCH --account=ACD115198
#SBATCH --partition=8gpus
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=200G
#SBATCH --gpus-per-node=8
#SBATCH --constraint=H200
#SBATCH --time=07:00:00
#SBATCH --signal=B:USR1@300
#SBATCH --no-requeue
#SBATCH --output=/work/asciibase64/Llama-3.2-1B-training/logs/%x-%j.out
#SBATCH --error=/work/asciibase64/Llama-3.2-1B-training/logs/%x-%j.err
set -euo pipefail
export NCCL_SOCKET_IFNAME='=lo'
export NCCL_SOCKET_FAMILY=AF_INET
source "${TRAIN_SOURCE_ROOT:?}/slurm/training-common.sh"
export DATA_ROOT TRAIN_ENV_PREFIX JOB_RECORD
CONFIGURATION=${2:?configuration}
export TORCHINDUCTOR_CACHE_DIR="$DATA_ROOT/caches/inductor/raw4b-$SLURM_JOB_ID"
export TORCHINDUCTOR_COMPILE_THREADS=1
python "${FOURB_CONTROLLER:?}" check
LAUNCH=(srun --ntasks=1 --kill-on-bad-exit=1 accelerate launch --multi_gpu --main_process_port "$((15000 + SLURM_JOB_ID % 40000))" --num_processes 8 --num_machines 1 --mixed_precision bf16 --dynamo_backend no --num_cpu_threads_per_process 1)
timeout --signal=TERM --kill-after=30s 600s "${LAUNCH[@]}" "$REPO/ppl_gate.py" "$CONFIGURATION"
python "$FOURB_CONTROLLER" preflight-check
"${LAUNCH[@]}" "$REPO/slurm/training_entry.py" --config "$CONFIGURATION" --prep-id dolma-raw-baseline-20261008 --run-id "$RUN_ID" --validate-export
python "$FOURB_CONTROLLER" training-check
"${LAUNCH[@]}" "$REPO/slurm/full_evaluation.py" "$CONFIGURATION" "$RUN_ID" dolma-raw-baseline-20261008
python "$FOURB_CONTROLLER" receipt
