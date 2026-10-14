#!/bin/bash
#SBATCH --job-name=horizon-wsd
#SBATCH --account=ACD115198
#SBATCH --partition=8gpus
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=200G
#SBATCH --gpus-per-node=8
#SBATCH --constraint=H200
#SBATCH --time=01:30:00
#SBATCH --signal=B:USR1@180
#SBATCH --no-requeue
#SBATCH --exclude=25a-hgpn001,25a-hgpn062,25a-hgpn151
#SBATCH --output=/work/asciibase64/Llama-3.2-1B-training/logs/%x-%j.out
#SBATCH --error=/work/asciibase64/Llama-3.2-1B-training/logs/%x-%j.err
set -euo pipefail
export NCCL_SOCKET_IFNAME='=lo'
export NCCL_SOCKET_FAMILY=AF_INET
source "${TRAIN_SOURCE_ROOT:?}/slurm/training-common.sh"
export DATA_ROOT TRAIN_ENV_PREFIX JOB_RECORD
printf 'NCCL_SOCKET_IFNAME=%s\nNCCL_SOCKET_FAMILY=%s\n' "$NCCL_SOCKET_IFNAME" "$NCCL_SOCKET_FAMILY" > "$JOB_RECORD/nccl-bootstrap-settings.txt"
CONFIGURATION=${2:?configuration}
MODE=${3:?mode}
PAUSE_STEP=${4:?pause step or zero}
PARENT=${5:?parent or dash}
FULL_EVAL=${6:?full evaluation boolean}
python3 "${HORIZON_CONTROLLER:?}" check
if [[ "$RUN_ID" == horizon-wsd-20261014-validation-recovery1 ]]; then
    srun --ntasks=1 --kill-on-bad-exit=1 accelerate launch --multi_gpu \
        --main_process_port "$((15000 + SLURM_JOB_ID % 40000))" \
        --num_processes 8 --num_machines 1 --mixed_precision bf16 \
        --dynamo_backend no --num_cpu_threads_per_process 1 "$REPO/horizon_gpu_checks.py"
fi
EXTRA=()
[[ "$MODE" != resume ]] || EXTRA+=(--resume)
[[ "$PAUSE_STEP" == 0 ]] || EXTRA+=(--pause-after-step "$PAUSE_STEP")
[[ "$PARENT" == - ]] || EXTRA+=(--fork-from-run "$PARENT")
[[ "$FULL_EVAL" != 1 ]] || EXTRA+=(--validate-export)
srun --ntasks=1 --kill-on-bad-exit=1 accelerate launch --multi_gpu \
    --main_process_port "$((15000 + SLURM_JOB_ID % 40000))" \
    --num_processes 8 --num_machines 1 --mixed_precision bf16 \
    --dynamo_backend no --num_cpu_threads_per_process 1 \
    "${HORIZON_REPO:?}/slurm/resplit_lr_entry.py" --config "$CONFIGURATION" \
    --prep-id dolma-raw-baseline-20261008 --run-id "$RUN_ID" "${EXTRA[@]}"
python3 "${HORIZON_CONTROLLER:?}" verify-job "$RUN_ID" --pause "$PAUSE_STEP"
if [[ "$FULL_EVAL" == 1 ]]; then
    srun --ntasks=1 --kill-on-bad-exit=1 accelerate launch --multi_gpu \
        --main_process_port "$((15001 + SLURM_JOB_ID % 40000))" \
        --num_processes 8 --num_machines 1 --mixed_precision bf16 \
        --dynamo_backend no --num_cpu_threads_per_process 1 \
        "$REPO/slurm/full_evaluation.py" "$CONFIGURATION" "$RUN_ID" dolma-raw-baseline-20261008
fi
