#!/bin/bash
#SBATCH --job-name=llama-pilot
#SBATCH --account=ACD115198
#SBATCH --partition=8gpus
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=200G
#SBATCH --gpus-per-node=4
#SBATCH --time=00:30:00
#SBATCH --signal=B:USR1@180
#SBATCH --no-requeue
#SBATCH --output=/work/asciibase64/Llama-3.2-1B-training/logs/%x-%j.out
#SBATCH --error=/work/asciibase64/Llama-3.2-1B-training/logs/%x-%j.err
set -euo pipefail
source /home/asciibase64/Llama-3.2-1B-training/slurm/training-common.sh
ACTION=${2:?Supply runtime, validate, pilot, scaling or resume}
PREPARATION_ID=${3:?Supply preparation ID or - for runtime}
TRAIN_CONFIG=${4:?Supply configuration path}
GPU_COUNT=$(wc -l < "$JOB_RECORD/gpus.txt")
launch() {
    DISTRIBUTED=()
    [[ "$GPU_COUNT" -le 1 ]] || DISTRIBUTED+=(--multi_gpu)
    srun --ntasks=1 --kill-on-bad-exit=1 accelerate launch \
        "${DISTRIBUTED[@]}" --num_processes "$GPU_COUNT" --num_machines 1 \
        --mixed_precision bf16 --dynamo_backend no --num_cpu_threads_per_process 1 "$@"
}
case "$ACTION" in
    runtime)
        launch "$REPO/tests/training_runtime.py" --run-id "$RUN_ID"
        ;;
    validate)
        launch "$REPO/tests/training_runtime.py" --run-id "$RUN_ID"
        launch "$REPO/train.py" --config "$TRAIN_CONFIG" --prep-id "$PREPARATION_ID" \
            --run-id "$RUN_ID" --stop-after-step 1
        launch "$REPO/train.py" --config "$TRAIN_CONFIG" --prep-id "$PREPARATION_ID" \
            --run-id "$RUN_ID" --resume --validate-export
        ;;
    pilot|scaling|resume)
        EXTRA=()
        [[ "$ACTION" != resume ]] || EXTRA+=(--resume)
        launch "$REPO/train.py" --config "$TRAIN_CONFIG" --prep-id "$PREPARATION_ID" \
            --run-id "$RUN_ID" "${EXTRA[@]}"
        ;;
    *) echo 'Invalid training action' >&2; exit 1;;
esac
