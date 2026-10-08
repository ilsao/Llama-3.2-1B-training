#!/bin/bash
#SBATCH --job-name=llama-auto
#SBATCH --account=ACD115198
#SBATCH --partition=8gpus
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=200G
#SBATCH --gpus-per-node=8
#SBATCH --time=01:20:00
#SBATCH --signal=B:USR1@180
#SBATCH --no-requeue
#SBATCH --output=/work/asciibase64/Llama-3.2-1B-training/logs/%x-%j.out
#SBATCH --error=/work/asciibase64/Llama-3.2-1B-training/logs/%x-%j.err
set -euo pipefail
PHASE=${2:?Supply phase}
export TRAIN_ENV_PREFIX=/work/asciibase64/Llama-3.2-1B-training/environments/training-fa3-py311-94e22c9
[[ "$PHASE" != setup && "$PHASE" != raw-prepare ]] || SETUP_TRAINING_ENV=1
source /home/asciibase64/Llama-3.2-1B-training/slurm/training-common.sh
export DATA_ROOT TRAIN_ENV_PREFIX JOB_RECORD
export TORCH_EXTENSIONS_DIR="$DATA_ROOT/caches/torch-extensions"
CONFIG=${4:?Supply resolved phase configuration}
CAMPAIGN_CONFIG=${5:?Supply campaign configuration}
# A queued job must not run training past the absolute delivery cutoff.
export CAMPAIGN_END_EPOCH
CAMPAIGN_END_EPOCH=$(python3 "$REPO/slurm/auto_baseline.py" guard --config "$CAMPAIGN_CONFIG" --job "$SLURM_JOB_ID")
[[ $(date +%s) -lt "$CAMPAIGN_END_EPOCH" ]] || exit 2
if [[ "$PHASE" == setup ]]; then
    [[ $(type -t module || true) == function ]] || source /etc/profile.d/lmod.sh
    module load cuda/12.6
    srun --ntasks=1 "$DATA_ROOT/environments/training-py311/bin/python" "$REPO/slurm/fa3_setup.py"
    exit
fi
if [[ "$PHASE" == raw-prepare ]]; then
    export PREP_STOP_EPOCH=$((TRAIN_END_EPOCH - 180))
    printf '%s\n' raw-prepare > "$JOB_RECORD/stage.txt"
    printf '%s\n' "${3}" > "$JOB_RECORD/preparation-id.txt"
    SOURCE_ID=$(python3 -c 'import json,sys; print(json.load(open(sys.argv[1]))["preparation_id"])' "$CAMPAIGN_CONFIG")
    srun --ntasks=1 "$DATA_ROOT/environments/training-py311/bin/python" "$REPO/raw_baseline.py" \
        --root "$DATA_ROOT" --source-id "$SOURCE_ID" --prep-id "${3}"
    exit
fi
launch() {
    local seconds=$((CAMPAIGN_END_EPOCH - $(date +%s)))
    [[ "$seconds" -gt 0 ]] || return 124
    timeout --signal=TERM --kill-after=15s "$seconds" \
        srun --ntasks=1 --kill-on-bad-exit=1 accelerate launch --multi_gpu \
        --main_process_port "$((15000 + SLURM_JOB_ID % 40000))" \
        --num_processes "$(wc -l < "$JOB_RECORD/gpus.txt")" --num_machines 1 \
        --mixed_precision bf16 --dynamo_backend no --num_cpu_threads_per_process 1 "$@"
}
COMMON=(--config "$CONFIG" --prep-id "${3:?Supply preparation ID}" --run-id "$RUN_ID")
case "$PHASE" in
    probe4)
        launch "$REPO/train.py" "${COMMON[@]}"
        ;;
    probe8)
        launch "$REPO/tests/fa3_runtime.py"
        launch "$REPO/tests/training_runtime.py" --run-id "$RUN_ID"
        launch "$REPO/train.py" "${COMMON[@]}" --stop-after-step 1
        launch "$REPO/train.py" "${COMMON[@]}" --resume --validate-export
        ;;
    lr-low|lr-high|baseline)
        EXTRA=()
        [[ ${6:-new} != resume ]] || EXTRA+=(--resume)
        set +e
        launch "$REPO/train.py" "${COMMON[@]}" "${EXTRA[@]}"
        RESULT=$?
        set -e
        [[ "$RESULT" == 0 || "$RESULT" == 75 ]] || exit "$RESULT"
        if [[ "$PHASE" == baseline ]]; then
            [[ $(date +%s) -lt $((CAMPAIGN_END_EPOCH - 60)) ]] || exit 75
            launch "$REPO/slurm/full_evaluation.py" "$CONFIG" "$RUN_ID" "${3}"
            exit 0
        fi
        exit "$RESULT"
        ;;
    *) echo 'Unknown automatic training phase' >&2; exit 2;;
esac
