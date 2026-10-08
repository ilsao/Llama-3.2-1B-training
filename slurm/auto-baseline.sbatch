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
        launch "$REPO/slurm/training_entry.py" "${COMMON[@]}"
        ;;
    probe8)
        launch "$REPO/tests/fa3_runtime.py"
        launch "$REPO/tests/training_runtime.py" --run-id "$RUN_ID"
        launch "$REPO/slurm/training_entry.py" "${COMMON[@]}" --stop-after-step 1
        launch "$REPO/slurm/training_entry.py" "${COMMON[@]}" --resume --validate-export
        ;;
    lr-low|lr-high|baseline)
        EXTRA=()
        [[ ${6:-new} != resume ]] || EXTRA+=(--resume)
        if [[ "$PHASE" == baseline && ${6:-new} != resume ]]; then
            launch "$REPO/slurm/training_entry.py" "${COMMON[@]}" --pause-after-step 1
            EXTRA+=(--resume)
        fi
        set +e
        launch "$REPO/slurm/training_entry.py" "${COMMON[@]}" "${EXTRA[@]}"
        RESULT=$?
        set -e
        if [[ "$RESULT" == 0 ]]; then
            RESULT=$(python3 - "$JOB_RECORD" "$GPU_COUNT" <<'PYCODE'
import json,sys
from pathlib import Path
codes=[json.loads((Path(sys.argv[1])/f'worker-rank-{i}-exit.json').read_text())['exit_code'] for i in range(int(sys.argv[2]))]
if len(set(codes))!=1: raise SystemExit('Distributed ranks disagreed on training stop status')
print(codes[0])
PYCODE
)
        fi
        [[ "$RESULT" == 0 || "$RESULT" == 75 ]] || exit "$RESULT"
        if [[ "$PHASE" == baseline ]]; then
            STOP_REASON=$(python3 -c 'import json,sys; print(json.load(open(sys.argv[1]))["stop_reason"])' "$DATA_ROOT/runs/experiments/$RUN_ID/summary.json")
            [[ "$STOP_REASON" != data_unavailable ]] || exit 75
            [[ $(date +%s) -lt $((CAMPAIGN_END_EPOCH - 60)) ]] || exit 75
            launch "$REPO/slurm/full_evaluation.py" "$CONFIG" "$RUN_ID" "${3}"
            exit 0
        fi
        exit "$RESULT"
        ;;
    *) echo 'Unknown automatic training phase' >&2; exit 2;;
esac
