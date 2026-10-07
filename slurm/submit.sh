#!/bin/bash
# Lightweight login-node submission wrapper. No automatic downstream chaining:
# exit 75 means resubmit the same stage before moving to the next stage.
set -euo pipefail
REPO=/home/asciibase64/Llama-3.2-1B-training
DATA_ROOT=${DATA_ROOT:-/work/asciibase64/Llama-3.2-1B-training}
[[ "$DATA_ROOT" == /work/asciibase64/* ]] || { echo 'Invalid DATA_ROOT' >&2; exit 1; }
STAGE=${1:?Supply setup or a dataset.py stage}
PREPARATION_ID=${2:-}
if [[ "$STAGE" != setup ]]; then
    [[ "$PREPARATION_ID" =~ ^[A-Za-z0-9][A-Za-z0-9_.-]{0,95}$ ]] || { echo 'Supply a valid preparation ID' >&2; exit 1; }
fi
mkdir -p "$DATA_ROOT/logs"
SUBMIT_ARGS=(--parsable --output="$DATA_ROOT/logs/%x-%j.out" --error="$DATA_ROOT/logs/%x-%j.err")
if [[ "$STAGE" == setup ]]; then
    JOB_ID=$(sbatch "${SUBMIT_ARGS[@]}" "$REPO/slurm/setup.sbatch")
elif [[ "$STAGE" == score ]]; then
    JOB_ID=$(sbatch "${SUBMIT_ARGS[@]}" "$REPO/slurm/score.sbatch" "$PREPARATION_ID")
else
    case "$STAGE" in download|split|clean|select|pack|verify|smoke) ;; *) echo 'Invalid stage' >&2; exit 1;; esac
    JOB_ID=$(sbatch "${SUBMIT_ARGS[@]}" "$REPO/slurm/prepare.sbatch" "$STAGE" "$PREPARATION_ID")
fi
printf '%s\t%s\t%s\t%s\n' "$(date -u +%FT%TZ)" "$JOB_ID" "$STAGE" "$PREPARATION_ID" >> "$DATA_ROOT/logs/submissions.tsv"
printf 'Submitted job %s (%s, preparation=%s)\n' "$JOB_ID" "$STAGE" "$PREPARATION_ID"
