#!/bin/bash
# User approved this exact fourth formal run, 4h/32 H200-hours on 2026-10-10.
# Submit through the shared-ledger wrapper; never run on login.
#SBATCH --job-name=be60p20-lrscaled-2b
#SBATCH --account=ACD115198
#SBATCH --partition=8gpus
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=200G
#SBATCH --gpus-per-node=8
#SBATCH --time=04:00:00
#SBATCH --nodelist=25a-hgpn078
#SBATCH --signal=B:USR1@300
#SBATCH --no-requeue
#SBATCH --output=/work/asciibase64/Llama-3.2-1B-training/logs/%x-%j.out
#SBATCH --error=/work/asciibase64/Llama-3.2-1B-training/logs/%x-%j.err
set -euo pipefail
source "${TRAIN_SOURCE_ROOT:?Submit from the sealed source snapshot}/slurm/training-common.sh"
PREPARATION_ID=${2:?Supply parent raw preparation}
TRAIN_CONFIG=${3:?Supply sealed configuration}
export JOB_RECORD
[[ "$RUN_ID" == be60p20-lrscaled-2b-20261010 && "$PREPARATION_ID" == dolma-raw-baseline-20261008 ]] || exit 1
python - "$REPO" "$TRAIN_CONFIG" <<'PY'
import sys
sys.path.insert(0, sys.argv[1])
from training_state import digest, load_config
if digest(load_config(sys.argv[2])) != 'c2ec561c2bae73988d292fa9fe1bc7093265a3c1a6f401930dceedea40da46bf':
    raise SystemExit('Configuration does not match the proposed BE60P20 2B run')
PY
srun --ntasks=1 --kill-on-bad-exit=1 accelerate launch --multi_gpu \
    --main_process_port "$((15000 + SLURM_JOB_ID % 40000))" \
    --num_processes 8 --num_machines 1 --mixed_precision bf16 --dynamo_backend no \
    --num_cpu_threads_per_process 1 "$REPO/slurm/training_entry.py" \
    --config "$TRAIN_CONFIG" --prep-id "$PREPARATION_ID" --run-id "$RUN_ID" --validate-export
python - "$DATA_ROOT" "$RUN_ID" <<'PY'
import json, sys
from pathlib import Path
s = json.loads((Path(sys.argv[1]) / 'runs/experiments' / sys.argv[2] / 'summary.json').read_text())
if not (s['training_token_target_reached'] and s['global_step'] == s['attempted_updates'] == s['last_eval_step'] == 15259
        and s['total_tokens_seen'] == s['budget_tokens_charged'] == 2000027648
        and s['export_global_step'] == 15259 and s['initialization']['pretrained_weights_loaded'] is False):
    raise SystemExit('Allocation ended without the complete 2B endpoint; no automatic continuation')
print('Complete 2B-token BE60P20 endpoint and HF export verified.')
PY
