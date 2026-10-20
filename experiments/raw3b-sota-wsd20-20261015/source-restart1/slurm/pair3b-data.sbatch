#!/bin/bash
#SBATCH --job-name=pair3b-data
#SBATCH --account=ACD115198
#SBATCH --partition=8gpus
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=200G
#SBATCH --gpus-per-node=1
#SBATCH --constraint=H200
#SBATCH --time=16:00:00
#SBATCH --signal=B:USR1@900
#SBATCH --no-requeue
#SBATCH --output=/work/asciibase64/Llama-3.2-1B-training/logs/%x-%j.out
#SBATCH --error=/work/asciibase64/Llama-3.2-1B-training/logs/%x-%j.err
set -euo pipefail
source "${TRAIN_SOURCE_ROOT:?}/slurm/common.sh"
export DATA_ROOT JOB_RECORD
STAGE=${1:?stage}
printf '%s\n' "$STAGE" > "$JOB_RECORD/stage.txt"
printf '%s\n' dolma-dclm10-005-rawmix3b-20261015 > "$JOB_RECORD/preparation-id.txt"
python "${PAIR3B_CONTROLLER:?}" check
if [[ "$STAGE" == rawprep ]]; then
    srun --ntasks=1 python "$REPO/slurm/raw_continue_once_v2.py" "${2:?raw extension authorization}"
else
    [[ "$STAGE" == mixprep ]] || exit 1
    srun --ntasks=1 python "$REPO/mixture_reweight.py" "$DATA_ROOT" dolma-dclm10-005-rawmix3b-20261015
fi
