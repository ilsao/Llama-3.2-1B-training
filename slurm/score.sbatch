#!/bin/bash
#SBATCH --job-name=dolma-fasttext
#SBATCH --account=ACD115198
#SBATCH --partition=8gpus
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=12
#SBATCH --mem=200G
#SBATCH --gpus-per-node=1
#SBATCH --time=24:00:00
#SBATCH --signal=B:USR1@900
#SBATCH --output=/work/asciibase64/Llama-3.2-1B-training/logs/%x-%j.out
#SBATCH --error=/work/asciibase64/Llama-3.2-1B-training/logs/%x-%j.err
set -euo pipefail
source /home/asciibase64/Llama-3.2-1B-training/slurm/common.sh
PREPARATION_ID=${1:?Supply the same preparation ID as the completed clean stage}
printf '%s\n' "$PREPARATION_ID" > "$JOB_RECORD/preparation-id.txt"
printf '%s\n' score > "$JOB_RECORD/stage.txt"
# fastText is CPU inference; the H200 allocation is the explicitly approved
# resource exception, not a GPU implementation of fastText.
srun --ntasks=1 python "$REPO/dataset.py" score --prep-id "$PREPARATION_ID" --root "$DATA_ROOT" \
    --config "${DATA_CONFIG:-$REPO/configs/dataset.json}"
