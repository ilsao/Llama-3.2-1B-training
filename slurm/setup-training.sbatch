#!/bin/bash
#SBATCH --job-name=llama-setup
#SBATCH --account=ACD115198
#SBATCH --partition=8gpus
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=200G
#SBATCH --gpus-per-node=1
#SBATCH --time=01:00:00
#SBATCH --no-requeue
#SBATCH --output=/work/asciibase64/Llama-3.2-1B-training/logs/%x-%j.out
#SBATCH --error=/work/asciibase64/Llama-3.2-1B-training/logs/%x-%j.err
set -euo pipefail
SETUP_TRAINING_ENV=1
source /home/asciibase64/Llama-3.2-1B-training/slurm/training-common.sh
if [[ ! -x "$TRAIN_ENV_PREFIX/bin/python" ]]; then
    srun --ntasks=1 /work/envstack/apps/miniconda3/26.1.1/condabin/conda \
        create --yes --prefix "$TRAIN_ENV_PREFIX" python=3.11 pip
fi
INSTALL_REQUIREMENTS="$REPO/requirements-training.txt"
if [[ -f "$TRAIN_ENV_PREFIX/requirements-resolved.txt" ]]; then
    INSTALL_REQUIREMENTS="$TRAIN_ENV_PREFIX/requirements-resolved.txt"
fi
srun --ntasks=1 "$TRAIN_ENV_PREFIX/bin/python" -m pip install \
    --cache-dir "$DATA_ROOT/caches/pip" -r "$INSTALL_REQUIREMENTS"
srun --ntasks=1 "$TRAIN_ENV_PREFIX/bin/python" -m pip check
srun --ntasks=1 "$TRAIN_ENV_PREFIX/bin/python" -c '
import importlib.metadata, pathlib, sys
path = pathlib.Path(sys.argv[1]) / "requirements-resolved.txt"
temporary = path.with_suffix(".partial")
temporary.write_text("".join("{}=={}\n".format(d.metadata["Name"], d.version) for d in sorted(importlib.metadata.distributions(), key=lambda d: d.metadata["Name"].lower())))
temporary.replace(path)
' "$TRAIN_ENV_PREFIX"
