#!/bin/bash
#SBATCH --job-name=dolma-data-env
#SBATCH --account=ACD115198
#SBATCH --partition=8gpus
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=200G
#SBATCH --gpus-per-node=1
#SBATCH --time=24:00:00
#SBATCH --signal=B:USR1@900
#SBATCH --output=/work/asciibase64/Llama-3.2-1B-training/logs/%x-%j.out
#SBATCH --error=/work/asciibase64/Llama-3.2-1B-training/logs/%x-%j.err
set -euo pipefail
SKIP_DATA_ENV=1
source /home/asciibase64/Llama-3.2-1B-training/slurm/common.sh
# Installation/builds execute through srun on the allocated compute node.
if [[ ! -x "$ENV_PREFIX/bin/python" ]]; then
    srun --ntasks=1 /work/envstack/apps/miniconda3/26.1.1/condabin/conda \
        create --yes --prefix "$ENV_PREFIX" python=3.11 pip
fi
INSTALL_REQUIREMENTS="$REPO/requirements-data.txt"
if [[ -f "$ENV_PREFIX/requirements-resolved.txt" ]]; then
    INSTALL_REQUIREMENTS="$ENV_PREFIX/requirements-resolved.txt"
fi
srun --ntasks=1 "$ENV_PREFIX/bin/python" -m pip install \
    --cache-dir "$DATA_ROOT/caches/pip" -r "$INSTALL_REQUIREMENTS"
srun --ntasks=1 "$ENV_PREFIX/bin/python" -m pip check
# Persist all resolved package versions without serializing URLs or credentials.
srun --ntasks=1 "$ENV_PREFIX/bin/python" -c '
import importlib.metadata, pathlib, sys
path = pathlib.Path(sys.argv[1]) / "requirements-resolved.txt"
temporary = path.with_suffix(".partial")
temporary.write_text("".join("{}=={}\n".format(d.metadata["Name"], d.version) for d in sorted(importlib.metadata.distributions(), key=lambda d: d.metadata["Name"].lower())))
temporary.replace(path)
' "$ENV_PREFIX"
