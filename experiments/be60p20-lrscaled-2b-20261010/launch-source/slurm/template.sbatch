#!/bin/bash
#SBATCH --account=ACD115198
#SBATCH --partition=8gpus
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --gpus-per-node=1
#SBATCH --time=4:00:00


source $HOME/.bashrc

echo "========================================"
echo "Start Time: $(date '+%Y-%m-%d %H:%M:%S')"
echo "========================================"
START_TIME=$(date +%s)


END_TIME=$(date +%s)
echo "========================================"
echo "End Time: $(date '+%Y-%m-%d %H:%M:%S')  "
echo "Total Cost: $((END_TIME - START_TIME)) s"
echo "========================================"
