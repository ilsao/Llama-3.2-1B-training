#!/bin/bash
# Lightweight durable submission wrapper; no automatic downstream chaining.
set -euo pipefail
exec python3 /home/asciibase64/Llama-3.2-1B-training/slurm/submit_preprocessing.py "$@"
