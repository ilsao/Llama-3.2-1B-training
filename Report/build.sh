#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")"

# This is a small CPU-only document build, never an experiment launcher.
REPORT_TOOLS_ROOT="${REPORT_TOOLS_ROOT:-/work/asciibase64/Llama-3.2-1B-training/report-tools-20261021}"
REPORT_PYTHON_EXPLICIT="${REPORT_PYTHON:-}"
REPORT_PYTHON="${REPORT_PYTHON:-python3}"
if [[ -d "$REPORT_TOOLS_ROOT/python" ]]; then
  export PYTHONPATH="$REPORT_TOOLS_ROOT/python${PYTHONPATH:+:$PYTHONPATH}"
  export MPLCONFIGDIR="$REPORT_TOOLS_ROOT/mpl-cache"
  export XDG_CACHE_HOME="$REPORT_TOOLS_ROOT/cache"
  if [[ -z "$REPORT_PYTHON_EXPLICIT" && -x /work/asciibase64/Llama-3.2-1B-training/environments/training-py311/bin/python ]]; then
    REPORT_PYTHON=/work/asciibase64/Llama-3.2-1B-training/environments/training-py311/bin/python
  fi
fi
export OPENBLAS_NUM_THREADS=1
export OMP_NUM_THREADS=1
"$REPORT_PYTHON" generate_tables.py
"$REPORT_PYTHON" figures/gen_fig_report.py
if command -v latexmk >/dev/null 2>&1; then
  latexmk -pdf -interaction=nonstopmode -halt-on-error main.tex
elif command -v tectonic >/dev/null 2>&1; then
  tectonic --keep-logs --keep-intermediates main.tex
elif [[ -x "$REPORT_TOOLS_ROOT/tectonic" ]]; then
  "$REPORT_TOOLS_ROOT/tectonic" --keep-logs --keep-intermediates main.tex
else
  echo 'Install latexmk with a TeX distribution or provide Tectonic.' >&2
  exit 1
fi
"$REPORT_PYTHON" verify_report.py
