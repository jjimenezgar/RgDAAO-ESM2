#!/usr/bin/env bash
set -euo pipefail
export PYTHONHASHSEED=42
export CUBLAS_WORKSPACE_CONFIG=:4096:8
OUTPUT_ROOT="${1:-results}"
python - <<'PY'
from rgdaao.config import load_config, check_comparison
check_comparison(load_config('configs/esm2_frozen.yaml'), load_config('configs/esm2_finetune.yaml'))
PY
# Both configurations are fixed and both checkpoints selected before test access.
python -m rgdaao.train --config configs/esm2_frozen.yaml --output-dir "$OUTPUT_ROOT/esm2_frozen"
python -m rgdaao.train --config configs/esm2_finetune.yaml --output-dir "$OUTPUT_ROOT/esm2_finetune"
python -m rgdaao.evaluate --run-dir "$OUTPUT_ROOT/esm2_frozen"
python -m rgdaao.evaluate --run-dir "$OUTPUT_ROOT/esm2_finetune"
python scripts/plot_results.py --results-dir "$OUTPUT_ROOT"
