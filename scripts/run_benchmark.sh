#!/usr/bin/env bash
set -euo pipefail

python -m rgdaao.train \
  --config configs/esm2_frozen.yaml \
  --data-dir data/processed \
  --output-dir results/esm2_frozen

python -m rgdaao.train \
  --config configs/esm2_finetune.yaml \
  --data-dir data/processed \
  --output-dir results/esm2_finetune
