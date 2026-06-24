#!/usr/bin/env bash
set -euo pipefail
python fase_v24_gate1_operator_genesis.py \
  --matrix quick \
  --workers auto \
  --resume \
  --out-dir runs/v24_gate1_quick
