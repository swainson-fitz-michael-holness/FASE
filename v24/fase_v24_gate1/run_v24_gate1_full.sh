#!/usr/bin/env bash
set -euo pipefail

# Apple Silicon: --workers auto resolves to performance-core count when available.
# The runner checkpoints after every completed condition. Re-run with --resume.
python fase_v24_gate1_operator_genesis.py \
  --matrix full \
  --workers auto \
  --resume \
  --out-dir runs/v24_gate1_full
