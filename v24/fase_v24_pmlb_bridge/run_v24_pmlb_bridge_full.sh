#!/usr/bin/env bash
set -euo pipefail

PYTHON="${PYTHON:-python}"
OUT="${OUT:-runs/v24_pmlb_bridge_529_pollen_full}"
PYSR_TIMEOUT="${PYSR_TIMEOUT:-180}"
PYSR_ITERATIONS="${PYSR_ITERATIONS:-1000}"
PYSR_MAXSIZE="${PYSR_MAXSIZE:-30}"
SEEDS=(42 1337 2025 9001)
FOLDS=(1 2 3 4 5)

mkdir -p "$OUT"
for seed in "${SEEDS[@]}"; do
  for fold in "${FOLDS[@]}"; do
    echo "=== 529_pollen seed=$seed fold=$fold ==="
    "$PYTHON" fase_v24_pmlb_bridge_529_pollen.py \
      --data-file data/529_pollen.tsv.gz \
      --matrix full \
      --seeds "$seed" \
      --single-fold "$fold" \
      --run-pysr \
      --pysr-timeout "$PYSR_TIMEOUT" \
      --pysr-iterations "$PYSR_ITERATIONS" \
      --pysr-maxsize "$PYSR_MAXSIZE" \
      --resume \
      --out-dir "$OUT"
  done
done


echo "Final JSON/CSV reports are refreshed by the last fold invocation."
echo "Optional Markdown: $PYTHON render_bridge_summary.py $OUT/v24_pmlb_bridge_report.json --out $OUT/V24_PMLB_BRIDGE_RESULT.md"
