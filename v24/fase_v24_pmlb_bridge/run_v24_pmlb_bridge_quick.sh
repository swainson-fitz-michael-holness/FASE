#!/usr/bin/env bash
set -euo pipefail
PYTHON="${PYTHON:-python}"
OUT="${OUT:-runs/v24_pmlb_bridge_529_pollen_quick}"
SEEDS=(42 1337)
FOLDS=(1 2 3)
mkdir -p "$OUT"
for seed in "${SEEDS[@]}"; do
  for fold in "${FOLDS[@]}"; do
    "$PYTHON" fase_v24_pmlb_bridge_529_pollen.py \
      --data-file data/529_pollen.tsv.gz \
      --matrix quick --seeds "$seed" --single-fold "$fold" \
      --resume --out-dir "$OUT"
  done
done

echo "Final report: $OUT/v24_pmlb_bridge_report.json"
echo "Optional Markdown: $PYTHON render_bridge_summary.py $OUT/v24_pmlb_bridge_report.json --out $OUT/V24_PMLB_BRIDGE_RESULT.md"
