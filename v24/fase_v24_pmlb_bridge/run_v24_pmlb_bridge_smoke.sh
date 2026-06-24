#!/usr/bin/env bash
set -euo pipefail
PYTHON="${PYTHON:-python}"
OUT="${OUT:-runs/v24_pmlb_bridge_smoke}"
mkdir -p "$OUT"
for fold in 1 2; do
  "$PYTHON" fase_v24_pmlb_bridge_529_pollen.py \
    --data-file data/529_pollen.tsv.gz \
    --matrix smoke --seeds 42 --single-fold "$fold" \
    --resume --out-dir "$OUT"
done

echo "Final report: $OUT/v24_pmlb_bridge_report.json"
echo "Optional Markdown: $PYTHON render_bridge_summary.py $OUT/v24_pmlb_bridge_report.json --out $OUT/V24_PMLB_BRIDGE_RESULT.md"
