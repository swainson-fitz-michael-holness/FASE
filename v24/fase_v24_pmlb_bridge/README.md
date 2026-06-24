# V24-PMLB Bridge Probe on `529_pollen`

This bundle is the terminal conversion experiment for the current FASE research arc.

## Files

- `fase_v24_pmlb_bridge_529_pollen.py` — full bridge engine and report writer.
- `run_v24_pmlb_bridge_full.sh` — resumable 20-fold run with PySR.
- `run_v24_pmlb_bridge_quick.sh` — two-seed quick field without PySR.
- `run_v24_pmlb_bridge_smoke.sh` — two-fold smoke run.
- `test_v24_pmlb_bridge_smoke.py` — executable regression test.
- `render_bridge_summary.py` — converts the final JSON into a manuscript-ready Markdown trajectory note.
- `V24_PMLB_BRIDGE.md` — frozen protocol and interpretation.
- `LOCAL_EVIDENCE.md` — smoke and preliminary quick-run findings, including observed failure boundaries.
- `data/529_pollen.tsv.gz` — offline copy converted from the public POLLEN data.
- v23/v23.1 modules required by the predictive layer.

## Environment

```bash
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

For the PySR baseline:

```bash
pip install pysr
```

The `pmlb` package is optional because the dataset is bundled. The supplied shell runners explicitly use the bundled file for reproducibility; direct runs without `--data-file` may use `pmlb.fetch_data` when available.

## Smoke run

```bash
./run_v24_pmlb_bridge_smoke.sh
```

Expected JSON/CSV outputs (the Markdown summary is generated separately with `render_bridge_summary.py`):

```text
runs/v24_pmlb_bridge_smoke/v24_pmlb_bridge_checkpoint.json
runs/v24_pmlb_bridge_smoke/v24_pmlb_bridge_report.json
runs/v24_pmlb_bridge_smoke/v24_pmlb_bridge_folds.csv
runs/v24_pmlb_bridge_smoke/v24_pmlb_bridge_operators.csv
```

## Full run

```bash
./run_v24_pmlb_bridge_full.sh
```

The shell script runs each `(seed, fold)` in a separate interpreter and resumes from the atomic checkpoint. Re-running the same command skips completed folds.

Environment overrides:

```bash
PYTHON=python3 OUT=runs/my_bridge PYSR_TIMEOUT=240 ./run_v24_pmlb_bridge_full.sh
```

## Direct command

One fold:

```bash
python fase_v24_pmlb_bridge_529_pollen.py \
  --data-file data/529_pollen.tsv.gz \
  --matrix full \
  --seeds 42 \
  --single-fold 1 \
  --run-pysr \
  --resume \
  --out-dir runs/v24_pmlb_bridge_529_pollen
```

Finalize an existing checkpoint:

```bash
python fase_v24_pmlb_bridge_529_pollen.py \
  --data-file data/529_pollen.tsv.gz \
  --matrix full \
  --seeds 42,1337,2025,9001 \
  --aggregate-only \
  --resume \
  --out-dir runs/v24_pmlb_bridge_529_pollen
```

## Important interpretation

`529_pollen` is a synthetic PMLB regression dataset, not a temporal or causal dataset. The probe constructs empirical local transformations and tests whether reusable rewrite schemas survive held-out transfer and shuffled controls. A positive report is an external representational indication, not proof of global generality or causality.
