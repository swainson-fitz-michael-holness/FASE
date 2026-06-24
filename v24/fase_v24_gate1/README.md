# FASE-v24 Gate 1 bundle

## Files

- `fase_v24_gate1_operator_genesis.py` — standalone engine and matrix runner.
- `test_fase_v24_gate1_smoke.py` — fast deterministic destroyer-clause test.
- `run_v24_gate1_quick.sh` — 192-condition quick field.
- `run_v24_gate1_full.sh` — 1,536-condition full field with checkpoint/resume.
- `V24_GATE_1.md` — formal gate specification and closure criteria.
- `smoke_run/` — locally generated passing smoke report.
- `quick_run/` — locally generated 192-condition report.

## Start

```bash
python test_fase_v24_gate1_smoke.py
```

Then:

```bash
./run_v24_gate1_full.sh
```

Outputs:

```text
runs/v24_gate1_full/v24_gate1_report.json
runs/v24_gate1_full/v24_gate1_rows.csv
runs/v24_gate1_full/v24_gate1_checkpoint.json
```

Interruptions are safe. Re-running the shell script resumes completed conditions.
