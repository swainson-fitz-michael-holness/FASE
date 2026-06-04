# FASE Quickstart Fixture

This directory contains committed tiny fixtures for documentation, CLI smoke
checks, and artifact validation.

Law:

```text
y = 1.0 + 2.0*x0 - 0.5*x1 + 0.25*x2^2
```

Files:

- `train.csv`: numeric training rows with target column `y`.
- `predict.csv`: held-out rows with `y` included for evaluation.
- `model.json`: hand-authored `FASEModel.to_dict` wrapper using raw `x0`,
  raw `x1`, and `x2^2` stage-1 features.

Run:

```bash
fase validate-model --model examples/quickstart/model.json
fase predict --model examples/quickstart/model.json --csv examples/quickstart/predict.csv --drop-column y --output outputs/example_predictions.csv --strict-schema
fase eval-model --model examples/quickstart/model.json --csv examples/quickstart/predict.csv --target y --output outputs/example_eval.json --strict-schema
```
