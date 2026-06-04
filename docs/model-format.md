# FASEModel JSON Format

This document defines the product-level JSON wrapper used by `fase export-model`
and `fase fit --model-output`. It describes the stable artifact contract around
`FASEModel.to_dict`; it does not guarantee every internal research block can be
serialized.

## Wrapper Object

New exported model files are JSON objects with this top-level shape:

```json
{
  "format": "FASEModel.to_dict",
  "schema_version": "fase-model-v1",
  "version": "0.1.0",
  "feature_schema": {
    "feature_names": ["x0", "x1"],
    "feature_count": 2,
    "target": "y"
  },
  "model": {
    "w": [1.0],
    "b0": 0.0,
    "stage1": [],
    "stage2": []
  }
}
```

The committed fixture at `examples/quickstart/model.json` is the smallest
checked example of this wrapper format. Installed packages can copy the same
fixture with `fase copy-examples --output outputs/fase_quickstart_fixture`.

Required top-level fields:

- `format`: must be `FASEModel.to_dict`.
- `schema_version`: current value is `fase-model-v1`.
- `version`: FASE package wrapper version that produced the export.
- `model`: serialized `FASEModel.to_dict` payload.

Optional top-level fields:

- `feature_schema`: present when export was created from CSV metadata.

## Model Payload

The `model` object is produced by `FASEModel.to_dict`. Product tooling expects:

- `w`: list of numeric linear weights.
- `b0`: numeric intercept.
- `stage1`: list of first-stage feature specs.
- `stage2`: list of second-stage blocks.

Stage 2 blocks should include:

- `kind`: non-empty block kind string.
- `mus` and `sds`: normalization vectors when present.
- `params`: block-specific parameters when present.

Supported reload paths are determined by `FASEModel.from_dict` in `FASE_v21.py`.
Current smoke tests cover bare/intercept models, ruliad hypergraph-state blocks,
closure-captured ruliad states, and group-invariant grammar blocks.

## Feature Schema

`feature_schema` is metadata for prediction-time column checks:

```json
{
  "feature_names": ["x0", "x1"],
  "feature_count": 2,
  "target": "y"
}
```

Rules:

- `feature_names` must be a list of strings.
- `feature_count` must equal `len(feature_names)`.
- `target` is optional and, when present, must be a string.

`fase predict` and `fase eval-model` do not reorder columns. They warn when the
CSV column order or names differ from `feature_schema`; pass `--strict-schema`
to fail instead of warning.

## Compatibility

Older bare `FASEModel.to_dict` JSON files without the wrapper remain readable by
`fase inspect-model`, `fase predict`, and `fase eval-model` when the internal
model payload can be reloaded. They are reported with a compatibility warning
because they lack `format`, `schema_version`, package `version`, and
`feature_schema` metadata.

## Validation

Use `fase validate-model --model model.json --json` when you only need the
structural validation result. It exits `0` for valid artifacts and `1` for
invalid artifacts.

Use `fase inspect-model --model model.json --json` when you also need metadata
such as weight counts and block kinds. The JSON output includes
`schema_validation`:

```json
{
  "status": "ok",
  "schema_version": "fase-model-v1",
  "format": "FASEModel.to_dict",
  "wrapped": true,
  "errors": [],
  "warnings": []
}
```

Validation is structural. A model can pass wrapper validation and still fail
reload if it contains an unsupported research block. Treat reload/predict smoke
tests as the stronger compatibility check.

`fase fit --model-output` and `fase export-model` also attach
`model_export.preflight` to their fit reports. Preflight diagnostics inspect
stage-2 blocks before writing JSON and report unsupported block indices, kinds,
and missing parameters when the issue can be detected without executing a fit.
`fase report` surfaces these diagnostics in the terminal summary for fit
reports.
