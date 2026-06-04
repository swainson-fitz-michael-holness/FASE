# FASE

FASE is a symbolic-regression and lab-harness toolkit for discovering compact,
auditable models from noisy data. The current product surface packages the FASE
v21 engine behind a Python API and CLI, while keeping the research scripts
available for direct experimentation.

The fastest product path is:

1. Use FASE v21 as the stable symbolic-regression engine.
2. Use the CLI for repeatable demos, reports, and lab harness runs.
3. Treat v7 regime-first control as a research harness until its agreement and
   compactness gates close.

## Current Product Status

Ready for use now:

- Installable Python package metadata through `pyproject.toml`.
- `fase` CLI entry point.
- Synthetic symbolic-regression demo command.
- Committed quickstart fixtures under `examples/quickstart` and packaged
  wheel fixtures through `fase copy-examples`.
- v7 lab harness command wrapper.
- JSON report summarizer.
- Deterministic quickstart dataset generator.
- Fast smoke tests for package and CLI behavior.
- Generated-output ignore rules for future runs.

Still research-grade:

- The v7 controller policy is not ready to ship as an autonomous controller.
- The latest checked v7 run failed the compactness gate for both teacher modes.
- FASE model serialization supports ruliad hypergraph-state blocks and
  closure-captured ruliad states, but not every opaque closure-backed block
  shape yet.

Latest observed v7 metrics from `v7_lab_outputs/v7_lab_summary.json`:

| Mode | Regime Accuracy | Exact Agreement | Avg Hamming | Template Purity | Gate |
| --- | ---: | ---: | ---: | ---: | --- |
| `v6_2_turbo` | 0.5647 | 0.5492 | 0.5231 | 0.9245 | fail |
| `v6_3_stock` | 0.3953 | 0.3847 | 0.8769 | 0.9099 | fail |

This is enough for a credible lab/demo product. It is not enough for a safety
claim around production control.

## Repository Layout

```text
.
├── FASE_v21.py                         # Current core symbolic-regression engine
├── FASE.py                             # Older v20-era implementation
├── FASE_Paper.md                       # Research/theory notes
├── fase/
│   ├── __init__.py                     # Product-facing package exports
│   ├── api.py                          # Stable wrapper around FASE_v21
│   └── cli.py                          # Command-line interface
├── pre_v7_fase_v21_distillation_pipeline.py
│                                       # Pre-v7 bitwise distillation harness
├── v7_regime_first_lab_harness.py      # v7 regime-first lab harness
├── tests/
│   └── test_cli_smoke.py               # Fast CLI smoke tests
├── pyproject.toml                      # Install/build metadata
├── requirements.txt                    # Legacy broad dependency list
└── README.md                           # This guide
```

Generated run directories are intentionally ignored for future work:

```text
outputs/
pre_v7_outputs/
v7_lab_outputs/
v7_lab_outputs_*/
```

Some historical generated files may already be tracked or present locally. Do
not delete them casually if they are useful for result comparison.

## Committed Quickstart Fixture

Use `examples/quickstart` when you want a runnable example without generating
files first.

```bash
fase validate-model --model examples/quickstart/model.json
fase predict \
  --model examples/quickstart/model.json \
  --csv examples/quickstart/predict.csv \
  --drop-column y \
  --output outputs/example_predictions.csv \
  --strict-schema
fase eval-model \
  --model examples/quickstart/model.json \
  --csv examples/quickstart/predict.csv \
  --target y \
  --output outputs/example_eval.json \
  --strict-schema
```

The fixture follows:

```text
y = 1.0 + 2.0*x0 - 0.5*x1 + 0.25*x2^2
```

Use `fase copy-examples` when you want to copy the packaged fixture out of an
installed wheel, and use `fase init-example` when you want to generate a fresh
synthetic quickstart dataset into `outputs/`.

## Installation

Use Python 3.10 or newer.

### Minimal Product Install

This install is enough for the package, CLI, synthetic demo, and report tools.

```bash
cd /Users/swainsonholness/dev/FASE
python3 -m venv venv
source venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e .
```

Check that the CLI is visible:

```bash
fase version
```

Expected shape:

```text
FASE 0.1.0
```

### Development Install

Use this when changing the package or running tests.

```bash
cd /Users/swainsonholness/dev/FASE
source venv/bin/activate
python -m pip install -e ".[dev]"
python -m unittest discover -s tests
```

### Optional Research Dependencies

The core product wrapper only requires `numpy`. The broader research scripts
may need optional dependencies.

Install PMLB demos and plotting:

```bash
python -m pip install -e ".[demo]"
```

Install PySR baseline support:

```bash
python -m pip install -e ".[baseline]"
```

Install the broad legacy/research stack:

```bash
python -m pip install -e ".[legacy]"
```

Install everything:

```bash
python -m pip install -e ".[all]"
```

## CLI Quickstart

The CLI command is `fase`.

```bash
fase --help
```

Available commands:

```text
fase version
fase status
fase check
fase doctor
fase clean
fase release-check
fase release-notes
fase release-bundle
fase artifact-check
fase artifact-manifest
fase init-example
fase copy-examples
fase demo
fase fit
fase export-model
fase inspect-model
fase validate-model
fase compare-models
fase predict
fase eval-model
fase v7-lab
fase report
```

## Command: `fase version`

Print the installed product wrapper version.

```bash
fase version
```

Use this as the first sanity check after installation.

## Command: `fase status`

Summarize product readiness without writing files or running a fit.

```bash
fase status
fase status --json
fase status --strict --json
fase status --strict --require-clean --json
fase status --output outputs/fase_status.json
```

The command reports:

- release-check status
- latest wheel/sdist artifact status
- artifact count
- git branch, commit, and dirty-state summary

By default this command is read-only and exits zero even when attention is
needed, so it can be used as a dashboard during development. Use `--strict` in
release automation when missing artifacts or failed release checks should fail
the command.

Use `--require-clean` for final release gates that must fail if the source tree
is dirty or git provenance is unavailable.

## Command: `fase check`

Run product smoke checks against the installed package.

Default check:

```bash
fase check
```

Deeper check with a real fast fit/export/predict round trip:

```bash
fase check --fit
```

The default check verifies:

- `FASE_v21` imports and exposes `run_fase_kfold`.
- Numeric CSV parsing works.
- Exported model JSON can be loaded.
- Prediction CSV output can be written.

`--fit` additionally trains on a tiny numeric CSV, exports a model, reloads it,
and writes predictions. It remains small enough for product smoke testing, but
it is slower than the default check.

## Command: `fase doctor`

Inspect the local runtime environment without launching a fit.

Human-readable diagnostics:

```bash
fase doctor
```

JSON diagnostics for support logs or automation:

```bash
fase doctor --json
fase doctor --output outputs/fase_doctor.json
```

The command reports:

- Python version and executable.
- Required package/module availability.
- Optional package availability for common data-science integrations.
- Platform and current working directory metadata.

Unlike `fase check`, this command does not train, export, or predict. Use it
when packaging or support needs to distinguish install/runtime issues from
model-fitting issues.

## Command: `fase clean`

Preview or remove generated FASE output artifacts.

Dry run is the default:

```bash
fase clean
```

Clean a specific workspace root:

```bash
fase clean --root /Users/swainsonholness/dev/FASE
```

Actually remove generated outputs:

```bash
fase clean --yes
```

JSON report:

```bash
fase clean --json
fase clean --yes --json
```

Only target stale generated outputs whose newest file is at least seven days
old:

```bash
fase clean --older-than-days 7
fase clean --older-than-days 7 --yes
```

The command targets generated artifacts only:

- `outputs/`
- `build/`
- `dist/`
- `pre_v7_outputs/`
- `v7_lab_outputs/`
- `v7_lab_outputs_*/`
- `*.egg-info/`
- `*.pkl.tmp`
- `*.pid`
- `__pycache__/`

Virtualenv, VCS, and dependency directories such as `venv/`, `.venv/`, `.git/`,
and `node_modules/` are skipped. Use the default dry run before passing `--yes`.

## Command: `fase release-check`

Validate local release/package readiness without building or running heavy
research harnesses.

Human-readable check:

```bash
fase release-check
```

JSON check for automation:

```bash
fase release-check --json
fase release-check --output outputs/fase_release_check.json
```

The command verifies:

- Required package files exist.
- `pyproject.toml` declares expected package metadata.
- Package version and pyproject version match.
- Classifiers, optional dependency groups, and quickstart package data are declared.
- `MANIFEST.in` excludes generated outputs, checkpoint temps, and PID files.
- README and changelog release sections exist.
- Expected CLI commands are registered.

This is intentionally lighter than a full build. Use it before a release
branch, then run the test suite and optional build tooling.

Full local build check:

```bash
python -m build --sdist --wheel
```

## Command: `fase release-notes`

Print compact release handoff notes without running a build or full research
harness.

Human-readable notes:

```bash
fase release-notes
```

JSON notes for automation or packaging records:

```bash
fase release-notes --json
fase release-notes --output outputs/fase_release_notes.json
```

The command reports:

- Package name, version, root, and release-check status.
- Newest local wheel and source distribution paths when present.
- Current product capabilities and known limitations.
- Registered CLI command list expected for the package shell.
- Recommended verification commands for a local release handoff.

Use this after `fase release-check` and before sharing a build artifact or
packaging status update.

## Command: `fase release-bundle`

Write release handoff JSON files into one directory.

```bash
fase release-bundle
fase release-bundle --json
fase release-bundle --output-dir outputs/fase_release_bundle
fase release-bundle --strict --json
fase release-bundle --strict --require-clean --json
```

The bundle contains:

- `release_bundle.json`: top-level status and file index.
- `release_notes.json`: the same handoff payload as `fase release-notes`.
- `artifact_manifest.json`: the latest wheel/sdist manifest by default.

`release_bundle.json` also records git provenance: branch, commit SHA, short
commit, dirty flag, and a compact dirty-status summary. This makes downloaded
CI artifacts traceable to the exact source state that built them.

By default the command writes the bundle even when artifacts are missing, so it
can be used before or after a build. Use `--strict` after building artifacts to
fail unless release checks pass and the artifact manifest is present.

Use `--require-clean` for release CI or final handoff checks that must fail if
the source tree is dirty or git provenance is unavailable. Local product-smoke
runs do not enable this by default because development worktrees are often
intentionally dirty.

## Command: `fase artifact-check`

Install a built wheel into a temporary environment and run installed CLI smoke
checks.

Validate the newest wheel in `dist/`:

```bash
fase artifact-check
```

Validate a specific wheel:

```bash
fase artifact-check --wheel dist/fase_symbolic-0.1.0-py3-none-any.whl
```

Preview without creating the temporary environment:

```bash
fase artifact-check --dry-run
```

By default the temporary environment is created with system site packages and
the wheel is installed with `--no-deps`. This keeps the check fast and avoids a
network dependency while still validating that the built wheel exposes the
installed `fase` command, can run `fase check` and `fase release-notes`, can
write the deterministic quickstart example with `fase init-example`, and can
copy packaged quickstart fixtures with `fase copy-examples`. It also verifies
installed exported-model inspection and validation with `fase inspect-model`
and `fase validate-model`.

Use `--isolated-deps` only when you intend to install dependencies separately
or are validating in an environment where dependency resolution is already
available.

## Command: `fase artifact-manifest`

Summarize built wheel and source distribution artifacts without installing
them. The manifest includes paths, sizes, SHA-256 hashes, archive member
counts, packaged quickstart fixture files, and a completeness check for the
expected quickstart members. It also validates lightweight artifact metadata,
including package name/version and the `fase` console script entry point.

```bash
fase artifact-manifest
fase artifact-manifest --json
fase artifact-manifest --latest-only --json
fase artifact-manifest --latest-only --output outputs/fase_artifact_manifest.json
```

Use `--latest-only` when `dist/` contains stale artifacts from older builds; it keeps
the newest wheel and newest source distribution without deleting any files.

Inspect specific artifacts:

```bash
fase artifact-manifest \
  --artifact dist/fase_symbolic-0.1.0-py3-none-any.whl \
  --artifact dist/fase_symbolic-0.1.0.tar.gz
```

## Command: `fase init-example`

Write a deterministic numeric quickstart dataset into a local directory.

```bash
fase init-example --output outputs/fase_quickstart
```

The command writes:

- `train.csv`: numeric training rows with target column `y`.
- `predict.csv`: held-out numeric rows with `y` included for comparison.
- `README.md`: local commands for fitting and predicting from that directory.

The generated law is:

```text
y = 1.0 + 2.0*x0 - 0.5*x1 + 0.25*x2^2
```

Run the full local quickstart:

```bash
fase init-example --output outputs/fase_quickstart --force
fase fit \
  --csv outputs/fase_quickstart/train.csv \
  --target y \
  --dry-run \
  --fast
fase export-model \
  --csv outputs/fase_quickstart/train.csv \
  --target y \
  --fast \
  --output outputs/fase_quickstart/model.json \
  --report-output outputs/fase_quickstart/report.json
fase predict \
  --model outputs/fase_quickstart/model.json \
  --csv outputs/fase_quickstart/predict.csv \
  --drop-column y \
  --output outputs/fase_quickstart/predictions.csv
fase eval-model \
  --model outputs/fase_quickstart/model.json \
  --csv outputs/fase_quickstart/predict.csv \
  --target y \
  --output outputs/fase_quickstart/eval.json
```

By default, the command refuses to overwrite existing quickstart files. Pass
`--force` only when you intend to replace them.

## Command: `fase copy-examples`

Copy the packaged quickstart fixture from the installed `fase` package into a
local directory. This is the wheel-safe path for users who installed FASE
without cloning the repository.

```bash
fase copy-examples --output outputs/fase_quickstart_fixture
```

The command writes:

- `README.md`: fixture notes and runnable commands.
- `train.csv`: tiny training fixture.
- `predict.csv`: held-out rows with `y` included for evaluation.
- `model.json`: validated `FASEModel.to_dict` wrapper for the fixture law.

Run the copied fixture:

```bash
fase copy-examples --output outputs/fase_quickstart_fixture --force
fase validate-model --model outputs/fase_quickstart_fixture/model.json
fase predict \
  --model outputs/fase_quickstart_fixture/model.json \
  --csv outputs/fase_quickstart_fixture/predict.csv \
  --drop-column y \
  --output outputs/fase_quickstart_fixture/predictions.csv \
  --strict-schema
fase eval-model \
  --model outputs/fase_quickstart_fixture/model.json \
  --csv outputs/fase_quickstart_fixture/predict.csv \
  --target y \
  --output outputs/fase_quickstart_fixture/eval.json \
  --strict-schema
```

By default, the command refuses to overwrite existing fixture files. Pass
`--force` only when replacement is intentional.

## Command: `fase demo`

Run a synthetic FASE v21 symbolic-regression demo and write a compact JSON
report.

Fast smoke demo:

```bash
fase demo --fast --output outputs/fase_demo_report.json
```

Custom synthetic demo:

```bash
fase demo \
  --n 300 \
  --d 8 \
  --k-folds 3 \
  --seed 1337 \
  --output outputs/demo_300x8.json
```

Enable PySR baseline if installed:

```bash
fase demo \
  --n 300 \
  --d 8 \
  --k-folds 3 \
  --compare-pysr \
  --output outputs/demo_with_pysr.json
```

Options:

| Option | Default | Meaning |
| --- | ---: | --- |
| `--output` | `outputs/fase_demo_report.json` | Compact JSON report path |
| `--n` | `120` | Synthetic sample count |
| `--d` | `6` | Synthetic feature count |
| `--k-folds` | `2` | Cross-validation folds |
| `--seed` | `42` | Random seed |
| `--fast` | off | Use small atoms, no grammar, no ruliad, no PySR |
| `--compare-pysr` | off | Run PySR baseline if dependency is installed |
| `--no-gls-noise` | off | Disable synthetic heteroskedastic noise |

Output JSON shape:

```json
{
  "R2_oof": 0.93,
  "R2_oof_gls": 0.93,
  "MSE_oof": 0.06,
  "num_consensus_ops": 4,
  "consensus_ops": ["og:tanh[x0]"],
  "og_stability": {},
  "og_min_bits": {},
  "folds": []
}
```

Interpretation:

- `R2_oof` is out-of-fold predictive fit.
- `R2_oof_gls` is the sigma-aware version when `Sigma` is supplied.
- `MSE_oof` is out-of-fold mean squared error.
- `num_consensus_ops` is the number of stable OG-SET operators retained.
- `consensus_ops` are the symbolic operator keys retained across folds.

## Command: `fase fit`

Fit FASE v21 on a numeric CSV dataset and write a compact JSON report.

Validate CSV settings without fitting:

```bash
fase fit \
  --csv outputs/fase_quickstart/train.csv \
  --target y \
  --dry-run
```

Fast CSV fit:

```bash
fase fit \
  --csv outputs/fase_quickstart/train.csv \
  --target y \
  --fast \
  --output outputs/example_fit_report.json
```

Fast CSV fit with best-effort model export:

```bash
fase fit \
  --csv outputs/fase_quickstart/train.csv \
  --target y \
  --fast \
  --output outputs/example_fit_report.json \
  --model-output outputs/example_model.json
```

Headerless CSV with target in the first column:

```bash
fase fit \
  --csv data/headerless.csv \
  --target 0 \
  --no-header \
  --fast \
  --output outputs/headerless_fit_report.json
```

Options:

| Option | Default | Meaning |
| --- | ---: | --- |
| `--csv` | required | Path to a numeric CSV file |
| `--target` | last column | Target column name or zero-based index |
| `--output` | `outputs/fase_fit_report.json` | Compact JSON report path |
| `--delimiter` | `,` | CSV delimiter |
| `--no-header` | off | Treat the CSV as headerless |
| `--k-folds` | `2` | Cross-validation folds |
| `--seed` | `42` | Random seed |
| `--fast` | off | Use small atoms, no grammar, no ruliad, no PySR |
| `--compare-pysr` | off | Run PySR baseline if dependency is installed |
| `--dry-run` | off | Validate CSV parsing and settings without fitting |
| `--model-output` | unset | Optional path for exported FASEModel JSON |

Input requirements:

- The CSV must be numeric after the optional header row.
- Missing values are not imputed.
- Categorical columns must be encoded before running `fase fit`.
- The target can be selected by header name, such as `--target y`, or by
  zero-based column index, such as `--target 3`.

Report output includes the standard compact FASE metrics plus dataset metadata:

```json
{
  "R2_oof": 0.91,
  "MSE_oof": 0.08,
  "num_consensus_ops": 3,
  "dataset": {
    "path": "outputs/fase_quickstart/train.csv",
    "rows": 27,
    "columns": 4,
    "target": "y",
    "target_index": 8,
    "feature_names": ["x0", "x1", "x2"]
  }
}
```

When `--model-output` is provided, the compact report includes:

```json
{
  "model_export": {
    "path": "outputs/example_model.json",
    "format": "FASEModel.to_dict",
    "status": "exported",
    "error": null,
    "feature_schema": {
      "feature_names": ["x0", "x1", "x2"],
      "feature_count": 3,
      "target": "y"
    },
    "preflight": {
      "status": "ok",
      "stage1_count": 3,
      "stage2_count": 0,
      "errors": [],
      "warnings": []
    }
  }
}
```

Exported model JSON also stores this compact `feature_schema` when the model is
created through `fase fit --model-output` or `fase export-model`. `fase predict`
and `fase eval-model` use it to warn when a later CSV has a different feature
count, feature names, or feature order.

Model export is best-effort in this release. Atomic, grammar, ruliad
hypergraph-state blocks, and closure-captured ruliad states can export. Opaque
closures that do not expose serializable state report `unsupported` or
`failed` instead of crashing the command. The `model_export.preflight` field
lists unsupported stage-2 block indices, kinds, and reasons when they can be
detected before serialization.

## Command: `fase export-model`

Fit a numeric CSV and export the consensus model JSON. This is a convenience
command for the common case where the model artifact is the primary output.

```bash
fase export-model \
  --csv outputs/fase_quickstart/train.csv \
  --target y \
  --fast \
  --output outputs/example_model.json \
  --report-output outputs/example_export_report.json
```

Dry run:

```bash
fase export-model \
  --csv outputs/fase_quickstart/train.csv \
  --target y \
  --output outputs/example_model.json \
  --dry-run
```

Options mostly match `fase fit`:

| Option | Default | Meaning |
| --- | ---: | --- |
| `--csv` | required | Path to a numeric CSV file |
| `--target` | last column | Target column name or zero-based index |
| `--output` | required | Exported FASEModel JSON path |
| `--report-output` | `outputs/fase_export_model_report.json` | Compact fit report path |
| `--delimiter` | `,` | CSV delimiter |
| `--no-header` | off | Treat the CSV as headerless |
| `--k-folds` | `2` | Cross-validation folds |
| `--seed` | `42` | Random seed |
| `--fast` | off | Use small atoms, no grammar, no ruliad, no PySR |
| `--compare-pysr` | off | Run PySR baseline if dependency is installed |
| `--dry-run` | off | Validate CSV parsing and settings without fitting |

## Command: `fase inspect-model`

Inspect exported model metadata without loading a dataset or executing the
model.

```bash
fase inspect-model --model outputs/example_model.json
```

JSON output for automation:

```bash
fase inspect-model \
  --model outputs/example_model.json \
  --json \
  --output outputs/example_model_inspection.json
```

The inspection reports:

- Export format and wrapper version.
- File size.
- Exported feature schema, when present.
- Linear weight count and nonzero weight count.
- Stage 1 and Stage 2 block counts.
- Stage 2 block kinds and output widths.

This command is intentionally read-only. It parses the exported JSON directly
and does not require prediction data.

## Command: `fase validate-model`

Validate exported model JSON structure without loading executable model code or
running predictions.

```bash
fase validate-model --model outputs/example_model.json
```

JSON output for automation:

```bash
fase validate-model \
  --model outputs/example_model.json \
  --json \
  --output outputs/example_model_validation.json
```

The command reports:

- Structural status: `ok` or `invalid`.
- Export format and schema version.
- Whether the payload uses the product wrapper or legacy bare model shape.
- Validation errors and compatibility warnings.

The exit code is `0` for structurally valid artifacts and `1` for invalid
artifacts. This is lighter than `fase inspect-model`; use it in artifact
handoff scripts when the only question is whether the JSON contract is valid.

## Command: `fase compare-models`

Compare two exported model JSON files using metadata only.

```bash
fase compare-models \
  --left outputs/baseline_model.json \
  --right outputs/candidate_model.json
```

JSON output for automation:

```bash
fase compare-models \
  --left outputs/baseline_model.json \
  --right outputs/candidate_model.json \
  --json \
  --output outputs/model_comparison.json
```

The comparison reports:

- Whether export format and wrapper version match.
- Whether feature schema names and counts match.
- Left-only and right-only feature names.
- Complexity deltas, computed as right minus left.
- Stage 2 block-kind count deltas.

This command uses the same metadata-only inspection path as
`fase inspect-model`; it does not load datasets or execute model prediction.

## Command: `fase predict`

Generate predictions from an exported `FASEModel.to_dict` JSON file.

```bash
fase predict \
  --model outputs/example_model.json \
  --csv outputs/fase_quickstart/predict.csv \
  --drop-column y \
  --output outputs/example_predictions.csv \
  --strict-schema
```

If the prediction CSV still contains the original target column, drop it before
prediction:

```bash
fase predict \
  --model outputs/example_model.json \
  --csv outputs/fase_quickstart/predict.csv \
  --drop-column y \
  --output outputs/example_predictions.csv
```

Dry run:

```bash
fase predict \
  --model outputs/example_model.json \
  --csv outputs/fase_quickstart/predict.csv \
  --drop-column y \
  --output outputs/example_predictions.csv \
  --dry-run
```

Options:

| Option | Default | Meaning |
| --- | ---: | --- |
| `--model` | required | Exported FASEModel JSON path |
| `--csv` | required | Numeric feature CSV path |
| `--output` | required | Prediction CSV output path |
| `--drop-column` | unset | Optional column name or index to remove before prediction |
| `--delimiter` | `,` | CSV delimiter |
| `--no-header` | off | Treat the CSV as headerless |
| `--dry-run` | off | Validate model/CSV settings without predicting |
| `--strict-schema` | off | Fail if exported model feature schema differs from the CSV |

Prediction output shape:

```csv
row,prediction
0,1.234
1,5.678
```

Prediction uses row order as the join key. Feature columns must match the
training feature order used when the model was exported.

When the exported model includes a `feature_schema`, `fase predict` prints a
warning to stderr if the prediction CSV feature count, names, or order differ
from the training schema. It does not auto-reorder columns.

Use `--strict-schema` for CI or production runs where schema warnings should
be treated as command failures.

## Command: `fase eval-model`

Evaluate an exported `FASEModel.to_dict` JSON file against a numeric CSV target
column.

```bash
fase eval-model \
  --model outputs/example_model.json \
  --csv outputs/fase_quickstart/predict.csv \
  --target y \
  --output outputs/example_eval.json \
  --strict-schema
```

Also write the prediction CSV used by the evaluation:

```bash
fase eval-model \
  --model outputs/example_model.json \
  --csv outputs/fase_quickstart/predict.csv \
  --target y \
  --output outputs/example_eval.json \
  --predictions-output outputs/example_predictions.csv
```

Dry run:

```bash
fase eval-model \
  --model outputs/example_model.json \
  --csv outputs/fase_quickstart/predict.csv \
  --target y \
  --output outputs/example_eval.json \
  --dry-run
```

Options:

| Option | Default | Meaning |
| --- | ---: | --- |
| `--model` | required | Exported FASEModel JSON path |
| `--csv` | required | Numeric CSV with features and target |
| `--target` | last column | Target column name or zero-based index |
| `--output` | required | Evaluation JSON output path |
| `--predictions-output` | unset | Optional prediction CSV output path |
| `--delimiter` | `,` | CSV delimiter |
| `--no-header` | off | Treat the CSV as headerless |
| `--dry-run` | off | Validate model/CSV settings without evaluating |
| `--strict-schema` | off | Fail if exported model feature schema differs from the CSV |

Evaluation JSON includes:

```json
{
  "schema_warnings": [],
  "metrics": {
    "rows": 5,
    "mse": 0.01,
    "rmse": 0.1,
    "mae": 0.08,
    "max_abs_error": 0.2,
    "r2": 0.99
  }
}
```

When the exported model includes a `feature_schema`, `fase eval-model` includes
any schema warnings in the JSON output and also prints them to stderr.

Use `--strict-schema` when evaluation should fail instead of writing a report
for a mismatched feature schema.

## Command: `fase report`

Summarize a JSON report in a terminal-friendly table.

Summarize a FASE demo report:

```bash
fase report outputs/fase_demo_report.json
```

When a fit/export report contains `model_export`, `fase report` prints the
export status, output path, format, and preflight status. Unsupported stage-2
blocks are shown in a separate preflight issues table with block index, kind,
and reason.

Summarize a v7 lab summary:

```bash
fase report v7_lab_outputs/v7_lab_summary.json
```

Example v7 table shape:

```text
mode         regime_acc  exact   hamming  purity  ops    gate
-----------  ----------  ------  -------  ------  -----  -----
v6_2_turbo  0.5647      0.5492  0.5231   0.9245  20.20  False
v6_3_stock  0.3953      0.3847  0.8769   0.9099  25.40  False
```

## Command: `fase v7-lab`

Run the v7 regime-first lab harness.

Dry run first:

```bash
fase v7-lab --dry-run
```

Fast check:

```bash
fase v7-lab \
  --fast \
  --episodes-per-combo 1 \
  --horizon 10 \
  --replay-rounds 0 \
  --output-dir v7_lab_outputs_fast
```

Full research run:

```bash
fase v7-lab \
  --fase-module FASE_v21.py \
  --output-dir v7_lab_outputs \
  --episodes-per-combo 8 \
  --horizon 100 \
  --replay-rounds 2 \
  --reuse-checkpoints
```

Options:

| Option | Default | Meaning |
| --- | ---: | --- |
| `--fase-module` | `FASE_v21.py` | Path to FASE v21 single-file module |
| `--output-dir` | `v7_lab_outputs` | Output directory |
| `--episodes-per-combo` | `8` | Teacher episodes per scenario combination |
| `--horizon` | `100` | Max ticks per episode |
| `--replay-rounds` | `2` | Weighted replay/refit rounds |
| `--k-folds` | `2` | Fast-mode FASE folds |
| `--seed` | `42` | Fast-mode seed |
| `--fast` | off | Use reduced FASE config |
| `--dry-run` | off | Print parameters without running |
| `--reuse-checkpoints` | on | Reuse available teacher/student checkpoints |
| `--no-reuse-checkpoints` | off | Rebuild checkpoints from scratch |

Expected output files:

```text
v7_lab_outputs/
├── teacher_v62_turbo.jsonl
├── teacher_v62_turbo.pkl
├── teacher_v63_stock.jsonl
├── teacher_v63_stock.pkl
├── v6_2_turbo/
│   ├── round_0_train_eval.json
│   ├── round_0_mine_stats.json
│   ├── round_1_train_eval.json
│   ├── round_1_mine_stats.json
│   ├── round_2_train_eval.json
│   └── final_eval_report.json
├── v6_3_stock/
│   └── ...
└── v7_lab_summary.json
```

Important runtime note:

- Full v7 runs are CPU-heavy.
- Run a dry run and a fast run before starting the full run.
- Keep the laptop awake and plugged in for full runs.
- Student checkpoint pickling may warn when FASE internals contain local
  closures. This is currently non-fatal; the harness continues and still
  writes JSON evaluation reports.

## Python API Usage

The product wrapper is intentionally small. It gives callers stable entry
points while the research files remain available.

### Run FASE v21 on Synthetic Data

```python
from fase.api import compact_report, make_synthetic, run_kfold, write_json

X, y, Sigma = make_synthetic(seed=42, n=300, d=8, gls_noise=True)

report = run_kfold(
    X,
    y,
    Sigma=Sigma,
    k_folds=3,
    seed=42,
    config_patch={
        "COMPARE_WITH_PYSR": False,
        "USE_RULIAD_STAGE25": False,
        "MAX_ATOMS": 12,
        "MAX_GRAMMAR": 0,
        "OGSET": {"bag_boots": 2},
    },
)

write_json(compact_report(report), "outputs/api_demo_report.json")
```

### Run FASE v21 on CSV Data

```python
from fase.api import (
    compact_report,
    export_model_from_report,
    load_csv_dataset,
    run_kfold,
    write_example_dataset,
    write_json,
)

write_example_dataset("outputs/fase_quickstart", force=True)
X, y, metadata = load_csv_dataset("outputs/fase_quickstart/train.csv", target="y")
report = run_kfold(
    X,
    y,
    k_folds=2,
    seed=42,
    config_patch={
        "COMPARE_WITH_PYSR": False,
        "USE_RULIAD_STAGE25": False,
        "MAX_ATOMS": 12,
        "MAX_GRAMMAR": 0,
        "OGSET": {"bag_boots": 2},
    },
)
compact = compact_report(report)
compact["dataset"] = metadata
compact["model_export"] = export_model_from_report(report, "outputs/example_model.json")
write_json(compact, "outputs/example_fit_report.json")
```

### Predict With an Exported Model

```python
from fase.api import predict_from_exported_model, write_predictions_csv

predictions, metadata = predict_from_exported_model(
    "outputs/example_model.json",
    "outputs/fase_quickstart/predict.csv",
    drop_column="y",
)
write_predictions_csv(predictions, "outputs/example_predictions.csv")
```

Drop a target column that is still present in the prediction CSV:

```python
predictions, metadata = predict_from_exported_model(
    "outputs/example_model.json",
    "outputs/fase_quickstart/predict.csv",
    drop_column="y",
)
```

### Evaluate an Exported Model

```python
from fase.api import evaluate_exported_model, write_json

eval_report = evaluate_exported_model(
    "outputs/example_model.json",
    "outputs/fase_quickstart/predict.csv",
    target="y",
)
write_json(eval_report, "outputs/example_eval.json")
```

### Load the Raw FASE v21 Module

```python
from fase.api import load_fase_v21

fase_v21 = load_fase_v21()
print(fase_v21.CONFIG["K_FOLDS"])
```

Use raw module access when you need lower-level research functions such as:

- `run_fase_kfold(...)`
- `run_fase_given_split(...)`
- `make_synthetic(...)`
- `FASEModel`

## Testing

Fast built-in tests:

```bash
python -m unittest discover -s tests
```

These tests verify:

- CLI version command.
- `fase check` and `fase check --fit`.
- `fase doctor` human-readable and JSON diagnostics.
- `fase clean` dry-run and guarded removal behavior.
- `fase release-check` package readiness reporting.
- `fase status` read-only product readiness summary.
- `fase release-notes` release handoff reporting.
- `fase release-bundle` handoff directory generation.
- `fase artifact-check --dry-run` wheel validation planning.
- `fase init-example` quickstart dataset generation.
- `fase copy-examples` packaged quickstart fixture extraction.
- Committed quickstart fixture validation and prediction.
- CSV dataset parsing and `fase fit --dry-run`.
- Model export helper and exported-model prediction.
- Exported-model metadata inspection through `fase inspect-model`.
- Exported-model structure validation through `fase validate-model`.
- Exported-model metadata comparison through `fase compare-models`.
- Exported-model feature schema warnings for prediction/evaluation.
- Exported-model evaluation metrics through `fase eval-model`.
- A real fast `export-model` plus `predict` round trip on a tiny numeric CSV.
- v7 dry-run command.
- JSON report summarization.

They intentionally avoid full v7 computation and heavyweight benchmarks.

Optional manual smoke run:

```bash
fase demo --fast --output outputs/fase_demo_report.json
fase check
fase doctor
fase clean
fase release-check
fase status --json
fase release-notes
fase release-bundle --json
fase artifact-check --dry-run
fase artifact-manifest --latest-only --json
fase init-example --output outputs/fase_quickstart --force
fase copy-examples --output outputs/fase_quickstart_fixture --force
fase report outputs/fase_demo_report.json
fase fit --csv outputs/fase_quickstart/train.csv --target y --dry-run
fase validate-model --model examples/quickstart/model.json --json
fase inspect-model --model examples/quickstart/model.json --json
fase compare-models --left examples/quickstart/model.json --right examples/quickstart/model.json --json
fase predict --model examples/quickstart/model.json --csv examples/quickstart/predict.csv --drop-column y --output outputs/example_predictions.csv --strict-schema
fase eval-model --model examples/quickstart/model.json --csv examples/quickstart/predict.csv --target y --output outputs/example_eval.json --strict-schema
```

## Development Workflow

Recommended product loop:

```bash
cd /Users/swainsonholness/dev/FASE
source venv/bin/activate
python -m pip install -e ".[dev]"
python -m unittest discover -s tests
fase version
fase check
fase doctor
fase clean
fase release-check
fase status --json
fase release-notes
fase release-bundle --json
fase artifact-check --dry-run
fase artifact-manifest --latest-only --json
fase init-example --output outputs/fase_quickstart --force
fase copy-examples --output outputs/fase_quickstart_fixture --force
fase v7-lab --dry-run
```

Run the complete local release smoke sequence:

```bash
bash scripts/product_smoke.sh
```

Before a commit, check status:

```bash
git status --short
```

Generated output directories are ignored for future runs. If older generated
files are already tracked, decide separately whether to keep them as benchmark
artifacts or remove them in a dedicated cleanup commit.

Preview generated cleanup at any time:

```bash
fase clean
fase clean --older-than-days 7
```

## Packaging Notes

Package metadata lives in `pyproject.toml`.

Release notes live in `CHANGELOG.md`.

Release decision criteria and the local handoff checklist live in
`docs/release.md`.

Exported model JSON format notes live in `docs/model-format.md`.

Committed CLI fixtures live in `examples/quickstart`. The installed wheel also
ships the same fixture under package data for `fase copy-examples`.

Source distribution inclusion/exclusion rules live in `MANIFEST.in`.

Build artifacts are written to `dist/` and ignored by git.

The installed console script is:

```text
fase = fase.cli:main
```

### Final Release Handback

Before tagging or publishing, create the release commit and make the worktree
clean, then run:

```bash
python -m unittest discover -s tests
bash scripts/product_smoke.sh
fase status --output outputs/fase_status_clean.json --strict --require-clean
fase release-bundle --output-dir outputs/fase_release_bundle_clean --strict --require-clean
git status --short
```

`git status --short` should print nothing. If it reports modified, deleted, or
untracked files, do not tag or publish yet. The clean status and clean bundle
commands intentionally fail on dirty worktrees so final handoff artifacts are
traceable to one committed source state.

### CI Smoke Workflow

The repository includes a lightweight product smoke workflow at
`.github/workflows/product-smoke.yml`. It mirrors the local release smoke
script and intentionally avoids full v7 lab runs.

The workflow runs on push, pull request, and manual dispatch across Python
3.10, 3.11, and 3.12. It performs:

- Editable package install with development tooling.
- Syntax compile checks for package and test files.
- Fast unit smoke tests.
- `fase release-check`.
- `fase status --json`.
- `fase release-notes --json`.
- Source distribution and wheel build with `--no-isolation`.
- Artifact manifest generation with `fase artifact-manifest --latest-only --json`.
- Release bundle generation with `fase release-bundle --strict --json`.
- Installed-wheel validation with `fase artifact-check`.
- Upload of `dist/*` and `outputs/fase_release_bundle/*` as CI handoff artifacts.

Run the same sequence locally with:

```bash
bash scripts/product_smoke.sh
```

Set `BUILD_NO_ISOLATION=0` when you explicitly want an isolated PEP 517 build
and the environment has package-index access.

The package deliberately wraps the existing single-file research modules rather
than moving them. This reduces breakage while creating a product surface.

Current packaging modules:

| Module | Purpose |
| --- | --- |
| `fase.api` | Stable Python wrapper and report helpers |
| `fase.cli` | CLI command implementation |
| `FASE_v21.py` | Current symbolic-regression engine |
| `v7_regime_first_lab_harness.py` | Regime-first lab harness |
| `pre_v7_fase_v21_distillation_pipeline.py` | Older bitwise harness |

## Troubleshooting

### `fase` command not found

Install the package in editable mode:

```bash
python -m pip install -e .
```

Then retry:

```bash
fase version
```

### Missing `pmlb` or `pysr`

The minimal product install does not include heavy optional dependencies.

Install demo dependencies:

```bash
python -m pip install -e ".[demo]"
```

Install PySR baseline dependencies:

```bash
python -m pip install -e ".[baseline]"
```

### Full v7 run takes a long time

Use a fast check first:

```bash
fase v7-lab --fast --episodes-per-combo 1 --horizon 10 --replay-rounds 0
```

Then run full only when the CLI path is working.

### Student checkpoint pickle warning

You may see a warning like:

```text
Warning: could not save v6_2_turbo student round 0 ...
```

This means an internal FASE model contains a local closure that cannot be
pickled. The harness continues and still writes JSON reports. Treat student
pickle persistence as best-effort; JSON export can now recover ruliad states
captured by closures, but Python pickle persistence is still more restrictive.

### Model export failed

`fase fit --model-output ...` and `fase export-model ...` export through
`FASEModel.to_dict`. Ruliad hypergraph-state blocks and closure-captured
ruliad states are serializable, but opaque closure-backed models can still
fail when no serializable state is exposed.

Use fast mode for the simplest export path:

```bash
fase init-example --output outputs/fase_quickstart --force
fase export-model --csv outputs/fase_quickstart/train.csv --target y --fast --output outputs/model.json
```

If export still fails, the fit report will include `model_export.status`,
`model_export.error`, and `model_export.preflight` so the failure can be
inspected without losing metrics.

### v7 gate fails

Gate failure means the discovered regime law is not compact or accurate enough
under the current thresholds. This is expected for the current lab state.

Use the report command:

```bash
fase report v7_lab_outputs/v7_lab_summary.json
```

Then inspect:

- `regime_accuracy`
- `exact_agreement`
- `avg_hamming`
- `mean_num_ops`
- `mean_median_bits`
- `compactness_gate`

## Product Roadmap

Near-term:

- Harden serialization for any remaining opaque closure-backed block variants.
- Expand exported-model regression fixtures beyond the current fast-fit case.
- Add isolated dependency artifact validation when release dependencies are pinned.
- Add a small dashboard or static HTML report for v7 runs.
- Split heavyweight research dependencies from core dependencies completely.
- Decide whether historical generated outputs should stay tracked or removed
  in a dedicated cleanup commit.

Medium-term:

- Stabilize a public Python API around `run_fase_kfold`.
- Add regression tests for known synthetic laws.
- Expand the quickstart dataset into examples, notebooks, or scripts.
- Add benchmark fixtures with small deterministic datasets.
- Improve v7 regime law quality before presenting it as a controller product.

## Safety Position

FASE is currently productizable as an interpretable model-discovery toolkit.
The v7 harness is productizable as a lab workflow and report generator.

Do not position the current v7 student as a production controller until it
passes agreement and compactness gates on held-out trajectories and the runtime
shield/policy interaction has a dedicated safety validation suite.
