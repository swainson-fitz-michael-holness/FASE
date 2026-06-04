#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

PYTHON_BIN="${PYTHON_BIN:-python3}"
if [[ -z "${BUILD_PYTHON:-}" ]]; then
  if [[ -x "$ROOT/venv/bin/python" ]]; then
    BUILD_PYTHON="$ROOT/venv/bin/python"
  else
    BUILD_PYTHON="$PYTHON_BIN"
  fi
fi

WHEEL_PATH="${WHEEL_PATH:-dist/fase_symbolic-0.1.0-py3-none-any.whl}"
BUILD_NO_ISOLATION="${BUILD_NO_ISOLATION:-1}"

echo "==> Syntax check"
"$PYTHON_BIN" -m py_compile fase/api.py fase/cli.py fase/__init__.py tests/test_cli_smoke.py

echo "==> Unit smoke tests"
"$PYTHON_BIN" -m unittest discover -s tests

echo "==> Release readiness"
"$PYTHON_BIN" -m fase.cli release-check
"$PYTHON_BIN" -m fase.cli release-notes --json

tmp_dir="$(mktemp -d)"
trap 'rm -rf "$tmp_dir"' EXIT
RELEASE_BUNDLE_DIR="${RELEASE_BUNDLE_DIR:-$tmp_dir/release_bundle}"

echo "==> Committed quickstart fixtures"
"$PYTHON_BIN" -m fase.cli validate-model --model examples/quickstart/model.json --json
"$PYTHON_BIN" -m fase.cli predict \
  --model examples/quickstart/model.json \
  --csv examples/quickstart/predict.csv \
  --drop-column y \
  --output "$tmp_dir/example_predictions.csv" \
  --strict-schema
"$PYTHON_BIN" -m fase.cli eval-model \
  --model examples/quickstart/model.json \
  --csv examples/quickstart/predict.csv \
  --target y \
  --output "$tmp_dir/example_eval.json" \
  --strict-schema

echo "==> Packaged quickstart copy"
"$PYTHON_BIN" -m fase.cli copy-examples --output "$tmp_dir/packaged_quickstart"
"$PYTHON_BIN" -m fase.cli validate-model --model "$tmp_dir/packaged_quickstart/model.json" --json
"$PYTHON_BIN" -m fase.cli predict \
  --model "$tmp_dir/packaged_quickstart/model.json" \
  --csv "$tmp_dir/packaged_quickstart/predict.csv" \
  --drop-column y \
  --output "$tmp_dir/packaged_predictions.csv" \
  --strict-schema

echo "==> Build artifacts"
build_args=(--sdist --wheel)
if [[ "$BUILD_NO_ISOLATION" == "1" ]]; then
  build_args+=(--no-isolation)
fi
"$BUILD_PYTHON" -m build "${build_args[@]}"

echo "==> Artifact manifest"
"$PYTHON_BIN" -m fase.cli artifact-manifest --root "$ROOT" --latest-only --json

echo "==> Product status"
"$PYTHON_BIN" -m fase.cli status --root "$ROOT" --output "$tmp_dir/fase_status.json" --strict

echo "==> Release bundle"
"$PYTHON_BIN" -m fase.cli release-bundle --root "$ROOT" --output-dir "$RELEASE_BUNDLE_DIR" --json --strict

echo "==> Installed artifact smoke"
"$PYTHON_BIN" -m fase.cli artifact-check --wheel "$WHEEL_PATH"

echo "Product smoke complete"
