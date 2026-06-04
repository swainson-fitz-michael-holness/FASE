# FASE Release Guide

This document defines the local release decision gate for the current FASE
product shell. It is intentionally scoped to the installable symbolic
regression package and CLI, not to claims about autonomous controller safety.

## Release Decision Checklist

A release candidate can be handed off when all required items pass:

- `python -m pip install -e ".[dev]"` succeeds in a clean virtual environment.
- `bash scripts/product_smoke.sh` completes without failures.
- `fase release-check` reports `status=ok`.
- `fase status --strict --json` reports consolidated release, artifact, and git
  status after artifacts are built; use `--require-clean` for final clean-tree
  gates.
- `fase release-notes --json` reports `release_checks_required_ok=true`.
- The built wheel passes `fase artifact-check`.
- `fase artifact-manifest --latest-only --json` records wheel/sdist hashes and
  fails if packaged quickstart fixture membership or package metadata is
  incomplete.
- `fase release-bundle --strict --json` writes release notes and the latest
  artifact manifest into one handoff directory with git provenance.
- `fase release-bundle --strict --require-clean --json` can be used for final
  release CI when the source tree must be clean.
- The product smoke CI workflow uploads `dist/*` and
  `outputs/fase_release_bundle/*` for download from the workflow run.
- `examples/quickstart/model.json` validates and predicts against
  `examples/quickstart/predict.csv`.
- `fase copy-examples` can copy the packaged quickstart fixture from an
  installed wheel.
- README, changelog, and this release guide describe the current product
  boundary accurately.
- `docs/model-format.md` describes the exported model JSON contract.
- Generated outputs remain ignored or are intentionally committed as benchmark
  artifacts in a separate change.

## Shippable Surface

The current shippable surface is:

- `fase` CLI installation and command dispatch.
- Numeric CSV fit, export, predict, and evaluate workflow.
- Committed quickstart fixtures under `examples/quickstart`, plus packaged
  fixture extraction through `fase copy-examples`.
- Exported `FASEModel.to_dict` JSON validation, inspection, comparison, reload,
  and prediction.
- Model export preflight diagnostics for unsupported stage-2 blocks.
- Terminal report summaries for model export preflight diagnostics.
- Ruliad hypergraph-state and common grammar block serialization paths covered
  by smoke tests.
- v7 lab dry-run/report tooling for research workflow orchestration.
- Generated-output cleanup, including age-filtered cleanup, and release/artifact
  validation commands.
- Artifact manifest reporting for wheel/sdist handoff metadata.

## Research-Grade Surface

The following should not be described as production-ready:

- Autonomous v7 controller deployment.
- Safety claims based on the current v7 compactness gate.
- Full v7 harness runs as release smoke tests.
- Opaque closure-backed model serialization paths that do not expose
  serializable state through `FASEModel.to_dict`.

## Version Bump Rules

Use a patch version bump when:

- CLI behavior is backward compatible.
- Model JSON schema remains loadable by the current reader.
- Exported model format changes are additive and documented in
  `docs/model-format.md`.
- Changes are documentation, packaging, cleanup, or smoke-test hardening.

Use a minor version bump when:

- New user-facing CLI commands are added.
- Exported model JSON gains new optional fields.
- CSV input behavior adds new accepted forms without breaking existing usage.

Use a major version bump before:

- Removing or renaming CLI commands.
- Breaking exported model JSON compatibility.
- Reframing v7 from research harness to production controller.

## Local Release Handoff

Run the product smoke script from the repository root:

```bash
bash scripts/product_smoke.sh
```

Then capture release notes:

```bash
fase release-notes --output outputs/fase_release_notes.json
fase artifact-manifest --latest-only --output outputs/fase_artifact_manifest.json
fase status --output outputs/fase_status.json --strict
fase status --output outputs/fase_status_clean.json --strict --require-clean
fase release-bundle --output-dir outputs/fase_release_bundle --strict
fase release-bundle --output-dir outputs/fase_release_bundle_clean --strict --require-clean
```

If package-index access is available and an isolated build is desired, run:

```bash
BUILD_NO_ISOLATION=0 bash scripts/product_smoke.sh
```

The default local script uses `--no-isolation` for the build phase so the
release smoke path can run after development dependencies are installed without
requiring network access.

## Final Release Handback

Use this sequence immediately before tagging or publishing. It assumes the
working tree is intended to be clean after the release commit is created.

```bash
python -m unittest discover -s tests
bash scripts/product_smoke.sh
fase status --output outputs/fase_status_clean.json --strict --require-clean
fase release-bundle --output-dir outputs/fase_release_bundle_clean --strict --require-clean
git status --short
```

Interpretation:

- `python -m unittest discover -s tests` verifies the fast product regression
  suite.
- `bash scripts/product_smoke.sh` builds wheel/sdist artifacts, validates the
  installed wheel, and writes release-bundle JSON when `RELEASE_BUNDLE_DIR` is
  provided.
- `fase status --strict --require-clean` is the read-only final gate. It fails
  when release checks fail, artifacts are missing, git provenance is
  unavailable, or the worktree is dirty.
- `fase release-bundle --strict --require-clean` writes the final handoff
  bundle and fails under the same clean-tree conditions.
- `git status --short` should print nothing before tagging. If it does not,
  do not tag or publish yet.

Only after those checks pass should a release tag be created. The tag command
is intentionally omitted from automated smoke scripts; tagging remains a manual
release decision.

## Stop Conditions

Do not publish or hand off a release candidate if:

- `scripts/product_smoke.sh` fails.
- `fase artifact-check` fails on the wheel in `dist/`.
- `fase artifact-manifest --latest-only --json` reports `needs_attention`.
- The README or release notes imply v7 production safety.
- A generated checkpoint or benchmark artifact is deleted unintentionally.
- Model export fails for the fixture paths covered by the smoke tests.
