# Changelog

All notable product-facing changes to FASE are tracked here.

## 0.1.0

Initial product shell for the FASE symbolic-regression toolkit.

- Added installable package metadata and `fase` console script.
- Added product wrapper package with stable API helpers.
- Added CLI commands for version, diagnostics, cleanup, reports, fitting, model export, prediction, and v7 dry runs.
- Added numeric CSV loading for `fase fit --csv`.
- Added best-effort `FASEModel.to_dict` JSON export plus reload/predict support.
- Added exported-model metadata inspection through `fase inspect-model`.
- Added exported-model structure validation through `fase validate-model`.
- Added exported-model metadata comparison through `fase compare-models`.
- Added exported-model feature-schema metadata and prediction/evaluation schema warnings.
- Added `--strict-schema` for fail-fast prediction/evaluation schema validation.
- Added ruliad hypergraph-state serialization for exported FASE models.
- Added recursive block-parameter serialization for grammar blocks such as `GroupSpec`.
- Added smoke checks through `fase check`, `fase check --fit`, and `fase doctor`.
- Added read-only product status summaries through `fase status`.
- Added `fase status --require-clean` for final clean-worktree status gates.
- Added guarded generated-output cleanup through `fase clean`.
- Added release and artifact validation through `fase release-check` and `fase artifact-check`.
- Added compact release handoff notes through `fase release-notes`.
- Added build artifact manifests through `fase artifact-manifest`, including `--latest-only` for stale `dist/` directories.
- Hardened artifact manifests to fail incomplete wheel/sdist quickstart fixture membership.
- Hardened artifact manifests to validate wheel/sdist package metadata and the `fase` console script entry point.
- Added release handoff bundles with git provenance through `fase release-bundle`.
- Added `fase release-bundle --require-clean` for final clean-worktree release gates.
- Added CI upload of wheel/sdist artifacts and release-bundle JSON handoff files.
- Added age-filtered generated-output cleanup through `fase clean --older-than-days`.
- Added machine-readable removal reports through `fase clean --yes --json`.
- Added a reusable product smoke script and CI workflow for release validation.
- Added release decision documentation in `docs/release.md`.
- Added final release handback commands for pre-tag/pre-publish checks.
- Hardened `fase release-check` to validate optional dependency groups, classifiers, package data, and generated-output manifest exclusions.
- Added exported `FASEModel` JSON format documentation and structural validation.
- Hardened ruliad serialization for closure-captured `HypergraphState` blocks.
- Added model export preflight diagnostics for unsupported stage-2 blocks.
- Added `fase report` summaries for model export preflight diagnostics.
- Added committed quickstart fixtures under `examples/quickstart`.
- Added packaged quickstart fixture extraction through `fase copy-examples`.
- Added deterministic quickstart dataset generation through `fase init-example`.
- Added exported-model evaluation metrics through `fase eval-model`.
- Added fast regression tests for CLI and exported-model behavior.
- Documented product usage, packaging notes, troubleshooting, and current v7 lab limitations.
