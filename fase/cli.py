from __future__ import annotations

import argparse
import importlib
import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any, Dict, Iterable

from . import api

EXPECTED_CLI_COMMANDS = (
    "version",
    "status",
    "check",
    "doctor",
    "clean",
    "release-check",
    "release-notes",
    "release-bundle",
    "artifact-check",
    "artifact-manifest",
    "init-example",
    "copy-examples",
    "demo",
    "fit",
    "export-model",
    "inspect-model",
    "validate-model",
    "compare-models",
    "predict",
    "eval-model",
    "v7-lab",
    "report",
)


def _fast_fase_config(k_folds: int, seed: int) -> Dict[str, Any]:
    return {
        "K_FOLDS": k_folds,
        "SEEDS": [seed],
        "COMPARE_WITH_PYSR": False,
        "USE_RULIAD_STAGE25": False,
        "MAX_ATOMS": 12,
        "MAX_GRAMMAR": 0,
        "OGSET": {
            "bag_boots": 2,
            "final_min_freq": 0.50,
            "final_min_sign_stab": 0.80,
            "final_min_bits": 3.0,
        },
    }


def _print_table(rows: Iterable[Dict[str, Any]]) -> None:
    rows = list(rows)
    if not rows:
        print("No rows.")
        return
    headers = list(rows[0].keys())
    widths = {
        h: max(len(str(h)), *(len(str(row.get(h, ""))) for row in rows))
        for h in headers
    }
    print("  ".join(h.ljust(widths[h]) for h in headers))
    print("  ".join("-" * widths[h] for h in headers))
    for row in rows:
        print("  ".join(str(row.get(h, "")).ljust(widths[h]) for h in headers))


def cmd_version(_: argparse.Namespace) -> int:
    print(f"FASE {api.__version__}")
    return 0


def cmd_demo(args: argparse.Namespace) -> int:
    X, y, Sigma = api.make_synthetic(seed=args.seed, n=args.n, d=args.d, gls_noise=not args.no_gls_noise)
    config_patch = _fast_fase_config(args.k_folds, args.seed) if args.fast else {
        "K_FOLDS": args.k_folds,
        "SEEDS": [args.seed],
        "COMPARE_WITH_PYSR": args.compare_pysr,
    }
    report = api.run_kfold(X, y, Sigma=Sigma, k_folds=args.k_folds, seed=args.seed, config_patch=config_patch)
    compact = api.compact_report(report)
    api.write_json(compact, args.output)
    print(f"Wrote demo report: {args.output}")
    print(
        "R2_oof={R2_oof:.4f} MSE_oof={MSE_oof:.6f} consensus_ops={num_consensus_ops}".format(
            **compact
        )
    )
    return 0


def cmd_init_example(args: argparse.Namespace) -> int:
    try:
        report = api.write_example_dataset(args.output, force=args.force)
    except FileExistsError as exc:
        print(str(exc), file=sys.stderr)
        return 1

    print("Wrote FASE quickstart example")
    print(f"directory={report['directory']}")
    print(f"train_csv={report['train_csv']}")
    print(f"predict_csv={report['predict_csv']}")
    print(f"readme={report['readme']}")
    print(f"law={report['law']}")
    print(f"train_rows={report['train_rows']} predict_rows={report['predict_rows']}")
    print("Next:")
    print(f"  fase export-model --csv {report['train_csv']} --target y --fast --output {report['directory']}/model.json --report-output {report['directory']}/report.json")
    print(f"  fase predict --model {report['directory']}/model.json --csv {report['predict_csv']} --drop-column y --output {report['directory']}/predictions.csv")
    print(f"  fase eval-model --model {report['directory']}/model.json --csv {report['predict_csv']} --target y --output {report['directory']}/eval.json")
    return 0


def cmd_copy_examples(args: argparse.Namespace) -> int:
    try:
        report = api.copy_quickstart_example(args.output, force=args.force)
    except (FileExistsError, FileNotFoundError) as exc:
        print(str(exc), file=sys.stderr)
        return 1

    print("Copied packaged FASE quickstart fixture")
    print(f"directory={report['directory']}")
    print(f"source={report['source']}")
    print(f"train_csv={report['train_csv']}")
    print(f"predict_csv={report['predict_csv']}")
    print(f"model_json={report['model_json']}")
    print(f"readme={report['readme']}")
    print(f"law={report['law']}")
    print("Next:")
    print(f"  fase validate-model --model {report['model_json']}")
    print(f"  fase predict --model {report['model_json']} --csv {report['predict_csv']} --drop-column y --output {report['directory']}/predictions.csv --strict-schema")
    print(f"  fase eval-model --model {report['model_json']} --csv {report['predict_csv']} --target y --output {report['directory']}/eval.json --strict-schema")
    return 0


def cmd_fit(args: argparse.Namespace) -> int:
    X, y, metadata = api.load_csv_dataset(
        args.csv,
        target=args.target,
        delimiter=args.delimiter,
        has_header=not args.no_header,
    )
    if args.dry_run:
        print("CSV fit dry run")
        print(f"path={metadata['path']}")
        print(f"rows={metadata['rows']}")
        print(f"features={len(metadata['feature_names'])}")
        print(f"target={metadata['target']}")
        print(f"k_folds={args.k_folds}")
        print(f"fast={args.fast}")
        return 0

    config_patch = _fast_fase_config(args.k_folds, args.seed) if args.fast else {
        "K_FOLDS": args.k_folds,
        "SEEDS": [args.seed],
        "COMPARE_WITH_PYSR": args.compare_pysr,
    }
    report = api.run_kfold(X, y, Sigma=None, k_folds=args.k_folds, seed=args.seed, config_patch=config_patch)
    compact = api.compact_report(report)
    compact["dataset"] = metadata
    if args.model_output:
        compact["model_export"] = api.export_model_from_report(report, args.model_output, dataset_metadata=metadata)
    api.write_json(compact, args.output)
    print(f"Wrote fit report: {args.output}")
    if args.model_output:
        export = compact["model_export"]
        if export["status"] == "exported":
            print(f"Wrote model: {export['path']}")
        else:
            print(f"Model export {export['status']}: {export['error']}")
            preflight_errors = (export.get("preflight") or {}).get("errors") or []
            if preflight_errors:
                print("Model export preflight errors:")
                for error in preflight_errors:
                    print(f"  - {error}")
    print(
        "rows={rows} features={features} target={target} R2_oof={R2_oof:.4f} MSE_oof={MSE_oof:.6f}".format(
            rows=metadata["rows"],
            features=len(metadata["feature_names"]),
            target=metadata["target"],
            **compact,
        )
    )
    return 0


def cmd_export_model(args: argparse.Namespace) -> int:
    args.model_output = args.output
    args.output = args.report_output
    return cmd_fit(args)


def cmd_inspect_model(args: argparse.Namespace) -> int:
    report = api.inspect_exported_model(args.model)
    if args.output:
        api.write_json(report, args.output)

    if args.json:
        print(json.dumps(report, indent=2))
        return 0

    schema = report.get("feature_schema") or {}
    model = report["model"]
    rows = [
        {"field": "path", "value": report["path"]},
        {"field": "format", "value": report["format"]},
        {"field": "schema_version", "value": report.get("schema_version") or "unknown"},
        {"field": "schema_status", "value": report["schema_validation"]["status"]},
        {"field": "version", "value": report.get("version") or "unknown"},
        {"field": "wrapped", "value": report["wrapped"]},
        {"field": "bytes", "value": report["file"]["bytes"]},
        {"field": "feature_count", "value": schema.get("feature_count", "unknown")},
        {"field": "target", "value": schema.get("target", "unknown")},
        {"field": "linear_weights", "value": model["linear_weights"]},
        {"field": "nonzero_weights", "value": model["nonzero_linear_weights"]},
        {"field": "stage1_count", "value": model["stage1_count"]},
        {"field": "stage2_count", "value": model["stage2_count"]},
        {"field": "stage2_total_width", "value": model["stage2_total_width"]},
    ]
    print("FASE model inspection")
    _print_table(rows)
    if schema.get("feature_names"):
        print("features=" + ",".join(schema["feature_names"]))
    if model["stage2_kinds"]:
        kind_rows = [
            {"kind": kind, "count": count}
            for kind, count in sorted(model["stage2_kinds"].items())
        ]
        print("Stage 2 block kinds")
        _print_table(kind_rows)
    if args.output:
        print(f"Wrote inspection: {args.output}")
    return 0


def cmd_validate_model(args: argparse.Namespace) -> int:
    payload = api.read_json(args.model)
    report = api.validate_exported_model_payload(payload)
    report["path"] = str(Path(args.model))

    if args.output:
        api.write_json(report, args.output)
    if args.json:
        print(json.dumps(report, indent=2))
        return 0 if report["status"] == "ok" else 1

    print("FASE model validation")
    print(f"status={report['status']}")
    print(f"path={report['path']}")
    print(f"format={report.get('format') or 'unknown'}")
    print(f"schema_version={report.get('schema_version') or 'unknown'}")
    if report["errors"]:
        print("Errors")
        for error in report["errors"]:
            print(f"- {error}")
    if report["warnings"]:
        print("Warnings")
        for warning in report["warnings"]:
            print(f"- {warning}")
    if args.output:
        print(f"Wrote validation: {args.output}")
    return 0 if report["status"] == "ok" else 1


def cmd_compare_models(args: argparse.Namespace) -> int:
    report = api.compare_exported_models(args.left, args.right)
    if args.output:
        api.write_json(report, args.output)

    if args.json:
        print(json.dumps(report, indent=2))
        return 0

    schema = report["feature_schema"]
    rows = [
        {"field": "left", "value": report["left_path"]},
        {"field": "right", "value": report["right_path"]},
        {"field": "same_format", "value": report["same_format"]},
        {"field": "same_version", "value": report["same_version"]},
        {"field": "same_feature_names", "value": schema["same_feature_names"]},
        {"field": "same_feature_count", "value": schema["same_feature_count"]},
        {"field": "left_feature_count", "value": schema["left_feature_count"]},
        {"field": "right_feature_count", "value": schema["right_feature_count"]},
    ]
    print("FASE model comparison")
    _print_table(rows)

    delta_rows = [
        {"metric": key, "delta_right_minus_left": value}
        for key, value in report["complexity_delta"].items()
    ]
    print("Complexity delta")
    _print_table(delta_rows)

    if schema["left_only_features"] or schema["right_only_features"]:
        print("left_only_features=" + ",".join(schema["left_only_features"]))
        print("right_only_features=" + ",".join(schema["right_only_features"]))

    if report["stage2_kind_delta"]:
        kind_rows = [
            {"kind": kind, **values}
            for kind, values in report["stage2_kind_delta"].items()
        ]
        print("Stage 2 kind delta")
        _print_table(kind_rows)

    if args.output:
        print(f"Wrote comparison: {args.output}")
    return 0


def cmd_predict(args: argparse.Namespace) -> int:
    if args.dry_run:
        X, metadata = api.load_csv_features(
            args.csv,
            drop_column=args.drop_column,
            delimiter=args.delimiter,
            has_header=not args.no_header,
        )
        print("Prediction dry run")
        print(f"model={args.model}")
        print(f"path={metadata['path']}")
        print(f"rows={metadata['rows']}")
        print(f"features={X.shape[1]}")
        print(f"drop_column={metadata['dropped_column']}")
        return 0

    predictions, metadata = api.predict_from_exported_model(
        args.model,
        args.csv,
        drop_column=args.drop_column,
        delimiter=args.delimiter,
        has_header=not args.no_header,
        strict_schema=args.strict_schema,
    )
    api.write_predictions_csv(predictions, args.output)
    print(f"Wrote predictions: {args.output}")
    print(f"rows={metadata['rows']} features={len(metadata['feature_names'])}")
    for warning in metadata.get("schema_warnings", []):
        print(f"WARNING: {warning}", file=sys.stderr)
    return 0


def cmd_eval_model(args: argparse.Namespace) -> int:
    if args.dry_run:
        X, y, metadata = api.load_csv_dataset(
            args.csv,
            target=args.target,
            delimiter=args.delimiter,
            has_header=not args.no_header,
        )
        print("Model evaluation dry run")
        print(f"model={args.model}")
        print(f"path={metadata['path']}")
        print(f"rows={metadata['rows']}")
        print(f"features={X.shape[1]}")
        print(f"target={metadata['target']}")
        print(f"predictions_output={args.predictions_output}")
        return 0

    report = api.evaluate_exported_model(
        args.model,
        args.csv,
        target=args.target,
        delimiter=args.delimiter,
        has_header=not args.no_header,
        strict_schema=args.strict_schema,
    )
    api.write_json(report, args.output)
    if args.predictions_output:
        api.write_predictions_csv(report["predictions"], args.predictions_output)

    metrics = report["metrics"]
    print(f"Wrote evaluation: {args.output}")
    if args.predictions_output:
        print(f"Wrote predictions: {args.predictions_output}")
    print(
        "rows={rows} target={target} R2={r2:.4f} RMSE={rmse:.6f} MAE={mae:.6f}".format(
            target=report["dataset"]["target"],
            **metrics,
        )
    )
    for warning in report.get("schema_warnings", []):
        print(f"WARNING: {warning}", file=sys.stderr)
    return 0


def _check(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)
    print(f"OK {message}")


def cmd_check(args: argparse.Namespace) -> int:
    print(f"FASE check {api.__version__}")
    module = api.load_fase_v21()
    _check(hasattr(module, "run_fase_kfold"), "FASE_v21 exposes run_fase_kfold")

    with tempfile.TemporaryDirectory() as td:
        root = Path(td)
        csv_path = root / "tiny.csv"
        model_path = root / "model.json"
        prediction_input = root / "predict.csv"
        prediction_output = root / "predictions.csv"

        csv_path.write_text("a,b,y\n1,2,3\n4,5,6\n", encoding="utf-8")
        X, y, metadata = api.load_csv_dataset(csv_path, target="y")
        _check(X.shape == (2, 2), "CSV feature parsing works")
        _check(y.tolist() == [3.0, 6.0], "CSV target parsing works")
        _check(metadata["target"] == "y", "CSV target metadata works")

        model_path.write_text(
            json.dumps(
                {
                    "format": "FASEModel.to_dict",
                    "schema_version": api.FASE_MODEL_SCHEMA_VERSION,
                    "version": api.__version__,
                    "model": {"w": [], "b0": 2.5, "stage1": [], "stage2": []},
                }
            ),
            encoding="utf-8",
        )
        prediction_input.write_text("a,b,y\n1,2,9\n4,5,8\n", encoding="utf-8")
        preds, pred_meta = api.predict_from_exported_model(model_path, prediction_input, drop_column="y")
        _check(preds.tolist() == [2.5, 2.5], "exported model reload and predict works")
        _check(pred_meta["dropped_column"] == "y", "prediction drop-column works")
        api.write_predictions_csv(preds, prediction_output)
        _check(prediction_output.exists(), "prediction CSV writer works")
        eval_report = api.evaluate_exported_model(model_path, prediction_input, target="y")
        _check(eval_report["metrics"]["rows"] == 2, "exported model evaluation works")

        if args.fit:
            train_path = root / "train.csv"
            fit_model = root / "fit_model.json"
            fit_report = root / "fit_report.json"
            fit_predictions = root / "fit_predictions.csv"
            train_path.write_text(
                "x0,x1,y\n"
                "0,0,1\n"
                "1,0,2\n"
                "0,1,3\n"
                "1,1,4\n"
                "2,0,3\n"
                "0,2,5\n",
                encoding="utf-8",
            )
            export_args = argparse.Namespace(
                csv=str(train_path),
                target="y",
                output=str(fit_model),
                report_output=str(fit_report),
                delimiter=",",
                no_header=False,
                k_folds=2,
                seed=42,
                fast=True,
                compare_pysr=False,
                dry_run=False,
            )
            predict_args = argparse.Namespace(
                model=str(fit_model),
                csv=str(train_path),
                output=str(fit_predictions),
                drop_column="y",
                delimiter=",",
                no_header=False,
                dry_run=False,
                strict_schema=False,
            )
            cmd_export_model(export_args)
            cmd_predict(predict_args)
            fit_payload = api.read_json(fit_report)
            _check(fit_payload["model_export"]["status"] == "exported", "real fast model export works")
            _check(fit_predictions.exists(), "real fast exported model prediction works")

    print("FASE check complete")
    return 0


def _doctor_status(available: bool, required: bool) -> str:
    if available:
        return "OK"
    return "FAIL" if required else "MISSING"


def cmd_doctor(args: argparse.Namespace) -> int:
    diagnostics = api.collect_diagnostics()
    if args.output:
        api.write_json(diagnostics, args.output)

    if args.json:
        print(json.dumps(diagnostics, indent=2))
        return 0 if diagnostics["required_ok"] else 1

    rows = [
        {
            "check": "python",
            "status": "OK" if diagnostics["python"]["ok"] else "FAIL",
            "detail": f"{diagnostics['python']['version']} at {diagnostics['python']['executable']}",
        }
    ]
    for package in diagnostics["packages"]:
        version = package["version"] or "not installed"
        rows.append(
            {
                "check": f"package:{package['name']}",
                "status": _doctor_status(package["available"], package["required"]),
                "detail": version,
            }
        )
    for module in diagnostics["modules"]:
        detail = module["origin"] or "not found"
        rows.append(
            {
                "check": f"module:{module['name']}",
                "status": _doctor_status(module["available"], module["required"]),
                "detail": detail,
            }
        )

    print(f"FASE doctor {diagnostics['fase_version']}")
    print(f"status={diagnostics['status']}")
    _print_table(rows)
    if args.output:
        print(f"Wrote diagnostics: {args.output}")
    return 0 if diagnostics["required_ok"] else 1


def _print_clean_report(report: Dict[str, Any], *, removed: bool) -> None:
    action = "Removed" if removed else "Would remove"
    print(f"root={report['root']}")
    if report.get("older_than_days") is not None:
        print(f"older_than_days={report['older_than_days']}")
    print(f"candidates={report['count']} files={report['total_files']} bytes={report['total_bytes']}")
    rows = [
        {
            "action": action,
            "kind": entry["kind"],
            "files": entry["files"],
            "bytes": entry["bytes"],
            "age_days": f"{float(entry.get('age_days', 0.0)):.2f}",
            "path": entry["relative_path"],
        }
        for entry in report["entries"]
    ]
    _print_table(rows)


def cmd_clean(args: argparse.Namespace) -> int:
    if args.yes:
        report = api.clean_generated_outputs(args.root, older_than_days=args.older_than_days)
        if args.json:
            print(json.dumps(report, indent=2))
            return 0

        print("FASE clean")
        _print_clean_report(report, removed=True)
        print(f"removed={report['removed_count']}")
        return 0

    report = api.discover_generated_outputs(args.root, older_than_days=args.older_than_days)
    if args.json:
        print(json.dumps(report, indent=2))
        return 0

    print("FASE clean dry run")
    _print_clean_report(report, removed=False)
    print("Pass --yes to remove these generated outputs.")
    return 0


def _parser_command_names(parser: argparse.ArgumentParser) -> set[str]:
    for action in parser._actions:
        if isinstance(action, argparse._SubParsersAction):
            return set(action.choices.keys())
    return set()


def _add_cli_command_checks(report: Dict[str, Any]) -> Dict[str, Any]:
    available_commands = _parser_command_names(build_parser())
    command_checks = []
    for command in EXPECTED_CLI_COMMANDS:
        command_checks.append(
            {
                "name": f"cli:{command}",
                "required": True,
                "status": "pass" if command in available_commands else "fail",
                "detail": "registered" if command in available_commands else "missing",
            }
        )
    report["checks"].extend(command_checks)
    report["required_ok"] = all(check["status"] == "pass" for check in report["checks"] if check["required"])
    report["status"] = "ok" if report["required_ok"] else "needs_attention"
    return report


def cmd_release_check(args: argparse.Namespace) -> int:
    report = api.collect_release_checks(args.root)
    report = _add_cli_command_checks(report)

    if args.output:
        api.write_json(report, args.output)
    if args.json:
        print(json.dumps(report, indent=2))
        return 0 if report["required_ok"] else 1

    rows = [
        {
            "check": check["name"],
            "status": check["status"].upper(),
            "detail": check["detail"],
        }
        for check in report["checks"]
    ]
    print("FASE release check")
    print(f"status={report['status']}")
    print(f"root={report['root']}")
    _print_table(rows)
    if args.output:
        print(f"Wrote release check: {args.output}")
    return 0 if report["required_ok"] else 1


def _git_summary_line(git: Dict[str, Any]) -> str:
    if git.get("available"):
        return f"{git.get('branch') or 'detached'}@{git.get('commit_short') or 'unknown'} dirty={git.get('dirty')} count={git.get('status_count')}"
    return f"unavailable {git.get('error') or ''}".rstrip()


def cmd_status(args: argparse.Namespace) -> int:
    report = api.collect_product_status(
        args.root,
        latest_only=not args.all_artifacts,
        require_clean=args.require_clean,
    )
    release = _add_cli_command_checks(report["release_checks"])
    report["release_status"] = release["status"]
    report["release_checks_required_ok"] = bool(release["required_ok"])
    report["status"] = (
        "ok"
        if release["required_ok"]
        and report["artifact_manifest_status"] == "ok"
        and report["clean_required_pass"]
        else "needs_attention"
    )

    if args.output:
        api.write_json(report, args.output)
    if args.json:
        print(json.dumps(report, indent=2))
        return 0 if report["status"] == "ok" or not (args.strict or args.require_clean) else 1

    package = report["package"]
    print("FASE product status")
    print(f"status={report['status']}")
    print(f"root={report['root']}")
    print(f"package={package['project_name']} version={package['project_version']}")
    print(f"release_checks={report['release_status']}")
    print(f"artifacts={report['artifact_manifest_status']} count={report['artifact_count']} latest_only={report['latest_only']}")
    print(f"require_clean={report['require_clean']} clean_required_pass={report['clean_required_pass']}")
    print(f"git={_git_summary_line(report.get('git') or {})}")
    failing = [
        {"check": check["name"], "status": check["status"].upper(), "detail": check["detail"]}
        for check in release["checks"]
        if check["required"] and check["status"] != "pass"
    ]
    if failing:
        print("Failing checks")
        _print_table(failing)
    artifacts = [
        {
            "kind": item["kind"],
            "name": item["name"],
            "sha256": (item["sha256"] or "missing")[:16],
            "quickstart": len(item["quickstart_files"]),
        }
        for item in report["artifact_manifest"]["artifacts"]
    ]
    if artifacts:
        print("Artifacts")
        _print_table(artifacts)
    if args.output:
        print(f"Wrote status: {args.output}")
    return 0 if report["status"] == "ok" or not (args.strict or args.require_clean) else 1


def cmd_release_notes(args: argparse.Namespace) -> int:
    report = api.collect_release_notes(args.root, cli_commands=list(EXPECTED_CLI_COMMANDS))

    if args.output:
        api.write_json(report, args.output)
    if args.json:
        print(json.dumps(report, indent=2))
        return 0 if report["release_checks_required_ok"] else 1

    artifacts = [
        {"artifact": "wheel", "path": report["artifacts"].get("wheel") or "missing"},
        {"artifact": "sdist", "path": report["artifacts"].get("sdist") or "missing"},
    ]
    print("FASE release notes")
    print(f"status={report['status']}")
    print(f"package={report['project_name']} version={report['project_version']}")
    print(f"root={report['root']}")
    print("Artifacts")
    _print_table(artifacts)
    print("Capabilities")
    for item in report["capabilities"]:
        print(f"- {item}")
    print("Limitations")
    for item in report["limitations"]:
        print(f"- {item}")
    print("CLI commands")
    print(", ".join(report["cli_commands"]))
    print("Verification commands")
    for item in report["verification_commands"]:
        print(f"- {item}")
    if args.output:
        print(f"Wrote release notes: {args.output}")
    return 0 if report["release_checks_required_ok"] else 1


def cmd_release_bundle(args: argparse.Namespace) -> int:
    report = api.create_release_bundle(
        args.root,
        output_dir=args.output_dir,
        latest_only=not args.all_artifacts,
        require_clean=args.require_clean,
        timestamp=args.timestamp,
        cli_commands=list(EXPECTED_CLI_COMMANDS),
    )
    if args.json:
        print(json.dumps(report, indent=2))
        return 0 if report["status"] == "ok" or not (args.strict or args.require_clean) else 1

    print("FASE release bundle")
    print(f"status={report['status']}")
    print(f"root={report['root']}")
    print(f"bundle_dir={report['bundle_dir']}")
    print(f"latest_only={report['latest_only']}")
    print(f"artifact_manifest_status={report['artifact_manifest_status']}")
    print(f"artifact_count={report['artifact_count']}")
    print(f"require_clean={report['require_clean']}")
    print(f"clean_required_pass={report['clean_required_pass']}")
    print(f"git={_git_summary_line(report.get('git') or {})}")
    rows = [{"file": key, "path": path} for key, path in report["files"].items()]
    _print_table(rows)
    if report["status"] != "ok":
        print("Bundle written, but release checks or artifact manifest still need attention.", file=sys.stderr)
    return 0 if report["status"] == "ok" or not (args.strict or args.require_clean) else 1


def cmd_artifact_manifest(args: argparse.Namespace) -> int:
    report = api.collect_artifact_manifest(
        args.root,
        artifact_paths=args.artifact,
        latest_only=args.latest_only,
    )
    if args.output:
        api.write_json(report, args.output)
    if args.json:
        print(json.dumps(report, indent=2))
        return 0 if report["status"] == "ok" else 1

    rows = [
        {
            "kind": item["kind"],
            "bytes": item["bytes"] if item["bytes"] is not None else "missing",
            "sha256": (item["sha256"] or "missing")[:16],
            "members": item["member_count"],
            "quickstart": len(item["quickstart_files"]),
            "quickstart_ok": "yes" if item.get("quickstart_complete") else "no",
            "metadata_ok": "yes" if item.get("artifact_metadata", {}).get("valid") else "no",
            "path": item["relative_path"],
        }
        for item in report["artifacts"]
    ]
    print("FASE artifact manifest")
    print(f"status={report['status']}")
    print(f"root={report['root']}")
    print(f"latest_only={report['latest_only']}")
    _print_table(rows)
    for item in report["artifacts"]:
        for warning in item.get("warnings", []):
            print(f"WARNING {item['relative_path']}: {warning}", file=sys.stderr)
    if args.output:
        print(f"Wrote artifact manifest: {args.output}")
    return 0 if report["status"] == "ok" else 1


def _latest_wheel(root: str | Path) -> Path:
    dist = Path(root).resolve() / "dist"
    wheels = sorted(dist.glob("*.whl"), key=lambda path: path.stat().st_mtime)
    if not wheels:
        raise FileNotFoundError(f"No wheel artifacts found in {dist}")
    return wheels[-1]


def _venv_executable(env_dir: Path, name: str) -> Path:
    scripts_dir = "Scripts" if os.name == "nt" else "bin"
    suffix = ".exe" if os.name == "nt" and name in {"python", "fase"} else ""
    return env_dir / scripts_dir / f"{name}{suffix}"


def _run_artifact_step(name: str, command: list[str], *, verbose: bool) -> Dict[str, Any]:
    proc = subprocess.run(command, check=False, text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    if verbose or proc.returncode != 0:
        if proc.stdout:
            print(proc.stdout, end="")
        if proc.stderr:
            print(proc.stderr, end="", file=sys.stderr)
    return {
        "step": name,
        "status": "pass" if proc.returncode == 0 else "fail",
        "returncode": proc.returncode,
        "command": " ".join(command),
    }


def cmd_artifact_check(args: argparse.Namespace) -> int:
    wheel = Path(args.wheel).resolve() if args.wheel else _latest_wheel(args.root)
    if not wheel.exists():
        raise FileNotFoundError(f"Wheel artifact does not exist: {wheel}")

    if args.dry_run:
        print("FASE artifact check dry run")
        print(f"wheel={wheel}")
        print(f"system_site_packages={not args.isolated_deps}")
        print("checks=venv, pip install --no-deps, fase version, fase check, fase release-notes, fase init-example, fase copy-examples, fase inspect-model, fase validate-model, fase compare-models")
        return 0

    with tempfile.TemporaryDirectory(prefix="fase-artifact-check-") as td:
        env_dir = Path(td) / "venv"
        example_dir = Path(td) / "quickstart"
        packaged_example_dir = Path(td) / "quickstart_fixture"
        inspect_model = Path(td) / "inspect_model.json"
        compare_model = Path(td) / "compare_model.json"
        inspect_model.write_text(
            json.dumps(
                {
                    "format": "FASEModel.to_dict",
                    "schema_version": api.FASE_MODEL_SCHEMA_VERSION,
                    "version": api.__version__,
                    "feature_schema": {
                        "feature_names": ["x0"],
                        "feature_count": 1,
                        "target": "y",
                    },
                    "model": {"w": [1.0], "b0": 0.0, "stage1": [], "stage2": []},
                }
            ),
            encoding="utf-8",
        )
        compare_model.write_text(
            json.dumps(
                {
                    "format": "FASEModel.to_dict",
                    "version": api.__version__,
                    "feature_schema": {
                        "feature_names": ["x0", "x1"],
                        "feature_count": 2,
                        "target": "y",
                    },
                    "model": {
                        "w": [1.0, 0.0],
                        "b0": 0.0,
                        "stage1": [],
                        "stage2": [{"kind": "bilinear", "params": {}, "Gamma": [], "mus": [0.0], "sds": [1.0]}],
                    },
                }
            ),
            encoding="utf-8",
        )
        create_cmd = [sys.executable, "-m", "venv"]
        if not args.isolated_deps:
            create_cmd.append("--system-site-packages")
        create_cmd.append(str(env_dir))
        steps = [_run_artifact_step("create-venv", create_cmd, verbose=args.verbose)]

        env_python = _venv_executable(env_dir, "python")
        env_fase = _venv_executable(env_dir, "fase")
        steps.append(
            _run_artifact_step(
                "install-wheel",
                [str(env_python), "-m", "pip", "install", "--no-deps", str(wheel)],
                verbose=args.verbose,
            )
        )
        steps.append(
            _run_artifact_step(
                "fase-inspect-model",
                [str(env_fase), "inspect-model", "--model", str(inspect_model), "--json"],
                verbose=args.verbose,
            )
        )
        steps.append(
            _run_artifact_step(
                "fase-validate-model",
                [str(env_fase), "validate-model", "--model", str(inspect_model), "--json"],
                verbose=args.verbose,
            )
        )
        steps.append(
            _run_artifact_step(
                "fase-compare-models",
                [str(env_fase), "compare-models", "--left", str(inspect_model), "--right", str(compare_model), "--json"],
                verbose=args.verbose,
            )
        )
        steps.append(_run_artifact_step("fase-version", [str(env_fase), "version"], verbose=args.verbose))
        steps.append(_run_artifact_step("fase-release-notes", [str(env_fase), "release-notes", "--json"], verbose=args.verbose))
        steps.append(_run_artifact_step("fase-check", [str(env_fase), "check"], verbose=args.verbose))
        steps.append(
            _run_artifact_step(
                "fase-init-example",
                [str(env_fase), "init-example", "--output", str(example_dir)],
                verbose=args.verbose,
            )
        )
        steps.append(
            _run_artifact_step(
                "fase-copy-examples",
                [str(env_fase), "copy-examples", "--output", str(packaged_example_dir)],
                verbose=args.verbose,
            )
        )

    rows = [
        {
            "step": step["step"],
            "status": step["status"].upper(),
            "returncode": step["returncode"],
        }
        for step in steps
    ]
    passed = all(step["status"] == "pass" for step in steps)
    print("FASE artifact check")
    print(f"status={'ok' if passed else 'needs_attention'}")
    print(f"wheel={wheel}")
    _print_table(rows)
    return 0 if passed else 1


def _print_v7_summary(report: Dict[str, Any]) -> None:
    compact = api.summarize_v7_report(report)
    rows = []
    for mode, metrics in compact.items():
        rows.append(
            {
                "mode": mode,
                "regime_acc": f"{metrics['regime_accuracy']:.4f}",
                "exact": f"{metrics['exact_agreement']:.4f}",
                "hamming": f"{metrics['avg_hamming']:.4f}",
                "purity": f"{metrics['mean_template_purity']:.4f}",
                "ops": f"{metrics['mean_num_ops']:.2f}",
                "gate": metrics["gate_all_pass"],
            }
        )
    _print_table(rows)


def _print_model_export_summary(export: Dict[str, Any]) -> None:
    rows = [
        {
            "field": "status",
            "value": export.get("status", "unknown"),
        },
        {
            "field": "path",
            "value": export.get("path", ""),
        },
        {
            "field": "format",
            "value": export.get("format", ""),
        },
    ]
    if export.get("error"):
        rows.append({"field": "error", "value": export["error"]})

    preflight = export.get("preflight") or {}
    if preflight:
        rows.extend(
            [
                {"field": "preflight", "value": preflight.get("status", "unknown")},
                {"field": "stage1_count", "value": preflight.get("stage1_count", "")},
                {"field": "stage2_count", "value": preflight.get("stage2_count", "")},
            ]
        )

    print("Model export")
    _print_table(rows)

    block_rows = []
    preflight_blocks = preflight.get("blocks", []) if isinstance(preflight, dict) else []
    for block in preflight_blocks:
        errors = block.get("errors") or []
        warnings = block.get("warnings") or []
        if errors or warnings or block.get("supported") is False:
            block_rows.append(
                {
                    "index": block.get("index"),
                    "kind": block.get("kind", ""),
                    "supported": block.get("supported"),
                    "issues": "; ".join([*errors, *warnings]),
                }
            )
    if block_rows:
        print("Model export preflight issues")
        _print_table(block_rows)


def cmd_v7_lab(args: argparse.Namespace) -> int:
    if args.dry_run:
        print("v7 lab dry run")
        print(f"fase_module={args.fase_module}")
        print(f"output_dir={args.output_dir}")
        print(f"episodes_per_combo={args.episodes_per_combo}")
        print(f"horizon={args.horizon}")
        print(f"replay_rounds={args.replay_rounds}")
        print(f"reuse_checkpoints={args.reuse_checkpoints}")
        print(f"fast={args.fast}")
        return 0

    harness = importlib.import_module("v7_regime_first_lab_harness")
    config_patch = _fast_fase_config(args.k_folds, args.seed) if args.fast else None
    report = harness.run_v7_lab_harness(
        fase_v21_path=args.fase_module,
        output_dir=args.output_dir,
        episodes_per_combo=args.episodes_per_combo,
        horizon=args.horizon,
        replay_rounds=args.replay_rounds,
        reuse_checkpoints=args.reuse_checkpoints,
        config_patch=config_patch,
    )
    _print_v7_summary(report)
    return 0


def cmd_report(args: argparse.Namespace) -> int:
    data = api.read_json(args.path)
    if all(isinstance(v, dict) and "metrics" in v for v in data.values()):
        _print_v7_summary(data)
    elif {"R2_oof", "MSE_oof"}.issubset(data.keys()):
        rows = [
            {
                "R2_oof": f"{float(data['R2_oof']):.4f}",
                "MSE_oof": f"{float(data['MSE_oof']):.6f}",
                "consensus_ops": data.get("num_consensus_ops", len(data.get("og_stability", {}))),
            }
        ]
        _print_table(rows)
        if isinstance(data.get("model_export"), dict):
            _print_model_export_summary(data["model_export"])
    else:
        print(f"Read JSON report: {Path(args.path)}")
        print(f"Top-level keys: {', '.join(sorted(data.keys()))}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="fase", description="FASE symbolic-regression and lab-harness CLI.")
    sub = parser.add_subparsers(dest="command", required=True)

    version = sub.add_parser("version", help="Print the installed FASE version.")
    version.set_defaults(func=cmd_version)

    demo = sub.add_parser("demo", help="Run a synthetic FASE v21 smoke/demo fit.")
    demo.add_argument("--output", default="outputs/fase_demo_report.json", help="Path for the compact JSON report.")
    demo.add_argument("--n", type=int, default=120, help="Synthetic sample count.")
    demo.add_argument("--d", type=int, default=6, help="Synthetic feature count.")
    demo.add_argument("--k-folds", type=int, default=2, help="Number of CV folds.")
    demo.add_argument("--seed", type=int, default=42, help="Random seed.")
    demo.add_argument("--fast", action="store_true", help="Use a smaller configuration for quick checks.")
    demo.add_argument("--compare-pysr", action="store_true", help="Enable PySR baseline if installed.")
    demo.add_argument("--no-gls-noise", action="store_true", help="Disable synthetic heteroskedastic noise.")
    demo.set_defaults(func=cmd_demo)

    init_example = sub.add_parser("init-example", help="Write a deterministic numeric quickstart dataset.")
    init_example.add_argument("--output", default="outputs/fase_quickstart", help="Directory to write train.csv, predict.csv, and README.md.")
    init_example.add_argument("--force", action="store_true", help="Overwrite existing quickstart example files.")
    init_example.set_defaults(func=cmd_init_example)

    copy_examples = sub.add_parser("copy-examples", help="Copy packaged quickstart fixtures to a local directory.")
    copy_examples.add_argument("--output", default="outputs/fase_quickstart_fixture", help="Directory to write README.md, train.csv, predict.csv, and model.json.")
    copy_examples.add_argument("--force", action="store_true", help="Overwrite existing quickstart fixture files.")
    copy_examples.set_defaults(func=cmd_copy_examples)

    fit = sub.add_parser("fit", help="Fit FASE v21 on a numeric CSV dataset.")
    fit.add_argument("--csv", required=True, help="Path to a numeric CSV file.")
    fit.add_argument("--target", help="Target column name or zero-based index. Defaults to the last column.")
    fit.add_argument("--output", default="outputs/fase_fit_report.json", help="Path for the compact JSON report.")
    fit.add_argument("--delimiter", default=",", help="CSV delimiter.")
    fit.add_argument("--no-header", action="store_true", help="Treat the CSV as headerless.")
    fit.add_argument("--k-folds", type=int, default=2, help="Number of CV folds.")
    fit.add_argument("--seed", type=int, default=42, help="Random seed.")
    fit.add_argument("--fast", action="store_true", help="Use a smaller configuration for quick checks.")
    fit.add_argument("--compare-pysr", action="store_true", help="Enable PySR baseline if installed.")
    fit.add_argument("--dry-run", action="store_true", help="Validate and summarize CSV settings without fitting.")
    fit.add_argument("--model-output", help="Optional path for exported FASEModel JSON.")
    fit.set_defaults(func=cmd_fit)

    export = sub.add_parser("export-model", help="Fit a numeric CSV and export the consensus FASE model JSON.")
    export.add_argument("--csv", required=True, help="Path to a numeric CSV file.")
    export.add_argument("--target", help="Target column name or zero-based index. Defaults to the last column.")
    export.add_argument("--output", required=True, help="Path for exported FASEModel JSON.")
    export.add_argument("--report-output", default="outputs/fase_export_model_report.json", help="Path for the compact JSON fit report.")
    export.add_argument("--delimiter", default=",", help="CSV delimiter.")
    export.add_argument("--no-header", action="store_true", help="Treat the CSV as headerless.")
    export.add_argument("--k-folds", type=int, default=2, help="Number of CV folds.")
    export.add_argument("--seed", type=int, default=42, help="Random seed.")
    export.add_argument("--fast", action="store_true", help="Use a smaller configuration for quick checks.")
    export.add_argument("--compare-pysr", action="store_true", help="Enable PySR baseline if installed.")
    export.add_argument("--dry-run", action="store_true", help="Validate and summarize CSV settings without fitting.")
    export.set_defaults(func=cmd_export_model)

    inspect = sub.add_parser("inspect-model", help="Inspect exported FASEModel JSON metadata without loading data.")
    inspect.add_argument("--model", required=True, help="Path to an exported FASEModel JSON.")
    inspect.add_argument("--json", action="store_true", help="Print inspection as JSON.")
    inspect.add_argument("--output", help="Optional path to write inspection JSON.")
    inspect.set_defaults(func=cmd_inspect_model)

    validate = sub.add_parser("validate-model", help="Validate exported FASEModel JSON structure without loading model code.")
    validate.add_argument("--model", required=True, help="Path to an exported FASEModel JSON.")
    validate.add_argument("--json", action="store_true", help="Print validation as JSON.")
    validate.add_argument("--output", help="Optional path to write validation JSON.")
    validate.set_defaults(func=cmd_validate_model)

    compare = sub.add_parser("compare-models", help="Compare two exported FASEModel JSON files using metadata only.")
    compare.add_argument("--left", required=True, help="Path to the baseline exported FASEModel JSON.")
    compare.add_argument("--right", required=True, help="Path to the candidate exported FASEModel JSON.")
    compare.add_argument("--json", action="store_true", help="Print comparison as JSON.")
    compare.add_argument("--output", help="Optional path to write comparison JSON.")
    compare.set_defaults(func=cmd_compare_models)

    predict = sub.add_parser("predict", help="Generate predictions from an exported FASEModel JSON.")
    predict.add_argument("--model", required=True, help="Path to an exported FASEModel JSON.")
    predict.add_argument("--csv", required=True, help="Path to a numeric feature CSV.")
    predict.add_argument("--output", required=True, help="Path for prediction CSV output.")
    predict.add_argument("--drop-column", help="Optional column name or zero-based index to drop before prediction.")
    predict.add_argument("--delimiter", default=",", help="CSV delimiter.")
    predict.add_argument("--no-header", action="store_true", help="Treat the CSV as headerless.")
    predict.add_argument("--dry-run", action="store_true", help="Validate model/CSV settings without predicting.")
    predict.add_argument("--strict-schema", action="store_true", help="Fail if exported model feature schema does not match the CSV.")
    predict.set_defaults(func=cmd_predict)

    eval_model = sub.add_parser("eval-model", help="Evaluate an exported FASEModel JSON against a numeric CSV target.")
    eval_model.add_argument("--model", required=True, help="Path to an exported FASEModel JSON.")
    eval_model.add_argument("--csv", required=True, help="Path to a numeric CSV containing features and target.")
    eval_model.add_argument("--target", help="Target column name or zero-based index. Defaults to the last column.")
    eval_model.add_argument("--output", required=True, help="Path for evaluation JSON output.")
    eval_model.add_argument("--predictions-output", help="Optional path for prediction CSV output.")
    eval_model.add_argument("--delimiter", default=",", help="CSV delimiter.")
    eval_model.add_argument("--no-header", action="store_true", help="Treat the CSV as headerless.")
    eval_model.add_argument("--dry-run", action="store_true", help="Validate model/CSV settings without evaluating.")
    eval_model.add_argument("--strict-schema", action="store_true", help="Fail if exported model feature schema does not match the CSV.")
    eval_model.set_defaults(func=cmd_eval_model)

    check = sub.add_parser("check", help="Run product smoke checks for the installed FASE package.")
    check.add_argument("--fit", action="store_true", help="Also run a real fast fit/export/predict round trip.")
    check.set_defaults(func=cmd_check)

    status = sub.add_parser("status", help="Summarize release checks, artifacts, and git provenance.")
    status.add_argument("--root", default=".", help="Repository root to inspect. Defaults to the current directory.")
    status.add_argument("--all-artifacts", action="store_true", help="Include all dist artifacts instead of only the newest artifact per kind.")
    status.add_argument("--require-clean", action="store_true", help="Mark status invalid and exit non-zero if the git work tree is dirty or unavailable.")
    status.add_argument("--json", action="store_true", help="Print product status as JSON.")
    status.add_argument("--output", help="Optional path to write product status JSON.")
    status.add_argument("--strict", action="store_true", help="Exit non-zero unless release checks pass and artifacts are present.")
    status.set_defaults(func=cmd_status)

    doctor = sub.add_parser("doctor", help="Inspect the FASE runtime environment without fitting.")
    doctor.add_argument("--json", action="store_true", help="Print diagnostics as JSON.")
    doctor.add_argument("--output", help="Optional path to write diagnostics JSON.")
    doctor.set_defaults(func=cmd_doctor)

    clean = sub.add_parser("clean", help="Preview or remove generated FASE output artifacts.")
    clean.add_argument("--root", default=".", help="Root directory to scan. Defaults to the current directory.")
    clean.add_argument("--older-than-days", type=float, help="Only include generated outputs whose newest file is at least this old.")
    clean.add_argument("--yes", action="store_true", help="Actually remove generated outputs. Default is dry-run.")
    clean.add_argument("--json", action="store_true", help="Print the cleanup report as JSON.")
    clean.set_defaults(func=cmd_clean)

    release = sub.add_parser("release-check", help="Validate local release/package readiness without building.")
    release.add_argument("--root", default=".", help="Repository root to inspect. Defaults to the current directory.")
    release.add_argument("--json", action="store_true", help="Print release readiness as JSON.")
    release.add_argument("--output", help="Optional path to write release readiness JSON.")
    release.set_defaults(func=cmd_release_check)

    release_notes = sub.add_parser("release-notes", help="Print compact release handoff notes.")
    release_notes.add_argument("--root", default=".", help="Repository root to inspect. Defaults to the current directory.")
    release_notes.add_argument("--json", action="store_true", help="Print release notes as JSON.")
    release_notes.add_argument("--output", help="Optional path to write release notes JSON.")
    release_notes.set_defaults(func=cmd_release_notes)

    release_bundle = sub.add_parser("release-bundle", help="Write release notes and artifact manifest into one handoff directory.")
    release_bundle.add_argument("--root", default=".", help="Repository root to inspect. Defaults to the current directory.")
    release_bundle.add_argument("--output-dir", help="Directory for release_bundle.json, release_notes.json, and artifact_manifest.json.")
    release_bundle.add_argument("--timestamp", help="Optional timestamp label for the default output directory.")
    release_bundle.add_argument("--all-artifacts", action="store_true", help="Include all dist artifacts instead of only the newest artifact per kind.")
    release_bundle.add_argument("--require-clean", action="store_true", help="Mark the bundle invalid and exit non-zero if the git work tree is dirty or unavailable.")
    release_bundle.add_argument("--json", action="store_true", help="Print release bundle summary as JSON.")
    release_bundle.add_argument("--strict", action="store_true", help="Exit non-zero unless release checks pass and artifacts are present.")
    release_bundle.set_defaults(func=cmd_release_bundle)

    artifact = sub.add_parser("artifact-check", help="Install a built wheel in a temporary environment and run smoke checks.")
    artifact.add_argument("--root", default=".", help="Repository root used to find dist/*.whl when --wheel is omitted.")
    artifact.add_argument("--wheel", help="Specific wheel artifact to validate. Defaults to the newest dist/*.whl.")
    artifact.add_argument("--isolated-deps", action="store_true", help="Do not expose system site packages to the temporary venv.")
    artifact.add_argument("--dry-run", action="store_true", help="Print the planned artifact validation without running it.")
    artifact.add_argument("--verbose", action="store_true", help="Print subprocess output even when checks pass.")
    artifact.set_defaults(func=cmd_artifact_check)

    artifact_manifest = sub.add_parser("artifact-manifest", help="Summarize built wheel/sdist artifacts with hashes.")
    artifact_manifest.add_argument("--root", default=".", help="Repository root used to find dist artifacts.")
    artifact_manifest.add_argument("--artifact", action="append", help="Specific artifact path to include. May be passed multiple times.")
    artifact_manifest.add_argument(
        "--latest-only",
        action="store_true",
        help="When auto-discovering dist artifacts, include only the newest artifact per kind.",
    )
    artifact_manifest.add_argument("--json", action="store_true", help="Print artifact manifest as JSON.")
    artifact_manifest.add_argument("--output", help="Optional path to write artifact manifest JSON.")
    artifact_manifest.set_defaults(func=cmd_artifact_manifest)

    lab = sub.add_parser("v7-lab", help="Run the v7 regime-first lab harness.")
    lab.add_argument("--fase-module", default="FASE_v21.py", help="Path to the FASE v21 module file.")
    lab.add_argument("--output-dir", default="v7_lab_outputs", help="Output directory.")
    lab.add_argument("--episodes-per-combo", type=int, default=8, help="Teacher episodes per sensor/topology combo.")
    lab.add_argument("--horizon", type=int, default=100, help="Episode horizon.")
    lab.add_argument("--replay-rounds", type=int, default=2, help="Weighted replay rounds.")
    lab.add_argument("--k-folds", type=int, default=2, help="Fast-mode FASE folds.")
    lab.add_argument("--seed", type=int, default=42, help="Fast-mode seed.")
    lab.add_argument("--fast", action="store_true", help="Use a smaller FASE config for fast checks.")
    lab.add_argument("--dry-run", action="store_true", help="Print resolved parameters without running.")
    lab.add_argument("--reuse-checkpoints", dest="reuse_checkpoints", action="store_true", default=True)
    lab.add_argument("--no-reuse-checkpoints", dest="reuse_checkpoints", action="store_false")
    lab.set_defaults(func=cmd_v7_lab)

    report = sub.add_parser("report", help="Summarize a FASE or v7 JSON report.")
    report.add_argument("path", help="Path to a JSON report.")
    report.set_defaults(func=cmd_report)
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    return int(args.func(args))


if __name__ == "__main__":
    raise SystemExit(main())
