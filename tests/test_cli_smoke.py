from __future__ import annotations

import io
import json
import math
import os
import subprocess
import sys
import tarfile
import tempfile
import time
import unittest
import zipfile
from pathlib import Path

import numpy as np

from fase.api import (
    collect_artifact_manifest,
    collect_product_status,
    collect_release_notes,
    compare_exported_models,
    copy_quickstart_example,
    create_release_bundle,
    evaluate_exported_model,
    export_model_from_report,
    inspect_exported_model,
    load_csv_dataset,
    load_exported_model,
    preflight_model_export,
    predict_from_exported_model,
    read_json,
    validate_exported_model_payload,
    write_example_dataset,
)


class DummyModel:
    def to_dict(self) -> dict:
        return {"w": [1.0], "b0": 0.0, "stage1": [], "stage2": []}


class FailingModel:
    def to_dict(self) -> dict:
        raise ValueError("unsupported block")


class CliSmokeTests(unittest.TestCase):
    def run_cli(self, *args: str) -> subprocess.CompletedProcess[str]:
        return subprocess.run(
            [sys.executable, "-m", "fase.cli", *args],
            cwd=Path(__file__).resolve().parents[1],
            check=True,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )

    def test_version(self) -> None:
        result = self.run_cli("version")
        self.assertIn("FASE", result.stdout)

    def test_check_default(self) -> None:
        result = self.run_cli("check")
        self.assertIn("FASE check", result.stdout)
        self.assertIn("FASE check complete", result.stdout)

    def test_check_with_real_fast_fit(self) -> None:
        result = self.run_cli("check", "--fit")
        self.assertIn("real fast model export works", result.stdout)
        self.assertIn("FASE check complete", result.stdout)

    def test_doctor_default(self) -> None:
        result = self.run_cli("doctor")
        self.assertIn("FASE doctor", result.stdout)
        self.assertIn("module:FASE_v21", result.stdout)

    def test_doctor_json(self) -> None:
        result = self.run_cli("doctor", "--json")
        payload = json.loads(result.stdout)
        self.assertEqual(payload["status"], "ok")
        self.assertTrue(payload["required_ok"])
        self.assertEqual(payload["fase_version"], "0.1.0")

    def test_clean_dry_run_preserves_outputs(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            generated = root / "outputs"
            generated.mkdir()
            (generated / "report.json").write_text("{}", encoding="utf-8")
            keep = root / "keep.txt"
            keep.write_text("keep", encoding="utf-8")
            result = self.run_cli("clean", "--root", str(root))

            self.assertTrue(generated.exists())
            self.assertTrue(keep.exists())
        self.assertIn("FASE clean dry run", result.stdout)
        self.assertIn("outputs", result.stdout)

    def test_clean_yes_removes_generated_outputs_only(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            generated = root / "v7_lab_outputs"
            generated.mkdir()
            (generated / "summary.json").write_text("{}", encoding="utf-8")
            tmp_file = root / "student_round_0.pkl.tmp"
            tmp_file.write_text("tmp", encoding="utf-8")
            venv_cache = root / "venv" / "pkg" / "__pycache__"
            venv_cache.mkdir(parents=True)
            (venv_cache / "module.pyc").write_text("cache", encoding="utf-8")
            keep = root / "keep.txt"
            keep.write_text("keep", encoding="utf-8")

            result = self.run_cli("clean", "--root", str(root), "--yes")

            self.assertFalse(generated.exists())
            self.assertFalse(tmp_file.exists())
            self.assertTrue(venv_cache.exists())
            self.assertTrue(keep.exists())
        self.assertIn("removed=", result.stdout)

    def test_clean_yes_json_reports_removed_entries(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            generated = root / "outputs"
            generated.mkdir()
            (generated / "report.json").write_text("{}", encoding="utf-8")

            result = self.run_cli("clean", "--root", str(root), "--yes", "--json")
            report = json.loads(result.stdout)

            self.assertFalse(generated.exists())
            self.assertEqual(report["removed_count"], 1)
            self.assertEqual(report["removed"][0]["relative_path"], "outputs")
            self.assertEqual(report["entries"][0]["relative_path"], "outputs")

    def test_clean_older_than_days_filters_fresh_outputs(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            old_generated = root / "v7_lab_outputs_old"
            fresh_generated = root / "v7_lab_outputs_fresh"
            old_generated.mkdir()
            fresh_generated.mkdir()
            old_file = old_generated / "summary.json"
            fresh_file = fresh_generated / "summary.json"
            old_file.write_text("{}", encoding="utf-8")
            fresh_file.write_text("{}", encoding="utf-8")

            old_mtime = time.time() - (3 * 86400)
            os.utime(old_file, (old_mtime, old_mtime))
            os.utime(old_generated, (old_mtime, old_mtime))

            dry_run = self.run_cli("clean", "--root", str(root), "--older-than-days", "1", "--json")
            result = self.run_cli("clean", "--root", str(root), "--older-than-days", "1", "--yes")
            report = json.loads(dry_run.stdout)

            self.assertFalse(old_generated.exists())
            self.assertTrue(fresh_generated.exists())
            self.assertEqual(report["older_than_days"], 1.0)
            self.assertEqual(report["count"], 1)
            self.assertGreaterEqual(report["entries"][0]["age_days"], 2.0)

        self.assertIn("older_than_days=1.0", result.stdout)
        self.assertIn("removed=1", result.stdout)

    def test_release_check_default(self) -> None:
        result = self.run_cli("release-check")
        self.assertIn("FASE release check", result.stdout)
        self.assertIn("cli:release-check", result.stdout)
        self.assertIn("metadata:version", result.stdout)
        self.assertIn("metadata:optional-dependencies", result.stdout)
        self.assertIn("manifest:generated-excludes", result.stdout)

    def test_release_check_json(self) -> None:
        result = self.run_cli("release-check", "--json")
        payload = json.loads(result.stdout)
        check_names = {check["name"] for check in payload["checks"]}

        self.assertEqual(payload["status"], "ok")
        self.assertTrue(payload["required_ok"])
        self.assertEqual(payload["project_version"], "0.1.0")
        self.assertIn("cli:status", check_names)
        self.assertIn("metadata:classifiers", check_names)
        self.assertIn("metadata:package-data", check_names)
        self.assertIn("manifest:generated-excludes", check_names)

    def test_collect_product_status(self) -> None:
        report = collect_product_status(Path(__file__).resolve().parents[1])

        self.assertIn(report["status"], {"ok", "needs_attention"})
        self.assertEqual(report["package"]["project_name"], "fase-symbolic")
        self.assertIn(report["artifact_manifest_status"], {"ok", "missing"})
        self.assertFalse(report["require_clean"])
        self.assertTrue(report["clean_required_pass"])
        self.assertIn("git", report)

    def test_status_default_and_json(self) -> None:
        human = self.run_cli("status")
        json_result = self.run_cli("status", "--json")
        payload = json.loads(json_result.stdout)

        self.assertIn("FASE product status", human.stdout)
        self.assertIn("release_checks=", human.stdout)
        self.assertIn("artifacts=", human.stdout)
        self.assertIn(payload["status"], {"ok", "needs_attention"})
        self.assertEqual(payload["package"]["project_name"], "fase-symbolic")
        self.assertFalse(payload["require_clean"])
        self.assertTrue(payload["clean_required_pass"])
        self.assertIn("release_checks", payload)
        self.assertIn("artifact_manifest", payload)

    def test_status_strict_fails_for_incomplete_root(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            result = subprocess.run(
                [
                    sys.executable,
                    "-m",
                    "fase.cli",
                    "status",
                    "--root",
                    td,
                    "--json",
                    "--strict",
                ],
                cwd=Path(__file__).resolve().parents[1],
                check=False,
                text=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
            )
            payload = json.loads(result.stdout)

        self.assertEqual(result.returncode, 1)
        self.assertEqual(payload["status"], "needs_attention")
        self.assertFalse(payload["release_checks_required_ok"])

    def test_status_require_clean_fails_without_git(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            result = subprocess.run(
                [
                    sys.executable,
                    "-m",
                    "fase.cli",
                    "status",
                    "--root",
                    td,
                    "--json",
                    "--require-clean",
                ],
                cwd=Path(__file__).resolve().parents[1],
                check=False,
                text=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
            )
            payload = json.loads(result.stdout)

        self.assertEqual(result.returncode, 1)
        self.assertEqual(payload["status"], "needs_attention")
        self.assertTrue(payload["require_clean"])
        self.assertFalse(payload["clean_required_pass"])
        self.assertFalse(payload["git"]["available"])

    def test_collect_release_notes(self) -> None:
        report = collect_release_notes(Path(__file__).resolve().parents[1], cli_commands=["version", "release-notes"])

        self.assertEqual(report["status"], "ok")
        self.assertEqual(report["project_name"], "fase-symbolic")
        self.assertIn("version", report["cli_commands"])
        self.assertTrue(any("unittest" in command for command in report["verification_commands"]))
        self.assertTrue(any("--no-isolation" in command for command in report["verification_commands"]))

    def test_release_notes_default(self) -> None:
        result = self.run_cli("release-notes")

        self.assertIn("FASE release notes", result.stdout)
        self.assertIn("package=fase-symbolic version=0.1.0", result.stdout)
        self.assertIn("Verification commands", result.stdout)
        self.assertIn("fase release-check", result.stdout)

    def test_release_notes_json_and_output(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            out_path = Path(td) / "release_notes.json"
            result = self.run_cli("release-notes", "--json", "--output", str(out_path))
            stdout_report = json.loads(result.stdout)
            file_report = read_json(out_path)

        self.assertEqual(stdout_report["status"], "ok")
        self.assertTrue(stdout_report["release_checks_required_ok"])
        self.assertEqual(stdout_report["project_version"], "0.1.0")
        self.assertEqual(file_report["project_name"], "fase-symbolic")

    def test_create_release_bundle_writes_handoff_files(self) -> None:
        root = Path(__file__).resolve().parents[1]
        with tempfile.TemporaryDirectory() as td:
            report = create_release_bundle(
                root,
                output_dir=Path(td) / "bundle",
                timestamp="20260510T000000Z",
                cli_commands=["version", "release-bundle"],
            )
            bundle = read_json(report["files"]["release_bundle"])
            release_notes = read_json(report["files"]["release_notes"])
            artifact_manifest = read_json(report["files"]["artifact_manifest"])

        self.assertEqual(bundle["timestamp"], "20260510T000000Z")
        self.assertTrue(bundle["latest_only"])
        self.assertFalse(bundle["require_clean"])
        self.assertTrue(bundle["clean_required_pass"])
        self.assertTrue(bundle["release_checks_required_ok"])
        self.assertIn("git", bundle)
        self.assertIn("dirty", bundle["git"])
        self.assertIn("status_count", bundle["git"])
        self.assertEqual(release_notes["project_name"], "fase-symbolic")
        self.assertIn(artifact_manifest["status"], {"ok", "missing"})

    def test_release_bundle_cli_json_and_output_dir(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            out_dir = Path(td) / "release_bundle"
            result = self.run_cli(
                "release-bundle",
                "--output-dir",
                str(out_dir),
                "--timestamp",
                "20260510T000000Z",
                "--json",
            )
            stdout_report = json.loads(result.stdout)
            bundle = read_json(out_dir / "release_bundle.json")
            release_notes_exists = (out_dir / "release_notes.json").exists()
            artifact_manifest_exists = (out_dir / "artifact_manifest.json").exists()

        self.assertEqual(stdout_report["timestamp"], "20260510T000000Z")
        self.assertEqual(bundle["files"]["release_bundle"], str(out_dir / "release_bundle.json"))
        self.assertIn("git", stdout_report)
        self.assertEqual(stdout_report["git"]["dirty"], bundle["git"]["dirty"])
        self.assertTrue(release_notes_exists)
        self.assertTrue(artifact_manifest_exists)

    def test_release_bundle_require_clean_fails_without_git(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            root = Path(td) / "not_git"
            root.mkdir()
            out_dir = Path(td) / "release_bundle"
            result = subprocess.run(
                [
                    sys.executable,
                    "-m",
                    "fase.cli",
                    "release-bundle",
                    "--root",
                    str(root),
                    "--output-dir",
                    str(out_dir),
                    "--require-clean",
                    "--json",
                ],
                cwd=Path(__file__).resolve().parents[1],
                check=False,
                text=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
            )
            payload = json.loads(result.stdout)

        self.assertEqual(result.returncode, 1)
        self.assertEqual(payload["status"], "needs_attention")
        self.assertTrue(payload["require_clean"])
        self.assertFalse(payload["clean_required_pass"])
        self.assertFalse(payload["git"]["available"])

    def test_product_smoke_script_and_ci_workflow_are_wired(self) -> None:
        root = Path(__file__).resolve().parents[1]
        script = root / "scripts" / "product_smoke.sh"
        workflow = root / ".github" / "workflows" / "product-smoke.yml"
        release_doc = root / "docs" / "release.md"
        model_format_doc = root / "docs" / "model-format.md"
        example_model = root / "examples" / "quickstart" / "model.json"

        self.assertTrue(script.exists())
        self.assertTrue(workflow.exists())
        self.assertTrue(release_doc.exists())
        self.assertTrue(model_format_doc.exists())
        self.assertTrue(example_model.exists())
        self.assertIn("status --root", script.read_text(encoding="utf-8"))
        self.assertIn("release-notes --json", script.read_text(encoding="utf-8"))
        self.assertIn("release-bundle", script.read_text(encoding="utf-8"))
        self.assertIn("RELEASE_BUNDLE_DIR", script.read_text(encoding="utf-8"))
        self.assertIn("copy-examples", script.read_text(encoding="utf-8"))
        self.assertIn("--no-isolation", script.read_text(encoding="utf-8"))
        workflow_text = workflow.read_text(encoding="utf-8")
        self.assertIn("bash scripts/product_smoke.sh", workflow_text)
        self.assertIn("actions/upload-artifact", workflow_text)
        self.assertIn("outputs/fase_release_bundle", workflow_text)
        self.assertIn("dist/*", workflow_text)
        release_doc_text = release_doc.read_text(encoding="utf-8")
        self.assertIn("Release Decision Checklist", release_doc_text)
        self.assertIn("Final Release Handback", release_doc_text)
        self.assertIn("fase status --output outputs/fase_status_clean.json --strict --require-clean", release_doc_text)
        self.assertIn("fase release-bundle --output-dir outputs/fase_release_bundle_clean --strict --require-clean", release_doc_text)
        self.assertIn("FASEModel JSON Format", model_format_doc.read_text(encoding="utf-8"))

    def test_committed_quickstart_fixture_is_runnable(self) -> None:
        root = Path(__file__).resolve().parents[1]
        example = root / "examples" / "quickstart"
        model_path = example / "model.json"
        train_path = example / "train.csv"
        predict_path = example / "predict.csv"

        X, y, metadata = load_csv_dataset(train_path, target="y")
        validation = self.run_cli("validate-model", "--model", str(model_path), "--json")
        model = load_exported_model(model_path)

        with tempfile.TemporaryDirectory() as td:
            out_path = Path(td) / "predictions.csv"
            eval_path = Path(td) / "eval.json"
            predict_result = self.run_cli(
                "predict",
                "--model",
                str(model_path),
                "--csv",
                str(predict_path),
                "--drop-column",
                "y",
                "--output",
                str(out_path),
                "--strict-schema",
            )
            eval_result = self.run_cli(
                "eval-model",
                "--model",
                str(model_path),
                "--csv",
                str(predict_path),
                "--target",
                "y",
                "--output",
                str(eval_path),
                "--strict-schema",
            )
            eval_report = read_json(eval_path)
            rows = out_path.read_text(encoding="utf-8").strip().splitlines()

        np.testing.assert_allclose(model.predict(X), y)
        self.assertEqual(metadata["feature_names"], ["x0", "x1", "x2"])
        self.assertEqual(json.loads(validation.stdout)["status"], "ok")
        self.assertIn("Wrote predictions", predict_result.stdout)
        self.assertIn("Wrote evaluation", eval_result.stdout)
        self.assertEqual(eval_report["metrics"]["mse"], 0.0)
        self.assertEqual(rows[0], "row,prediction")
        self.assertEqual(len(rows), 5)

    def test_artifact_check_dry_run(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            wheel = Path(td) / "fase_symbolic-0.1.0-py3-none-any.whl"
            wheel.write_text("not a real wheel", encoding="utf-8")
            result = self.run_cli("artifact-check", "--wheel", str(wheel), "--dry-run")

        self.assertIn("FASE artifact check dry run", result.stdout)
        self.assertIn("fase_symbolic-0.1.0-py3-none-any.whl", result.stdout)
        self.assertIn("fase init-example", result.stdout)
        self.assertIn("fase copy-examples", result.stdout)
        self.assertIn("fase inspect-model", result.stdout)
        self.assertIn("fase validate-model", result.stdout)
        self.assertIn("fase compare-models", result.stdout)
        self.assertIn("fase release-notes", result.stdout)

    def test_artifact_manifest_reports_hashes_and_fixture_members(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            dist = root / "dist"
            dist.mkdir()
            wheel = dist / "fase_symbolic-0.1.0-py3-none-any.whl"
            sdist = dist / "fase_symbolic-0.1.0.tar.gz"

            with zipfile.ZipFile(wheel, "w") as zf:
                zf.writestr("fase/__init__.py", "")
                zf.writestr("fase/data/quickstart/__init__.py", "")
                zf.writestr(
                    "fase_symbolic-0.1.0.dist-info/METADATA",
                    "Metadata-Version: 2.4\nName: fase-symbolic\nVersion: 0.1.0\n",
                )
                zf.writestr(
                    "fase_symbolic-0.1.0.dist-info/entry_points.txt",
                    "[console_scripts]\nfase = fase.cli:main\n",
                )
                for name in ["README.md", "train.csv", "predict.csv", "model.json"]:
                    zf.writestr(f"fase/data/quickstart/{name}", "{}")
            payload = b"# fixture\n"
            with tarfile.open(sdist, "w:gz") as tf:
                pyproject_payload = (
                    '[project]\nname = "fase-symbolic"\nversion = "0.1.0"\n'
                    '[project.scripts]\nfase = "fase.cli:main"\n'
                ).encode("utf-8")
                info = tarfile.TarInfo("fase_symbolic-0.1.0/pyproject.toml")
                info.size = len(pyproject_payload)
                tf.addfile(info, io.BytesIO(pyproject_payload))
                for member in [
                    "examples/quickstart/README.md",
                    "examples/quickstart/train.csv",
                    "examples/quickstart/predict.csv",
                    "examples/quickstart/model.json",
                    "fase/data/quickstart/README.md",
                    "fase/data/quickstart/__init__.py",
                    "fase/data/quickstart/train.csv",
                    "fase/data/quickstart/predict.csv",
                    "fase/data/quickstart/model.json",
                ]:
                    info = tarfile.TarInfo(f"fase_symbolic-0.1.0/{member}")
                    info.size = len(payload)
                    tf.addfile(info, io.BytesIO(payload))

            report = collect_artifact_manifest(root)
            result = self.run_cli("artifact-manifest", "--root", str(root), "--json")
            stdout_report = json.loads(result.stdout)

        self.assertEqual(report["status"], "ok")
        self.assertEqual(report["artifact_count"], 2)
        self.assertEqual(stdout_report["artifact_count"], 2)
        self.assertTrue(all(len(item["sha256"]) == 64 for item in report["artifacts"]))
        self.assertTrue(all(item["quickstart_complete"] for item in report["artifacts"]))
        self.assertTrue(all(item["artifact_metadata"]["valid"] for item in report["artifacts"]))
        self.assertTrue(any("fase/data/quickstart/model.json" in item["quickstart_files"] for item in report["artifacts"]))
        self.assertTrue(any("examples/quickstart/README.md" in item["quickstart_files"][0] for item in report["artifacts"] if item["quickstart_files"]))

    def test_artifact_manifest_flags_missing_quickstart_members(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            dist = root / "dist"
            dist.mkdir()
            wheel = dist / "fase_symbolic-0.1.0-py3-none-any.whl"

            with zipfile.ZipFile(wheel, "w") as zf:
                zf.writestr("fase/__init__.py", "")
                zf.writestr("fase/data/quickstart/model.json", "{}")

            report = collect_artifact_manifest(root)
            result = subprocess.run(
                [sys.executable, "-m", "fase.cli", "artifact-manifest", "--root", str(root), "--json"],
                cwd=Path(__file__).resolve().parents[1],
                check=False,
                text=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
            )
            stdout_report = json.loads(result.stdout)

        self.assertEqual(report["status"], "needs_attention")
        self.assertEqual(result.returncode, 1)
        self.assertFalse(report["artifacts"][0]["quickstart_complete"])
        self.assertIn("fase/data/quickstart/README.md", report["artifacts"][0]["quickstart_missing"])
        self.assertEqual(stdout_report["status"], "needs_attention")

    def test_artifact_manifest_flags_bad_wheel_metadata(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            dist = root / "dist"
            dist.mkdir()
            wheel = dist / "fase_symbolic-0.1.0-py3-none-any.whl"

            with zipfile.ZipFile(wheel, "w") as zf:
                zf.writestr("fase/__init__.py", "")
                zf.writestr("fase/data/quickstart/__init__.py", "")
                zf.writestr(
                    "fase_symbolic-0.1.0.dist-info/METADATA",
                    "Metadata-Version: 2.4\nName: wrong-name\nVersion: 0.1.0\n",
                )
                zf.writestr(
                    "fase_symbolic-0.1.0.dist-info/entry_points.txt",
                    "[console_scripts]\nfase = fase.cli:main\n",
                )
                for name in ["README.md", "train.csv", "predict.csv", "model.json"]:
                    zf.writestr(f"fase/data/quickstart/{name}", "{}")

            report = collect_artifact_manifest(root)

        self.assertEqual(report["status"], "needs_attention")
        self.assertTrue(report["artifacts"][0]["quickstart_complete"])
        self.assertFalse(report["artifacts"][0]["artifact_metadata"]["valid"])
        self.assertIn("Wheel metadata Name mismatch: wrong-name", report["artifacts"][0]["warnings"])

    def test_artifact_manifest_latest_only_keeps_newest_per_kind(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            dist = root / "dist"
            dist.mkdir()
            old_wheel = dist / "fase_symbolic-0.1.0-py3-none-any.whl"
            new_wheel = dist / "fase_symbolic-0.1.1-py3-none-any.whl"
            old_sdist = dist / "fase_symbolic-0.1.0.tar.gz"
            new_sdist = dist / "fase_symbolic-0.1.1.tar.gz"

            with zipfile.ZipFile(old_wheel, "w") as zf:
                zf.writestr("old.txt", "")
            with zipfile.ZipFile(new_wheel, "w") as zf:
                zf.writestr("fase/data/quickstart/__init__.py", "")
                zf.writestr(
                    "fase_symbolic-0.1.1.dist-info/METADATA",
                    "Metadata-Version: 2.4\nName: fase-symbolic\nVersion: 0.1.0\n",
                )
                zf.writestr(
                    "fase_symbolic-0.1.1.dist-info/entry_points.txt",
                    "[console_scripts]\nfase = fase.cli:main\n",
                )
                for name in ["README.md", "train.csv", "predict.csv", "model.json"]:
                    zf.writestr(f"fase/data/quickstart/{name}", "{}")
            payload = b"# fixture\n"
            info = tarfile.TarInfo("fase_symbolic-0.1.0/old.txt")
            info.size = len(payload)
            with tarfile.open(old_sdist, "w:gz") as tf:
                tf.addfile(info, io.BytesIO(payload))
            with tarfile.open(new_sdist, "w:gz") as tf:
                pyproject_payload = (
                    '[project]\nname = "fase-symbolic"\nversion = "0.1.0"\n'
                    '[project.scripts]\nfase = "fase.cli:main"\n'
                ).encode("utf-8")
                info = tarfile.TarInfo("fase_symbolic-0.1.1/pyproject.toml")
                info.size = len(pyproject_payload)
                tf.addfile(info, io.BytesIO(pyproject_payload))
                for member in [
                    "examples/quickstart/README.md",
                    "examples/quickstart/train.csv",
                    "examples/quickstart/predict.csv",
                    "examples/quickstart/model.json",
                    "fase/data/quickstart/README.md",
                    "fase/data/quickstart/__init__.py",
                    "fase/data/quickstart/train.csv",
                    "fase/data/quickstart/predict.csv",
                    "fase/data/quickstart/model.json",
                ]:
                    info = tarfile.TarInfo(f"fase_symbolic-0.1.1/{member}")
                    info.size = len(payload)
                    tf.addfile(info, io.BytesIO(payload))

            now = time.time()
            for path in [old_wheel, old_sdist]:
                os.utime(path, (now - 1000, now - 1000))
            for path in [new_wheel, new_sdist]:
                os.utime(path, (now, now))

            report = collect_artifact_manifest(root, latest_only=True)
            result = self.run_cli("artifact-manifest", "--root", str(root), "--latest-only", "--json")
            stdout_report = json.loads(result.stdout)

        self.assertTrue(report["latest_only"])
        self.assertEqual(report["artifact_count"], 2)
        self.assertEqual({item["name"] for item in report["artifacts"]}, {new_wheel.name, new_sdist.name})
        self.assertEqual(stdout_report["artifact_count"], 2)
        self.assertEqual({item["name"] for item in stdout_report["artifacts"]}, {new_wheel.name, new_sdist.name})

    def test_write_example_dataset_guards_overwrite(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            root = Path(td) / "quickstart"
            report = write_example_dataset(root)
            X, y, meta = load_csv_dataset(root / "train.csv", target="y")
            with self.assertRaises(FileExistsError):
                write_example_dataset(root)

        self.assertEqual(report["train_rows"], 27)
        self.assertEqual(X.shape, (27, 3))
        self.assertEqual(y.shape, (27,))
        self.assertEqual(meta["feature_names"], ["x0", "x1", "x2"])

    def test_init_example_command_writes_runnable_dataset(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            root = Path(td) / "quickstart"
            predictions_path = Path(td) / "predictions.csv"
            result = self.run_cli("init-example", "--output", str(root))
            fit_dry_run = self.run_cli(
                "fit",
                "--csv",
                str(root / "train.csv"),
                "--target",
                "y",
                "--dry-run",
                "--fast",
            )
            predict_dry_run = self.run_cli(
                "predict",
                "--model",
                str(root / "model.json"),
                "--csv",
                str(root / "predict.csv"),
                "--drop-column",
                "y",
                "--output",
                str(predictions_path),
                "--dry-run",
            )

        self.assertIn("Wrote FASE quickstart example", result.stdout)
        self.assertIn("train_rows=27", result.stdout)
        self.assertIn("CSV fit dry run", fit_dry_run.stdout)
        self.assertIn("rows=27", fit_dry_run.stdout)
        self.assertIn("Prediction dry run", predict_dry_run.stdout)
        self.assertIn("features=3", predict_dry_run.stdout)

    def test_copy_examples_command_writes_packaged_fixture(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            root = Path(td) / "quickstart_fixture"
            predictions_path = Path(td) / "predictions.csv"
            eval_path = Path(td) / "eval.json"
            result = self.run_cli("copy-examples", "--output", str(root))
            with self.assertRaises(FileExistsError):
                copy_quickstart_example(root)
            predict_result = self.run_cli(
                "predict",
                "--model",
                str(root / "model.json"),
                "--csv",
                str(root / "predict.csv"),
                "--drop-column",
                "y",
                "--output",
                str(predictions_path),
                "--strict-schema",
            )
            eval_result = self.run_cli(
                "eval-model",
                "--model",
                str(root / "model.json"),
                "--csv",
                str(root / "predict.csv"),
                "--target",
                "y",
                "--output",
                str(eval_path),
                "--strict-schema",
            )
            eval_report = read_json(eval_path)

        self.assertIn("Copied packaged FASE quickstart fixture", result.stdout)
        self.assertIn("model_json=", result.stdout)
        self.assertIn("Wrote predictions", predict_result.stdout)
        self.assertIn("Wrote evaluation", eval_result.stdout)
        self.assertEqual(eval_report["metrics"]["mse"], 0.0)

    def test_v7_dry_run(self) -> None:
        result = self.run_cli("v7-lab", "--dry-run", "--fast")
        self.assertIn("v7 lab dry run", result.stdout)
        self.assertIn("fast=True", result.stdout)

    def test_load_csv_dataset_target_name(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            path = Path(td) / "tiny.csv"
            path.write_text("a,b,y\n1,2,3\n4,5,6\n", encoding="utf-8")
            X, y, meta = load_csv_dataset(path, target="y")
        self.assertEqual(X.shape, (2, 2))
        self.assertEqual(y.tolist(), [3.0, 6.0])
        self.assertEqual(meta["target"], "y")
        self.assertEqual(meta["feature_names"], ["a", "b"])

    def test_load_csv_dataset_headerless_target_index(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            path = Path(td) / "tiny.csv"
            path.write_text("1,2,3\n4,5,6\n", encoding="utf-8")
            X, y, meta = load_csv_dataset(path, target="0", has_header=False)
        self.assertEqual(X.tolist(), [[2.0, 3.0], [5.0, 6.0]])
        self.assertEqual(y.tolist(), [1.0, 4.0])
        self.assertEqual(meta["target"], "x0")

    def test_fit_csv_dry_run(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            path = Path(td) / "tiny.csv"
            path.write_text("a,b,y\n1,2,3\n4,5,6\n", encoding="utf-8")
            result = self.run_cli("fit", "--csv", str(path), "--target", "y", "--dry-run", "--fast")
        self.assertIn("CSV fit dry run", result.stdout)
        self.assertIn("target=y", result.stdout)

    def test_export_model_from_report_success(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            path = Path(td) / "model.json"
            status = export_model_from_report(
                {"model": DummyModel()},
                path,
                dataset_metadata={"feature_names": ["a"], "target": "y"},
            )
            payload = read_json(path)
        self.assertEqual(status["status"], "exported")
        self.assertEqual(payload["format"], "FASEModel.to_dict")
        self.assertEqual(payload["schema_version"], "fase-model-v1")
        self.assertEqual(payload["model"]["w"], [1.0])
        self.assertEqual(payload["feature_schema"]["feature_names"], ["a"])
        self.assertEqual(payload["feature_schema"]["target"], "y")

    def test_validate_exported_model_payload(self) -> None:
        valid = {
            "format": "FASEModel.to_dict",
            "schema_version": "fase-model-v1",
            "version": "0.1.0",
            "feature_schema": {"feature_names": ["a"], "feature_count": 1, "target": "y"},
            "model": {"w": [1.0], "b0": 0.0, "stage1": [], "stage2": []},
        }
        invalid = {
            "format": "FASEModel.to_dict",
            "schema_version": "fase-model-v1",
            "feature_schema": {"feature_names": ["a", "b"], "feature_count": 1},
            "model": {"w": "bad", "b0": "bad", "stage1": {}, "stage2": []},
        }
        bare = {"w": [], "b0": 2.5, "stage1": [], "stage2": []}

        self.assertEqual(validate_exported_model_payload(valid)["status"], "ok")
        invalid_report = validate_exported_model_payload(invalid)
        bare_report = validate_exported_model_payload(bare)
        self.assertEqual(invalid_report["status"], "invalid")
        self.assertTrue(any("feature_count" in error for error in invalid_report["errors"]))
        self.assertEqual(bare_report["status"], "ok")
        self.assertTrue(any("Bare FASEModel" in warning for warning in bare_report["warnings"]))

    def test_validate_model_cli_json_output_and_failure(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            valid_path = root / "valid.json"
            invalid_path = root / "invalid.json"
            out_path = root / "validation.json"
            valid_path.write_text(
                json.dumps(
                    {
                        "format": "FASEModel.to_dict",
                        "schema_version": "fase-model-v1",
                        "version": "0.1.0",
                        "feature_schema": {"feature_names": ["a"], "feature_count": 1},
                        "model": {"w": [1.0], "b0": 0.0, "stage1": [], "stage2": []},
                    }
                ),
                encoding="utf-8",
            )
            invalid_path.write_text(
                json.dumps(
                    {
                        "format": "FASEModel.to_dict",
                        "schema_version": "fase-model-v1",
                        "model": {"w": "bad", "b0": 0.0, "stage1": [], "stage2": []},
                    }
                ),
                encoding="utf-8",
            )
            result = self.run_cli("validate-model", "--model", str(valid_path), "--json", "--output", str(out_path))
            stdout_report = json.loads(result.stdout)
            file_report = read_json(out_path)
            failed = subprocess.run(
                [sys.executable, "-m", "fase.cli", "validate-model", "--model", str(invalid_path), "--json"],
                cwd=Path(__file__).resolve().parents[1],
                check=False,
                text=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
            )

        self.assertEqual(stdout_report["status"], "ok")
        self.assertEqual(file_report["status"], "ok")
        self.assertNotEqual(failed.returncode, 0)
        self.assertEqual(json.loads(failed.stdout)["status"], "invalid")

    def test_export_model_from_report_failure(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            path = Path(td) / "model.json"
            status = export_model_from_report({"model": FailingModel()}, path)
        self.assertEqual(status["status"], "failed")
        self.assertIn("unsupported block", status["error"])
        self.assertEqual(status["preflight"]["status"], "ok")

    def test_export_model_preflight_reports_unsupported_stage2_block(self) -> None:
        import FASE_v21 as fase_v21

        model = fase_v21.FASEModel(
            stage1_specs=[],
            stage2_blocks=[
                {
                    "kind": "mystery_block",
                    "params": {},
                    "Gamma": np.zeros((0, 1)),
                    "mus": np.zeros(1),
                    "sds": np.ones(1),
                    "apply": lambda X: (np.zeros((0, 1)), np.zeros(1), np.ones(1), X[:, [0]]),
                }
            ],
            w=np.array([1.0]),
            b0=0.0,
        )

        with tempfile.TemporaryDirectory() as td:
            path = Path(td) / "unsupported_model.json"
            preflight = preflight_model_export(model)
            status = export_model_from_report({"model": model}, path)

        self.assertEqual(preflight["status"], "unsupported")
        self.assertEqual(status["status"], "unsupported")
        self.assertFalse(path.exists())
        self.assertIn("stage2[0]", status["error"])
        self.assertIn("mystery_block", status["error"])
        self.assertEqual(status["preflight"]["blocks"][0]["kind"], "mystery_block")

    def test_inspect_exported_model_api(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            path = Path(td) / "model.json"
            path.write_text(
                json.dumps(
                    {
                        "format": "FASEModel.to_dict",
                        "version": "0.1.0",
                        "feature_schema": {
                            "feature_names": ["a", "b"],
                            "feature_count": 2,
                            "target": "y",
                        },
                        "model": {
                            "w": [1.0, 0.0, -2.0],
                            "b0": 0.5,
                            "stage1": [{"spec": {"kind": "raw", "i": 0}, "Gamma": [], "mu": [0.0], "sd": [1.0]}],
                            "stage2": [
                                {"kind": "bilinear", "params": {}, "Gamma": [], "mus": [0.0, 0.0], "sds": [1.0, 1.0]},
                                {"kind": "ruliad", "params": {}, "Gamma": [], "mus": [0.0], "sds": [1.0]},
                            ],
                        },
                    }
                ),
                encoding="utf-8",
            )
            report = inspect_exported_model(path)

        self.assertEqual(report["format"], "FASEModel.to_dict")
        self.assertEqual(report["schema_validation"]["status"], "ok")
        self.assertEqual(report["feature_schema"]["feature_names"], ["a", "b"])
        self.assertEqual(report["model"]["linear_weights"], 3)
        self.assertEqual(report["model"]["nonzero_linear_weights"], 2)
        self.assertEqual(report["model"]["stage1_count"], 1)
        self.assertEqual(report["model"]["stage2_count"], 2)
        self.assertEqual(report["model"]["stage2_kinds"], {"bilinear": 1, "ruliad": 1})
        self.assertEqual(report["model"]["stage2_total_width"], 3)

    def test_inspect_model_cli_json_and_output(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            model_path = Path(td) / "model.json"
            out_path = Path(td) / "inspect.json"
            model_path.write_text(
                json.dumps(
                    {
                        "format": "FASEModel.to_dict",
                        "version": "0.1.0",
                        "feature_schema": {
                            "feature_names": ["a"],
                            "feature_count": 1,
                            "target": "y",
                        },
                        "model": {"w": [1.0], "b0": 0.0, "stage1": [], "stage2": []},
                    }
                ),
                encoding="utf-8",
            )
            result = self.run_cli("inspect-model", "--model", str(model_path), "--json", "--output", str(out_path))
            stdout_report = json.loads(result.stdout)
            file_report = read_json(out_path)

        self.assertEqual(stdout_report["feature_schema"]["feature_names"], ["a"])
        self.assertEqual(file_report["model"]["linear_weights"], 1)

    def test_inspect_model_cli_human_output(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            model_path = Path(td) / "model.json"
            model_path.write_text(
                json.dumps(
                    {
                        "format": "FASEModel.to_dict",
                        "version": "0.1.0",
                        "feature_schema": {
                            "feature_names": ["a"],
                            "feature_count": 1,
                            "target": "y",
                        },
                        "model": {
                            "w": [1.0, 2.0],
                            "b0": 0.0,
                            "stage1": [],
                            "stage2": [{"kind": "bilinear", "params": {}, "Gamma": [], "mus": [0.0], "sds": [1.0]}],
                        },
                    }
                ),
                encoding="utf-8",
            )
            result = self.run_cli("inspect-model", "--model", str(model_path))

        self.assertIn("FASE model inspection", result.stdout)
        self.assertIn("features=a", result.stdout)
        self.assertIn("bilinear", result.stdout)

    def test_compare_exported_models_api(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            left = Path(td) / "left.json"
            right = Path(td) / "right.json"
            left.write_text(
                json.dumps(
                    {
                        "format": "FASEModel.to_dict",
                        "version": "0.1.0",
                        "feature_schema": {
                            "feature_names": ["a"],
                            "feature_count": 1,
                            "target": "y",
                        },
                        "model": {"w": [1.0], "b0": 0.0, "stage1": [], "stage2": []},
                    }
                ),
                encoding="utf-8",
            )
            right.write_text(
                json.dumps(
                    {
                        "format": "FASEModel.to_dict",
                        "version": "0.1.0",
                        "feature_schema": {
                            "feature_names": ["a", "b"],
                            "feature_count": 2,
                            "target": "y",
                        },
                        "model": {
                            "w": [1.0, 0.0],
                            "b0": 0.0,
                            "stage1": [{"spec": {"kind": "raw", "i": 0}, "Gamma": [], "mu": [0.0], "sd": [1.0]}],
                            "stage2": [{"kind": "bilinear", "params": {}, "Gamma": [], "mus": [0.0], "sds": [1.0]}],
                        },
                    }
                ),
                encoding="utf-8",
            )
            report = compare_exported_models(left, right)

        self.assertFalse(report["feature_schema"]["same_feature_names"])
        self.assertEqual(report["feature_schema"]["right_only_features"], ["b"])
        self.assertEqual(report["complexity_delta"]["linear_weights"], 1)
        self.assertEqual(report["complexity_delta"]["stage1_count"], 1)
        self.assertEqual(report["stage2_kind_delta"]["bilinear"]["delta"], 1)

    def test_compare_models_cli_json_and_output(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            left = Path(td) / "left.json"
            right = Path(td) / "right.json"
            out_path = Path(td) / "compare.json"
            left.write_text(
                json.dumps(
                    {
                        "format": "FASEModel.to_dict",
                        "version": "0.1.0",
                        "feature_schema": {"feature_names": ["a"], "feature_count": 1},
                        "model": {"w": [1.0], "b0": 0.0, "stage1": [], "stage2": []},
                    }
                ),
                encoding="utf-8",
            )
            right.write_text(
                json.dumps(
                    {
                        "format": "FASEModel.to_dict",
                        "version": "0.1.0",
                        "feature_schema": {"feature_names": ["a"], "feature_count": 1},
                        "model": {"w": [1.0, 2.0], "b0": 0.0, "stage1": [], "stage2": []},
                    }
                ),
                encoding="utf-8",
            )
            result = self.run_cli("compare-models", "--left", str(left), "--right", str(right), "--json", "--output", str(out_path))
            stdout_report = json.loads(result.stdout)
            file_report = read_json(out_path)

        self.assertEqual(stdout_report["complexity_delta"]["linear_weights"], 1)
        self.assertTrue(file_report["feature_schema"]["same_feature_names"])

    def test_compare_models_cli_human_output(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            left = Path(td) / "left.json"
            right = Path(td) / "right.json"
            left.write_text(
                json.dumps(
                    {
                        "format": "FASEModel.to_dict",
                        "version": "0.1.0",
                        "feature_schema": {"feature_names": ["a"], "feature_count": 1},
                        "model": {"w": [1.0], "b0": 0.0, "stage1": [], "stage2": []},
                    }
                ),
                encoding="utf-8",
            )
            right.write_text(
                json.dumps(
                    {
                        "format": "FASEModel.to_dict",
                        "version": "0.1.0",
                        "feature_schema": {"feature_names": ["a", "b"], "feature_count": 2},
                        "model": {
                            "w": [1.0, 2.0],
                            "b0": 0.0,
                            "stage1": [],
                            "stage2": [{"kind": "bilinear", "params": {}, "Gamma": [], "mus": [0.0], "sds": [1.0]}],
                        },
                    }
                ),
                encoding="utf-8",
            )
            result = self.run_cli("compare-models", "--left", str(left), "--right", str(right))

        self.assertIn("FASE model comparison", result.stdout)
        self.assertIn("Complexity delta", result.stdout)
        self.assertIn("right_only_features=b", result.stdout)
        self.assertIn("bilinear", result.stdout)

    def test_ruliad_model_export_reload_predict_round_trip(self) -> None:
        import FASE_v21 as fase_v21

        registry = fase_v21.OpRegistry()
        nodes = {
            0: fase_v21.Node(id=0, op="id", parents=(0,), is_input=True),
            1: fase_v21.Node(id=1, op="square", parents=(0,)),
        }
        state = fase_v21.HypergraphState(
            registry=registry,
            nodes=nodes,
            output_ids=[1],
            input_ids=[0],
            rule_history=["test-square"],
        )
        gamma = np.zeros((0, 1))
        mus = np.array([0.0])
        sds = np.array([1.0])

        def apply_block(X, Gamma=gamma, mus=mus, sds=sds, state=state):
            return Gamma, mus, sds, state.features(X)

        model = fase_v21.FASEModel(
            stage1_specs=[],
            stage2_blocks=[
                {
                    "kind": "ruliad",
                    "params": {
                        "state": state,
                        "output_indices": None,
                        "nodes": len(nodes),
                        "rule_history_tail": state.rule_history[-8:],
                    },
                    "Gamma": gamma,
                    "mus": mus,
                    "sds": sds,
                    "apply": apply_block,
                }
            ],
            w=np.array([2.0]),
            b0=1.0,
        )

        with tempfile.TemporaryDirectory() as td:
            path = Path(td) / "ruliad_model.json"
            status = export_model_from_report({"model": model}, path)
            reloaded = load_exported_model(path)

        X = np.array([[2.0], [3.0]])
        self.assertEqual(status["status"], "exported")
        self.assertEqual(reloaded.predict(X).tolist(), [9.0, 19.0])

    def test_ruliad_closure_only_model_export_reload_predict_round_trip(self) -> None:
        import FASE_v21 as fase_v21

        registry = fase_v21.OpRegistry()
        nodes = {
            0: fase_v21.Node(id=0, op="id", parents=(0,), is_input=True),
            1: fase_v21.Node(id=1, op="square", parents=(0,)),
        }
        state = fase_v21.HypergraphState(
            registry=registry,
            nodes=nodes,
            output_ids=[0, 1],
            input_ids=[0],
            rule_history=["test-closure-square"],
        )
        output_indices = [1]
        gamma = np.zeros((0, 1))
        mus = np.array([0.0])
        sds = np.array([1.0])

        def state_features(X):
            features = state.features(X)
            return features[:, output_indices]

        def apply_block(X, Gamma=gamma, mus=mus, sds=sds):
            return Gamma, mus, sds, state_features(X)

        model = fase_v21.FASEModel(
            stage1_specs=[],
            stage2_blocks=[
                {
                    "kind": "ruliad",
                    "params": {
                        "state_features": state_features,
                        "nodes": len(nodes),
                        "rule_history_tail": state.rule_history[-8:],
                    },
                    "Gamma": gamma,
                    "mus": mus,
                    "sds": sds,
                    "apply": apply_block,
                }
            ],
            w=np.array([2.0]),
            b0=1.0,
        )

        with tempfile.TemporaryDirectory() as td:
            path = Path(td) / "ruliad_closure_model.json"
            status = export_model_from_report({"model": model}, path)
            payload = read_json(path)
            reloaded = load_exported_model(path)

        X = np.array([[2.0], [3.0]])
        self.assertEqual(status["status"], "exported")
        self.assertEqual(status["preflight"]["status"], "ok")
        self.assertTrue(status["preflight"]["blocks"][0]["ruliad"]["state_source"].startswith("state_features.closure["))
        self.assertEqual(payload["model"]["stage2"][0]["params"]["output_indices"], [1])
        self.assertEqual(reloaded.predict(X).tolist(), [9.0, 19.0])

    def test_group_invariant_block_export_reload_predict_round_trip(self) -> None:
        import FASE_v21 as fase_v21

        spec = fase_v21.GroupSpec(sign_groups=[[0, 1]], rot2d_pairs=[(0, 1)])
        width = fase_v21.block_group_invar(np.zeros((1, 2)), spec).shape[1]
        gamma = np.zeros((0, width))
        mus = np.zeros(width)
        sds = np.ones(width)

        def apply_block(X, Gamma=gamma, mus=mus, sds=sds, spec=spec):
            return Gamma, mus, sds, fase_v21.block_group_invar(X, spec)

        model = fase_v21.FASEModel(
            stage1_specs=[],
            stage2_blocks=[
                {
                    "kind": "group_invar",
                    "params": {"spec": spec},
                    "Gamma": gamma,
                    "mus": mus,
                    "sds": sds,
                    "apply": apply_block,
                }
            ],
            w=np.arange(1.0, float(width) + 1.0),
            b0=-0.5,
        )

        with tempfile.TemporaryDirectory() as td:
            path = Path(td) / "group_model.json"
            status = export_model_from_report({"model": model}, path)
            reloaded = load_exported_model(path)

        X = np.array([[1.0, 2.0], [-3.0, 4.0]])
        self.assertEqual(status["status"], "exported")
        np.testing.assert_allclose(reloaded.predict(X), model.predict(X))

    def test_export_model_dry_run(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            csv_path = Path(td) / "tiny.csv"
            model_path = Path(td) / "model.json"
            csv_path.write_text("a,b,y\n1,2,3\n4,5,6\n", encoding="utf-8")
            result = self.run_cli(
                "export-model",
                "--csv",
                str(csv_path),
                "--target",
                "y",
                "--output",
                str(model_path),
                "--dry-run",
                "--fast",
            )
        self.assertIn("CSV fit dry run", result.stdout)
        self.assertIn("target=y", result.stdout)

    def test_predict_from_exported_intercept_model(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            model_path = Path(td) / "model.json"
            csv_path = Path(td) / "features.csv"
            model_path.write_text(
                json.dumps(
                    {
                        "format": "FASEModel.to_dict",
                        "version": "0.1.0",
                        "model": {"w": [], "b0": 2.5, "stage1": [], "stage2": []},
                    }
                ),
                encoding="utf-8",
            )
            csv_path.write_text("a,b,y\n1,2,9\n4,5,8\n", encoding="utf-8")
            preds, meta = predict_from_exported_model(model_path, csv_path, drop_column="y")
        self.assertEqual(preds.tolist(), [2.5, 2.5])
        self.assertEqual(meta["dropped_column"], "y")

    def test_predict_warns_on_feature_schema_order_mismatch(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            model_path = Path(td) / "model.json"
            csv_path = Path(td) / "features.csv"
            model_path.write_text(
                json.dumps(
                    {
                        "format": "FASEModel.to_dict",
                        "version": "0.1.0",
                        "feature_schema": {
                            "feature_names": ["a", "b"],
                            "feature_count": 2,
                            "target": "y",
                        },
                        "model": {"w": [], "b0": 2.5, "stage1": [], "stage2": []},
                    }
                ),
                encoding="utf-8",
            )
            csv_path.write_text("b,a,y\n2,1,9\n5,4,8\n", encoding="utf-8")
            preds, meta = predict_from_exported_model(model_path, csv_path, drop_column="y")

        self.assertEqual(preds.tolist(), [2.5, 2.5])
        self.assertEqual(meta["model_feature_schema"]["feature_names"], ["a", "b"])
        self.assertTrue(any("order mismatch" in warning for warning in meta["schema_warnings"]))

    def test_predict_strict_schema_raises_on_order_mismatch(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            model_path = Path(td) / "model.json"
            csv_path = Path(td) / "features.csv"
            model_path.write_text(
                json.dumps(
                    {
                        "format": "FASEModel.to_dict",
                        "version": "0.1.0",
                        "feature_schema": {
                            "feature_names": ["a", "b"],
                            "feature_count": 2,
                            "target": "y",
                        },
                        "model": {"w": [], "b0": 2.5, "stage1": [], "stage2": []},
                    }
                ),
                encoding="utf-8",
            )
            csv_path.write_text("b,a,y\n2,1,9\n5,4,8\n", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "feature schema mismatch"):
                predict_from_exported_model(model_path, csv_path, drop_column="y", strict_schema=True)

    def test_predict_cli_with_exported_intercept_model(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            model_path = Path(td) / "model.json"
            csv_path = Path(td) / "features.csv"
            out_path = Path(td) / "predictions.csv"
            model_path.write_text(
                json.dumps(
                    {
                        "format": "FASEModel.to_dict",
                        "version": "0.1.0",
                        "model": {"w": [], "b0": 2.5, "stage1": [], "stage2": []},
                    }
                ),
                encoding="utf-8",
            )
            csv_path.write_text("a,b,y\n1,2,9\n4,5,8\n", encoding="utf-8")
            result = self.run_cli(
                "predict",
                "--model",
                str(model_path),
                "--csv",
                str(csv_path),
                "--drop-column",
                "y",
                "--output",
                str(out_path),
            )
            rows = out_path.read_text(encoding="utf-8").strip().splitlines()
        self.assertIn("Wrote predictions", result.stdout)
        self.assertEqual(rows, ["row,prediction", "0,2.5", "1,2.5"])

    def test_evaluate_exported_intercept_model(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            model_path = Path(td) / "model.json"
            csv_path = Path(td) / "features.csv"
            model_path.write_text(
                json.dumps(
                    {
                        "format": "FASEModel.to_dict",
                        "version": "0.1.0",
                        "model": {"w": [], "b0": 2.5, "stage1": [], "stage2": []},
                    }
                ),
                encoding="utf-8",
            )
            csv_path.write_text("a,b,y\n1,2,2.5\n4,5,4.5\n", encoding="utf-8")
            report = evaluate_exported_model(model_path, csv_path, target="y")

        self.assertEqual(report["metrics"]["rows"], 2)
        self.assertEqual(report["metrics"]["mse"], 2.0)
        self.assertEqual(report["metrics"]["mae"], 1.0)
        self.assertEqual(report["dataset"]["target"], "y")
        self.assertEqual(report["predictions"], [2.5, 2.5])

    def test_eval_model_cli_with_exported_intercept_model(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            model_path = Path(td) / "model.json"
            csv_path = Path(td) / "features.csv"
            eval_path = Path(td) / "eval.json"
            pred_path = Path(td) / "predictions.csv"
            model_path.write_text(
                json.dumps(
                    {
                        "format": "FASEModel.to_dict",
                        "version": "0.1.0",
                        "model": {"w": [], "b0": 2.5, "stage1": [], "stage2": []},
                    }
                ),
                encoding="utf-8",
            )
            csv_path.write_text("a,b,y\n1,2,2.5\n4,5,4.5\n", encoding="utf-8")
            result = self.run_cli(
                "eval-model",
                "--model",
                str(model_path),
                "--csv",
                str(csv_path),
                "--target",
                "y",
                "--output",
                str(eval_path),
                "--predictions-output",
                str(pred_path),
            )
            report = read_json(eval_path)
            rows = pred_path.read_text(encoding="utf-8").strip().splitlines()

        self.assertIn("Wrote evaluation", result.stdout)
        self.assertEqual(report["metrics"]["mse"], 2.0)
        self.assertEqual(rows, ["row,prediction", "0,2.5", "1,2.5"])

    def test_eval_model_cli_warns_on_feature_schema_name_mismatch(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            model_path = Path(td) / "model.json"
            csv_path = Path(td) / "features.csv"
            eval_path = Path(td) / "eval.json"
            model_path.write_text(
                json.dumps(
                    {
                        "format": "FASEModel.to_dict",
                        "version": "0.1.0",
                        "feature_schema": {
                            "feature_names": ["a", "b"],
                            "feature_count": 2,
                            "target": "y",
                        },
                        "model": {"w": [], "b0": 2.5, "stage1": [], "stage2": []},
                    }
                ),
                encoding="utf-8",
            )
            csv_path.write_text("a,c,y\n1,2,2.5\n4,5,4.5\n", encoding="utf-8")
            result = self.run_cli(
                "eval-model",
                "--model",
                str(model_path),
                "--csv",
                str(csv_path),
                "--target",
                "y",
                "--output",
                str(eval_path),
            )
            report = read_json(eval_path)

        self.assertIn("Feature name mismatch", result.stderr)
        self.assertTrue(any("Feature name mismatch" in warning for warning in report["schema_warnings"]))

    def test_eval_model_cli_strict_schema_fails_on_name_mismatch(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            model_path = Path(td) / "model.json"
            csv_path = Path(td) / "features.csv"
            eval_path = Path(td) / "eval.json"
            model_path.write_text(
                json.dumps(
                    {
                        "format": "FASEModel.to_dict",
                        "version": "0.1.0",
                        "feature_schema": {
                            "feature_names": ["a", "b"],
                            "feature_count": 2,
                            "target": "y",
                        },
                        "model": {"w": [], "b0": 2.5, "stage1": [], "stage2": []},
                    }
                ),
                encoding="utf-8",
            )
            csv_path.write_text("a,c,y\n1,2,2.5\n4,5,4.5\n", encoding="utf-8")
            result = subprocess.run(
                [
                    sys.executable,
                    "-m",
                    "fase.cli",
                    "eval-model",
                    "--model",
                    str(model_path),
                    "--csv",
                    str(csv_path),
                    "--target",
                    "y",
                    "--output",
                    str(eval_path),
                    "--strict-schema",
                ],
                cwd=Path(__file__).resolve().parents[1],
                check=False,
                text=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
            )

        self.assertNotEqual(result.returncode, 0)
        self.assertIn("feature schema mismatch", result.stderr)
        self.assertFalse(eval_path.exists())

    def test_real_fast_fit_export_predict_round_trip(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            train_path = root / "train.csv"
            model_path = root / "model.json"
            report_path = root / "report.json"
            predict_path = root / "predict.csv"
            out_path = root / "predictions.csv"
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
            predict_path.write_text("x0,x1,y\n0,0,1\n1,1,4\n", encoding="utf-8")

            export_result = self.run_cli(
                "export-model",
                "--csv",
                str(train_path),
                "--target",
                "y",
                "--fast",
                "--k-folds",
                "2",
                "--output",
                str(model_path),
                "--report-output",
                str(report_path),
            )
            predict_result = self.run_cli(
                "predict",
                "--model",
                str(model_path),
                "--csv",
                str(predict_path),
                "--drop-column",
                "y",
                "--output",
                str(out_path),
            )
            report = read_json(report_path)
            exported_model = read_json(model_path)
            rows = out_path.read_text(encoding="utf-8").strip().splitlines()
            model_exists = model_path.exists()

        self.assertIn("Wrote model", export_result.stdout)
        self.assertIn("Wrote predictions", predict_result.stdout)
        self.assertEqual(report["model_export"]["status"], "exported")
        self.assertEqual(exported_model["feature_schema"]["feature_names"], ["x0", "x1"])
        self.assertTrue(model_exists)
        self.assertEqual(rows[0], "row,prediction")
        self.assertEqual(len(rows), 3)
        for row in rows[1:]:
            self.assertTrue(math.isfinite(float(row.split(",", 1)[1])))

    def test_report_v7_summary(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            path = Path(td) / "summary.json"
            path.write_text(
                json.dumps(
                    {
                        "v6_2_turbo": {
                            "metrics": {
                                "regime_accuracy": 0.5,
                                "exact_agreement": 0.4,
                                "avg_hamming": 0.7,
                                "mean_template_purity": 0.9,
                                "mean_num_ops": 3.0,
                            },
                            "compactness_gate": {"all_pass": False},
                        }
                    }
                ),
                encoding="utf-8",
            )
            result = self.run_cli("report", str(path))
        self.assertIn("v6_2_turbo", result.stdout)
        self.assertIn("regime_acc", result.stdout)

    def test_report_surfaces_model_export_preflight(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            path = Path(td) / "fit_report.json"
            path.write_text(
                json.dumps(
                    {
                        "R2_oof": 0.75,
                        "MSE_oof": 0.125,
                        "num_consensus_ops": 2,
                        "model_export": {
                            "path": "outputs/bad_model.json",
                            "format": "FASEModel.to_dict",
                            "status": "unsupported",
                            "error": "stage2[0] kind 'mystery_block' is not supported",
                            "preflight": {
                                "status": "unsupported",
                                "stage1_count": 0,
                                "stage2_count": 1,
                                "errors": ["stage2[0] kind 'mystery_block' is not supported"],
                                "warnings": [],
                                "blocks": [
                                    {
                                        "index": 0,
                                        "kind": "mystery_block",
                                        "supported": False,
                                        "errors": ["stage2[0] kind 'mystery_block' is not supported"],
                                        "warnings": [],
                                    }
                                ],
                            },
                        },
                    }
                ),
                encoding="utf-8",
            )
            result = self.run_cli("report", str(path))

        self.assertIn("R2_oof", result.stdout)
        self.assertIn("Model export", result.stdout)
        self.assertIn("unsupported", result.stdout)
        self.assertIn("Model export preflight issues", result.stdout)
        self.assertIn("mystery_block", result.stdout)


if __name__ == "__main__":
    unittest.main()
