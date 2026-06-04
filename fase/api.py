from __future__ import annotations

import copy
import csv
import hashlib
import importlib
import importlib.metadata
import importlib.util
import json
import platform
import re
import shutil
import subprocess
import sys
import tarfile
from contextlib import contextmanager
from datetime import datetime, timezone
from importlib import resources
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional, Tuple
import zipfile

import numpy as np

__version__ = "0.1.0"

GENERATED_OUTPUT_PATTERNS: Tuple[str, ...] = (
    "build",
    "dist",
    "outputs",
    "pre_v7_outputs",
    "v7_lab_outputs",
    "v7_lab_outputs_*",
    "*.egg-info",
    "**/*.pkl.tmp",
    "**/*.pid",
    "**/__pycache__",
)
CLEAN_IGNORE_DIRS: Tuple[str, ...] = (
    ".git",
    ".hg",
    ".mypy_cache",
    ".tox",
    ".venv",
    "node_modules",
    "venv",
)
EXPECTED_PACKAGE_FILES: Tuple[str, ...] = (
    "pyproject.toml",
    "MANIFEST.in",
    "README.md",
    "CHANGELOG.md",
    "docs/release.md",
    "docs/model-format.md",
    "examples/quickstart/README.md",
    "examples/quickstart/train.csv",
    "examples/quickstart/predict.csv",
    "examples/quickstart/model.json",
    "fase/data/__init__.py",
    "fase/data/quickstart/README.md",
    "fase/data/quickstart/__init__.py",
    "fase/data/quickstart/train.csv",
    "fase/data/quickstart/predict.csv",
    "fase/data/quickstart/model.json",
    "fase/__init__.py",
    "fase/api.py",
    "fase/cli.py",
    "FASE_v21.py",
    "v7_regime_first_lab_harness.py",
    "scripts/product_smoke.sh",
    ".github/workflows/product-smoke.yml",
)
EXAMPLE_LAW = "y = 1.0 + 2.0*x0 - 0.5*x1 + 0.25*x2^2"
FASE_MODEL_FORMAT = "FASEModel.to_dict"
FASE_MODEL_SCHEMA_VERSION = "fase-model-v1"
QUICKSTART_EXAMPLE_FILES: Tuple[str, ...] = (
    "README.md",
    "train.csv",
    "predict.csv",
    "model.json",
)
REQUIRED_WHEEL_QUICKSTART_MEMBERS: Tuple[str, ...] = tuple(
    [f"fase/data/quickstart/{name}" for name in QUICKSTART_EXAMPLE_FILES]
    + ["fase/data/quickstart/__init__.py"]
)
REQUIRED_SDIST_QUICKSTART_SUFFIXES: Tuple[str, ...] = tuple(
    [f"examples/quickstart/{name}" for name in QUICKSTART_EXAMPLE_FILES]
    + [f"fase/data/quickstart/{name}" for name in QUICKSTART_EXAMPLE_FILES]
    + ["fase/data/quickstart/__init__.py"]
)
SUPPORTED_STAGE2_BLOCK_KINDS: Tuple[str, ...] = (
    "relu_proj",
    "sinproj",
    "fct",
    "bilinear",
    "dihedral_invar",
    "perm_invar",
    "group_invar",
    "combo_vec",
    "ruliad",
)
REQUIRED_STAGE2_PARAMS: Dict[str, Tuple[str, ...]] = {
    "relu_proj": ("w", "t"),
    "sinproj": ("w", "b"),
    "fct": ("w", "b0", "kappa"),
    "bilinear": ("i", "j"),
    "dihedral_invar": ("i", "j"),
    "perm_invar": ("group_idx",),
    "group_invar": ("spec",),
    "combo_vec": ("w1", "w2"),
}


def _read_text_if_exists(path: Path) -> str:
    if not path.exists():
        return ""
    return path.read_text(encoding="utf-8")


def _extract_pyproject_value(text: str, key: str) -> Optional[str]:
    match = re.search(rf"^{re.escape(key)}\s*=\s*[\"']([^\"']+)[\"']", text, flags=re.MULTILINE)
    return match.group(1) if match else None


def _contains_all(text: str, snippets: Tuple[str, ...]) -> bool:
    return all(snippet in text for snippet in snippets)


def _release_check(name: str, passed: bool, detail: str, *, required: bool = True) -> Dict[str, Any]:
    return {
        "name": name,
        "required": bool(required),
        "status": "pass" if passed else "fail",
        "detail": detail,
    }


def collect_release_checks(root: str | Path = ".") -> Dict[str, Any]:
    """Collect dependency-free release/package readiness checks."""
    root_path = Path(root).resolve()
    pyproject_path = root_path / "pyproject.toml"
    pyproject_text = _read_text_if_exists(pyproject_path)
    manifest_text = _read_text_if_exists(root_path / "MANIFEST.in")
    readme_text = _read_text_if_exists(root_path / "README.md")
    changelog_text = _read_text_if_exists(root_path / "CHANGELOG.md")

    project_name = _extract_pyproject_value(pyproject_text, "name")
    project_version = _extract_pyproject_value(pyproject_text, "version")
    required_files = [
        _release_check(
            f"file:{relative}",
            (root_path / relative).exists(),
            relative,
        )
        for relative in EXPECTED_PACKAGE_FILES
    ]
    checks = [
        *required_files,
        _release_check("metadata:name", project_name == "fase-symbolic", project_name or "missing"),
        _release_check("metadata:version", project_version == __version__, f"pyproject={project_version} package={__version__}"),
        _release_check("metadata:readme", 'readme = "README.md"' in pyproject_text, "README.md declared"),
        _release_check("metadata:requires-python", 'requires-python = ">=3.10"' in pyproject_text, "requires-python >=3.10"),
        _release_check(
            "metadata:license",
            'license = "LicenseRef-Proprietary"' in pyproject_text
            or 'license = { text = "LicenseRef-Proprietary" }' in pyproject_text,
            "LicenseRef-Proprietary declared",
        ),
        _release_check("metadata:numpy-dependency", '"numpy>=1.23"' in pyproject_text, "numpy>=1.23 declared"),
        _release_check("metadata:console-script", 'fase = "fase.cli:main"' in pyproject_text, "fase CLI entry point"),
        _release_check(
            "metadata:classifiers",
            _contains_all(
                pyproject_text,
                (
                    '"Development Status :: 3 - Alpha"',
                    '"Programming Language :: Python :: 3.10"',
                    '"Programming Language :: Python :: 3.11"',
                    '"Programming Language :: Python :: 3.12"',
                    '"Topic :: Scientific/Engineering :: Artificial Intelligence"',
                ),
            ),
            "alpha, supported Python, and AI classifiers declared",
        ),
        _release_check(
            "metadata:optional-dependencies",
            _contains_all(
                pyproject_text,
                (
                    "[project.optional-dependencies]",
                    "demo = [",
                    "baseline = [",
                    "legacy = [",
                    "dev = [",
                    "all = [",
                ),
            ),
            "demo/baseline/legacy/dev/all extras declared",
        ),
        _release_check(
            "metadata:package-data",
            _contains_all(
                pyproject_text,
                (
                    "[tool.setuptools.package-data]",
                    '"data/quickstart/*.csv"',
                    '"data/quickstart/*.json"',
                    '"data/quickstart/*.md"',
                ),
            ),
            "quickstart package data declared",
        ),
        _release_check(
            "manifest:generated-excludes",
            _contains_all(
                manifest_text,
                (
                    "prune outputs",
                    "prune pre_v7_outputs",
                    "prune v7_lab_outputs",
                    "prune v7_lab_outputs_*",
                    "global-exclude *.pkl.tmp",
                    "global-exclude *.pid",
                ),
            ),
            "generated output and checkpoint exclusions declared",
        ),
        _release_check("docs:cli-quickstart", "## CLI Quickstart" in readme_text, "README CLI quickstart section"),
        _release_check("docs:testing", "## Testing" in readme_text, "README testing section"),
        _release_check("docs:packaging-notes", "## Packaging Notes" in readme_text, "README packaging notes section"),
        _release_check("docs:release-guide", "docs/release.md" in readme_text, "README release guide link"),
        _release_check("docs:model-format", "docs/model-format.md" in readme_text, "README model format link"),
        _release_check("docs:changelog-unreleased", "## 0.1.0" in changelog_text, "CHANGELOG 0.1.0 section"),
    ]

    required_ok = all(check["status"] == "pass" for check in checks if check["required"])
    return {
        "root": str(root_path),
        "status": "ok" if required_ok else "needs_attention",
        "required_ok": bool(required_ok),
        "project_name": project_name,
        "project_version": project_version,
        "package_version": __version__,
        "checks": checks,
    }


def _newest_path(paths: List[Path]) -> Optional[str]:
    existing = [path for path in paths if path.exists()]
    if not existing:
        return None
    return str(sorted(existing, key=lambda path: path.stat().st_mtime)[-1])


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _artifact_kind(path: Path) -> str:
    name = path.name
    if name.endswith(".whl"):
        return "wheel"
    if name.endswith(".tar.gz"):
        return "sdist"
    return path.suffix.lstrip(".") or "file"


def _artifact_members(path: Path) -> Tuple[List[str], List[str], List[str]]:
    """Return archive members, quickstart fixture members, and warnings."""
    warnings: List[str] = []
    members: List[str] = []
    try:
        if path.suffix == ".whl" or path.name.endswith(".whl"):
            with zipfile.ZipFile(path) as zf:
                members = sorted(zf.namelist())
        elif path.name.endswith(".tar.gz"):
            with tarfile.open(path, "r:gz") as tf:
                members = sorted(tf.getnames())
    except (tarfile.TarError, zipfile.BadZipFile, OSError) as exc:
        warnings.append(f"Could not inspect archive members: {exc}")

    quickstart = [
        name
        for name in members
        if "/examples/quickstart/" in f"/{name}"
        or "/fase/data/quickstart/" in f"/{name}"
    ]
    return members, quickstart, warnings


def _read_archive_member_text(path: Path, member: str) -> Optional[str]:
    try:
        if path.name.endswith(".whl"):
            with zipfile.ZipFile(path) as zf:
                return zf.read(member).decode("utf-8", errors="replace")
        if path.name.endswith(".tar.gz"):
            with tarfile.open(path, "r:gz") as tf:
                fh = tf.extractfile(member)
                if fh is None:
                    return None
                return fh.read().decode("utf-8", errors="replace")
    except (KeyError, tarfile.TarError, zipfile.BadZipFile, OSError):
        return None
    return None


def _extract_metadata_field(text: str, field: str) -> Optional[str]:
    match = re.search(rf"^{re.escape(field)}:\s*(.+)$", text, flags=re.MULTILINE)
    return match.group(1).strip() if match else None


def _artifact_metadata(path: Path, kind: str, members: List[str]) -> Tuple[Dict[str, Any], List[str]]:
    """Return lightweight artifact metadata and validation warnings."""
    metadata: Dict[str, Any] = {
        "valid": kind not in ("wheel", "sdist"),
        "name": None,
        "version": None,
        "console_script": None,
        "metadata_member": None,
        "entry_points_member": None,
        "pyproject_member": None,
    }
    warnings: List[str] = []

    if kind == "wheel":
        metadata_member = next((name for name in members if name.endswith(".dist-info/METADATA")), None)
        entry_points_member = next((name for name in members if name.endswith(".dist-info/entry_points.txt")), None)
        metadata["metadata_member"] = metadata_member
        metadata["entry_points_member"] = entry_points_member

        metadata_text = _read_archive_member_text(path, metadata_member) if metadata_member else None
        entry_points_text = _read_archive_member_text(path, entry_points_member) if entry_points_member else None
        metadata["name"] = _extract_metadata_field(metadata_text or "", "Name")
        metadata["version"] = _extract_metadata_field(metadata_text or "", "Version")
        metadata["console_script"] = "fase = fase.cli:main" if "fase = fase.cli:main" in (entry_points_text or "") else None

        if metadata["name"] != "fase-symbolic":
            warnings.append(f"Wheel metadata Name mismatch: {metadata['name'] or 'missing'}")
        if metadata["version"] != __version__:
            warnings.append(f"Wheel metadata Version mismatch: {metadata['version'] or 'missing'}")
        if metadata["console_script"] is None:
            warnings.append("Wheel entry_points.txt is missing fase = fase.cli:main")
        metadata["valid"] = not warnings
        return metadata, warnings

    if kind == "sdist":
        pyproject_member = next((name for name in members if name == "pyproject.toml" or name.endswith("/pyproject.toml")), None)
        metadata["pyproject_member"] = pyproject_member
        pyproject_text = _read_archive_member_text(path, pyproject_member) if pyproject_member else None
        metadata["name"] = _extract_pyproject_value(pyproject_text or "", "name")
        metadata["version"] = _extract_pyproject_value(pyproject_text or "", "version")
        metadata["console_script"] = "fase = \"fase.cli:main\"" if 'fase = "fase.cli:main"' in (pyproject_text or "") else None

        if metadata["name"] != "fase-symbolic":
            warnings.append(f"sdist pyproject name mismatch: {metadata['name'] or 'missing'}")
        if metadata["version"] != __version__:
            warnings.append(f"sdist pyproject version mismatch: {metadata['version'] or 'missing'}")
        if metadata["console_script"] is None:
            warnings.append("sdist pyproject is missing fase console script")
        metadata["valid"] = not warnings
        return metadata, warnings

    return metadata, warnings


def _missing_required_quickstart(kind: str, members: List[str]) -> List[str]:
    if kind == "wheel":
        present = set(members)
        return [name for name in REQUIRED_WHEEL_QUICKSTART_MEMBERS if name not in present]
    if kind == "sdist":
        return [
            suffix
            for suffix in REQUIRED_SDIST_QUICKSTART_SUFFIXES
            if not any(member.endswith(suffix) for member in members)
        ]
    return []


def collect_artifact_manifest(
    root: str | Path = ".",
    *,
    artifact_paths: Optional[List[str | Path]] = None,
    latest_only: bool = False,
) -> Dict[str, Any]:
    """Collect paths, hashes, sizes, and fixture membership for build artifacts."""
    root_path = Path(root).resolve()
    dist = root_path / "dist"
    if artifact_paths:
        paths = []
        for raw in artifact_paths:
            candidate = Path(raw)
            paths.append(candidate if candidate.is_absolute() else root_path / candidate)
    elif dist.exists():
        paths = sorted([*dist.glob("*.whl"), *dist.glob("*.tar.gz")])
        if latest_only:
            latest_by_kind: Dict[str, Path] = {}
            for path in paths:
                kind = _artifact_kind(path)
                current = latest_by_kind.get(kind)
                if current is None or path.stat().st_mtime > current.stat().st_mtime:
                    latest_by_kind[kind] = path
            paths = sorted(latest_by_kind.values())
    else:
        paths = []

    artifacts: List[Dict[str, Any]] = []
    for path in paths:
        resolved = path.resolve()
        exists = resolved.exists()
        kind = _artifact_kind(resolved)
        try:
            relative_path = str(resolved.relative_to(root_path))
        except ValueError:
            relative_path = str(resolved)
        item: Dict[str, Any] = {
            "path": str(resolved),
            "relative_path": relative_path,
            "name": resolved.name,
            "kind": kind,
            "exists": bool(exists),
            "bytes": None,
            "sha256": None,
            "modified_iso": None,
            "member_count": 0,
            "quickstart_files": [],
            "quickstart_required": list(
                REQUIRED_WHEEL_QUICKSTART_MEMBERS
                if kind == "wheel"
                else REQUIRED_SDIST_QUICKSTART_SUFFIXES
                if kind == "sdist"
                else ()
            ),
            "quickstart_missing": [],
            "quickstart_complete": kind not in ("wheel", "sdist"),
            "artifact_metadata": {
                "valid": kind not in ("wheel", "sdist"),
                "name": None,
                "version": None,
                "console_script": None,
                "metadata_member": None,
                "entry_points_member": None,
                "pyproject_member": None,
            },
            "warnings": [],
        }
        if exists:
            stat = resolved.stat()
            members, quickstart, warnings = _artifact_members(resolved)
            missing_quickstart = _missing_required_quickstart(kind, members)
            artifact_metadata, metadata_warnings = _artifact_metadata(resolved, kind, members)
            if missing_quickstart:
                warnings = [
                    *warnings,
                    "Missing required quickstart fixture members: " + ", ".join(missing_quickstart),
                ]
            warnings = [*warnings, *metadata_warnings]
            item.update(
                {
                    "bytes": int(stat.st_size),
                    "sha256": _sha256_file(resolved),
                    "modified_iso": datetime.fromtimestamp(stat.st_mtime, tz=timezone.utc).isoformat(),
                    "member_count": len(members),
                    "quickstart_files": quickstart,
                    "quickstart_missing": missing_quickstart,
                    "quickstart_complete": not missing_quickstart,
                    "artifact_metadata": artifact_metadata,
                    "warnings": warnings,
                }
            )
        else:
            item["warnings"].append("Artifact does not exist.")
        artifacts.append(item)

    existing = [item for item in artifacts if item["exists"]]
    complete = bool(existing) and all(
        item["exists"]
        and item.get("quickstart_complete", True)
        and item.get("artifact_metadata", {}).get("valid", True)
        and not item.get("warnings")
        for item in artifacts
    )
    return {
        "root": str(root_path),
        "dist": str(dist),
        "latest_only": bool(latest_only),
        "status": "ok" if complete else "missing" if not existing else "needs_attention",
        "artifact_count": len(existing),
        "artifacts": artifacts,
    }


def collect_release_notes(root: str | Path = ".", *, cli_commands: Optional[List[str]] = None) -> Dict[str, Any]:
    """Collect a compact release handoff summary without running heavy checks."""
    root_path = Path(root).resolve()
    release = collect_release_checks(root_path)
    dist = root_path / "dist"
    wheel = _newest_path(list(dist.glob("*.whl"))) if dist.exists() else None
    sdist = _newest_path(list(dist.glob("*.tar.gz"))) if dist.exists() else None
    commands = list(cli_commands or [])

    verification_commands = [
        "python3 -m unittest discover -s tests",
        "venv/bin/fase release-check",
        "venv/bin/fase status --json",
        "venv/bin/fase copy-examples --output outputs/fase_quickstart_fixture --force",
        "venv/bin/python -m build --sdist --wheel --no-isolation",
        "venv/bin/fase artifact-manifest --latest-only --json",
        "venv/bin/fase release-bundle --strict --json",
    ]
    if wheel:
        verification_commands.append(f"python3 -m fase.cli artifact-check --wheel {Path(wheel).relative_to(root_path)}")
    else:
        verification_commands.append("python3 -m fase.cli artifact-check --wheel dist/fase_symbolic-0.1.0-py3-none-any.whl")

    return {
        "root": str(root_path),
        "status": release["status"],
        "project_name": release["project_name"],
        "project_version": release["project_version"],
        "package_version": release["package_version"],
        "release_checks_required_ok": release["required_ok"],
        "artifacts": {
            "wheel": wheel,
            "sdist": sdist,
        },
        "capabilities": [
            "Installable Python package and `fase` CLI.",
            "Numeric CSV fit/export/predict/evaluate workflow.",
            "Committed and packaged quickstart fixtures for validation and prediction.",
            "Ruliad HypergraphState and closure-captured ruliad model serialization.",
            "Exported model inspection, comparison, and structural validation.",
            "Model export preflight diagnostics for unsupported stage-2 blocks.",
            "`fase report` summaries for fit, export, and v7 JSON reports.",
            "`fase status` read-only product readiness summary.",
            "Generated-output cleanup with optional age filtering and release/artifact checks.",
            "Build artifact manifest with hashes and packaged fixture membership.",
            "Release bundle handoff with notes, artifact manifest JSON, and git provenance.",
            "v7 regime-first lab harness dry-run and report summarization.",
        ],
        "limitations": [
            "v7 controller output is research-grade and not a production controller.",
            "Model export is best-effort for opaque closure-backed blocks without serializable state.",
            "Full v7 lab runs are CPU-heavy and intentionally excluded from release smoke checks.",
        ],
        "cli_commands": commands,
        "verification_commands": verification_commands,
    }


def collect_product_status(
    root: str | Path = ".",
    *,
    latest_only: bool = True,
    require_clean: bool = False,
) -> Dict[str, Any]:
    """Collect read-only product status across release checks, artifacts, and git."""
    root_path = Path(root).resolve()
    release = collect_release_checks(root_path)
    artifact_manifest = collect_artifact_manifest(root_path, latest_only=latest_only)
    git_provenance = collect_git_provenance(root_path)
    clean_required_pass = (
        not require_clean
        or (git_provenance.get("available") and not git_provenance.get("dirty"))
    )
    status = (
        "ok"
        if release["required_ok"]
        and artifact_manifest["status"] == "ok"
        and clean_required_pass
        else "needs_attention"
    )
    return {
        "root": str(root_path),
        "status": status,
        "package": {
            "project_name": release["project_name"],
            "project_version": release["project_version"],
            "package_version": release["package_version"],
        },
        "release_status": release["status"],
        "release_checks_required_ok": bool(release["required_ok"]),
        "artifact_manifest_status": artifact_manifest["status"],
        "artifact_count": artifact_manifest["artifact_count"],
        "latest_only": bool(latest_only),
        "require_clean": bool(require_clean),
        "clean_required_pass": bool(clean_required_pass),
        "git": git_provenance,
        "release_checks": release,
        "artifact_manifest": artifact_manifest,
    }


def _run_git(root: Path, args: List[str]) -> Tuple[Optional[str], Optional[str]]:
    try:
        completed = subprocess.run(
            ["git", *args],
            cwd=root,
            check=False,
            capture_output=True,
            text=True,
        )
    except (OSError, subprocess.SubprocessError) as exc:
        return None, str(exc)
    if completed.returncode != 0:
        return None, (completed.stderr or completed.stdout).strip() or f"git exited {completed.returncode}"
    return completed.stdout.strip(), None


def _git_status_summary(lines: List[str]) -> Dict[str, int]:
    summary = {
        "modified": 0,
        "added": 0,
        "deleted": 0,
        "renamed": 0,
        "copied": 0,
        "untracked": 0,
        "other": 0,
    }
    for line in lines:
        code = line[:2]
        if code == "??":
            summary["untracked"] += 1
        elif "D" in code:
            summary["deleted"] += 1
        elif "R" in code:
            summary["renamed"] += 1
        elif "C" in code:
            summary["copied"] += 1
        elif "A" in code:
            summary["added"] += 1
        elif "M" in code:
            summary["modified"] += 1
        else:
            summary["other"] += 1
    return summary


def collect_git_provenance(root: str | Path = ".") -> Dict[str, Any]:
    """Collect lightweight git provenance for release handoff metadata."""
    root_path = Path(root).resolve()
    inside, error = _run_git(root_path, ["rev-parse", "--is-inside-work-tree"])
    if inside != "true":
        return {
            "available": False,
            "error": error or "Not inside a git work tree.",
        }

    branch, branch_error = _run_git(root_path, ["branch", "--show-current"])
    commit, commit_error = _run_git(root_path, ["rev-parse", "HEAD"])
    short_commit, short_error = _run_git(root_path, ["rev-parse", "--short", "HEAD"])
    status, status_error = _run_git(root_path, ["status", "--short"])
    status_lines = [line for line in (status or "").splitlines() if line]

    errors = [err for err in [branch_error, commit_error, short_error, status_error] if err]
    return {
        "available": True,
        "branch": branch or None,
        "commit_sha": commit or None,
        "commit_short": short_commit or None,
        "dirty": bool(status_lines),
        "status_count": len(status_lines),
        "status_summary": _git_status_summary(status_lines),
        "errors": errors,
    }


def create_release_bundle(
    root: str | Path = ".",
    *,
    output_dir: str | Path | None = None,
    latest_only: bool = True,
    require_clean: bool = False,
    timestamp: Optional[str] = None,
    cli_commands: Optional[List[str]] = None,
) -> Dict[str, Any]:
    """Write release notes and artifact manifest JSON into one handoff directory."""
    root_path = Path(root).resolve()
    stamp = timestamp or datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    if output_dir is None:
        bundle_dir = root_path / "outputs" / f"fase_release_bundle_{stamp}"
    else:
        raw_output_dir = Path(output_dir)
        bundle_dir = raw_output_dir if raw_output_dir.is_absolute() else root_path / raw_output_dir
    bundle_dir.mkdir(parents=True, exist_ok=True)

    release_notes = collect_release_notes(root_path, cli_commands=cli_commands)
    artifact_manifest = collect_artifact_manifest(root_path, latest_only=latest_only)
    git_provenance = collect_git_provenance(root_path)
    clean_required_pass = (
        not require_clean
        or (git_provenance.get("available") and not git_provenance.get("dirty"))
    )

    release_notes_path = bundle_dir / "release_notes.json"
    artifact_manifest_path = bundle_dir / "artifact_manifest.json"
    summary_path = bundle_dir / "release_bundle.json"
    status = (
        "ok"
        if release_notes["release_checks_required_ok"]
        and artifact_manifest["status"] == "ok"
        and clean_required_pass
        else "needs_attention"
    )
    summary = {
        "root": str(root_path),
        "bundle_dir": str(bundle_dir),
        "created_iso": datetime.now(timezone.utc).isoformat(),
        "timestamp": stamp,
        "status": status,
        "release_checks_required_ok": bool(release_notes["release_checks_required_ok"]),
        "artifact_manifest_status": artifact_manifest["status"],
        "artifact_count": artifact_manifest["artifact_count"],
        "latest_only": bool(latest_only),
        "require_clean": bool(require_clean),
        "clean_required_pass": bool(clean_required_pass),
        "git": git_provenance,
        "files": {
            "release_notes": str(release_notes_path),
            "artifact_manifest": str(artifact_manifest_path),
            "release_bundle": str(summary_path),
        },
    }

    write_json(release_notes, release_notes_path)
    write_json(artifact_manifest, artifact_manifest_path)
    write_json(summary, summary_path)
    return summary


def load_fase_v21():
    """Load the current FASE v21 implementation lazily."""
    return importlib.import_module("FASE_v21")


def _package_diagnostic(package_name: str, import_name: Optional[str] = None, *, required: bool = False) -> Dict[str, Any]:
    module_name = import_name or package_name
    spec = importlib.util.find_spec(module_name)
    try:
        version = importlib.metadata.version(package_name)
    except importlib.metadata.PackageNotFoundError:
        version = None
    return {
        "name": package_name,
        "import_name": module_name,
        "required": bool(required),
        "available": spec is not None,
        "version": version,
        "origin": getattr(spec, "origin", None) if spec is not None else None,
    }


def _module_diagnostic(module_name: str, *, required: bool = False) -> Dict[str, Any]:
    spec = importlib.util.find_spec(module_name)
    return {
        "name": module_name,
        "required": bool(required),
        "available": spec is not None,
        "origin": getattr(spec, "origin", None) if spec is not None else None,
    }


def collect_diagnostics() -> Dict[str, Any]:
    """Collect fast, read-only diagnostics for support and install checks."""
    python_ok = sys.version_info >= (3, 10)
    packages = [
        _package_diagnostic("numpy", required=True),
        _package_diagnostic("scipy"),
        _package_diagnostic("scikit-learn", import_name="sklearn"),
        _package_diagnostic("pandas"),
        _package_diagnostic("pysr"),
    ]
    modules = [
        _module_diagnostic("FASE_v21", required=True),
        _module_diagnostic("v7_regime_first_lab_harness"),
    ]
    required_ok = python_ok and all(
        item["available"]
        for item in [*packages, *modules]
        if item.get("required")
    )
    return {
        "status": "ok" if required_ok else "needs_attention",
        "required_ok": bool(required_ok),
        "fase_version": __version__,
        "python": {
            "executable": sys.executable,
            "version": sys.version.split()[0],
            "version_info": list(sys.version_info[:3]),
            "required": ">=3.10",
            "ok": bool(python_ok),
        },
        "platform": {
            "system": platform.system(),
            "release": platform.release(),
            "machine": platform.machine(),
            "platform": platform.platform(),
        },
        "cwd": str(Path.cwd()),
        "packages": packages,
        "modules": modules,
    }


def _path_size(path: Path) -> Tuple[int, int]:
    if path.is_file():
        return 1, path.stat().st_size

    file_count = 0
    byte_count = 0
    for item in path.rglob("*"):
        if item.is_file():
            file_count += 1
            byte_count += item.stat().st_size
    return file_count, byte_count


def _path_latest_mtime(path: Path) -> float:
    latest = path.stat().st_mtime
    if path.is_dir():
        for item in path.rglob("*"):
            try:
                latest = max(latest, item.stat().st_mtime)
            except FileNotFoundError:
                continue
    return latest


def _dedupe_parent_paths(paths: List[Tuple[Path, str]], root: Path) -> List[Tuple[Path, str]]:
    unique: Dict[Path, str] = {}
    for path, pattern in paths:
        unique.setdefault(path, pattern)

    selected: List[Tuple[Path, str]] = []
    for path, pattern in sorted(unique.items(), key=lambda item: len(item[0].relative_to(root).parts)):
        if any(path == parent or parent in path.parents for parent, _ in selected):
            continue
        selected.append((path, pattern))
    return selected


def _is_ignored_cleanup_path(path: Path, root: Path) -> bool:
    return any(part in CLEAN_IGNORE_DIRS for part in path.relative_to(root).parts)


def discover_generated_outputs(
    root: str | Path = ".",
    *,
    patterns: Tuple[str, ...] = GENERATED_OUTPUT_PATTERNS,
    older_than_days: Optional[float] = None,
) -> Dict[str, Any]:
    """Find generated FASE outputs that are safe candidates for cleanup."""
    root_path = Path(root).resolve()
    if not root_path.exists():
        raise FileNotFoundError(f"Cleanup root does not exist: {root_path}")
    if not root_path.is_dir():
        raise NotADirectoryError(f"Cleanup root is not a directory: {root_path}")

    matches: List[Tuple[Path, str]] = []
    for pattern in patterns:
        for path in root_path.glob(pattern):
            if not path.exists():
                continue
            resolved = path.resolve()
            if resolved == root_path:
                continue
            try:
                resolved.relative_to(root_path)
            except ValueError as exc:
                raise ValueError(f"Refusing to inspect path outside cleanup root: {resolved}") from exc
            if _is_ignored_cleanup_path(resolved, root_path):
                continue
            matches.append((resolved, pattern))

    entries = []
    total_files = 0
    total_bytes = 0
    now = datetime.now(timezone.utc).timestamp()
    min_age_seconds = None if older_than_days is None else float(older_than_days) * 86400.0
    for path, pattern in _dedupe_parent_paths(matches, root_path):
        latest_mtime = _path_latest_mtime(path)
        age_seconds = max(0.0, now - latest_mtime)
        if min_age_seconds is not None and age_seconds < min_age_seconds:
            continue
        file_count, byte_count = _path_size(path)
        total_files += file_count
        total_bytes += byte_count
        entries.append(
            {
                "path": str(path),
                "relative_path": str(path.relative_to(root_path)),
                "kind": "dir" if path.is_dir() else "file",
                "pattern": pattern,
                "files": file_count,
                "bytes": byte_count,
                "modified_epoch": float(latest_mtime),
                "modified_iso": datetime.fromtimestamp(latest_mtime, tz=timezone.utc).isoformat(),
                "age_days": float(age_seconds / 86400.0),
            }
        )

    return {
        "root": str(root_path),
        "patterns": list(patterns),
        "ignored_dirs": list(CLEAN_IGNORE_DIRS),
        "older_than_days": older_than_days,
        "count": len(entries),
        "total_files": total_files,
        "total_bytes": total_bytes,
        "entries": entries,
    }


def clean_generated_outputs(
    root: str | Path = ".",
    *,
    patterns: Tuple[str, ...] = GENERATED_OUTPUT_PATTERNS,
    older_than_days: Optional[float] = None,
) -> Dict[str, Any]:
    """Remove generated FASE outputs discovered under root."""
    report = discover_generated_outputs(root, patterns=patterns, older_than_days=older_than_days)
    removed = []
    for entry in report["entries"]:
        path = Path(entry["path"])
        if not path.exists():
            continue
        if path.is_dir():
            shutil.rmtree(path)
        else:
            path.unlink()
        removed.append(entry)

    report["removed"] = removed
    report["removed_count"] = len(removed)
    return report


def _deep_merge(base: Dict[str, Any], patch: Dict[str, Any]) -> Dict[str, Any]:
    out = copy.deepcopy(base)
    for key, value in patch.items():
        if isinstance(value, dict) and isinstance(out.get(key), dict):
            out[key] = _deep_merge(out[key], value)
        else:
            out[key] = value
    return out


def _quickstart_y(x0: float, x1: float, x2: float) -> float:
    return 1.0 + 2.0 * x0 - 0.5 * x1 + 0.25 * x2 * x2


def _write_example_csv(path: Path, rows: List[List[float]]) -> None:
    with open(path, "w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["x0", "x1", "x2", "y"])
        writer.writerows(rows)


def _example_train_rows() -> List[List[float]]:
    rows: List[List[float]] = []
    for x0 in (-1.0, 0.0, 1.0):
        for x1 in (-2.0, 0.0, 2.0):
            for x2 in (-1.0, 0.0, 1.0):
                rows.append([x0, x1, x2, _quickstart_y(x0, x1, x2)])
    return rows


def _example_predict_rows() -> List[List[float]]:
    points = [
        (-1.5, -1.0, 0.5),
        (-0.5, 1.0, -0.5),
        (0.5, -2.0, 1.5),
        (1.5, 0.0, -1.5),
        (2.0, 2.0, 0.0),
    ]
    return [[x0, x1, x2, _quickstart_y(x0, x1, x2)] for x0, x1, x2 in points]


def write_example_dataset(output_dir: str | Path = "outputs/fase_quickstart", *, force: bool = False) -> Dict[str, Any]:
    """Write a deterministic numeric quickstart dataset for CLI/API examples."""
    root = Path(output_dir)
    train_path = root / "train.csv"
    predict_path = root / "predict.csv"
    readme_path = root / "README.md"
    planned = (train_path, predict_path, readme_path)
    existing = [str(path) for path in planned if path.exists()]
    if existing and not force:
        raise FileExistsError(
            "Example output already exists. Pass force=True or choose a new directory: "
            + ", ".join(existing)
        )

    root.mkdir(parents=True, exist_ok=True)
    train_rows = _example_train_rows()
    predict_rows = _example_predict_rows()
    _write_example_csv(train_path, train_rows)
    _write_example_csv(predict_path, predict_rows)
    readme_path.write_text(
        "\n".join(
            [
                "# FASE Quickstart Example",
                "",
                f"Law: `{EXAMPLE_LAW}`",
                "",
                "Files:",
                "",
                "- `train.csv`: numeric training data with target column `y`.",
                "- `predict.csv`: held-out numeric rows with `y` included for comparison.",
                "",
                "Run:",
                "",
                "```bash",
                "fase export-model --csv train.csv --target y --fast --output model.json --report-output report.json",
                "fase predict --model model.json --csv predict.csv --drop-column y --output predictions.csv",
                "fase eval-model --model model.json --csv predict.csv --target y --output eval.json",
                "```",
                "",
            ]
        ),
        encoding="utf-8",
    )

    return {
        "directory": str(root),
        "train_csv": str(train_path),
        "predict_csv": str(predict_path),
        "readme": str(readme_path),
        "law": EXAMPLE_LAW,
        "train_rows": len(train_rows),
        "predict_rows": len(predict_rows),
        "target": "y",
        "feature_names": ["x0", "x1", "x2"],
    }


def copy_quickstart_example(
    output_dir: str | Path = "outputs/fase_quickstart_fixture",
    *,
    force: bool = False,
) -> Dict[str, Any]:
    """Copy the packaged quickstart fixture to a user-writable directory."""
    root = Path(output_dir)
    planned = [root / name for name in QUICKSTART_EXAMPLE_FILES]
    existing = [str(path) for path in planned if path.exists()]
    if existing and not force:
        raise FileExistsError(
            "Quickstart fixture output already exists. Pass force=True or choose a new directory: "
            + ", ".join(existing)
        )

    root.mkdir(parents=True, exist_ok=True)
    package_root = resources.files(__package__).joinpath("data", "quickstart")
    copied: Dict[str, str] = {}
    for name in QUICKSTART_EXAMPLE_FILES:
        source = package_root.joinpath(name)
        if not source.is_file():
            raise FileNotFoundError(f"Packaged quickstart fixture is missing: {name}")
        destination = root / name
        destination.write_bytes(source.read_bytes())
        copied[name] = str(destination)

    return {
        "directory": str(root),
        "source": "packaged",
        "files": copied,
        "law": EXAMPLE_LAW,
        "train_csv": copied["train.csv"],
        "predict_csv": copied["predict.csv"],
        "model_json": copied["model.json"],
        "readme": copied["README.md"],
    }


@contextmanager
def temporary_config(module: Any, patch: Optional[Dict[str, Any]]) -> Iterator[Dict[str, Any]]:
    """Temporarily patch module.CONFIG for legacy code paths that read globals."""
    original = copy.deepcopy(module.CONFIG)
    if patch:
        module.CONFIG = _deep_merge(module.CONFIG, patch)
    try:
        yield module.CONFIG
    finally:
        module.CONFIG = original


def make_synthetic(seed: int = 42, n: int = 800, d: int = 10, gls_noise: bool = True):
    module = load_fase_v21()
    return module.make_synthetic(seed=seed, n=n, d=d, gls_noise=gls_noise)


def _resolve_column(column: Optional[str], names: List[str], default: int) -> int:
    if column is None:
        return default
    try:
        idx = int(column)
    except ValueError:
        if column not in names:
            raise ValueError(f"Column '{column}' was not found in CSV header: {names}")
        return names.index(column)
    if idx < 0:
        idx += len(names)
    if idx < 0 or idx >= len(names):
        raise ValueError(f"Column index {column} is outside CSV width {len(names)}")
    return idx


def _read_numeric_csv(
    path: str | Path,
    *,
    delimiter: str = ",",
    has_header: bool = True,
) -> Tuple[Path, List[str], np.ndarray]:
    csv_path = Path(path)
    with open(csv_path, "r", encoding="utf-8", newline="") as f:
        reader = csv.reader(f, delimiter=delimiter)
        raw_rows = [row for row in reader if row and any(cell.strip() for cell in row)]

    if not raw_rows:
        raise ValueError(f"CSV file has no data rows: {csv_path}")

    if has_header:
        header = [cell.strip() for cell in raw_rows[0]]
        data_rows = raw_rows[1:]
    else:
        header = [f"x{i}" for i in range(len(raw_rows[0]))]
        data_rows = raw_rows

    if not data_rows:
        raise ValueError(f"CSV file has no numeric data rows: {csv_path}")

    width = len(header)
    matrix: List[List[float]] = []
    for row_num, row in enumerate(data_rows, start=2 if has_header else 1):
        if len(row) != width:
            raise ValueError(f"CSV row {row_num} has {len(row)} columns; expected {width}")
        try:
            matrix.append([float(cell.strip()) for cell in row])
        except ValueError as exc:
            raise ValueError(f"CSV row {row_num} contains a non-numeric value: {row}") from exc

    return csv_path, header, np.asarray(matrix, dtype=float)


def load_csv_dataset(
    path: str | Path,
    *,
    target: Optional[str] = None,
    delimiter: str = ",",
    has_header: bool = True,
) -> Tuple[np.ndarray, np.ndarray, Dict[str, Any]]:
    """Load a numeric CSV into X, y, and metadata.

    The target can be a header name or zero-based column index. If omitted, the
    last column is used.
    """
    csv_path, header, arr = _read_numeric_csv(path, delimiter=delimiter, has_header=has_header)
    width = len(header)
    if width < 2:
        raise ValueError("CSV must contain at least one feature column and one target column")
    target_idx = _resolve_column(target, header, default=width - 1)
    feature_idx = [i for i in range(width) if i != target_idx]
    X = arr[:, feature_idx]
    y = arr[:, target_idx].reshape(-1)
    feature_names = [header[i] for i in feature_idx]
    metadata = {
        "path": str(csv_path),
        "rows": int(arr.shape[0]),
        "columns": int(width),
        "target": header[target_idx],
        "target_index": int(target_idx),
        "feature_names": feature_names,
        "delimiter": delimiter,
        "has_header": bool(has_header),
    }
    return X, y, metadata


def load_csv_features(
    path: str | Path,
    *,
    drop_column: Optional[str] = None,
    delimiter: str = ",",
    has_header: bool = True,
) -> Tuple[np.ndarray, Dict[str, Any]]:
    """Load numeric feature columns for prediction.

    drop_column can remove a target/id column by header name or zero-based index.
    """
    csv_path, header, arr = _read_numeric_csv(path, delimiter=delimiter, has_header=has_header)
    width = len(header)
    drop_idx = None
    if drop_column is not None:
        drop_idx = _resolve_column(drop_column, header, default=width - 1)
    feature_idx = [i for i in range(width) if i != drop_idx]
    X = arr[:, feature_idx]
    metadata = {
        "path": str(csv_path),
        "rows": int(arr.shape[0]),
        "columns": int(width),
        "dropped_column": header[drop_idx] if drop_idx is not None else None,
        "dropped_index": int(drop_idx) if drop_idx is not None else None,
        "feature_names": [header[i] for i in feature_idx],
        "delimiter": delimiter,
        "has_header": bool(has_header),
    }
    return X, metadata


def run_kfold(
    X: np.ndarray,
    y: np.ndarray,
    Sigma: Optional[np.ndarray] = None,
    *,
    k_folds: int = 5,
    seed: int = 42,
    config_patch: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    module = load_fase_v21()
    patch = _deep_merge({"K_FOLDS": k_folds, "SEEDS": [seed]}, config_patch or {})
    with temporary_config(module, patch) as config:
        return module.run_fase_kfold(X, y, Sigma=Sigma, K=k_folds, seed=seed, config=config)


def compact_report(report: Dict[str, Any]) -> Dict[str, Any]:
    """Return a JSON-safe summary from a full FASE k-fold report."""
    stability = report.get("og_stability", {}) or {}
    min_bits = report.get("og_min_bits", {}) or {}
    return {
        "R2_oof": float(report.get("R2_oof", np.nan)),
        "R2_oof_gls": float(report.get("R2_oof_gls", np.nan)),
        "MSE_oof": float(report.get("MSE_oof", np.nan)),
        "num_consensus_ops": int(len(stability)),
        "consensus_ops": sorted(stability.keys()),
        "og_stability": {str(k): float(v) for k, v in stability.items()},
        "og_min_bits": {str(k): float(v) for k, v in min_bits.items()},
        "folds": report.get("folds", []),
    }


def _feature_schema_from_metadata(metadata: Optional[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    if not metadata:
        return None
    feature_names = [str(name) for name in metadata.get("feature_names", [])]
    if not feature_names:
        return None
    schema = {
        "feature_names": feature_names,
        "feature_count": len(feature_names),
    }
    if metadata.get("target") is not None:
        schema["target"] = str(metadata["target"])
    return schema


def _schema_warnings(feature_schema: Optional[Dict[str, Any]], metadata: Dict[str, Any]) -> List[str]:
    if not feature_schema:
        return []

    warnings: List[str] = []
    expected_count = feature_schema.get("feature_count")
    actual_names = [str(name) for name in metadata.get("feature_names", [])]
    actual_count = len(actual_names)
    if expected_count is not None and int(expected_count) != actual_count:
        warnings.append(f"Feature count mismatch: model expects {expected_count}, CSV provides {actual_count}.")

    expected_names = [str(name) for name in feature_schema.get("feature_names", [])]
    if expected_names and actual_names and expected_names != actual_names:
        if sorted(expected_names) == sorted(actual_names):
            warnings.append("Feature name order mismatch: CSV columns are not in the exported model feature order.")
        else:
            warnings.append("Feature name mismatch: CSV feature names differ from the exported model feature names.")
    return warnings


def _raise_if_strict_schema_failed(schema_warnings: List[str], *, strict_schema: bool) -> None:
    if strict_schema and schema_warnings:
        raise ValueError("Exported model feature schema mismatch: " + " ".join(schema_warnings))


def _load_exported_model_payload(path: str | Path) -> Tuple[Any, Optional[Dict[str, Any]]]:
    payload = read_json(path)
    model_dict = payload.get("model", payload)
    feature_schema = payload.get("feature_schema") if isinstance(payload, dict) else None
    module = load_fase_v21()
    return module.FASEModel.from_dict(model_dict), feature_schema


def _is_number(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool)


def validate_exported_model_payload(payload: Any) -> Dict[str, Any]:
    """Validate the product-level exported FASEModel JSON shape."""
    errors: List[str] = []
    warnings: List[str] = []
    wrapped = isinstance(payload, dict) and isinstance(payload.get("model"), dict)

    if not isinstance(payload, dict):
        errors.append("Payload must be a JSON object.")
        model_dict: Dict[str, Any] = {}
        feature_schema = None
    elif wrapped:
        model_dict = payload["model"]
        feature_schema = payload.get("feature_schema")
        if payload.get("format") != FASE_MODEL_FORMAT:
            errors.append(f"Wrapped payload format must be {FASE_MODEL_FORMAT!r}.")
        if payload.get("schema_version") not in (None, FASE_MODEL_SCHEMA_VERSION):
            errors.append(f"Unsupported schema_version: {payload.get('schema_version')!r}.")
        if payload.get("schema_version") is None:
            warnings.append("Wrapped payload does not declare schema_version.")
        if payload.get("version") is None:
            warnings.append("Wrapped payload does not declare exporter version.")
    else:
        model_dict = payload
        feature_schema = None
        warnings.append("Bare FASEModel.to_dict payload is accepted for compatibility but lacks wrapper metadata.")

    weights = model_dict.get("w") if isinstance(model_dict, dict) else None
    if not isinstance(weights, list):
        errors.append("Model field 'w' must be a list of linear weights.")
    else:
        for idx, value in enumerate(weights):
            if not _is_number(value):
                errors.append(f"Model weight w[{idx}] must be numeric.")
                break

    if "b0" not in model_dict or not _is_number(model_dict.get("b0")):
        errors.append("Model field 'b0' must be numeric.")

    for key in ("stage1", "stage2"):
        if not isinstance(model_dict.get(key), list):
            errors.append(f"Model field '{key}' must be a list.")

    stage2 = model_dict.get("stage2", [])
    if isinstance(stage2, list):
        for idx, block in enumerate(stage2):
            if not isinstance(block, dict):
                errors.append(f"stage2[{idx}] must be an object.")
                continue
            if not isinstance(block.get("kind"), str) or not block.get("kind"):
                errors.append(f"stage2[{idx}].kind must be a non-empty string.")
            if "mus" in block and not isinstance(block.get("mus"), list):
                errors.append(f"stage2[{idx}].mus must be a list when present.")
            if "sds" in block and not isinstance(block.get("sds"), list):
                errors.append(f"stage2[{idx}].sds must be a list when present.")

    if feature_schema is not None:
        if not isinstance(feature_schema, dict):
            errors.append("feature_schema must be an object when present.")
        else:
            feature_names = feature_schema.get("feature_names")
            feature_count = feature_schema.get("feature_count")
            if not isinstance(feature_names, list) or not all(isinstance(name, str) for name in feature_names):
                errors.append("feature_schema.feature_names must be a list of strings.")
            if not isinstance(feature_count, int) or isinstance(feature_count, bool):
                errors.append("feature_schema.feature_count must be an integer.")
            elif isinstance(feature_names, list) and feature_count != len(feature_names):
                errors.append("feature_schema.feature_count must equal len(feature_schema.feature_names).")
            if "target" in feature_schema and not isinstance(feature_schema.get("target"), str):
                errors.append("feature_schema.target must be a string when present.")

    return {
        "status": "ok" if not errors else "invalid",
        "schema_version": payload.get("schema_version") if isinstance(payload, dict) else None,
        "format": payload.get("format") if isinstance(payload, dict) else None,
        "wrapped": bool(wrapped),
        "errors": errors,
        "warnings": warnings,
    }


def _is_int_sequence(value: Any) -> bool:
    if isinstance(value, np.ndarray):
        return np.issubdtype(value.dtype, np.integer)
    if isinstance(value, (list, tuple)):
        return all(isinstance(item, (int, np.integer)) and not isinstance(item, bool) for item in value)
    return False


def _int_sequence(value: Any) -> List[int]:
    if isinstance(value, np.ndarray):
        return [int(item) for item in value.ravel().tolist()]
    return [int(item) for item in value]


def _callable_captures(fn: Any) -> Iterator[Tuple[str, Any]]:
    for idx, item in enumerate(getattr(fn, "__defaults__", None) or ()):
        yield f"default[{idx}]", item
    for idx, cell in enumerate(getattr(fn, "__closure__", None) or ()):
        try:
            yield f"closure[{idx}]", cell.cell_contents
        except ValueError:
            continue


def _diagnose_ruliad_export(params: Any) -> Dict[str, Any]:
    diagnostics: Dict[str, Any] = {
        "state_source": None,
        "output_indices_source": None,
        "output_indices": None,
        "errors": [],
        "warnings": [],
    }
    if not isinstance(params, dict):
        diagnostics["errors"].append("params must be an object for ruliad blocks.")
        return diagnostics

    module = load_fase_v21()
    hypergraph_state = getattr(module, "HypergraphState", None)
    state = params.get("state")
    if hypergraph_state is not None and isinstance(state, hypergraph_state):
        diagnostics["state_source"] = "params.state"
    elif state is not None:
        diagnostics["errors"].append("params.state is present but is not a HypergraphState.")

    output_indices = params.get("output_indices")
    if output_indices is not None:
        if _is_int_sequence(output_indices):
            diagnostics["output_indices"] = _int_sequence(output_indices)
            diagnostics["output_indices_source"] = "params.output_indices"
        else:
            diagnostics["errors"].append("params.output_indices must be an integer sequence when present.")

    fn = params.get("state_features")
    if diagnostics["state_source"] is None and callable(fn):
        for source, item in _callable_captures(fn):
            if hypergraph_state is not None and isinstance(item, hypergraph_state):
                diagnostics["state_source"] = f"state_features.{source}"
            elif diagnostics["output_indices_source"] is None and _is_int_sequence(item):
                diagnostics["output_indices"] = _int_sequence(item)
                diagnostics["output_indices_source"] = f"state_features.{source}"
    elif diagnostics["state_source"] is None:
        diagnostics["warnings"].append("params.state_features is not callable, so closure state cannot be inspected.")

    if diagnostics["state_source"] is None:
        diagnostics["errors"].append(
            "ruliad export requires params.state or a state_features default/closure containing HypergraphState."
        )
    return diagnostics


def preflight_model_export(model: Any) -> Dict[str, Any]:
    """Inspect model exportability before calling model.to_dict()."""
    errors: List[str] = []
    warnings: List[str] = []
    blocks: List[Dict[str, Any]] = []

    if model is None:
        return {
            "status": "missing",
            "supported": False,
            "model_type": None,
            "has_to_dict": False,
            "stage1_count": None,
            "stage2_count": None,
            "errors": ["Report does not contain a 'model' entry."],
            "warnings": [],
            "blocks": [],
        }

    has_to_dict = hasattr(model, "to_dict")
    if not has_to_dict:
        errors.append("Model object does not expose to_dict().")

    stage1_specs = getattr(model, "stage1_specs", None)
    stage2_blocks = getattr(model, "stage2_blocks", None)
    if stage1_specs is not None and not isinstance(stage1_specs, list):
        errors.append("model.stage1_specs must be a list when present.")
    if isinstance(stage1_specs, list):
        for idx, spec in enumerate(stage1_specs):
            if not isinstance(spec, dict):
                errors.append(f"stage1[{idx}] must be an object.")
                continue
            for key in ("spec", "Gamma", "mu", "sd"):
                if key not in spec:
                    errors.append(f"stage1[{idx}] is missing {key!r}.")

    if stage2_blocks is None:
        warnings.append("Model does not expose stage2_blocks; export support can only be tested by to_dict().")
    elif not isinstance(stage2_blocks, list):
        errors.append("model.stage2_blocks must be a list when present.")
    else:
        for idx, block in enumerate(stage2_blocks):
            block_report: Dict[str, Any] = {
                "index": idx,
                "kind": None,
                "supported": True,
                "errors": [],
                "warnings": [],
            }
            if not isinstance(block, dict):
                block_report["supported"] = False
                block_report["errors"].append(f"stage2[{idx}] must be an object.")
                blocks.append(block_report)
                continue

            kind = block.get("kind")
            block_report["kind"] = kind
            if not isinstance(kind, str) or not kind:
                block_report["errors"].append(f"stage2[{idx}].kind must be a non-empty string.")
            elif kind not in SUPPORTED_STAGE2_BLOCK_KINDS:
                block_report["errors"].append(
                    f"stage2[{idx}] kind {kind!r} is not supported by FASEModel.from_dict reload."
                )

            for key in ("Gamma", "mus", "sds"):
                if key not in block:
                    block_report["errors"].append(f"stage2[{idx}] is missing {key!r}.")

            params = block.get("params", {})
            if kind == "ruliad":
                ruliad = _diagnose_ruliad_export(params)
                block_report["ruliad"] = {
                    "state_source": ruliad["state_source"],
                    "output_indices_source": ruliad["output_indices_source"],
                    "output_indices": ruliad["output_indices"],
                }
                block_report["errors"].extend(f"stage2[{idx}]: {msg}" for msg in ruliad["errors"])
                block_report["warnings"].extend(f"stage2[{idx}]: {msg}" for msg in ruliad["warnings"])
            elif isinstance(kind, str) and kind in REQUIRED_STAGE2_PARAMS:
                if not isinstance(params, dict):
                    block_report["errors"].append(f"stage2[{idx}].params must be an object.")
                else:
                    for key in REQUIRED_STAGE2_PARAMS[kind]:
                        if key not in params:
                            block_report["errors"].append(f"stage2[{idx}].params is missing {key!r}.")

            block_report["supported"] = not block_report["errors"]
            blocks.append(block_report)

    for block in blocks:
        errors.extend(block["errors"])
        warnings.extend(block["warnings"])

    return {
        "status": "ok" if not errors else "unsupported",
        "supported": not errors,
        "model_type": f"{type(model).__module__}.{type(model).__qualname__}",
        "has_to_dict": bool(has_to_dict),
        "stage1_count": len(stage1_specs) if isinstance(stage1_specs, list) else None,
        "stage2_count": len(stage2_blocks) if isinstance(stage2_blocks, list) else None,
        "supported_stage2_kinds": list(SUPPORTED_STAGE2_BLOCK_KINDS),
        "errors": errors,
        "warnings": warnings,
        "blocks": blocks,
    }


def export_model_from_report(
    report: Dict[str, Any],
    path: str | Path,
    *,
    dataset_metadata: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Export report['model'] to JSON when FASEModel serialization supports it."""
    out = Path(path)
    model = report.get("model")
    feature_schema = _feature_schema_from_metadata(dataset_metadata)
    status = {
        "path": str(out),
        "format": "FASEModel.to_dict",
        "status": "missing",
        "error": None,
        "feature_schema": feature_schema,
        "preflight": preflight_model_export(model),
    }
    if model is None:
        status["error"] = "Report does not contain a 'model' entry."
        return status
    if not hasattr(model, "to_dict"):
        status["status"] = "unsupported"
        status["error"] = "Model object does not expose to_dict()."
        return status
    if status["preflight"]["status"] == "unsupported":
        status["status"] = "unsupported"
        status["error"] = "; ".join(status["preflight"]["errors"])
        return status

    try:
        payload = {
            "format": FASE_MODEL_FORMAT,
            "schema_version": FASE_MODEL_SCHEMA_VERSION,
            "version": __version__,
            "model": model.to_dict(),
        }
        if feature_schema is not None:
            payload["feature_schema"] = feature_schema
        write_json(payload, out)
    except Exception as exc:
        status["status"] = "failed"
        status["error"] = str(exc)
        return status

    status["status"] = "exported"
    return status


def read_json(path: str | Path) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _count_nonzero(values: Any) -> int:
    if not isinstance(values, list):
        return 0
    count = 0
    for value in values:
        try:
            count += int(abs(float(value)) > 0.0)
        except (TypeError, ValueError):
            continue
    return count


def _block_width(block: Dict[str, Any]) -> int:
    for key in ("mus", "sds"):
        values = block.get(key)
        if isinstance(values, list):
            return len(values)
    return 0


def inspect_exported_model(path: str | Path) -> Dict[str, Any]:
    """Inspect an exported FASE model JSON without loading executable model code."""
    model_path = Path(path)
    payload = read_json(model_path)
    if not isinstance(payload, dict):
        raise ValueError(f"Exported model payload must be a JSON object: {model_path}")

    schema_validation = validate_exported_model_payload(payload)
    wrapped = isinstance(payload.get("model"), dict)
    model_dict = payload["model"] if wrapped else payload
    if not isinstance(model_dict, dict):
        raise ValueError(f"Exported model payload does not contain a model object: {model_path}")

    stage1 = model_dict.get("stage1", []) or []
    stage2 = model_dict.get("stage2", []) or []
    weights = model_dict.get("w", []) or []
    stage2_kinds: Dict[str, int] = {}
    stage2_widths: List[int] = []
    for block in stage2:
        if not isinstance(block, dict):
            kind = "unknown"
            width = 0
        else:
            kind = str(block.get("kind", "unknown"))
            width = _block_width(block)
        stage2_kinds[kind] = stage2_kinds.get(kind, 0) + 1
        stage2_widths.append(width)

    feature_schema = payload.get("feature_schema") if wrapped else None
    return {
        "path": str(model_path),
        "format": payload.get("format", "bare-FASEModel.to_dict") if wrapped else "bare-FASEModel.to_dict",
        "schema_version": payload.get("schema_version") if wrapped else None,
        "version": payload.get("version") if wrapped else None,
        "wrapped": bool(wrapped),
        "schema_validation": schema_validation,
        "file": {
            "bytes": model_path.stat().st_size if model_path.exists() else None,
        },
        "feature_schema": feature_schema,
        "has_feature_schema": feature_schema is not None,
        "model": {
            "linear_weights": len(weights) if isinstance(weights, list) else 0,
            "nonzero_linear_weights": _count_nonzero(weights),
            "intercept": model_dict.get("b0"),
            "stage1_count": len(stage1) if isinstance(stage1, list) else 0,
            "stage2_count": len(stage2) if isinstance(stage2, list) else 0,
            "stage2_kinds": stage2_kinds,
            "stage2_widths": stage2_widths,
            "stage2_total_width": int(sum(stage2_widths)),
        },
    }


def _dict_delta(left: Dict[str, int], right: Dict[str, int]) -> Dict[str, Dict[str, int]]:
    delta: Dict[str, Dict[str, int]] = {}
    for key in sorted(set(left) | set(right)):
        left_value = int(left.get(key, 0))
        right_value = int(right.get(key, 0))
        delta[key] = {
            "left": left_value,
            "right": right_value,
            "delta": right_value - left_value,
        }
    return delta


def compare_exported_models(left_path: str | Path, right_path: str | Path) -> Dict[str, Any]:
    """Compare two exported FASE model JSON files using metadata only."""
    left = inspect_exported_model(left_path)
    right = inspect_exported_model(right_path)
    left_schema = left.get("feature_schema") or {}
    right_schema = right.get("feature_schema") or {}
    left_features = [str(name) for name in left_schema.get("feature_names", [])]
    right_features = [str(name) for name in right_schema.get("feature_names", [])]
    left_model = left["model"]
    right_model = right["model"]

    return {
        "left_path": left["path"],
        "right_path": right["path"],
        "same_format": left.get("format") == right.get("format"),
        "same_version": left.get("version") == right.get("version"),
        "feature_schema": {
            "left_has_schema": bool(left.get("has_feature_schema")),
            "right_has_schema": bool(right.get("has_feature_schema")),
            "same_feature_names": left_features == right_features,
            "same_feature_count": left_schema.get("feature_count") == right_schema.get("feature_count"),
            "left_feature_count": left_schema.get("feature_count"),
            "right_feature_count": right_schema.get("feature_count"),
            "left_only_features": sorted(set(left_features) - set(right_features)),
            "right_only_features": sorted(set(right_features) - set(left_features)),
        },
        "complexity_delta": {
            "linear_weights": int(right_model["linear_weights"] - left_model["linear_weights"]),
            "nonzero_linear_weights": int(right_model["nonzero_linear_weights"] - left_model["nonzero_linear_weights"]),
            "stage1_count": int(right_model["stage1_count"] - left_model["stage1_count"]),
            "stage2_count": int(right_model["stage2_count"] - left_model["stage2_count"]),
            "stage2_total_width": int(right_model["stage2_total_width"] - left_model["stage2_total_width"]),
        },
        "stage2_kind_delta": _dict_delta(left_model["stage2_kinds"], right_model["stage2_kinds"]),
        "left": left,
        "right": right,
    }


def load_exported_model(path: str | Path) -> Any:
    model, _ = _load_exported_model_payload(path)
    return model


def predict_from_exported_model(
    model_path: str | Path,
    csv_path: str | Path,
    *,
    drop_column: Optional[str] = None,
    delimiter: str = ",",
    has_header: bool = True,
    strict_schema: bool = False,
) -> Tuple[np.ndarray, Dict[str, Any]]:
    model, feature_schema = _load_exported_model_payload(model_path)
    X, metadata = load_csv_features(
        csv_path,
        drop_column=drop_column,
        delimiter=delimiter,
        has_header=has_header,
    )
    metadata["model_path"] = str(model_path)
    metadata["model_feature_schema"] = feature_schema
    metadata["schema_warnings"] = _schema_warnings(feature_schema, metadata)
    _raise_if_strict_schema_failed(metadata["schema_warnings"], strict_schema=strict_schema)
    yhat = np.asarray(model.predict(X), dtype=float).reshape(-1)
    return yhat, metadata


def evaluate_exported_model(
    model_path: str | Path,
    csv_path: str | Path,
    *,
    target: Optional[str] = None,
    delimiter: str = ",",
    has_header: bool = True,
    strict_schema: bool = False,
) -> Dict[str, Any]:
    """Evaluate an exported FASEModel against a numeric CSV target column."""
    model, feature_schema = _load_exported_model_payload(model_path)
    X, y_true, metadata = load_csv_dataset(
        csv_path,
        target=target,
        delimiter=delimiter,
        has_header=has_header,
    )
    schema_warnings = _schema_warnings(feature_schema, metadata)
    _raise_if_strict_schema_failed(schema_warnings, strict_schema=strict_schema)
    y_pred = np.asarray(model.predict(X), dtype=float).reshape(-1)
    y_true = np.asarray(y_true, dtype=float).reshape(-1)
    if y_pred.shape[0] != y_true.shape[0]:
        raise ValueError(f"Prediction length {y_pred.shape[0]} does not match target length {y_true.shape[0]}")

    residuals = y_pred - y_true
    mse = float(np.mean(residuals ** 2))
    mae = float(np.mean(np.abs(residuals)))
    rmse = float(np.sqrt(mse))
    max_abs_error = float(np.max(np.abs(residuals))) if residuals.size else 0.0
    total = float(np.sum((y_true - np.mean(y_true)) ** 2))
    r2 = float(1.0 - (np.sum(residuals ** 2) / total)) if total > 0.0 else float("nan")

    return {
        "model_path": str(model_path),
        "dataset": metadata,
        "model_feature_schema": feature_schema,
        "schema_warnings": schema_warnings,
        "metrics": {
            "rows": int(y_true.shape[0]),
            "mse": mse,
            "rmse": rmse,
            "mae": mae,
            "max_abs_error": max_abs_error,
            "r2": r2,
        },
        "predictions": [float(x) for x in y_pred],
    }


def write_predictions_csv(predictions: np.ndarray, path: str | Path) -> None:
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    tmp = out.with_suffix(out.suffix + ".tmp")
    with open(tmp, "w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["row", "prediction"])
        for i, pred in enumerate(np.asarray(predictions).reshape(-1)):
            writer.writerow([i, float(pred)])
    tmp.replace(out)


def write_json(data: Dict[str, Any], path: str | Path) -> None:
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    tmp = out.with_suffix(out.suffix + ".tmp")
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)
    tmp.replace(out)


def summarize_v7_report(report: Dict[str, Any]) -> Dict[str, Any]:
    summary: Dict[str, Any] = {}
    for mode_name, mode_report in report.items():
        metrics = mode_report.get("metrics", {})
        gate = mode_report.get("compactness_gate") or {}
        summary[mode_name] = {
            "regime_accuracy": float(metrics.get("regime_accuracy", np.nan)),
            "exact_agreement": float(metrics.get("exact_agreement", np.nan)),
            "avg_hamming": float(metrics.get("avg_hamming", np.nan)),
            "mean_template_purity": float(metrics.get("mean_template_purity", np.nan)),
            "mean_num_ops": float(metrics.get("mean_num_ops", np.nan)),
            "mean_median_bits": float(metrics.get("mean_median_bits", np.nan)),
            "gate_all_pass": bool(gate.get("all_pass", False)),
        }
    return summary
