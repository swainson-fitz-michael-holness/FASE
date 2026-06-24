# FASE_v22.py — Audited benchmark harness for FASE vs PySR (Gate 0).
#
# This file deliberately contains ONLY the protocol corrections identified in
# the v21 audit; everything else is imported unmodified from FASE_v21.py, so
# this file IS the diff. Reviewers should read it top to bottom as a changelog.
#
# The six fixes:
#
#   1. LEAKAGE (fatal in v21): forward_atomic_search, stage-2, and stage-2.5
#      selected features by scoring candidates on the same fold later reported
#      as "OOF". v22 nests selection: each outer-training fold is split again,
#      ALL selection (stage-1 atoms, OG-SET, stage-2 grammar, stage-2.5
#      ruliad, the linear-lock decision) sees only the inner split, the linear
#      head is refit on the full outer-training fold with the selected
#      features frozen, and the outer validation fold is touched exactly once
#      — by predict(). See run_fase_kfold_nested / refit_linear_head.
#
#   2. PROTOCOL UNIFICATION: v21 compared FASE 5-fold OOF R^2 against PySR
#      single-split (20%) validation R^2. v22 runs both methods on the SAME
#      outer folds and assembles out-of-fold predictions identically.
#      See run_pysr_kfold.
#
#   3. AUGMENTATION -> LABELED ABLATION: v21's demo path injected FASE's
#      OG-SET features into PySR's input matrix unconditionally (the
#      OGSET["augment_pysr"] flag existed but was never consulted — the same
#      decorative-config failure class as fix #4). v22's headline baseline is
#      PySR on raw X; the augmented run survives only as an explicitly
#      labeled, default-OFF ablation ("pysr_plus_fase_features_ablation").
#
#   4. PYSR PASS-THROUGH: v21 defined config keys (deterministic, parallelism)
#      that were never forwarded to PySRRegressor — silently dropped, so runs
#      were not reproducible despite random_state. v22 forwards every key and
#      RECORDS anything the installed PySR version rejects
#      (kwargs_dropped_by_version in the result JSON). Nothing is silent.
#
#   5. NO INVENTED SIGMA: v21's demo synthesized a heteroskedastic weight
#      vector (0.8 + 0.4*|sin t|) for PMLB targets with no noise model. v22
#      uses Sigma=None for all real data; Sigma-aware mode is reserved for
#      synthetic data with a KNOWN noise covariance.
#
#   6. ARTIFACT = CLAIM: benchmark runs are driven by a committed manifest
#      (benchmarks/pmlb_manifest.json), produce one JSON per (dataset, seed)
#      plus summary.json and environment.json (see scripts/run_pysr_bench.py),
#      and a dataset that fails to load is recorded as load_failed — never
#      silently replaced by synthetic data.
#
# Excluded from the audited path: v21's fit_consensus_model. It refits with
# train == val == the full dataset (in-sample by construction). It remains
# available for deployment artifacts but must never feed a reported metric.
#
# Empirical check: scripts/null_leakage_check.py runs the v21 and v22
# protocols on pure noise (y independent of X). Any materially positive OOF
# R^2 under v21 on that data is the leak made visible.

from __future__ import annotations

import hashlib
import json
import re
import time
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

import FASE_v21 as v21
from FASE_v21 import (
    CONFIG,
    FASEModel,
    gls_r2,
    kfold_indices,
    load_pmlb_dataset,
    r2_score,
    ridge_with_intercept,
    run_fase_given_split,
    subset_sigma,
    train_val_split_indices,
)

# =========================
# Pinned PySR baseline configuration
# =========================
# Operator set follows the SRBench convention (+,-,*,/ with sin, cos, exp,
# log, sqrt, abs); the budget is 4x v21's 250 iterations, serial and
# deterministic. PROVENANCE NOTE: SRBench's exact PySR entry could not be
# fetched at build time (GitHub API rate-limited; candidate raw paths
# returned non-200), so these values are explicit v22 defaults to be
# cross-checked against the SRBench repository — or against whatever
# configuration the PySR maintainers would call fair — before the public
# run. Every kwarg actually used, and every kwarg dropped by the installed
# PySR version, is recorded in the per-dataset result JSON, so the run is
# auditable regardless of which values end up pinned.
PYSR_BENCH_CONFIG: Dict[str, Any] = {
    "niterations": 1000,
    "binary_operators": ["+", "-", "*", "/"],
    "unary_operators": ["sin", "cos", "exp", "log", "sqrt", "abs"],
    "maxsize": 30,
    "model_selection": "best",
    "deterministic": True,
    "parallelism": "serial",
    "procs": 0,
    "batching": False,
    "timeout_in_seconds": None,
}


def apply_bench_config(quick: bool = False,
                       pysr_overrides: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """Mutate the shared v21 CONFIG into benchmark mode.

    CONFIG is the same dict object FASE_v21's internals read, so changes here
    propagate into stage-1/2/2.5 and OG-SET. Returns the resolved PySR config.
    """
    CONFIG["COMPARE_WITH_PYSR"] = False          # v22 owns the baseline; v21's demo path is retired
    CONFIG["OGSET"]["augment_pysr"] = False      # fix #3: ablation only, never a default
    CONFIG["PYSR"] = dict(PYSR_BENCH_CONFIG)
    if pysr_overrides:
        CONFIG["PYSR"].update(pysr_overrides)
    if quick:
        # Smoke-test budget only. Never publish numbers from quick mode.
        CONFIG["MAX_ATOMS"] = 8
        CONFIG["MAX_GRAMMAR"] = 0
        CONFIG["USE_RULIAD_STAGE25"] = False
        CONFIG["OGSET"]["bag_boots"] = 2
        CONFIG["PYSR"].update({"niterations": 30, "maxsize": 15,
                               "timeout_in_seconds": 300})
    return CONFIG["PYSR"]


# =========================
# Small shared utilities
# =========================

def _json_safe(v):
    if isinstance(v, dict):
        return {str(k): _json_safe(x) for k, x in v.items()}
    if isinstance(v, (list, tuple)):
        return [_json_safe(x) for x in v]
    if isinstance(v, np.ndarray):
        return v.tolist()
    if isinstance(v, np.generic):
        return v.item()
    if isinstance(v, float) and not np.isfinite(v):
        return None
    return v


def _sanitize_predictions(yhat: np.ndarray, fill_value: float) -> Tuple[np.ndarray, int]:
    """Replace non-finite predictions with the training-fold mean.

    Applied symmetrically to FASE and PySR; the count is recorded per fold so
    the substitution is visible in the artifact rather than silent."""
    yhat = np.asarray(yhat, float).ravel()
    bad = ~np.isfinite(yhat)
    n_bad = int(bad.sum())
    if n_bad:
        yhat = yhat.copy()
        yhat[bad] = fill_value
    return yhat, n_bad


def fold_index_hash(folds) -> str:
    """Stable fingerprint of the exact validation-fold indices, recorded in
    every result JSON so cross-machine runs can verify identical splits."""
    h = hashlib.sha256()
    for _, va in folds:
        h.update(np.asarray(va, dtype=np.int64).tobytes())
    return h.hexdigest()[:16]


# =========================
# Fix #1 — nested selection
# =========================

def refit_linear_head(model: FASEModel, X_tr, y_tr, Sigma_tr=None,
                      alpha: Optional[float] = None) -> FASEModel:
    """Refit ONLY the final ridge weights on the full outer-training fold.

    The feature pipeline (stage-1 specs, OG-SET columns, stage-2/2.5 blocks)
    stays frozen exactly as selected on the inner split; no selection
    decision is revisited, so the outer validation fold remains untouched by
    selection."""
    if alpha is None:
        alpha = CONFIG["ALPHA_RIDGE"]
    X_tr = np.asarray(X_tr, float)
    y_tr = np.asarray(y_tr, float).ravel()
    F = model._build_features(X_tr)
    w, b0, _ = ridge_with_intercept(F, y_tr, alpha, Sigma=Sigma_tr)
    model.w = w
    model.b0 = b0
    return model


def run_fase_kfold_nested(X, y, Sigma=None, K: int = 5, seed: int = 1337,
                          inner_val_frac: float = 0.25, folds=None,
                          return_models: bool = False) -> Dict[str, Any]:
    """Leakage-fixed replacement for v21.run_fase_kfold (fix #1).

    Per outer fold: the training indices are split again
    (train_val_split_indices, seed offset +1000+k). run_fase_given_split —
    and therefore every selection decision in the pipeline — sees ONLY that
    inner split. The linear head is then refit on the full outer-training
    fold with the selected features frozen, and the outer validation fold is
    used exactly once: predict().

    Data exposure is symmetric with the PySR baseline: each method sees the
    outer-training fold and nothing else."""
    X = np.asarray(X, float)
    y = np.asarray(y, float).ravel()
    n = len(y)
    if folds is None:
        folds = kfold_indices(n, K, seed=seed, shuffle=True)
    K = len(folds)
    oof = np.full(n, np.nan)
    fold_reports: List[Dict[str, Any]] = []
    models: List[FASEModel] = []
    t0 = time.time()
    for k, (tr, va) in enumerate(folds, 1):
        itr, iva = train_val_split_indices(len(tr), val_frac=inner_val_frac,
                                           seed=seed + 1000 + k)
        sel_tr, sel_va = tr[itr], tr[iva]
        out = run_fase_given_split(
            X[sel_tr], y[sel_tr], X[sel_va], y[sel_va],
            seed=seed + k, name=f"fold{k}_inner",
            Sigma_tr=subset_sigma(Sigma, sel_tr),
            Sigma_va=subset_sigma(Sigma, sel_va),
        )
        model = refit_linear_head(out["model"], X[tr], y[tr],
                                  Sigma_tr=subset_sigma(Sigma, tr))
        fill = float(np.mean(y[tr]))
        yhat, n_bad = _sanitize_predictions(model.predict(X[va]), fill)
        oof[va] = yhat
        fold_reports.append(dict(
            fold=k,
            R2=r2_score(y[va], yhat),
            MSE=float(np.mean((y[va] - yhat) ** 2)),
            kept=out["kept"],
            og_names=out["og_names"],
            blocks=[b["kind"] for b in out["blocks"]],
            n_outer_train=int(len(tr)),
            n_inner_select_train=int(len(sel_tr)),
            n_inner_select_val=int(len(sel_va)),
            n_outer_val=int(len(va)),
            n_nonfinite_predictions=n_bad,
        ))
        models.append(model)
    res: Dict[str, Any] = dict(
        protocol="v22_nested_kfold",
        R2_oof=r2_score(y, oof),
        MSE_oof=float(np.mean((y - oof) ** 2)),
        R2_oof_gls=(gls_r2(y, oof, Sigma) if Sigma is not None else None),
        folds=fold_reports,
        k_folds=K,
        seed=seed,
        inner_val_frac=inner_val_frac,
        wall_time_s=time.time() - t0,
        oof_pred=oof,
    )
    if return_models:
        res["models"] = models
    return res


# =========================
# Fixes #2/#3/#4 — PySR under the identical protocol
# =========================

def _build_pysr_kwargs(cfg: Dict[str, Any], seed: int, n_train: int) -> Dict[str, Any]:
    kw: Dict[str, Any] = {
        "niterations": int(cfg.get("niterations", 1000)),
        "binary_operators": list(cfg.get("binary_operators", ["+", "-", "*", "/"])),
        "unary_operators": list(cfg.get("unary_operators",
                                        ["sin", "cos", "exp", "log", "sqrt", "abs"])),
        "maxsize": int(cfg.get("maxsize", 30)),
        "model_selection": cfg.get("model_selection", "best"),
        "random_state": int(seed),
        "deterministic": bool(cfg.get("deterministic", True)),
        "parallelism": cfg.get("parallelism", "serial"),
        "procs": int(cfg.get("procs", 0)),
        "verbosity": 0,
        "progress": False,
    }
    if cfg.get("maxdepth") is not None:
        kw["maxdepth"] = int(cfg["maxdepth"])
    if cfg.get("populations") is not None:
        kw["populations"] = int(cfg["populations"])
    if cfg.get("population_size") is not None:
        kw["population_size"] = int(cfg["population_size"])
    if cfg.get("timeout_in_seconds") is not None:
        kw["timeout_in_seconds"] = int(cfg["timeout_in_seconds"])
    if cfg.get("batching", False):
        kw["batching"] = True
        kw["batch_size"] = int(min(cfg.get("batch_size", 2048), max(n_train, 1)))
    return kw


def _construct_pysr(kwargs: Dict[str, Any]):
    """Construct PySRRegressor forwarding ALL kwargs; auto-drop (and record)
    any keyword the installed PySR version does not accept (fix #4 — v21
    defined deterministic/parallelism in config but never passed them)."""
    from pysr import PySRRegressor

    kw = dict(kwargs)
    dropped: Dict[str, Any] = {}
    for _ in range(len(kwargs) + 1):
        try:
            return PySRRegressor(**kw), dropped
        except TypeError as e:
            m = re.search(r"unexpected keyword argument '([^']+)'", str(e))
            if m is None or m.group(1) not in kw:
                raise
            bad = m.group(1)
            dropped[bad] = kw.pop(bad)
    raise RuntimeError("could not construct PySRRegressor after dropping all rejected kwargs")


def fit_pysr_once(Xtr, ytr, Xva, seed: int, cfg: Dict[str, Any]) -> Dict[str, Any]:
    t0 = time.time()
    try:
        import pysr
    except Exception as e:
        return {"status": "pysr_unavailable", "error": f"{type(e).__name__}: {e}"}
    kwargs = _build_pysr_kwargs(cfg, seed=seed, n_train=len(ytr))
    try:
        model, dropped = _construct_pysr(kwargs)
        model.fit(np.asarray(Xtr, float), np.asarray(ytr, float).ravel())
        yhat = np.asarray(model.predict(np.asarray(Xva, float)), float).ravel()
    except Exception as e:
        return {"status": "fit_failed", "error": f"{type(e).__name__}: {e}",
                "fit_time_s": time.time() - t0}
    expr = None
    try:
        expr = str(model.sympy())
    except Exception:
        pass
    return {
        "status": "ok",
        "yhat": yhat,
        "expr": expr,
        "pysr_version": getattr(pysr, "__version__", "unknown"),
        "kwargs_used": {k: v for k, v in kwargs.items() if k not in dropped},
        "kwargs_dropped_by_version": dropped,
        "fit_time_s": time.time() - t0,
    }


def run_pysr_kfold(X, y, folds, seed: int, cfg: Dict[str, Any],
                   augment: Optional[Dict[str, List[np.ndarray]]] = None,
                   label: str = "pysr_raw") -> Dict[str, Any]:
    """Identical-protocol PySR baseline (fixes #2/#3/#4).

    Same outer folds as FASE, fit on raw X only (`augment` is reserved for
    the explicitly labeled ablation), OOF predictions assembled identically,
    non-finite predictions handled identically to FASE. This replaces v21's
    single-split, FASE-feature-augmented comparison."""
    X = np.asarray(X, float)
    y = np.asarray(y, float).ravel()
    oof = np.full(len(y), np.nan)
    per_fold: List[Dict[str, Any]] = []
    pysr_version = None
    t0 = time.time()
    for k, (tr, va) in enumerate(folds, 1):
        Xtr, Xva = X[tr], X[va]
        if augment is not None:
            Xtr = np.hstack([Xtr, np.asarray(augment["train"][k - 1], float)])
            Xva = np.hstack([Xva, np.asarray(augment["val"][k - 1], float)])
        res = fit_pysr_once(Xtr, y[tr], Xva, seed=seed + k, cfg=cfg)
        if res.get("status") != "ok":
            per_fold.append({"fold": k,
                             **{kk: vv for kk, vv in res.items() if kk != "yhat"}})
            continue
        fill = float(np.mean(y[tr]))
        yhat, n_bad = _sanitize_predictions(res.pop("yhat"), fill)
        oof[va] = yhat
        pysr_version = res.get("pysr_version", pysr_version)
        per_fold.append({
            "fold": k,
            "R2": r2_score(y[va], yhat),
            "MSE": float(np.mean((y[va] - yhat) ** 2)),
            "n_nonfinite_predictions": n_bad,
            **res,
        })
    ok = np.isfinite(oof)
    out: Dict[str, Any] = {
        "label": label,
        "protocol": "v22_same_folds_oof",
        "folds": per_fold,
        "n_folds_ok": int(sum(1 for f in per_fold if "R2" in f)),
        "n_folds_total": len(folds),
        "oof_coverage": float(np.mean(ok)),
        "seed": seed,
        "wall_time_s": time.time() - t0,
    }
    if pysr_version:
        out["pysr_version"] = pysr_version
    if ok.all():
        out["R2_oof"] = r2_score(y, oof)
        out["MSE_oof"] = float(np.mean((y - oof) ** 2))
    elif ok.any():
        out["R2_oof_partial"] = r2_score(y[ok], oof[ok])
        out["MSE_oof_partial"] = float(np.mean((y[ok] - oof[ok]) ** 2))
    return out


def build_fase_feature_augment(models: List[FASEModel], X, folds) -> Dict[str, List[np.ndarray]]:
    """Per-fold FASE feature matrices for the labeled ablation (fix #3).

    Each fold model's pipeline was selected on inner-train only and is
    frozen; applying it to the outer validation fold is transform-only (no
    target involvement), so the ablation inherits the nested protocol's
    guarantees. The ablation answers a different question from the headline
    comparison: not "is FASE better than PySR" but "does PySR improve when
    handed FASE's discovered feature space"."""
    X = np.asarray(X, float)
    aug: Dict[str, List[np.ndarray]] = {"train": [], "val": []}
    for model, (tr, va) in zip(models, folds):
        aug["train"].append(model._build_features(X[tr]))
        aug["val"].append(model._build_features(X[va]))
    return aug


# =========================
# Fixes #5/#6 — one dataset, one seed, one protocol
# =========================

def run_benchmark_on_dataset(name: str, k_folds: int = 5, seed: int = 42,
                             pysr_cfg: Optional[Dict[str, Any]] = None,
                             run_pysr: bool = True,
                             run_ablation: bool = False) -> Dict[str, Any]:
    """Benchmark one PMLB dataset under the unified nested protocol.

    Sigma is None unconditionally for real data (fix #5): v21's demo
    synthesized a heteroskedastic weight vector (0.8 + 0.4*|sin t|) for PMLB
    targets; no noise model is known for these datasets, so none is assumed.
    A dataset that fails to load is recorded as status="load_failed" and
    skipped — never silently replaced by synthetic data (fix #6)."""
    info: Dict[str, Any] = {
        "dataset": name,
        "seed": int(seed),
        "k_folds": int(k_folds),
        "protocol": "v22_unified_nested_oof",
        "sigma": None,
    }
    X, y = load_pmlb_dataset(name)
    if X is None or y is None:
        info["status"] = "load_failed"
        return info
    X = np.asarray(X, float)
    y = np.asarray(y, float).ravel()
    n_unique = int(len(np.unique(y)))
    info.update(n=int(len(y)), d=int(X.shape[1]), y_n_unique=n_unique,
                target_looks_categorical=bool(n_unique <= 10))
    folds = kfold_indices(len(y), k_folds, seed=seed, shuffle=True)
    info["fold_sizes"] = [int(len(va)) for _, va in folds]
    info["fold_index_hash"] = fold_index_hash(folds)

    fase = run_fase_kfold_nested(X, y, Sigma=None, K=k_folds, seed=seed,
                                 folds=folds, return_models=run_ablation)
    models = fase.pop("models", None)
    fase.pop("oof_pred", None)
    info["fase"] = fase

    if run_pysr:
        cfg = pysr_cfg or CONFIG["PYSR"]
        info["pysr_raw"] = run_pysr_kfold(X, y, folds, seed=seed, cfg=cfg,
                                          label="pysr_raw")
        if run_ablation and models:
            aug = build_fase_feature_augment(models, X, folds)
            info["pysr_plus_fase_features_ablation"] = run_pysr_kfold(
                X, y, folds, seed=seed, cfg=cfg, augment=aug,
                label="pysr_plus_fase_features_ablation")
        f_r2 = fase.get("R2_oof")
        p_r2 = info["pysr_raw"].get("R2_oof")
        if f_r2 is not None and p_r2 is not None:
            info["delta_R2_oof_fase_minus_pysr"] = float(f_r2 - p_r2)
    info["status"] = "ok"
    return info


if __name__ == "__main__":
    # Minimal self-check on synthetic data (no PySR/PMLB required).
    from FASE_v21 import make_synthetic
    apply_bench_config(quick=True)
    X, y, _Sigma_true = make_synthetic(seed=7, n=240, d=6, gls_noise=False)
    res = run_fase_kfold_nested(X, y, Sigma=None, K=3, seed=7)
    print(f"[v22 smoke] nested OOF R2={res['R2_oof']:.4f} "
          f"MSE={res['MSE_oof']:.6f} ({res['wall_time_s']:.1f}s, "
          f"protocol={res['protocol']})")
