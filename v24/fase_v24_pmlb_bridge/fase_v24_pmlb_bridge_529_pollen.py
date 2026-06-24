#!/usr/bin/env python3
"""
FASE-v24 PMLB Bridge Probe — 529_pollen
=======================================

Purpose
-------
This is the terminal locality-breach experiment for the v23.1/v24 research arc.
It does NOT pretend that an iid tabular regression dataset already contains an
observed recursive trajectory. Instead it constructs a support-aware empirical
perturbation graph, asks whether repeated constant feature operations induce a
reusable target/inference operator, validates those operators on untouched
empirical pairs, and requires target/pair-shuffle destroyer controls to abstain.

Pipeline
--------
PMLB table
  -> nested predictive baselines (Ridge, v23, v23.1, optional PySR)
  -> fold-local standardization
  -> k-nearest-neighbor empirical graph
  -> axis-dominant constant perturbation modes g_{j,+/-}
  -> numerical typed rewrite schemas omega_g
  -> held-out empirical-transition validation
  -> support-safe counterfactual model-consistency audit
  -> target/pair-shuffle destroyer controls
  -> cross-fold stability and conversion-horizon classification

The bridge operator has the typed form

    omega_g : State(x, y) -> State(x + delta_g, psi_g(x, y))

where delta_g is induced from repeated feature-space transformations and psi_g
is selected from a small, complexity-penalized numerical rewrite language.
This is a bridge probe, not a claim that the original dataset contains physical
time or causal interventions.

Dependencies
------------
Python 3.10+, numpy, pandas, scipy, scikit-learn.
Optional: pmlb (data fetch), pysr (PySR baseline).
Bundled: v23.1 and its base modules, plus an offline 529_pollen TSV.
"""
from __future__ import annotations

import argparse
import copy
import csv
import gzip
import hashlib
import json
import math
import os
import platform
import pickle
import subprocess
import random
import re
import sys
import tempfile
import time
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

# Prevent nested numerical oversubscription before importing numpy/sklearn.
for _name in (
    "OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS",
):
    os.environ.setdefault(_name, "1")

import numpy as np
import pandas as pd
from scipy.optimize import minimize_scalar
from sklearn.base import clone
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import GroupKFold, KFold
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

import fase_v23_g1a_coordinate_gate as v23base
import fase_v23_1_joint_refined_gate as v231

EPS = 1e-12
DATASET = "529_pollen"
FEATURE_NAMES = ["RIDGE", "NUB", "CRACK", "WEIGHT"]


# ======================================================================================
# Utilities
# ======================================================================================

def json_safe(x: Any) -> Any:
    if isinstance(x, dict):
        return {str(k): json_safe(v) for k, v in x.items()}
    if isinstance(x, (list, tuple, set)):
        return [json_safe(v) for v in x]
    if isinstance(x, np.ndarray):
        return x.tolist()
    if isinstance(x, np.generic):
        return x.item()
    if isinstance(x, Path):
        return str(x)
    if isinstance(x, float) and not math.isfinite(x):
        return None
    # SymPy/PySR/pandas objects: preserve a readable form rather than fail.
    module = getattr(type(x), "__module__", "")
    if module.startswith(("sympy", "pysr", "pandas")):
        return str(x)
    return x


def atomic_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    text = json.dumps(json_safe(payload), indent=2, allow_nan=False)
    with tempfile.NamedTemporaryFile("w", encoding="utf-8", dir=path.parent, delete=False) as f:
        f.write(text)
        tmp = Path(f.name)
    os.replace(tmp, path)


def stable_hash(*items: Any) -> int:
    h = hashlib.sha256()
    for item in items:
        h.update(str(item).encode("utf-8"))
        h.update(b"\0")
    return int.from_bytes(h.digest()[:8], "little") & 0x7FFFFFFF


def nrmse(y: np.ndarray, pred: np.ndarray) -> float:
    y = np.asarray(y, float).reshape(-1)
    pred = np.asarray(pred, float).reshape(-1)
    scale = float(np.std(y))
    if scale < EPS:
        scale = max(float(np.mean(np.abs(y))), 1.0)
    return float(np.sqrt(np.mean((y - pred) ** 2)) / scale)


def safe_r2(y: np.ndarray, pred: np.ndarray) -> float:
    try:
        val = float(r2_score(np.asarray(y, float), np.asarray(pred, float)))
        return val if math.isfinite(val) else -1e9
    except Exception:
        return -1e9


def median_mad(values: Sequence[float]) -> Tuple[float, float]:
    a = np.asarray(list(values), float)
    if len(a) == 0:
        return float("nan"), float("nan")
    med = float(np.median(a))
    return med, float(np.median(np.abs(a - med)))


def confidence_upper_zero_events(n: int, alpha: float = 0.05) -> Optional[float]:
    if n <= 0:
        return None
    # Exact one-sided Clopper-Pearson upper bound with k=0.
    return float(1.0 - alpha ** (1.0 / n))


# ======================================================================================
# Dataset loading
# ======================================================================================

def _read_table(path: Path) -> pd.DataFrame:
    suffixes = "".join(path.suffixes).lower()
    sep = "\t" if ".tsv" in suffixes else ","
    return pd.read_csv(path, sep=sep)


def load_dataset(dataset: str, data_file: Optional[str], cache_dir: Optional[str]) -> Tuple[np.ndarray, np.ndarray, List[str], Dict[str, Any]]:
    if dataset != DATASET:
        raise ValueError(f"This frozen probe only accepts {DATASET!r}; got {dataset!r}")

    provenance: Dict[str, Any] = {
        "dataset": dataset,
        "expected_rows": 3848,
        "expected_features": 4,
        "source_note": "POLLEN is a synthetic benchmark; iid rows are not temporal trajectories.",
    }

    if data_file:
        path = Path(data_file).expanduser().resolve()
        if not path.exists():
            raise FileNotFoundError(path)
        df = _read_table(path)
        provenance.update({"loader": "explicit_file", "path": str(path)})
    else:
        df = None
        try:
            from pmlb import fetch_data
            fetched = fetch_data(dataset, return_X_y=False, local_cache_dir=cache_dir)
            if hasattr(fetched, "columns"):
                df = fetched.copy()
                provenance.update({"loader": "pmlb.fetch_data", "cache_dir": cache_dir})
        except Exception as exc:
            provenance["pmlb_error"] = f"{type(exc).__name__}: {exc}"

        if df is None:
            bundled = HERE / "data" / "529_pollen.tsv.gz"
            if not bundled.exists():
                raise RuntimeError("PMLB load failed and bundled data is missing")
            df = pd.read_csv(bundled, sep="\t")
            provenance.update({"loader": "bundled_offline_copy", "path": str(bundled)})

    target_col = "target" if "target" in df.columns else df.columns[-1]
    features = [c for c in df.columns if c != target_col]
    X = np.asarray(df[features], float)
    y = np.asarray(df[target_col], float).reshape(-1)

    if X.shape != (3848, 4):
        provenance["shape_warning"] = f"Observed shape {X.shape}; canonical profile is (3848, 4)."
    if not np.all(np.isfinite(X)) or not np.all(np.isfinite(y)):
        raise ValueError("Dataset contains non-finite values")

    provenance.update({
        "n": int(len(y)), "d": int(X.shape[1]), "feature_names": features,
        "target_name": str(target_col), "missing": int(np.isnan(np.column_stack([X, y])).sum()),
        "sha256_numeric": hashlib.sha256(np.column_stack([X, y]).astype(np.float64).tobytes()).hexdigest(),
    })
    return X, y, [str(c) for c in features], provenance


# ======================================================================================
# v23.1 fold model: original branch retained, refined branch selected only by inner CV
# ======================================================================================

@dataclass
class V231FoldConfig:
    quick: bool = False
    max_candidates_scored: Optional[int] = None
    depth1_beam: Optional[int] = None
    max_layers: Optional[int] = None
    accept_per_layer: Optional[int] = None
    nondegrade_tolerance: float = 1e-3


class V231FoldModel:
    def __init__(self, seed: int, cfg: V231FoldConfig):
        self.seed = int(seed)
        self.cfg = cfg
        self.record: Dict[str, Any] = {}
        self._raw = None
        self._baseline = None
        self._refined = None
        self._recommended = "raw_ridge"
        self._blend_weight = 0.0
        self._blend_challenger = "baseline"
        self._prediction_bound = float("inf")
        self._fallback_count = 0

    @staticmethod
    def _fit_expr_model(exprs: Sequence[v23base.Expr], X: np.ndarray, y: np.ndarray, alpha: float):
        Z, fit = v23base.fit_design(list(exprs), X)
        w, b0 = v23base.ridge_fit(Z, y, alpha)
        return (list(exprs), fit, np.asarray(w, float), float(b0))

    @staticmethod
    def _predict(model, X: np.ndarray) -> np.ndarray:
        _, fit, w, b0 = model
        Z = fit.transform(np.asarray(X, float))
        pred = np.asarray(v23base.ridge_predict(Z, w, b0), float).reshape(-1)
        pred[~np.isfinite(pred)] = b0
        return pred

    @classmethod
    def _oof_predictions(cls, exprs: Sequence[v23base.Expr], X: np.ndarray, y: np.ndarray,
                         splits: Sequence[Tuple[np.ndarray, np.ndarray]], alpha: float) -> np.ndarray:
        out = np.full(len(y), np.nan, float)
        for tr, va in splits:
            model = cls._fit_expr_model(exprs, X[tr], y[tr], alpha)
            out[va] = cls._predict(model, X[va])
        return out

    def fit(self, X: np.ndarray, y: np.ndarray) -> "V231FoldModel":
        X = np.asarray(X, float)
        y = np.asarray(y, float).reshape(-1)
        cfg = v231.make_config(self.seed, self.cfg.quick, include_sqrt=True)
        if self.cfg.max_candidates_scored is not None:
            cfg.base.max_candidates_scored = int(self.cfg.max_candidates_scored)
        if self.cfg.depth1_beam is not None:
            cfg.base.depth1_beam = int(self.cfg.depth1_beam)
        if self.cfg.max_layers is not None:
            cfg.base.max_layers = int(self.cfg.max_layers)
        if self.cfg.accept_per_layer is not None:
            cfg.base.accept_per_layer = int(self.cfg.accept_per_layer)

        idx = np.arange(len(y))
        inner_tr, inner_va = v23base.split_inner(idx, cfg.base.inner_val_frac, self.seed + 1000)
        selected, discovery = v23base.discover_coordinates(
            X[inner_tr], y[inner_tr], X[inner_va], y[inner_va], cfg.base
        )
        base_context = v23base.base_exprs(X.shape[1], cfg.base)
        baseline_exprs = base_context + [s.expr for s in selected]

        rcfg = copy.deepcopy(cfg.refine)
        rcfg.seed = self.seed + 10000
        refined_sel, refine_records = v231.refine_selected_nested_cv(
            copy.deepcopy(selected), base_context, X, y, rcfg
        )
        template_record: Dict[str, Any] = {"tested": 0, "accepted": False}
        if cfg.promote_parameterized_templates:
            current_context = base_context + [s.expr for s in refined_sel]
            promoted, template_record = v231.promote_best_template(
                current_context, X, y, cfg.base, rcfg, cfg.template_top_k
            )
            if promoted is not None:
                refined_sel.append(v23base.SelectedCoord(
                    expr=promoted,
                    layer=cfg.base.max_layers + 1,
                    val_gain=float(template_record.get("best_cv_r2", 0.0) - template_record.get("base_holdout_r2", 0.0)),
                    val_r2_after=float(template_record.get("best_cv_r2", 0.0)),
                    complexity=promoted.complexity(),
                    residual_alignment=0.0,
                ))
        refined_exprs = base_context + [s.expr for s in refined_sel]

        splits = v231._cv_splits(len(y), max(2, rcfg.inner_folds), self.seed + 20000)
        raw_cv = float(v231._score_exprs_cv(base_context, X, y, splits, max(cfg.base.ridge_alpha, 1e-4)))
        baseline_cv = float(v231._score_exprs_cv(baseline_exprs, X, y, splits, max(cfg.base.ridge_alpha, 1e-4)))
        refined_cv = float(v231._score_exprs_cv(refined_exprs, X, y, splits, max(cfg.base.ridge_alpha, rcfg.ridge_alpha)))
        scores = {"raw_ridge": raw_cv, "baseline": baseline_cv, "refined": refined_cv}
        challenger = "refined" if refined_cv >= baseline_cv else "baseline"
        challenger_exprs = refined_exprs if challenger == "refined" else baseline_exprs
        challenger_alpha = max(cfg.base.ridge_alpha, rcfg.ridge_alpha) if challenger == "refined" else cfg.base.ridge_alpha

        # Convex champion/challenger blend selected only from inner-CV predictions.
        # w=0 is the raw-feature Ridge anchor, so the candidate space explicitly contains
        # the non-degraded baseline rather than forcing a symbolic branch to win.
        raw_oof = self._oof_predictions(base_context, X, y, splits, max(cfg.base.ridge_alpha, 1e-4))
        ch_oof = self._oof_predictions(challenger_exprs, X, y, splits, challenger_alpha)
        valid = np.isfinite(raw_oof) & np.isfinite(ch_oof)
        diff = ch_oof[valid] - raw_oof[valid]
        denom = float(np.dot(diff, diff))
        if denom > 1e-12:
            blend_weight = float(np.clip(np.dot(diff, y[valid] - raw_oof[valid]) / denom, 0.0, 1.0))
        else:
            blend_weight = 0.0
        blend_oof = raw_oof.copy()
        blend_oof[valid] = raw_oof[valid] + blend_weight * diff
        blend_cv = safe_r2(y[valid], blend_oof[valid]) if np.any(valid) else -1e9
        raw_oof_r2 = safe_r2(y[valid], raw_oof[valid]) if np.any(valid) else -1e9
        if blend_weight > 1e-6 and blend_cv > raw_oof_r2 + 1e-8:
            recommended = f"blend_{challenger}"
        else:
            recommended = "raw_ridge"
            blend_weight = 0.0

        self._raw = self._fit_expr_model(base_context, X, y, max(cfg.base.ridge_alpha, 1e-4))
        self._baseline = self._fit_expr_model(baseline_exprs, X, y, cfg.base.ridge_alpha)
        self._refined = self._fit_expr_model(refined_exprs, X, y, max(cfg.base.ridge_alpha, rcfg.ridge_alpha))
        self._recommended = recommended
        self._blend_weight = blend_weight
        self._blend_challenger = challenger
        y_med = float(np.median(y))
        y_mad = float(np.median(np.abs(y - y_med)))
        robust_scale = max(1.4826 * y_mad, float(np.std(y)) * 0.25, 1e-3)
        self._prediction_bound = float(abs(y_med) + 25.0 * robust_scale)
        self.record = {
            "raw_ridge_inner_cv_r2": raw_cv,
            "baseline_inner_cv_r2": baseline_cv,
            "refined_inner_cv_r2": refined_cv,
            "recommended_branch": recommended,
            "blend_challenger": challenger,
            "blend_weight": blend_weight,
            "blend_inner_oof_r2": blend_cv,
            "raw_inner_oof_r2": raw_oof_r2,
            "prediction_bound": self._prediction_bound,
            "nondegrade_inner_pass": bool(blend_cv >= raw_oof_r2 - self.cfg.nondegrade_tolerance),
            "baseline_selected": [str(s.expr) for s in selected],
            "refined_selected": [str(s.expr) for s in refined_sel],
            "refinement_records": refine_records,
            "template_promotion": template_record,
            "discovery_trace": discovery.get("trace", []),
            "base_config": cfg.base.__dict__,
            "refine_config": asdict(rcfg),
        }
        return self

    def predict_raw(self, X: np.ndarray) -> np.ndarray:
        return self._predict(self._raw, X)

    def predict_baseline(self, X: np.ndarray) -> np.ndarray:
        return self._predict(self._baseline, X)

    def predict_refined(self, X: np.ndarray) -> np.ndarray:
        return self._predict(self._refined, X)

    def predict(self, X: np.ndarray) -> np.ndarray:
        raw = self.predict_raw(X)
        if self._recommended.startswith("blend_"):
            challenger = self.predict_refined(X) if self._blend_challenger == "refined" else self.predict_baseline(X)
            unstable_challenger = (not np.all(np.isfinite(challenger))) or (len(challenger) and float(np.max(np.abs(challenger))) > self._prediction_bound)
            if unstable_challenger:
                self._fallback_count += 1
                return raw
            pred = raw + self._blend_weight * (challenger - raw)
        else:
            pred = raw
        unstable = (not np.all(np.isfinite(pred))) or (len(pred) and float(np.max(np.abs(pred))) > self._prediction_bound)
        if unstable:
            self._fallback_count += 1
            return raw
        return pred


# ======================================================================================
# Optional PySR fold baseline
# ======================================================================================

def _construct_pysr(kwargs: Dict[str, Any]):
    from pysr import PySRRegressor
    kw = dict(kwargs)
    dropped: Dict[str, Any] = {}
    for _ in range(len(kwargs) + 1):
        try:
            return PySRRegressor(**kw), dropped
        except TypeError as exc:
            m = re.search(r"unexpected keyword argument '([^']+)'", str(exc))
            if m is None or m.group(1) not in kw:
                raise
            name = m.group(1)
            dropped[name] = kw.pop(name)
    raise RuntimeError("Could not construct PySRRegressor")


def fit_pysr_fold(Xtr: np.ndarray, ytr: np.ndarray, Xte: np.ndarray, seed: int,
                   niterations: int, maxsize: int, timeout: int) -> Dict[str, Any]:
    t0 = time.time()
    try:
        import pysr
    except Exception as exc:
        return {"status": "unavailable", "error": f"{type(exc).__name__}: {exc}"}
    kwargs = {
        "niterations": int(niterations),
        "binary_operators": ["+", "-", "*", "/"],
        "unary_operators": ["sin", "cos", "exp", "log", "sqrt", "abs"],
        "maxsize": int(maxsize),
        "model_selection": "best",
        "random_state": int(seed),
        "deterministic": True,
        "parallelism": "serial",
        "procs": 0,
        "batching": False,
        "timeout_in_seconds": int(timeout),
        "verbosity": 0,
        "progress": False,
    }
    try:
        model, dropped = _construct_pysr(kwargs)
        model.fit(np.asarray(Xtr, float), np.asarray(ytr, float))
        pred = np.asarray(model.predict(np.asarray(Xte, float)), float).reshape(-1)
        expression = None
        try:
            expression = str(model.sympy())
        except Exception:
            pass
        return {
            "status": "ok", "pred": pred, "expression": expression,
            "pysr_version": getattr(pysr, "__version__", "unknown"),
            "kwargs_used": kwargs, "kwargs_dropped_by_version": dropped,
            "elapsed_sec": float(time.time() - t0),
        }
    except Exception as exc:
        return {"status": "failed", "error": f"{type(exc).__name__}: {exc}", "elapsed_sec": float(time.time() - t0)}


# ======================================================================================
# Empirical perturbation graph
# ======================================================================================

@dataclass
class EdgeGraph:
    src: np.ndarray
    dst: np.ndarray
    delta: np.ndarray
    norm: np.ndarray
    dominant_feature: np.ndarray
    sign: np.ndarray
    dominance_ratio: np.ndarray


def build_edge_graph(X: np.ndarray, k: int, dominance_min: float) -> EdgeGraph:
    X = np.asarray(X, float)
    k = max(1, min(int(k), len(X) - 1))
    nn = NearestNeighbors(n_neighbors=k + 1, algorithm="kd_tree").fit(X)
    _, idx = nn.kneighbors(X)
    src = np.repeat(np.arange(len(X)), k)
    dst = idx[:, 1:].reshape(-1)
    delta = X[dst] - X[src]
    norm = np.linalg.norm(delta, axis=1)
    dominant = np.argmax(np.abs(delta), axis=1)
    max_abs = np.abs(delta[np.arange(len(delta)), dominant])
    ratio = max_abs / (norm + EPS)
    sign = np.where(delta[np.arange(len(delta)), dominant] >= 0.0, 1, -1)
    positive = norm > 1e-10
    if np.any(positive):
        lo, hi = np.quantile(norm[positive], [0.02, 0.98])
    else:
        lo, hi = 0.0, float("inf")
    keep = positive & (norm >= lo) & (norm <= hi) & (ratio >= dominance_min)
    return EdgeGraph(src[keep], dst[keep], delta[keep], norm[keep], dominant[keep], sign[keep], ratio[keep])


def select_mode_edges(graph: EdgeGraph, feature: int, sign: int,
                      reference_step: Optional[float], step_tolerance: float) -> Tuple[np.ndarray, float]:
    idx = np.where((graph.dominant_feature == feature) & (graph.sign == sign))[0]
    if len(idx) == 0:
        return np.empty(0, dtype=int), float("nan")
    step = float(np.median(graph.delta[idx, feature])) if reference_step is None else float(reference_step)
    tol = max(abs(step) * float(step_tolerance), 0.03)
    idx = idx[np.abs(graph.delta[idx, feature] - step) <= tol]

    # One representative edge per source, closest to the constant step. This prevents
    # high-degree observations from dominating the operator fit.
    best: Dict[int, Tuple[float, int]] = {}
    for z in idx:
        source = int(graph.src[z])
        score = abs(float(graph.delta[z, feature]) - step)
        if source not in best or score < best[source][0]:
            best[source] = (score, int(z))
    out = np.asarray([v[1] for v in best.values()], dtype=int)
    return out, step


# ======================================================================================
# Numerical typed rewrite schema
# ======================================================================================

OPERATOR_FAMILIES = ("identity", "shift", "affine_y", "context_shift", "context_affine")
FAMILY_COMPLEXITY = {
    "identity": 0.5,
    "shift": 1.5,
    "affine_y": 2.5,
    "context_shift": 2.5,
    "context_affine": 3.5,
}


@dataclass
class NumericRewrite:
    family: str
    coef: np.ndarray
    delta: np.ndarray
    feature: int
    sign: int
    feature_names: List[str]

    def predict_y(self, y_source: np.ndarray, x_source: np.ndarray) -> np.ndarray:
        y_source = np.asarray(y_source, float).reshape(-1)
        x_source = np.asarray(x_source, float)
        c = np.asarray(self.coef, float)
        if self.family == "identity":
            return y_source.copy()
        if self.family == "shift":
            return y_source + c[0]
        if self.family == "affine_y":
            return c[0] * y_source + c[1]
        if self.family == "context_shift":
            return y_source + c[0] + x_source @ c[1:]
        if self.family == "context_affine":
            return c[0] * y_source + c[1] + x_source @ c[2:]
        raise ValueError(self.family)

    def n_parameters(self) -> int:
        return int(len(self.coef) + len(self.delta))

    def schema(self) -> str:
        fname = self.feature_names[self.feature] if self.feature < len(self.feature_names) else f"x{self.feature}"
        step = float(self.delta[self.feature])
        if self.family == "identity":
            rhs = "y"
        elif self.family == "shift":
            rhs = f"y + ({self.coef[0]:.8g})"
        elif self.family == "affine_y":
            rhs = f"({self.coef[0]:.8g})*y + ({self.coef[1]:.8g})"
        elif self.family == "context_shift":
            terms = [f"({self.coef[0]:.8g})"] + [f"({b:.8g})*x{i}" for i, b in enumerate(self.coef[1:])]
            rhs = "y + " + " + ".join(terms)
        else:
            terms = [f"({self.coef[1]:.8g})"] + [f"({b:.8g})*x{i}" for i, b in enumerate(self.coef[2:])]
            rhs = f"({self.coef[0]:.8g})*y + " + " + ".join(terms)
        return f"omega_{fname}_{'plus' if self.sign > 0 else 'minus'}: State(x,y) -> State(x + {step:.8g}*e_{self.feature}, {rhs})"


def _fit_family(family: str, ys: np.ndarray, x: np.ndarray, yd: np.ndarray, alpha: float) -> NumericRewrite:
    ys = np.asarray(ys, float).reshape(-1)
    yd = np.asarray(yd, float).reshape(-1)
    x = np.asarray(x, float)
    if family == "identity":
        coef = np.empty(0, float)
    elif family == "shift":
        coef = np.asarray([float(np.mean(yd - ys))])
    elif family == "affine_y":
        D = np.column_stack([ys, np.ones(len(ys))])
        coef = Ridge(alpha=alpha, fit_intercept=False).fit(D, yd).coef_
    elif family == "context_shift":
        D = np.column_stack([np.ones(len(ys)), x])
        coef = Ridge(alpha=alpha, fit_intercept=False).fit(D, yd - ys).coef_
    elif family == "context_affine":
        D = np.column_stack([ys, np.ones(len(ys)), x])
        coef = Ridge(alpha=alpha, fit_intercept=False).fit(D, yd).coef_
    else:
        raise ValueError(family)
    return NumericRewrite(family, np.asarray(coef, float), np.empty(x.shape[1]), -1, 0, [])


def _family_predict(family: str, coef: np.ndarray, ys: np.ndarray, x: np.ndarray) -> np.ndarray:
    tmp = NumericRewrite(family, np.asarray(coef, float), np.empty(x.shape[1]), -1, 0, [])
    return tmp.predict_y(ys, x)


def fit_operator_candidates(ys: np.ndarray, x: np.ndarray, yd: np.ndarray, groups: np.ndarray,
                            alpha: float, complexity_weight: float,
                            active_payload: Optional[Dict[str, np.ndarray]], active_weight: float,
                            seed: int) -> Tuple[NumericRewrite, List[Dict[str, Any]]]:
    ys = np.asarray(ys, float); x = np.asarray(x, float); yd = np.asarray(yd, float)
    groups = np.asarray(groups, int)
    unique_groups = np.unique(groups)
    k = min(3, len(unique_groups))
    rows: List[Dict[str, Any]] = []

    for family in OPERATOR_FAMILIES:
        fold_scores: List[float] = []
        if k >= 2:
            splitter = GroupKFold(n_splits=k)
            for itr, iva in splitter.split(x, yd, groups):
                model = _fit_family(family, ys[itr], x[itr], yd[itr], alpha)
                pred = model.predict_y(ys[iva], x[iva])
                fold_scores.append(nrmse(yd[iva], pred))
        else:
            model = _fit_family(family, ys, x, yd, alpha)
            fold_scores.append(nrmse(yd, model.predict_y(ys, x)))

        fitted = _fit_family(family, ys, x, yd, alpha)
        active_nrmse = None
        if active_payload is not None and len(active_payload.get("y_source", [])):
            active_pred = fitted.predict_y(active_payload["y_source"], active_payload["x_source"])
            active_nrmse = nrmse(active_payload["y_dest"], active_pred)
        empirical = float(np.mean(fold_scores))
        score = empirical + complexity_weight * FAMILY_COMPLEXITY[family]
        if active_nrmse is not None:
            score += active_weight * float(active_nrmse)
        rows.append({
            "family": family, "coef": fitted.coef.tolist(),
            "cv_nrmse": empirical, "active_model_nrmse": active_nrmse,
            "complexity": FAMILY_COMPLEXITY[family], "selection_score": float(score),
        })

    rows.sort(key=lambda r: (r["selection_score"], r["complexity"]))
    best = rows[0]
    model = _fit_family(best["family"], ys, x, yd, alpha)
    return model, rows


def bootstrap_family_stability(ys: np.ndarray, x: np.ndarray, yd: np.ndarray, groups: np.ndarray,
                               alpha: float, complexity_weight: float, reps: int, seed: int) -> Dict[str, Any]:
    rng = np.random.default_rng(seed)
    unique = np.unique(groups)
    chosen: List[str] = []
    if len(unique) < 3 or reps <= 0:
        return {"reps": 0, "modal_family": None, "modal_fraction": 0.0, "counts": {}}
    for _ in range(reps):
        sampled_groups = rng.choice(unique, size=len(unique), replace=True)
        idxs: List[int] = []
        for g in sampled_groups:
            candidates = np.where(groups == g)[0]
            if len(candidates):
                idxs.extend(candidates.tolist())
        idx = np.asarray(idxs, int)
        if len(idx) < 10:
            continue
        model, _ = fit_operator_candidates(
            ys[idx], x[idx], yd[idx], groups[idx], alpha, complexity_weight,
            active_payload=None, active_weight=0.0, seed=int(rng.integers(0, 2**31 - 1))
        )
        chosen.append(model.family)
    counts = Counter(chosen)
    if not counts:
        return {"reps": 0, "modal_family": None, "modal_fraction": 0.0, "counts": {}}
    modal, count = counts.most_common(1)[0]
    return {"reps": len(chosen), "modal_family": modal, "modal_fraction": float(count / len(chosen)), "counts": dict(counts)}


# ======================================================================================
# Bridge fold
# ======================================================================================

@dataclass
class BridgeConfig:
    knn_k: int = 8
    dominance_min: float = 0.60
    step_tolerance_train: float = 0.50
    step_tolerance_test: float = 0.75
    min_train_edges: int = 80
    min_test_edges: int = 20
    ridge_alpha: float = 1e-3
    complexity_weight: float = 0.005
    active_queries: int = 16
    active_weight: float = 0.25
    support_quantile: float = 0.95
    support_multiplier: float = 1.50
    bootstrap_reps: int = 20
    heldout_r2_min: float = 0.25
    identity_improvement_min: float = 0.10
    pooled_improvement_min: float = 0.02
    model_r2_min: float = 0.80
    model_nrmse_max: float = 0.35
    bootstrap_stability_min: float = 0.60
    compression_min: float = 1.20
    support_fraction_min: float = 0.50
    geometry_dominance_min: float = 0.70
    step_transfer_relative_max: float = 0.50


def _mode_name(feature_names: Sequence[str], feature: int, sign: int) -> str:
    return f"{feature_names[feature]}_{'plus' if sign > 0 else 'minus'}"


def _prepare_mode_arrays(graph: EdgeGraph, idx: np.ndarray, X: np.ndarray, y: np.ndarray,
                         pair_shuffle: bool, rng: np.random.Generator):
    src = graph.src[idx]
    dst = graph.dst[idx]
    ys = y[src]
    yd = y[dst]
    if pair_shuffle and len(yd):
        yd = yd[rng.permutation(len(yd))]
    return src, dst, ys, X[src], yd


def build_active_payload(model: Any, candidate_rows: List[Dict[str, Any]],
                         X_train: np.ndarray, delta: np.ndarray,
                         support_nn: NearestNeighbors, support_threshold: float,
                         q: int, seed: int) -> Optional[Dict[str, np.ndarray]]:
    if q <= 0 or len(X_train) == 0 or len(candidate_rows) < 2:
        return None
    cf = X_train + delta.reshape(1, -1)
    dist, _ = support_nn.kneighbors(cf)
    ok = dist[:, 0] <= support_threshold
    idx = np.where(ok)[0]
    if len(idx) == 0:
        return None

    # Initial model predictions establish the frozen inference environment.
    y0 = model.predict(X_train[idx])
    ycf = model.predict(cf[idx])
    preds = []
    for row in candidate_rows:
        preds.append(_family_predict(row["family"], np.asarray(row["coef"], float), y0, X_train[idx]))
    disagreement = np.var(np.vstack(preds), axis=0)
    order = np.argsort(disagreement)[::-1][: min(q, len(idx))]
    sel = idx[order]
    return {
        "x_source": X_train[sel],
        "y_source": model.predict(X_train[sel]),
        "y_dest": model.predict(X_train[sel] + delta.reshape(1, -1)),
        "disagreement": disagreement[order],
    }


def evaluate_bridge_fold(Xtr: np.ndarray, ytr: np.ndarray, Xte: np.ndarray, yte: np.ndarray,
                         model: Any, feature_names: List[str], cfg: BridgeConfig,
                         seed: int, pair_shuffle: bool = False) -> Dict[str, Any]:
    rng = np.random.default_rng(seed)
    graph_tr = build_edge_graph(Xtr, cfg.knn_k, cfg.dominance_min)
    graph_te = build_edge_graph(Xte, cfg.knn_k, cfg.dominance_min)

    support_nn = NearestNeighbors(n_neighbors=2, algorithm="kd_tree").fit(Xtr)
    dtrain, _ = support_nn.kneighbors(Xtr)
    support_threshold = float(np.quantile(dtrain[:, 1], cfg.support_quantile) * cfg.support_multiplier)
    support_query_nn = NearestNeighbors(n_neighbors=1, algorithm="kd_tree").fit(Xtr)

    # Construct mode-specific edge sets first; the pooled operator is a required
    # destroyer for the claim that a feature perturbation deserves its own schema.
    prepared: Dict[str, Dict[str, Any]] = {}
    pooled_train_parts = []
    for feature in range(Xtr.shape[1]):
        for sign in (-1, 1):
            name = _mode_name(feature_names, feature, sign)
            itr, step = select_mode_edges(graph_tr, feature, sign, None, cfg.step_tolerance_train)
            ite, _ = select_mode_edges(graph_te, feature, sign, step, cfg.step_tolerance_test)
            prepared[name] = {"feature": feature, "sign": sign, "step": step, "itr": itr, "ite": ite}
            if len(itr):
                pooled_train_parts.append((name, itr))

    if pooled_train_parts:
        p_ys, p_x, p_yd, p_groups = [], [], [], []
        for name, idx in pooled_train_parts:
            src, _, ys, xs, yd = _prepare_mode_arrays(graph_tr, idx, Xtr, ytr, pair_shuffle, rng)
            p_ys.append(ys); p_x.append(xs); p_yd.append(yd)
            # Mode-qualified groups avoid accidental integer collision across modes.
            p_groups.append(np.asarray([stable_hash(name, int(s)) for s in src], dtype=np.int64))
        p_ys = np.concatenate(p_ys); p_x = np.vstack(p_x); p_yd = np.concatenate(p_yd); p_groups = np.concatenate(p_groups)
        pooled_operator, pooled_candidates = fit_operator_candidates(
            p_ys, p_x, p_yd, p_groups, cfg.ridge_alpha, cfg.complexity_weight,
            active_payload=None, active_weight=0.0, seed=seed + 17
        )
    else:
        pooled_operator = None
        pooled_candidates = []

    mode_rows: List[Dict[str, Any]] = []
    for name, info in prepared.items():
        feature, sign, step = info["feature"], info["sign"], info["step"]
        itr, ite = info["itr"], info["ite"]
        base_row: Dict[str, Any] = {
            "mode": name, "feature": feature, "feature_name": feature_names[feature], "sign": sign,
            "step_standardized": step, "n_train_edges": int(len(itr)), "n_test_edges": int(len(ite)),
            "status": "insufficient_support", "promoted": False,
        }
        if len(itr) < cfg.min_train_edges or len(ite) < cfg.min_test_edges or not math.isfinite(step):
            mode_rows.append(base_row)
            continue

        src_tr, _, ys_tr, xs_tr, yd_tr = _prepare_mode_arrays(graph_tr, itr, Xtr, ytr, pair_shuffle, rng)
        src_te, _, ys_te, xs_te, yd_te = _prepare_mode_arrays(graph_te, ite, Xte, yte, pair_shuffle, rng)

        # First pass without active counterfactual evidence.
        first, first_candidates = fit_operator_candidates(
            ys_tr, xs_tr, yd_tr, src_tr, cfg.ridge_alpha, cfg.complexity_weight,
            active_payload=None, active_weight=0.0, seed=seed + stable_hash(name, "first")
        )
        delta = np.zeros(Xtr.shape[1], float)
        delta[feature] = step
        active_payload = build_active_payload(
            model, first_candidates, Xtr, delta, support_query_nn, support_threshold,
            cfg.active_queries, seed + stable_hash(name, "active")
        )
        selected, candidate_rows = fit_operator_candidates(
            ys_tr, xs_tr, yd_tr, src_tr, cfg.ridge_alpha, cfg.complexity_weight,
            active_payload=active_payload, active_weight=cfg.active_weight,
            seed=seed + stable_hash(name, "selected")
        )
        selected.delta = delta
        selected.feature = feature
        selected.sign = sign
        selected.feature_names = feature_names

        pred_test = selected.predict_y(ys_te, xs_te)
        empirical_r2 = safe_r2(yd_te, pred_test)
        empirical_nrmse = nrmse(yd_te, pred_test)
        identity_pred = ys_te
        identity_nrmse = nrmse(yd_te, identity_pred)
        identity_improvement = float((identity_nrmse - empirical_nrmse) / max(identity_nrmse, EPS))

        pooled_nrmse = None
        pooled_r2 = None
        pooled_improvement = None
        if pooled_operator is not None:
            pp = pooled_operator.predict_y(ys_te, xs_te)
            pooled_nrmse = nrmse(yd_te, pp)
            pooled_r2 = safe_r2(yd_te, pp)
            pooled_improvement = float((pooled_nrmse - empirical_nrmse) / max(pooled_nrmse, EPS))

        # Counterfactual commutative-square audit on untouched test anchors:
        #   f_hat(g x) ~= omega_g(f_hat(x), x)
        cf = Xte + delta.reshape(1, -1)
        dist, _ = support_query_nn.kneighbors(cf)
        supported = dist[:, 0] <= support_threshold
        support_fraction = float(np.mean(supported))
        if np.any(supported):
            f0 = model.predict(Xte[supported])
            fg = model.predict(cf[supported])
            fomega = selected.predict_y(f0, Xte[supported])
            model_r2 = safe_r2(fg, fomega)
            model_nrmse = nrmse(fg, fomega)
        else:
            model_r2, model_nrmse = -1e9, 1e9

        stability = bootstrap_family_stability(
            ys_tr, xs_tr, yd_tr, src_tr, cfg.ridge_alpha, cfg.complexity_weight,
            cfg.bootstrap_reps, seed + stable_hash(name, "bootstrap")
        )
        train_dominance_median = float(np.median(graph_tr.dominance_ratio[itr]))
        test_dominance_median = float(np.median(graph_te.dominance_ratio[ite]))
        train_geometry_residual = float(np.median(
            np.linalg.norm(graph_tr.delta[itr] - delta.reshape(1, -1), axis=1) / (graph_tr.norm[itr] + EPS)
        ))
        test_geometry_residual = float(np.median(
            np.linalg.norm(graph_te.delta[ite] - delta.reshape(1, -1), axis=1) / (graph_te.norm[ite] + EPS)
        ))
        test_step_median = float(np.median(graph_te.delta[ite, feature]))
        step_transfer_relative_error = float(abs(test_step_median - step) / max(abs(step), 0.03))
        concrete_cost = float(len(itr) * (Xtr.shape[1] + 2))
        schema_cost = float(selected.n_parameters() + FAMILY_COMPLEXITY[selected.family] + 2)
        compression = concrete_cost / max(schema_cost, EPS)

        reasons = {
            "train_support": len(itr) >= cfg.min_train_edges,
            "test_support": len(ite) >= cfg.min_test_edges,
            "heldout_r2": empirical_r2 >= cfg.heldout_r2_min,
            "identity_destroyer": identity_improvement >= cfg.identity_improvement_min,
            "pooled_destroyer": pooled_improvement is not None and pooled_improvement >= cfg.pooled_improvement_min,
            "model_consistency_r2": model_r2 >= cfg.model_r2_min,
            "model_consistency_nrmse": model_nrmse <= cfg.model_nrmse_max,
            "bootstrap_stability": stability["modal_fraction"] >= cfg.bootstrap_stability_min,
            "compression": compression >= cfg.compression_min,
            "support_safe": support_fraction >= cfg.support_fraction_min,
            "geometry_dominance": min(train_dominance_median, test_dominance_median) >= cfg.geometry_dominance_min,
            "step_transfer": step_transfer_relative_error <= cfg.step_transfer_relative_max,
        }
        promoted = bool(all(reasons.values()))
        candidate = bool(
            empirical_r2 >= 0.10 and identity_improvement > 0.0 and
            model_r2 >= 0.50 and support_fraction >= 0.25
        )

        row = {
            **base_row,
            "status": "promoted" if promoted else ("candidate" if candidate else "rejected"),
            "promoted": promoted,
            "schema": selected.schema(),
            "operator_family": selected.family,
            "operator_coef": selected.coef.tolist(),
            "delta_standardized": delta.tolist(),
            "empirical_test_R2": empirical_r2,
            "empirical_test_nrmse": empirical_nrmse,
            "identity_nrmse": identity_nrmse,
            "identity_relative_improvement": identity_improvement,
            "pooled_test_R2": pooled_r2,
            "pooled_test_nrmse": pooled_nrmse,
            "pooled_relative_improvement": pooled_improvement,
            "model_counterfactual_R2": model_r2,
            "model_counterfactual_nrmse": model_nrmse,
            "support_fraction": support_fraction,
            "support_threshold": support_threshold,
            "train_dominance_median": train_dominance_median,
            "test_dominance_median": test_dominance_median,
            "train_geometry_residual": train_geometry_residual,
            "test_geometry_residual": test_geometry_residual,
            "test_step_median": test_step_median,
            "step_transfer_relative_error": step_transfer_relative_error,
            "bootstrap": stability,
            "compression_ratio": compression,
            "promotion_checks": reasons,
            "candidate_families": candidate_rows,
            "active_queries_used": 0 if active_payload is None else int(len(active_payload["y_source"])),
            "active_disagreement_median": None if active_payload is None else float(np.median(active_payload["disagreement"])),
            "active_family_before": first.family,
            "active_family_after": selected.family,
        }
        mode_rows.append(row)

    promoted = [r for r in mode_rows if r.get("promoted")]
    candidates = [r for r in mode_rows if r.get("status") == "candidate"]
    covered_sources = set()
    for row in mode_rows:
        if row.get("status") in ("promoted", "candidate"):
            info = prepared[row["mode"]]
            for z in info["ite"]:
                covered_sources.add(int(graph_te.src[z]))

    return {
        "pair_shuffle": pair_shuffle,
        "n_train_graph_edges": int(len(graph_tr.src)),
        "n_test_graph_edges": int(len(graph_te.src)),
        "support_threshold": support_threshold,
        "pooled_operator": None if pooled_operator is None else {
            "family": pooled_operator.family, "coef": pooled_operator.coef.tolist(),
            "candidates": pooled_candidates,
        },
        "modes": mode_rows,
        "n_promoted": len(promoted),
        "n_candidates": len(candidates),
        "transformation_coverage": float(len(covered_sources) / max(1, len(Xte))),
    }


# ======================================================================================
# One fold / one seed
# ======================================================================================

@dataclass
class MatrixConfig:
    name: str
    seeds: List[int]
    k_folds: int
    subsample: Optional[int]
    v23_quick: bool
    v23_candidates: int
    v23_beam: int
    v23_layers: int
    v23_accept: int
    bridge: BridgeConfig


def matrix_config(name: str) -> MatrixConfig:
    if name == "smoke":
        return MatrixConfig(
            name, [42], 2, 300, True, 80, 20, 1, 2,
            BridgeConfig(knn_k=6, min_train_edges=20, min_test_edges=8,
                         active_queries=4, bootstrap_reps=6,
                         heldout_r2_min=0.05, identity_improvement_min=0.02,
                         pooled_improvement_min=-0.05, model_r2_min=0.50,
                         bootstrap_stability_min=0.40, support_fraction_min=0.25),
        )
    if name == "quick":
        return MatrixConfig(
            name, [42, 1337], 3, 1200, True, 260, 40, 2, 2,
            BridgeConfig(knn_k=8, min_train_edges=50, min_test_edges=15,
                         active_queries=8, bootstrap_reps=12),
        )
    if name == "full":
        return MatrixConfig(
            name, [42, 1337, 2025, 9001], 5, None, False, 1800, 100, 3, 2,
            BridgeConfig(knn_k=8, min_train_edges=80, min_test_edges=20,
                         active_queries=16, bootstrap_reps=20),
        )
    raise ValueError(name)


def run_fold(X: np.ndarray, y: np.ndarray, feature_names: List[str], train_idx: np.ndarray, test_idx: np.ndarray,
             seed: int, fold_id: int, mcfg: MatrixConfig, run_pysr: bool,
             pysr_iterations: int, pysr_maxsize: int, pysr_timeout: int) -> Dict[str, Any]:
    t0 = time.time()
    Xtr_raw, Xte_raw = X[train_idx], X[test_idx]
    ytr, yte = y[train_idx], y[test_idx]
    scaler = StandardScaler().fit(Xtr_raw)
    Xtr, Xte = scaler.transform(Xtr_raw), scaler.transform(Xte_raw)

    # Simple prediction baseline.
    ridge = Ridge(alpha=1e-4).fit(Xtr, ytr)
    pred_ridge = ridge.predict(Xte)

    vcfg = V231FoldConfig(
        quick=mcfg.v23_quick,
        max_candidates_scored=mcfg.v23_candidates,
        depth1_beam=mcfg.v23_beam,
        max_layers=mcfg.v23_layers,
        accept_per_layer=mcfg.v23_accept,
        nondegrade_tolerance=1e-3,
    )
    vmodel = V231FoldModel(seed + 1000 * fold_id, vcfg).fit(Xtr, ytr)
    pred_v23 = vmodel.predict_baseline(Xte)
    pred_v231 = vmodel.predict_refined(Xte)
    pred_rec = vmodel.predict(Xte)

    # Main bridge on true labels and the frozen recommended v23 branch.
    bridge = evaluate_bridge_fold(
        Xtr, ytr, Xte, yte, vmodel, feature_names, copy.deepcopy(mcfg.bridge),
        seed + stable_hash(fold_id, "bridge"), pair_shuffle=False,
    )

    # Pair-shuffle control preserves X geometry and target marginal values but
    # destroys source->destination association.
    pair_control = evaluate_bridge_fold(
        Xtr, ytr, Xte, yte, vmodel, feature_names, copy.deepcopy(mcfg.bridge),
        seed + stable_hash(fold_id, "pair_shuffle"), pair_shuffle=True,
    )

    # Target-shuffle control: labels are globally permuted within each split and a
    # fresh predictor is fitted. This prevents the v23 model from carrying the true
    # target relation into the null audit.
    rng = np.random.default_rng(seed + stable_hash(fold_id, "target_shuffle"))
    ytr_s = ytr[rng.permutation(len(ytr))]
    yte_s = yte[rng.permutation(len(yte))]
    # The destroyer is aimed at the bridge promotion mechanism, not at re-running
    # an expensive FASE search on random labels. A fresh Ridge predictor supplies
    # the null inference environment while the empirical target relation is destroyed.
    null_model = Ridge(alpha=1e-4).fit(Xtr, ytr_s)
    target_control_cfg = copy.deepcopy(mcfg.bridge)
    target_control_cfg.bootstrap_reps = min(target_control_cfg.bootstrap_reps, 8)
    target_control = evaluate_bridge_fold(
        Xtr, ytr_s, Xte, yte_s, null_model, feature_names, target_control_cfg,
        seed + stable_hash(fold_id, "target_control"), pair_shuffle=False,
    )

    pysr_row = None
    if run_pysr:
        pysr_row = fit_pysr_fold(
            Xtr, ytr, Xte, seed + 90000 + fold_id,
            niterations=pysr_iterations, maxsize=pysr_maxsize, timeout=pysr_timeout,
        )
        if pysr_row.get("status") == "ok":
            pred = np.asarray(pysr_row.pop("pred"), float)
            bad = ~np.isfinite(pred)
            if np.any(bad):
                pred[bad] = float(np.mean(ytr))
            pysr_row.update({"R2": safe_r2(yte, pred), "MSE": float(mean_squared_error(yte, pred))})

    return {
        "seed": int(seed), "fold": int(fold_id),
        "n_train": int(len(train_idx)), "n_test": int(len(test_idx)),
        "train_indices_hash": hashlib.sha256(np.asarray(train_idx, np.int64).tobytes()).hexdigest()[:16],
        "test_indices_hash": hashlib.sha256(np.asarray(test_idx, np.int64).tobytes()).hexdigest()[:16],
        "scaler_mean": scaler.mean_.tolist(), "scaler_scale": scaler.scale_.tolist(),
        "predictive": {
            "ridge": {"R2": safe_r2(yte, pred_ridge), "MSE": float(mean_squared_error(yte, pred_ridge))},
            "v23_baseline": {"R2": safe_r2(yte, pred_v23), "MSE": float(mean_squared_error(yte, pred_v23))},
            "v23_1_refined": {"R2": safe_r2(yte, pred_v231), "MSE": float(mean_squared_error(yte, pred_v231))},
            "v23_1_recommended": {"R2": safe_r2(yte, pred_rec), "MSE": float(mean_squared_error(yte, pred_rec)), "branch": vmodel.record["recommended_branch"], "stability_fallback_count": vmodel._fallback_count},
            "pysr": pysr_row,
        },
        "v23_1_model": vmodel.record,
        "bridge": bridge,
        "controls": {"pair_shuffle": pair_control, "target_shuffle": target_control},
        "elapsed_sec": float(time.time() - t0),
    }


def _subsample(X: np.ndarray, y: np.ndarray, n: Optional[int], seed: int):
    if n is None or n >= len(y):
        return X, y, np.arange(len(y))
    rng = np.random.default_rng(seed)
    idx = np.sort(rng.choice(len(y), size=int(n), replace=False))
    return X[idx], y[idx], idx


# ======================================================================================
# Aggregation / horizon classification
# ======================================================================================

def aggregate_report(protocol: Dict[str, Any], fold_rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    protocol = copy.deepcopy(protocol)
    protocol["observed_seeds"] = sorted({int(f["seed"]) for f in fold_rows})
    protocol["observed_folds"] = len(fold_rows)
    predictive: Dict[str, List[float]] = defaultdict(list)
    all_real_modes: List[Dict[str, Any]] = []
    all_pair_modes: List[Dict[str, Any]] = []
    all_target_modes: List[Dict[str, Any]] = []
    v23_nondegrade = []

    for fold in fold_rows:
        for name, row in fold["predictive"].items():
            if isinstance(row, dict) and row.get("R2") is not None:
                predictive[name].append(float(row["R2"]))
        v23_nondegrade.append(bool(fold["v23_1_model"].get("nondegrade_inner_pass")))
        all_real_modes.extend([{**m, "seed": fold["seed"], "fold": fold["fold"]} for m in fold["bridge"]["modes"]])
        all_pair_modes.extend([{**m, "seed": fold["seed"], "fold": fold["fold"]} for m in fold["controls"]["pair_shuffle"]["modes"]])
        all_target_modes.extend([{**m, "seed": fold["seed"], "fold": fold["fold"]} for m in fold["controls"]["target_shuffle"]["modes"]])

    predictive_summary = {}
    for name, vals in predictive.items():
        predictive_summary[name] = {
            "folds": len(vals), "R2_mean": float(np.mean(vals)), "R2_median": float(np.median(vals)),
            "R2_min": float(np.min(vals)), "R2_max": float(np.max(vals)),
        }

    mode_groups: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for row in all_real_modes:
        mode_groups[row["mode"]].append(row)
    mode_summary = []
    for mode, rows in sorted(mode_groups.items()):
        eligible = [r for r in rows if r.get("status") != "insufficient_support"]
        promoted = [r for r in eligible if r.get("promoted")]
        mode_summary.append({
            "mode": mode,
            "folds": len(rows), "eligible_folds": len(eligible), "promoted_folds": len(promoted),
            "promotion_rate": float(len(promoted) / max(1, len(eligible))),
            "empirical_R2_median": None if not eligible else float(np.median([r.get("empirical_test_R2", -1e9) for r in eligible])),
            "model_counterfactual_R2_median": None if not eligible else float(np.median([r.get("model_counterfactual_R2", -1e9) for r in eligible])),
            "identity_improvement_median": None if not eligible else float(np.median([r.get("identity_relative_improvement", -1e9) for r in eligible])),
            "pooled_improvement_median": None if not eligible else float(np.median([r.get("pooled_relative_improvement", -1e9) for r in eligible])),
            "families": dict(Counter(r.get("operator_family") for r in promoted)),
            "schemas": list(dict.fromkeys(r.get("schema") for r in promoted if r.get("schema")))[:10],
        })

    real_eligible = [r for r in all_real_modes if r.get("status") != "insufficient_support"]
    real_promoted = [r for r in real_eligible if r.get("promoted")]
    pair_eligible = [r for r in all_pair_modes if r.get("status") != "insufficient_support"]
    pair_promoted = [r for r in pair_eligible if r.get("promoted")]
    target_eligible = [r for r in all_target_modes if r.get("status") != "insufficient_support"]
    target_promoted = [r for r in target_eligible if r.get("promoted")]

    stable_modes = [r for r in mode_summary if r["eligible_folds"] >= 2 and r["promotion_rate"] >= 0.60]
    coverages = [float(f["bridge"]["transformation_coverage"]) for f in fold_rows]
    false_promotions = len(pair_promoted) + len(target_promoted)
    null_trials = len(pair_eligible) + len(target_eligible)
    null_rate = float(false_promotions / max(1, null_trials))

    if false_promotions > 0:
        horizon = "INVALIDATED_BY_NULL_PROMOTION"
    elif stable_modes and np.median(coverages) >= 0.20 and all(v23_nondegrade):
        horizon = "SUPPORTED_EXTERNAL_INDICATION"
    elif real_promoted or any(r["promotion_rate"] > 0 for r in mode_summary):
        horizon = "LOCAL_CANDIDATE_ONLY"
    else:
        horizon = "ABSTENTION"

    return {
        "protocol": protocol,
        "predictive_summary": predictive_summary,
        "bridge_summary": {
            "real_eligible_mode_trials": len(real_eligible),
            "real_promotions": len(real_promoted),
            "real_promotion_rate": float(len(real_promoted) / max(1, len(real_eligible))),
            "stable_modes": stable_modes,
            "mode_summary": mode_summary,
            "transformation_coverage_median": float(np.median(coverages)) if coverages else 0.0,
            "transformation_coverage_min": float(np.min(coverages)) if coverages else 0.0,
        },
        "destroyer_controls": {
            "pair_shuffle": {"eligible_trials": len(pair_eligible), "false_promotions": len(pair_promoted)},
            "target_shuffle": {"eligible_trials": len(target_eligible), "false_promotions": len(target_promoted)},
            "combined_false_promotion_rate": null_rate,
            "combined_zero_event_95pct_upper": confidence_upper_zero_events(null_trials) if false_promotions == 0 else None,
        },
        "v23_1_nondegrade_rate": float(np.mean(v23_nondegrade)) if v23_nondegrade else 0.0,
        "horizon_classification": horizon,
        "interpretation": {
            "SUPPORTED_EXTERNAL_INDICATION": "At least one constant perturbation mode survives empirical, inference, compression, stability, and null-control tests across folds. This indicates that v24-style operator promotion survives a non-native iid tabular setting; it does not establish causality or physical recursion.",
            "LOCAL_CANDIDATE_ONLY": "Some fold-local operators are supported, but cross-fold stability or coverage is insufficient for a global external indication.",
            "ABSTENTION": "The predictive model may be useful, but the iid table does not license a stable recursive operator under the tested constant perturbation modes.",
            "INVALIDATED_BY_NULL_PROMOTION": "At least one shuffled destroyer control promoted an operator; the bridge evidence is not trustworthy.",
        }[horizon],
        "folds": fold_rows,
    }


def flatten_rows(report: Dict[str, Any]) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    fold_csv: List[Dict[str, Any]] = []
    mode_csv: List[Dict[str, Any]] = []
    for f in report["folds"]:
        row = {"seed": f["seed"], "fold": f["fold"], "n_train": f["n_train"], "n_test": f["n_test"], "elapsed_sec": f["elapsed_sec"]}
        for model, metrics in f["predictive"].items():
            if isinstance(metrics, dict):
                row[f"{model}_R2"] = metrics.get("R2")
                row[f"{model}_MSE"] = metrics.get("MSE")
                row[f"{model}_status"] = metrics.get("status")
        row["v23_recommended_branch"] = f["v23_1_model"].get("recommended_branch")
        row["real_promotions"] = f["bridge"]["n_promoted"]
        row["pair_shuffle_promotions"] = f["controls"]["pair_shuffle"]["n_promoted"]
        row["target_shuffle_promotions"] = f["controls"]["target_shuffle"]["n_promoted"]
        row["transformation_coverage"] = f["bridge"]["transformation_coverage"]
        fold_csv.append(row)

        for control_name, bridge in [("real", f["bridge"]), ("pair_shuffle", f["controls"]["pair_shuffle"]), ("target_shuffle", f["controls"]["target_shuffle"])]:
            for m in bridge["modes"]:
                mode_csv.append({
                    "seed": f["seed"], "fold": f["fold"], "control": control_name,
                    "mode": m.get("mode"), "status": m.get("status"), "promoted": m.get("promoted"),
                    "operator_family": m.get("operator_family"), "schema": m.get("schema"),
                    "n_train_edges": m.get("n_train_edges"), "n_test_edges": m.get("n_test_edges"),
                    "step_standardized": m.get("step_standardized"),
                    "empirical_test_R2": m.get("empirical_test_R2"),
                    "empirical_test_nrmse": m.get("empirical_test_nrmse"),
                    "identity_relative_improvement": m.get("identity_relative_improvement"),
                    "pooled_relative_improvement": m.get("pooled_relative_improvement"),
                    "model_counterfactual_R2": m.get("model_counterfactual_R2"),
                    "model_counterfactual_nrmse": m.get("model_counterfactual_nrmse"),
                    "support_fraction": m.get("support_fraction"),
                    "train_dominance_median": m.get("train_dominance_median"),
                    "test_dominance_median": m.get("test_dominance_median"),
                    "test_geometry_residual": m.get("test_geometry_residual"),
                    "step_transfer_relative_error": m.get("step_transfer_relative_error"),
                    "bootstrap_modal_fraction": (m.get("bootstrap") or {}).get("modal_fraction"),
                    "compression_ratio": m.get("compression_ratio"),
                    "active_queries_used": m.get("active_queries_used"),
                    "promotion_checks_json": json.dumps(m.get("promotion_checks", {}), sort_keys=True),
                })
    return fold_csv, mode_csv


def write_csv(path: Path, rows: List[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fields: List[str] = []
    seen = set()
    for row in rows:
        for key in row:
            if key not in seen:
                seen.add(key); fields.append(key)
    with tempfile.NamedTemporaryFile("w", encoding="utf-8", newline="", dir=path.parent, delete=False) as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow(json_safe(row))
        tmp = Path(f.name)
    os.replace(tmp, path)


def run_fold_isolated(payload: Dict[str, Any], timeout_sec: int) -> Dict[str, Any]:
    """Run one fold in a completely fresh Python interpreter.

    Repeated in-process symbolic searches can retain mutable global state, and repeated
    multiprocessing-spawn calls can deadlock in some BLAS/scipy combinations. A plain
    subprocess gives each fold a clean interpreter and is the most robust checkpoint unit.
    """
    with tempfile.TemporaryDirectory(prefix="v24_bridge_fold_") as td:
        td_path = Path(td)
        payload_path = td_path / "payload.pkl"
        result_path = td_path / "result.json"
        with payload_path.open("wb") as f:
            pickle.dump(payload, f, protocol=pickle.HIGHEST_PROTOCOL)
        cmd = [sys.executable, str(Path(__file__).resolve()),
               "--fold-payload", str(payload_path), "--fold-result", str(result_path)]
        try:
            completed = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                                       text=True, timeout=timeout_sec, check=False)
        except subprocess.TimeoutExpired as exc:
            raise TimeoutError(f"Fold exceeded timeout of {timeout_sec} seconds") from exc
        if completed.returncode != 0:
            raise RuntimeError(
                f"Fold worker exited {completed.returncode}\nSTDOUT:\n{completed.stdout}\nSTDERR:\n{completed.stderr}"
            )
        if not result_path.exists():
            raise RuntimeError("Fold worker completed without a result file")
        message = json.loads(result_path.read_text(encoding="utf-8"))
        if message.get("status") != "ok":
            raise RuntimeError(message.get("error", "fold worker failed") + "\n" + message.get("traceback", ""))
        return message["result"]


# ======================================================================================
# Main
# ======================================================================================

def main() -> None:
    ap = argparse.ArgumentParser(description="FASE-v24 PMLB Bridge Probe on 529_pollen")
    ap.add_argument("--fold-payload", default=None, help=argparse.SUPPRESS)
    ap.add_argument("--fold-result", default=None, help=argparse.SUPPRESS)
    ap.add_argument("--dataset", default=DATASET)
    ap.add_argument("--data-file", default=None)
    ap.add_argument("--cache-dir", default=None)
    ap.add_argument("--matrix", choices=["smoke", "quick", "full"], default="full")
    ap.add_argument("--seeds", default=None, help="Comma-separated override")
    ap.add_argument("--run-pysr", action="store_true")
    ap.add_argument("--pysr-iterations", type=int, default=1000)
    ap.add_argument("--pysr-maxsize", type=int, default=30)
    ap.add_argument("--pysr-timeout", type=int, default=180)
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--single-fold", type=int, default=None, help="Run exactly one fold per selected seed; intended for robust shell orchestration")
    ap.add_argument("--aggregate-only", action="store_true", help="Build final report from the existing checkpoint without running folds")
    ap.add_argument("--isolate-folds", action=argparse.BooleanOptionalAction, default=False)
    ap.add_argument("--fold-timeout", type=int, default=1800)
    ap.add_argument("--out-dir", default="runs/v24_pmlb_bridge_529_pollen")
    args = ap.parse_args()

    if args.fold_payload:
        result_path = Path(args.fold_result)
        try:
            with Path(args.fold_payload).open("rb") as f:
                payload = pickle.load(f)
            result = run_fold(**payload)
            atomic_json(result_path, {"status": "ok", "result": result})
        except BaseException as exc:
            import traceback
            atomic_json(result_path, {"status": "error", "error": f"{type(exc).__name__}: {exc}", "traceback": traceback.format_exc()})
            raise
        return

    t0 = time.time()
    mcfg = matrix_config(args.matrix)
    if args.seeds:
        mcfg.seeds = [int(x.strip()) for x in args.seeds.split(",") if x.strip()]
    X0, y0, feature_names, provenance = load_dataset(args.dataset, args.data_file, args.cache_dir)

    out_dir = Path(args.out_dir).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = out_dir / "v24_pmlb_bridge_checkpoint.json"
    report_path = out_dir / "v24_pmlb_bridge_report.json"
    fold_csv_path = out_dir / "v24_pmlb_bridge_folds.csv"
    mode_csv_path = out_dir / "v24_pmlb_bridge_operators.csv"

    protocol = {
        "protocol": "FASE-v24 PMLB Bridge Probe / 529_pollen",
        "status": "conversion-trajectory probe; not a generality proof and not a causal claim",
        "dataset_provenance": provenance,
        "matrix": args.matrix,
        "seeds": mcfg.seeds,
        "k_folds": mcfg.k_folds,
        "subsample": mcfg.subsample,
        "predictive_baselines": ["Ridge", "v23 baseline", "v23.1 refined", "v23.1 nondegrading recommendation"] + (["PySR"] if args.run_pysr else []),
        "bridge_operator": "State(x,y) -> State(x + constant axis perturbation, psi(x,y))",
        "operator_language": list(OPERATOR_FAMILIES),
        "destroyer_controls": ["pair shuffle", "target shuffle"],
        "perturbation_audit": "f_hat(gx) ~= omega_g(f_hat(x),x), evaluated only inside empirical support",
        "bridge_config": asdict(mcfg.bridge),
        "environment": {
            "python": sys.version, "platform": platform.platform(), "machine": platform.machine(),
            "numpy": np.__version__, "pandas": pd.__version__,
        },
    }

    checkpoint: Dict[str, Any] = {"protocol": protocol, "folds": {}}
    if args.resume and checkpoint_path.exists():
        try:
            checkpoint = json.loads(checkpoint_path.read_text(encoding="utf-8"))
            checkpoint.setdefault("folds", {})
        except Exception:
            pass

    if not args.aggregate_only:
      for seed in mcfg.seeds:
        X, y, subset_idx = _subsample(X0, y0, mcfg.subsample, seed)
        kf = KFold(n_splits=mcfg.k_folds, shuffle=True, random_state=seed)
        for fold_id, (tr, te) in enumerate(kf.split(X), start=1):
            if args.single_fold is not None and fold_id != args.single_fold:
                continue
            key = f"seed={seed}|fold={fold_id}"
            if key in checkpoint["folds"]:
                print(f"[resume] {key}", flush=True)
                continue
            print(f"[run] {key} n_train={len(tr)} n_test={len(te)}", flush=True)
            payload = dict(
                X=X, y=y, feature_names=feature_names, train_idx=tr, test_idx=te,
                seed=seed, fold_id=fold_id, mcfg=mcfg, run_pysr=args.run_pysr,
                pysr_iterations=args.pysr_iterations, pysr_maxsize=args.pysr_maxsize,
                pysr_timeout=args.pysr_timeout,
            )
            row = run_fold_isolated(payload, args.fold_timeout) if args.isolate_folds else run_fold(**payload)
            row["subset_indices_hash"] = hashlib.sha256(np.asarray(subset_idx, np.int64).tobytes()).hexdigest()[:16]
            checkpoint["folds"][key] = row
            checkpoint["elapsed_sec"] = float(time.time() - t0)
            atomic_json(checkpoint_path, checkpoint)
            print(
                f"  ridge={row['predictive']['ridge']['R2']:.4f} "
                f"v23={row['predictive']['v23_baseline']['R2']:.4f} "
                f"v23.1={row['predictive']['v23_1_refined']['R2']:.4f} "
                f"operators={row['bridge']['n_promoted']} "
                f"null={row['controls']['pair_shuffle']['n_promoted'] + row['controls']['target_shuffle']['n_promoted']}",
                flush=True,
            )

    fold_rows = [checkpoint["folds"][k] for k in sorted(checkpoint["folds"])]
    report = aggregate_report(protocol, fold_rows)
    report["elapsed_sec_total"] = float(time.time() - t0)
    report["checkpoint_path"] = str(checkpoint_path)
    atomic_json(report_path, report)
    fold_rows_csv, mode_rows_csv = flatten_rows(report)
    write_csv(fold_csv_path, fold_rows_csv)
    write_csv(mode_csv_path, mode_rows_csv)

    print(json.dumps(json_safe({
        "report": str(report_path), "fold_csv": str(fold_csv_path), "operator_csv": str(mode_csv_path),
        "horizon_classification": report["horizon_classification"],
        "predictive_summary": report["predictive_summary"],
        "bridge_summary": {k: v for k, v in report["bridge_summary"].items() if k != "mode_summary"},
        "destroyer_controls": report["destroyer_controls"],
        "elapsed_sec_total": report["elapsed_sec_total"],
    }), indent=2))


if __name__ == "__main__":
    main()
