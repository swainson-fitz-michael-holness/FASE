#!/usr/bin/env python3
"""
FASE-G1a-v23 Viète Full Killing Field — Audited Numerical-Stability Patch
=========================================================================

This runner preserves the previous Viète field protocol while correcting the
six issues identified after the first full field run:

1. Separate model-discovered expressions from oracle/post-hoc abstractions.
2. Reject numerically implausible validation predictions.
3. Protect division by scale-aware denominator floors and domain checks.
4. Use an SVD ridge solver with stronger/adaptive regularization.
5. Reject candidate coordinates with extreme train/validation distribution shift.
6. Report unstable_prediction_fail separately from coordinate_recovery_fail.

Apple Silicon
-------------
On macOS arm64, independent FASE conditions can run in deterministic spawned
processes. BLAS threads are pinned to one per process to avoid oversubscription.
NumPy will use the BLAS implementation installed in the active Python build
(typically Apple Accelerate for a native arm64 NumPy build).

The branchy symbolic search is CPU-oriented; Metal/MPS is deliberately not used.

Required sibling modules:
    fase_v23_g1a_coordinate_gate.py
    fase_v23_g1a_hierarchy_gate.py
    fase_v23_g1a_vieta_killing_field.py
    fase_v23_g1a_vieta_full_field_with_pysr.py

Recommended FASE rerun on an M1-class Mac:
    python fase_v23_g1a_vieta_full_field_audited_m1.py \
      --matrix full --pysr-scope none --workers auto \
      --out-dir runs/vieta_full_fase_audited

PySR control should normally be run separately:
    python fase_v23_g1a_vieta_full_field_audited_m1.py \
      --matrix full --skip-fase --pysr-scope anchor --pysr-regimes both \
      --pysr-iterations 1000 --pysr-maxsize 30 --pysr-timeout 120 \
      --out-dir runs/vieta_pysr_anchor_audited
"""

from __future__ import annotations

# Set thread controls before importing NumPy or sibling modules.
import os
import platform

_IS_APPLE_ARM = platform.system() == "Darwin" and platform.machine().lower() in {"arm64", "aarch64"}
if _IS_APPLE_ARM:
    os.environ.setdefault("VECLIB_MAXIMUM_THREADS", "1")
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
    os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("PYTHONHASHSEED", "0")

import argparse
import contextlib
import csv
import io
import json
import math
import multiprocessing as mp
import subprocess
import sys
import time
import traceback
import tempfile
from pathlib import Path
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import asdict, dataclass
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

import fase_v23_g1a_coordinate_gate as base
import fase_v23_g1a_hierarchy_gate as hierarchy
import fase_v23_g1a_vieta_killing_field as vkf
import fase_v23_g1a_vieta_full_field_with_pysr as legacy_runner

EPS = 1e-12


@dataclass
class RobustConfig:
    # Numerical linear head.
    ridge_alpha_min: float = 1e-4
    ridge_condition_trigger: float = 1e8
    ridge_condition_alpha_multiplier: float = 100.0

    # Protected division. floor = max(abs floor, relative floor * robust denom scale).
    division_abs_floor: float = 1e-4
    division_relative_floor: float = 1e-3
    division_max_unsafe_frac: float = 0.02

    # Candidate train/validation distribution shift.
    shift_location_z_max: float = 6.0
    shift_scale_ratio_min: float = 0.10
    shift_scale_ratio_max: float = 10.0
    shift_val_abs_z_q99_max: float = 20.0
    shift_val_abs_z_max: float = 100.0

    # Prediction sanity relative to robust target location/scale.
    prediction_bound_mad: float = 25.0
    prediction_bound_min_scale: float = 1e-3

    # Final stack repair.
    max_prune_steps: int = 32


ROBUST = RobustConfig()


def configure_robust(cfg: Dict[str, Any] | RobustConfig) -> None:
    global ROBUST
    ROBUST = cfg if isinstance(cfg, RobustConfig) else RobustConfig(**cfg)


def robust_scale(v: np.ndarray) -> float:
    x = np.asarray(v, float).reshape(-1)
    x = x[np.isfinite(x)]
    if x.size == 0:
        return ROBUST.prediction_bound_min_scale
    med = float(np.median(x))
    mad = float(np.median(np.abs(x - med)))
    # 1.4826 * MAD estimates standard deviation under Gaussian noise.
    scale = 1.4826 * mad
    if not math.isfinite(scale) or scale < ROBUST.prediction_bound_min_scale:
        q25, q75 = np.quantile(x, [0.25, 0.75])
        scale = float((q75 - q25) / 1.349)
    if not math.isfinite(scale) or scale < ROBUST.prediction_bound_min_scale:
        scale = float(np.std(x))
    return max(scale if math.isfinite(scale) else 0.0, ROBUST.prediction_bound_min_scale)


def prediction_sanity(pred: np.ndarray, y_reference: np.ndarray) -> Tuple[bool, Dict[str, Any]]:
    p = np.asarray(pred, float).reshape(-1)
    y = np.asarray(y_reference, float).reshape(-1)
    med = float(np.median(y))
    scale = robust_scale(y)
    bound = float(abs(med) + ROBUST.prediction_bound_mad * scale)
    finite = bool(np.all(np.isfinite(p)))
    max_abs = float(np.max(np.abs(p))) if p.size else 0.0
    exceed_frac = float(np.mean(np.abs(p) > bound)) if p.size else 0.0
    ok = bool(finite and exceed_frac == 0.0)
    return ok, {
        "finite": finite,
        "prediction_abs_bound": bound,
        "prediction_max_abs": max_abs,
        "prediction_exceed_frac": exceed_frac,
        "target_median": med,
        "target_robust_scale": scale,
    }


def protected_div(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Scale-aware protected division.

    This keeps evaluation finite, while candidate admission separately rejects
    expressions whose denominator is too often near zero.
    """
    aa = np.asarray(a, float)
    bb = np.asarray(b, float)
    finite_b = bb[np.isfinite(bb)]
    scale = float(np.mean(np.abs(finite_b))) if finite_b.size else 0.0
    floor = max(ROBUST.division_abs_floor, ROBUST.division_relative_floor * max(scale, EPS))
    signs = np.where(bb < 0.0, -1.0, 1.0)
    denom = np.where(np.abs(bb) < floor, signs * floor, bb)
    out = aa / denom
    return np.nan_to_num(out, nan=0.0, posinf=1e12, neginf=-1e12)


def ridge_fit_svd(Z: np.ndarray, y: np.ndarray, alpha: float = 1e-6) -> Tuple[np.ndarray, float]:
    """Stable ridge: regularized normal solve, SVD fallback only when needed.

    Candidate scoring calls this thousands of times, so the common path must be
    cheap. The stronger alpha stabilizes the solve; SVD is reserved for failed
    or non-finite solutions and therefore remains a genuine fallback.
    """
    X = np.asarray(Z, float)
    target = np.asarray(y, float).reshape(-1)
    if X.ndim == 1:
        X = X.reshape(-1, 1)
    if X.shape[1] == 0:
        return np.zeros(0), float(np.mean(target))

    muX = np.mean(X, axis=0, keepdims=True)
    muy = float(np.mean(target))
    Xc = X - muX
    yc = target - muy
    alpha_eff = max(float(alpha), ROBUST.ridge_alpha_min)
    A = Xc.T @ Xc + alpha_eff * np.eye(X.shape[1])
    b = Xc.T @ yc
    try:
        w = np.linalg.solve(A, b)
        if not np.all(np.isfinite(w)):
            raise np.linalg.LinAlgError("non-finite ridge solution")
    except np.linalg.LinAlgError:
        try:
            U, sv, Vt = np.linalg.svd(Xc, full_matrices=False)
            filt = sv / (sv * sv + alpha_eff * ROBUST.ridge_condition_alpha_multiplier)
            w = Vt.T @ (filt * (U.T @ yc))
        except np.linalg.LinAlgError:
            aug_X = np.vstack([Xc, math.sqrt(alpha_eff) * np.eye(X.shape[1])])
            aug_y = np.concatenate([yc, np.zeros(X.shape[1])])
            w, *_ = np.linalg.lstsq(aug_X, aug_y, rcond=1e-10)
    b0 = float(muy - (muX @ w.reshape(-1, 1)).item())
    return np.asarray(w, float), b0


def expr_contains_op(expr: base.Expr, op: str) -> bool:
    if expr.op == op:
        return True
    return bool((expr.left is not None and expr_contains_op(expr.left, op)) or
                (expr.right is not None and expr_contains_op(expr.right, op)))


def expr_contains_var(expr: base.Expr, idx: int) -> bool:
    if expr.op == "x" and int(expr.idx) == idx:
        return True
    return bool((expr.left is not None and expr_contains_var(expr.left, idx)) or
                (expr.right is not None and expr_contains_var(expr.right, idx)))


def is_radical_product_expr(expr: base.Expr, multiplier_idx: int = 1) -> bool:
    if expr.op == "mul" and expr.left is not None and expr.right is not None:
        left_sqrt = expr_contains_op(expr.left, "sqrt")
        right_sqrt = expr_contains_op(expr.right, "sqrt")
        left_var = expr_contains_var(expr.left, multiplier_idx)
        right_var = expr_contains_var(expr.right, multiplier_idx)
        if (left_sqrt and right_var) or (right_sqrt and left_var):
            return True
    return bool((expr.left is not None and is_radical_product_expr(expr.left, multiplier_idx)) or
                (expr.right is not None and is_radical_product_expr(expr.right, multiplier_idx)))


def denominator_domain_stats(expr: base.Expr, Xmat: np.ndarray) -> Dict[str, Any]:
    """Inspect every division denominator in an expression tree."""
    rows: List[Dict[str, float]] = []

    def visit(e: base.Expr) -> None:
        if e.op == "div" and e.right is not None:
            den = np.asarray(base.eval_expr(e.right, Xmat), float).reshape(-1)
            finite = den[np.isfinite(den)]
            scale = float(np.median(np.abs(finite))) if finite.size else 0.0
            floor = max(ROBUST.division_abs_floor, ROBUST.division_relative_floor * max(scale, EPS))
            unsafe = float(np.mean((~np.isfinite(den)) | (np.abs(den) < floor)))
            rows.append({"unsafe_frac": unsafe, "floor": floor, "scale": scale})
        if e.left is not None:
            visit(e.left)
        if e.right is not None:
            visit(e.right)

    visit(expr)
    return {
        "n_divisions": len(rows),
        "max_unsafe_frac": max((r["unsafe_frac"] for r in rows), default=0.0),
        "rows": rows,
    }


def distribution_shift_stats(vtr: np.ndarray, vva: np.ndarray) -> Dict[str, Any]:
    """Fast standardized train/validation shift audit."""
    tr = np.asarray(vtr, float).reshape(-1)
    va = np.asarray(vva, float).reshape(-1)
    if not np.all(np.isfinite(tr)) or not np.all(np.isfinite(va)):
        return {"pass": False, "reason": "nonfinite"}
    mu_tr = float(np.mean(tr))
    mu_va = float(np.mean(va))
    sd_tr = max(float(np.std(tr)), 1e-10)
    sd_va = max(float(np.std(va)), 1e-10)
    location_z = abs(mu_va - mu_tr) / sd_tr
    ratio = sd_va / sd_tr
    zva = (va - mu_tr) / sd_tr
    abs_z = np.abs(zva)
    # partition avoids a full sort and is sufficient for a 99th-percentile guard.
    q_index = min(len(abs_z) - 1, max(0, int(0.99 * (len(abs_z) - 1))))
    q99 = float(np.partition(abs_z, q_index)[q_index]) if len(abs_z) else 0.0
    max_abs = float(np.max(abs_z)) if len(abs_z) else 0.0
    ok = bool(
        location_z <= ROBUST.shift_location_z_max
        and ROBUST.shift_scale_ratio_min <= ratio <= ROBUST.shift_scale_ratio_max
        and q99 <= ROBUST.shift_val_abs_z_q99_max
        and max_abs <= ROBUST.shift_val_abs_z_max
    )
    return {
        "pass": ok,
        "location_z": float(location_z),
        "scale_ratio": float(ratio),
        "val_abs_z_q99": q99,
        "val_abs_z_max": max_abs,
    }


def discover_coordinates_robust(
    X_select: np.ndarray,
    y_select: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    cfg: base.GateConfig,
) -> Tuple[List[base.SelectedCoord], Dict[str, object]]:
    selected_exprs = base.base_exprs(X_select.shape[1], cfg)
    discovered: List[base.SelectedCoord] = []
    trace: List[Dict[str, object]] = []

    for layer in range(cfg.max_layers):
        Ztr, fit = base.fit_design(selected_exprs, X_select)
        Zva = fit.transform(X_val)
        w, b0 = ridge_fit_svd(Ztr, y_select, cfg.ridge_alpha)
        pred_tr = base.ridge_predict(Ztr, w, b0)
        pred_val = base.ridge_predict(Zva, w, b0)
        base_ok, base_sanity = prediction_sanity(pred_val, y_select)
        base_r2 = base.r2_score(y_val, pred_val) if base_ok else -math.inf
        resid = y_select - pred_tr

        cands = base.generate_candidates(selected_exprs, X_select, resid, cfg)
        scored: List[Tuple[float, base.SelectedCoord]] = []
        existing_raw = [base.sanitize(base.eval_expr(e0, X_select)) for e0 in selected_exprs]
        rejected = defaultdict(int)
        y_med = float(np.median(y_select))
        y_scale = robust_scale(y_select)
        candidate_prediction_bound = float(abs(y_med) + ROBUST.prediction_bound_mad * y_scale)

        for e in cands:
            # Protected division already applies a scale-aware denominator floor.
            # To keep the killing field computationally tractable, candidate
            # admission uses the resulting coordinate distribution rather than
            # recursively re-evaluating every denominator subtree.
            vtr_raw = base.sanitize(base.eval_expr(e, X_select), clip=1e12)
            vva_raw = base.sanitize(base.eval_expr(e, X_val), clip=1e12)
            if expr_contains_op(e, "div"):
                tr_sd = max(float(np.std(vtr_raw)), 1e-10)
                va_sd = max(float(np.std(vva_raw)), 1e-10)
                if (float(np.max(np.abs(vtr_raw - np.mean(vtr_raw)))) > 1000.0 * tr_sd or
                    float(np.max(np.abs(vva_raw - np.mean(vva_raw)))) > 1000.0 * va_sd):
                    rejected["unsafe_division_output"] += 1
                    continue
            if vtr_raw.std() < 1e-10:
                rejected["constant"] += 1
                continue
            shift = distribution_shift_stats(vtr_raw, vva_raw)
            if not shift.get("pass", False):
                rejected["distribution_shift"] += 1
                continue
            if existing_raw and max(base.corr_abs(vtr_raw, v0) for v0 in existing_raw) >= cfg.redundancy_corr:
                rejected["redundant"] += 1
                continue

            mu = float(vtr_raw.mean())
            sd = float(vtr_raw.std())
            if sd < 1e-10 or not math.isfinite(sd):
                rejected["invalid_scale"] += 1
                continue
            ztr = ((vtr_raw - mu) / sd).reshape(-1, 1)
            zva = ((vva_raw - mu) / sd).reshape(-1, 1)
            Ztr_c = np.hstack([Ztr, ztr])
            Zva_c = np.hstack([Zva, zva])
            wc, b0c = ridge_fit_svd(Ztr_c, y_select, cfg.ridge_alpha)
            pred_c = base.ridge_predict(Zva_c, wc, b0c)
            sane = bool(np.all(np.isfinite(pred_c)) and np.max(np.abs(pred_c)) <= candidate_prediction_bound)
            if not sane:
                rejected["prediction_sanity"] += 1
                continue
            r2_c = base.r2_score(y_val, pred_c)
            gain = r2_c - base_r2
            align = base.corr_abs(vtr_raw, resid)
            objective = gain - cfg.complexity_penalty * e.complexity() + 0.01 * align
            if gain > cfg.min_gain:
                scored.append((objective, base.SelectedCoord(
                    expr=e,
                    layer=layer + 1,
                    val_gain=float(gain),
                    val_r2_after=float(r2_c),
                    complexity=e.complexity(),
                    residual_alignment=float(align),
                )))
            else:
                rejected["insufficient_gain"] += 1

        scored.sort(key=lambda t: t[0], reverse=True)
        accepted: List[base.SelectedCoord] = []
        for _, s in scored:
            trial_existing = selected_exprs + [a.expr for a in accepted]
            if base.max_redundancy(s.expr, trial_existing, X_select) >= cfg.redundancy_corr:
                rejected["same_layer_redundant"] += 1
                continue
            selected_exprs.append(s.expr)
            discovered.append(s)
            accepted.append(s)
            if len(accepted) >= cfg.accept_per_layer:
                break

        trace.append({
            "layer": layer + 1,
            "base_inner_val_r2": float(base_r2),
            "base_prediction_sanity": base_sanity,
            "n_candidates": len(cands),
            "rejected_counts": dict(rejected),
            "accepted": [{
                "expr": str(s.expr),
                "gain": s.val_gain,
                "r2_after": s.val_r2_after,
                "complexity": s.complexity,
                "residual_alignment": s.residual_alignment,
            } for s in accepted],
        })
        if not accepted:
            break
    return discovered, {"trace": trace}


def stable_final_fit(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    base_expr_list: List[base.Expr],
    selected: List[base.SelectedCoord],
    ridge_alpha: float,
) -> Tuple[np.ndarray, List[base.SelectedCoord], Dict[str, Any]]:
    """Fit the final head and prune newest coordinates until predictions are sane."""
    retained = list(selected)
    initial_count = len(retained)
    attempts: List[Dict[str, Any]] = []
    max_steps = min(ROBUST.max_prune_steps, initial_count + 1)

    for step in range(max_steps + 1):
        exprs = base_expr_list + [s.expr for s in retained]
        Ztr, fit = base.fit_design(exprs, X_train)
        Zva = fit.transform(X_val)
        w, b0 = ridge_fit_svd(Ztr, y_train, ridge_alpha)
        pred = base.ridge_predict(Zva, w, b0)
        sane, sanity = prediction_sanity(pred, y_train)
        attempts.append({
            "step": step,
            "n_selected": len(retained),
            "sane": sane,
            **sanity,
        })
        if sane:
            return pred, retained, {
                "unstable_prediction_detected": step > 0,
                "unstable_prediction_fail": False,
                "n_pruned": initial_count - len(retained),
                "attempts": attempts,
                "fallback": None,
            }
        if not retained:
            break
        retained.pop()  # newest coordinate is least established

    fallback = np.full(X_val.shape[0], float(np.mean(y_train)))
    return fallback, [], {
        "unstable_prediction_detected": True,
        "unstable_prediction_fail": True,
        "n_pruned": initial_count,
        "attempts": attempts,
        "fallback": "training_mean",
    }


def unique_exprs(exprs: Sequence[base.Expr]) -> List[base.Expr]:
    out: List[base.Expr] = []
    seen = set()
    for e in exprs:
        if e.key() not in seen:
            seen.add(e.key())
            out.append(e)
    return out


def discovered_expression_audit(
    X: np.ndarray,
    hidden: Dict[str, np.ndarray],
    selected_exprs: Sequence[base.Expr],
) -> Dict[str, Any]:
    all_sub = unique_exprs([
        s for e in selected_exprs for s in e.subexpressions() if s.op != "const"
    ])
    zrad = hidden.get("z_sqrt_2_plus_a")
    znext = hidden.get("z_p_next")

    best_rad = {"corr": 0.0, "expr": None}
    best_prod = {"corr": 0.0, "expr": None}
    for e in all_sub:
        if expr_contains_op(e, "sqrt") and zrad is not None:
            c = base.corr_abs(base.sanitize(base.eval_expr(e, X), clip=1e12), zrad)
            if c > best_rad["corr"]:
                best_rad = {"corr": float(c), "expr": str(e)}
        if is_radical_product_expr(e, multiplier_idx=1) and znext is not None:
            c = base.corr_abs(base.sanitize(base.eval_expr(e, X), clip=1e12), znext)
            if c > best_prod["corr"]:
                best_prod = {"corr": float(c), "expr": str(e)}

    return {
        "n_selected_unique": len(unique_exprs(selected_exprs)),
        "n_selected_subexpressions": len(all_sub),
        "sqrt_syntax_present": any(expr_contains_op(e, "sqrt") for e in all_sub),
        "best_discovered_radical_coordinate": best_rad,
        "best_discovered_radical_product": best_prod,
        "radical_coordinate_pass": bool(best_rad["expr"] is not None and best_rad["corr"] >= 0.95),
        "radical_product_pass": bool(best_prod["expr"] is not None and best_prod["corr"] >= 0.95),
    }


def run_hierarchy_gate_robust(
    X: np.ndarray,
    y: np.ndarray,
    hidden: Dict[str, np.ndarray],
    cfg: hierarchy.HierarchyConfig,
) -> Dict[str, object]:
    t0 = time.time()
    bcfg = hierarchy.to_base_config(cfg)
    bcfg.ridge_alpha = max(float(bcfg.ridge_alpha), ROBUST.ridge_alpha_min)
    n = len(y)
    folds = base.kfold_indices(n, bcfg.k_folds, bcfg.seed)
    yhat = np.zeros(n)
    all_selected: List[base.SelectedCoord] = []
    fold_summaries: List[Dict[str, Any]] = []
    unstable_detected = False
    unstable_fail = False
    total_pruned = 0

    for fold_id, val_idx in enumerate(folds, start=1):
        train_idx = np.setdiff1d(np.arange(n), val_idx)
        inner_train_idx, inner_val_idx = base.split_inner(train_idx, bcfg.inner_val_frac, bcfg.seed + 1000 * fold_id)
        selected, info = discover_coordinates_robust(
            X[inner_train_idx], y[inner_train_idx],
            X[inner_val_idx], y[inner_val_idx],
            bcfg,
        )
        pred, retained, stability = stable_final_fit(
            X[train_idx], y[train_idx], X[val_idx],
            base.base_exprs(X.shape[1], bcfg), selected, bcfg.ridge_alpha,
        )
        yhat[val_idx] = pred
        all_selected.extend(retained)
        unstable_detected = unstable_detected or bool(stability["unstable_prediction_detected"])
        unstable_fail = unstable_fail or bool(stability["unstable_prediction_fail"])
        total_pruned += int(stability["n_pruned"])
        fold_summaries.append({
            "fold": fold_id,
            "val_R2": float(base.r2_score(y[val_idx], yhat[val_idx])),
            "selected_proposed": [str(s.expr) for s in selected],
            "selected_retained": [str(s.expr) for s in retained],
            "trace": info["trace"],
            "final_prediction_stability": stability,
        })

    selected_exprs_only = unique_exprs([s.expr for s in all_selected])
    selected_plus_subexprs = unique_exprs([
        se for e in selected_exprs_only for se in e.subexpressions() if se.op != "const"
    ])
    single = base.coordinate_recovery(X, selected_exprs_only, hidden)
    selected_span = hierarchy.span_recovery(X, selected_exprs_only, hidden, alpha=bcfg.ridge_alpha)
    subexpr_span = hierarchy.span_recovery(X, selected_plus_subexprs, hidden, alpha=bcfg.ridge_alpha)
    abstraction = hierarchy.derive_rank1_bilinear_abstractions(X, yhat, hidden=hidden)
    discovery_audit = discovered_expression_audit(X, hidden, selected_exprs_only)

    return {
        "protocol": "FASE-G1a-v23 audited hierarchy gate / robust numeric patch",
        "primitive_algebra": ["+", "-", "*", "/", "compose", "normalize", "project"] + (["sqrt_pos"] if bcfg.include_sqrt else []),
        "n": int(n),
        "d": int(X.shape[1]),
        "R2_oof": float(base.r2_score(y, yhat)),
        "MSE_oof": float(base.mse(y, yhat)),
        "elapsed_sec": float(time.time() - t0),
        "top_selected_expressions": base.top_expression_counts(all_selected, 16),
        "selected_expression_strings": [str(e) for e in selected_exprs_only],
        "single_coordinate_recovery": single,
        "selected_span_recovery": selected_span,
        "subexpression_span_recovery": subexpr_span,
        "discovered_expression_audit": discovery_audit,
        "higher_hierarchy_abstraction": abstraction,
        "numerical_stability": {
            "unstable_prediction_detected": unstable_detected,
            "unstable_prediction_fail": unstable_fail,
            "n_coordinates_pruned": total_pruned,
        },
        "folds": fold_summaries,
        "config": {**cfg.__dict__, "effective_ridge_alpha": bcfg.ridge_alpha},
        "robust_config": asdict(ROBUST),
    }


def summarize_robust(result: Dict[str, object], r2_thr: float = 0.95) -> Dict[str, object]:
    selected_span = result.get("selected_span_recovery", {})
    sub_span = result.get("subexpression_span_recovery", {})
    terminal_names = [name for name in selected_span if any(m in name for m in ("product", "p_next", "next", "mul", "ratio"))]
    if not terminal_names and selected_span:
        terminal_names = [list(selected_span.keys())[-1]]

    def row_corr(container: Dict[str, Any], name: str) -> float:
        row = container.get(name, {})
        return float(max(row.get("span_corr", 0.0), row.get("span_R2", 0.0)))

    terminal_score = max(
        [row_corr(selected_span, n) for n in terminal_names] +
        [row_corr(sub_span, n) for n in terminal_names] + [0.0]
    )
    stability = result.get("numerical_stability", {})
    return {
        "R2_oof": result["R2_oof"],
        "MSE_oof": result["MSE_oof"],
        "prediction_pass": bool(float(result["R2_oof"]) >= r2_thr),
        "terminal_coordinate_names": terminal_names,
        "terminal_span_score": float(terminal_score),
        "terminal_span_pass": bool(terminal_score >= 0.95),
        "unstable_prediction_detected": bool(stability.get("unstable_prediction_detected", False)),
        "unstable_prediction_fail": bool(stability.get("unstable_prediction_fail", False)),
        "n_coordinates_pruned": int(stability.get("n_coordinates_pruned", 0)),
        "top_selected_expressions": result.get("top_selected_expressions", [])[:8],
        "elapsed_sec": result["elapsed_sec"],
    }


def run_once_robust(seed: int, n: int, noise: float, include_sqrt: bool, quick: bool) -> Dict[str, object]:
    X, y, hidden = vkf.make_vieta(seed=seed, n=n, noise=noise)
    cfg = vkf.cfg_for(seed=seed, n=n, include_sqrt=include_sqrt, quick=quick)
    cfg.ridge_alpha = max(float(cfg.ridge_alpha), ROBUST.ridge_alpha_min)
    result = run_hierarchy_gate_robust(X, y, hidden, cfg)
    summary = summarize_robust(result)
    oracle_audit = vkf.radical_product_audit(X, y, hidden)
    return {
        "condition": {
            "seed": seed,
            "n": n,
            "noise": noise,
            "include_sqrt": include_sqrt,
            "primitive_regime": "A0r = A0 + sqrt_pos" if include_sqrt else "strict A0",
            "quick": quick,
        },
        "summary": summary,
        "discovered_expression_audit": result.get("discovered_expression_audit", {}),
        "oracle_posthoc_abstraction_audit": oracle_audit,
        "top_selected_expressions": result.get("top_selected_expressions", [])[:12],
        "selected_expression_strings": result.get("selected_expression_strings", []),
        "selected_span_recovery": result.get("selected_span_recovery", {}),
        "subexpression_span_recovery": result.get("subexpression_span_recovery", {}),
        "single_coordinate_recovery": result.get("single_coordinate_recovery", {}),
        "numerical_stability": result.get("numerical_stability", {}),
        "folds": result.get("folds", []),
        "elapsed_sec": result.get("elapsed_sec"),
    }


def gate_verdict_robust(case: Dict[str, object]) -> Dict[str, object]:
    summary = case.get("summary", {})
    cond = case.get("condition", {})
    include_sqrt = bool(cond.get("include_sqrt", False))
    discovery = case.get("discovered_expression_audit", {})
    oracle = case.get("oracle_posthoc_abstraction_audit", {})

    pred = bool(summary.get("prediction_pass", False))
    terminal = bool(summary.get("terminal_span_pass", False))
    rad_discovered = bool(discovery.get("radical_coordinate_pass", False))
    product_discovered = bool(discovery.get("radical_product_pass", False))
    unstable_fail = bool(summary.get("unstable_prediction_fail", False))
    coordinate_fail = None if not include_sqrt else bool(not (terminal and rad_discovered and product_discovered))
    gate = bool(include_sqrt and pred and terminal and rad_discovered and product_discovered and not unstable_fail)

    return {
        "prediction_pass": pred,
        "terminal_span_pass": terminal,
        "terminal_span_score": summary.get("terminal_span_score"),
        "radical_coordinate_present_in_discovery": rad_discovered,
        "radical_coordinate_discovered_corr": discovery.get("best_discovered_radical_coordinate", {}).get("corr"),
        "radical_coordinate_discovered_expr": discovery.get("best_discovered_radical_coordinate", {}).get("expr"),
        "radical_product_present_in_discovery": product_discovered,
        "radical_product_discovered_corr": discovery.get("best_discovered_radical_product", {}).get("corr"),
        "radical_product_discovered_expr": discovery.get("best_discovered_radical_product", {}).get("expr"),
        "oracle_posthoc_radical_product_pass": bool(oracle.get("pass_radical_product", False)),
        "oracle_posthoc_best_abstraction": (oracle.get("best_candidate") or {}).get("expr") if isinstance(oracle, dict) else None,
        "unstable_prediction_detected": bool(summary.get("unstable_prediction_detected", False)),
        "unstable_prediction_fail": unstable_fail,
        "n_coordinates_pruned": int(summary.get("n_coordinates_pruned", 0)),
        "coordinate_recovery_fail": coordinate_fail,
        "strict_A0_not_eligible_for_radical_gate": not include_sqrt,
        "vieta_gate_pass": gate,
    }


def flatten_fase_case(case: Dict[str, Any], label: str) -> Dict[str, Any]:
    cond = case.get("condition", {})
    gv = case.get("gate_verdict", {})
    summary = case.get("summary", {})
    tops = case.get("top_selected_expressions", []) or []
    best_discovered = gv.get("radical_product_discovered_expr")
    if best_discovered is None and tops:
        best_discovered = tops[0].get("expr")
    return {
        "method": "FASE",
        "label": label,
        "primitive_regime": cond.get("primitive_regime"),
        "include_sqrt": cond.get("include_sqrt"),
        "seed": cond.get("seed"),
        "n": cond.get("n"),
        "noise": cond.get("noise"),
        "k_folds": None,
        "R2_oof": summary.get("R2_oof"),
        "MSE_oof": summary.get("MSE_oof"),
        "elapsed_sec": case.get("elapsed_sec"),
        "prediction_pass": gv.get("prediction_pass"),
        "terminal_span_pass": gv.get("terminal_span_pass"),
        "terminal_span_score": gv.get("terminal_span_score"),
        "radical_in_discovery": gv.get("radical_coordinate_present_in_discovery"),
        "radical_discovered_corr": gv.get("radical_coordinate_discovered_corr"),
        "radical_product_in_discovery": gv.get("radical_product_present_in_discovery"),
        "radical_product_discovered_corr": gv.get("radical_product_discovered_corr"),
        "oracle_posthoc_radical_product_pass": gv.get("oracle_posthoc_radical_product_pass"),
        "vieta_gate_pass": gv.get("vieta_gate_pass"),
        "strict_A0_not_eligible": gv.get("strict_A0_not_eligible_for_radical_gate"),
        "unstable_prediction_detected": gv.get("unstable_prediction_detected"),
        "unstable_prediction_fail": gv.get("unstable_prediction_fail"),
        "coordinate_recovery_fail": gv.get("coordinate_recovery_fail"),
        "n_coordinates_pruned": gv.get("n_coordinates_pruned"),
        "best_discovered_expression": best_discovered,
        "best_oracle_posthoc_abstraction": gv.get("oracle_posthoc_best_abstraction"),
        "top_expressions_json": json.dumps(tops[:8]),
        "error": None,
    }


def apply_patches() -> None:
    # All sibling modules share the same imported base module object.
    base.safe_div = protected_div
    base.ridge_fit = ridge_fit_svd
    base.discover_coordinates = discover_coordinates_robust
    hierarchy.run_hierarchy_gate = run_hierarchy_gate_robust


def condition_grid(matrix: str) -> Tuple[List[int], List[int], List[float]]:
    return legacy_runner.condition_grid(matrix)


def detect_performance_cores() -> Optional[int]:
    if not _IS_APPLE_ARM:
        return None
    for key in ("hw.perflevel0.physicalcpu", "hw.physicalcpu"):
        try:
            out = subprocess.check_output(["sysctl", "-n", key], text=True, stderr=subprocess.DEVNULL).strip()
            n = int(out)
            if n > 0:
                return n
        except Exception:
            pass
    return None


def resolve_workers(spec: str) -> int:
    if spec != "auto":
        return max(1, int(spec))
    cpu = os.cpu_count() or 1
    if _IS_APPLE_ARM:
        perf = detect_performance_cores()
        return max(1, min(perf or max(1, cpu // 2), 8))
    return max(1, min(max(1, cpu // 2), 8))


def capture_numpy_config() -> str:
    buf = io.StringIO()
    try:
        with contextlib.redirect_stdout(buf):
            np.show_config()
        return buf.getvalue()
    except Exception as e:
        return repr(e)


def worker_fase_condition(payload: Tuple[int, int, int, float, str, Dict[str, Any]]) -> Tuple[int, List[Tuple[str, Dict[str, Any]]]]:
    index, seed, n, noise, matrix, robust_dict = payload
    configure_robust(robust_dict)
    apply_patches()
    np.seterr(all="ignore")
    out: List[Tuple[str, Dict[str, Any]]] = []
    for include_sqrt, label in ((False, "FASE_strict_A0"), (True, "FASE_A0r")):
        case = run_once_robust(seed, n, noise, include_sqrt=include_sqrt, quick=(matrix == "quick"))
        case["gate_verdict"] = gate_verdict_robust(case)
        out.append((label, case))
    return index, out



def json_safe(value: Any) -> Any:
    """Recursively convert scientific/PySR/SymPy objects to strict JSON values.

    PySR's pandas rows may contain SymPy objects such as Mul in fields like
    ``sympy_format``.  Those objects are useful diagnostically but are not
    natively serializable by ``json``.  Preserve their readable form as text.
    """
    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float):
        return value if math.isfinite(value) else str(value)
    if isinstance(value, np.generic):
        return json_safe(value.item())
    if isinstance(value, np.ndarray):
        return [json_safe(v) for v in value.tolist()]
    if isinstance(value, dict):
        return {str(k): json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple, set, frozenset)):
        return [json_safe(v) for v in value]
    if hasattr(value, "to_dict"):
        try:
            return json_safe(value.to_dict())
        except Exception:
            pass
    if hasattr(value, "tolist"):
        try:
            return json_safe(value.tolist())
        except Exception:
            pass
    # SymPy Basic subclasses (Mul, Add, Pow, Float, Symbol, ...), pathlib
    # values, Julia/PySR wrapper values, and other scientific scalars.
    return str(value)


def atomic_json_dump(path: str, payload: Any) -> None:
    """Write a JSON document atomically so interruption cannot corrupt it."""
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(prefix=target.name + ".", suffix=".tmp", dir=str(target.parent))
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            json.dump(json_safe(payload), f, indent=2, allow_nan=False)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp_name, target)
    except Exception:
        try:
            os.unlink(tmp_name)
        except OSError:
            pass
        raise


def case_key(row: Dict[str, Any]) -> Tuple[Any, ...]:
    return (
        row.get("method"), row.get("label"), row.get("seed"),
        row.get("n"), row.get("noise"), row.get("include_sqrt"),
    )


def load_checkpoint(path: str) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    if not os.path.exists(path):
        return [], []
    with open(path, "r", encoding="utf-8") as f:
        obj = json.load(f)
    rows = obj.get("rows", []) if isinstance(obj, dict) else []
    detailed = obj.get("detailed", []) if isinstance(obj, dict) else []
    return list(rows), list(detailed)


def write_checkpoint(
    path: str,
    protocol: Dict[str, Any],
    rows: Sequence[Dict[str, Any]],
    detailed: Sequence[Dict[str, Any]],
    started_at: float,
) -> None:
    payload = {
        "protocol": protocol,
        "rows": list(rows),
        "aggregates": aggregate_rows(rows),
        "detailed": list(detailed),
        "elapsed_sec_total": float(time.time() - started_at),
        "checkpoint": True,
    }
    atomic_json_dump(path, payload)


def aggregate_rows(rows: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    groups: Dict[Tuple[Any, ...], List[Dict[str, Any]]] = defaultdict(list)
    for r in rows:
        key = (r.get("method"), r.get("label"), r.get("primitive_regime"), r.get("n"), r.get("noise"))
        groups[key].append(r)
    out = []
    for key, rs in sorted(groups.items(), key=lambda kv: tuple(str(x) for x in kv[0])):
        vals = [float(r["R2_oof"]) for r in rs if r.get("R2_oof") is not None and math.isfinite(float(r["R2_oof"]))]
        gates = [bool(r.get("vieta_gate_pass")) for r in rs if r.get("vieta_gate_pass") is not None]
        unstable = [bool(r.get("unstable_prediction_fail")) for r in rs if r.get("unstable_prediction_fail") is not None]
        coordfails = [bool(r.get("coordinate_recovery_fail")) for r in rs if r.get("coordinate_recovery_fail") is not None]
        elapsed = [float(r["elapsed_sec"]) for r in rs if r.get("elapsed_sec") is not None and math.isfinite(float(r["elapsed_sec"]))]
        out.append({
            "method": key[0], "label": key[1], "primitive_regime": key[2], "n": key[3], "noise": key[4],
            "runs": len(rs),
            "R2_mean": float(np.mean(vals)) if vals else None,
            "R2_min": float(np.min(vals)) if vals else None,
            "R2_median": float(np.median(vals)) if vals else None,
            "R2_max": float(np.max(vals)) if vals else None,
            "gate_pass_rate": float(np.mean(gates)) if gates else None,
            "unstable_prediction_fail_rate": float(np.mean(unstable)) if unstable else None,
            "coordinate_recovery_fail_rate": float(np.mean(coordfails)) if coordfails else None,
            "elapsed_total_sec": float(np.sum(elapsed)) if elapsed else None,
        })
    return {"by_method_n_noise": out}


def write_csv(path: str, rows: Sequence[Dict[str, Any]]) -> None:
    fieldnames = [
        "method", "label", "primitive_regime", "include_sqrt", "seed", "n", "noise", "k_folds",
        "R2_oof", "MSE_oof", "elapsed_sec", "prediction_pass", "terminal_span_pass", "terminal_span_score",
        "radical_in_discovery", "radical_discovered_corr", "radical_product_in_discovery",
        "radical_product_discovered_corr", "oracle_posthoc_radical_product_pass", "vieta_gate_pass",
        "strict_A0_not_eligible", "unstable_prediction_detected", "unstable_prediction_fail",
        "coordinate_recovery_fail", "n_coordinates_pruned", "best_discovered_expression",
        "best_oracle_posthoc_abstraction", "error",
    ]
    with open(path, "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        w.writeheader()
        for r in rows:
            w.writerow(r)


def add_robust_args(ap: argparse.ArgumentParser) -> None:
    ap.add_argument("--ridge-alpha-min", type=float, default=1e-4)
    ap.add_argument("--division-abs-floor", type=float, default=1e-4)
    ap.add_argument("--division-relative-floor", type=float, default=1e-3)
    ap.add_argument("--division-max-unsafe-frac", type=float, default=0.02)
    ap.add_argument("--prediction-bound-mad", type=float, default=25.0)
    ap.add_argument("--shift-location-z-max", type=float, default=6.0)
    ap.add_argument("--shift-scale-ratio-min", type=float, default=0.10)
    ap.add_argument("--shift-scale-ratio-max", type=float, default=10.0)
    ap.add_argument("--shift-val-abs-z-q99-max", type=float, default=20.0)
    ap.add_argument("--shift-val-abs-z-max", type=float, default=100.0)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--matrix", choices=["quick", "medium", "full"], default="full")
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--out-dir", default="./vieta_field_audited_out")
    ap.add_argument("--k-folds", type=int, default=5)
    ap.add_argument("--r2-threshold", type=float, default=0.95)
    ap.add_argument("--workers", default="auto", help="FASE condition workers: auto or positive integer.")
    ap.add_argument("--pysr-scope", choices=["none", "minimal", "anchor", "full"], default="none")
    ap.add_argument("--pysr-regimes", choices=["none", "no_sqrt", "with_sqrt", "both"], default="both")
    ap.add_argument("--pysr-iterations", type=int, default=1000)
    ap.add_argument("--pysr-maxsize", type=int, default=30)
    ap.add_argument("--pysr-timeout", type=int, default=120)
    ap.add_argument("--pysr-extra-unary", default="")
    ap.add_argument("--skip-fase", action="store_true")
    ap.add_argument("--resume", action="store_true", help="Resume completed cases from the audited checkpoint in --out-dir.")
    ap.add_argument("--checkpoint-every", type=int, default=1, help="Write an atomic checkpoint every N completed PySR cases.")
    add_robust_args(ap)
    args = ap.parse_args()
    if args.quick:
        args.matrix = "quick"

    robust_cfg = RobustConfig(
        ridge_alpha_min=args.ridge_alpha_min,
        division_abs_floor=args.division_abs_floor,
        division_relative_floor=args.division_relative_floor,
        division_max_unsafe_frac=args.division_max_unsafe_frac,
        prediction_bound_mad=args.prediction_bound_mad,
        shift_location_z_max=args.shift_location_z_max,
        shift_scale_ratio_min=args.shift_scale_ratio_min,
        shift_scale_ratio_max=args.shift_scale_ratio_max,
        shift_val_abs_z_q99_max=args.shift_val_abs_z_q99_max,
        shift_val_abs_z_max=args.shift_val_abs_z_max,
    )
    configure_robust(robust_cfg)
    apply_patches()
    np.seterr(all="ignore")

    os.makedirs(args.out_dir, exist_ok=True)
    seeds, sample_sizes, noise_levels = condition_grid(args.matrix)
    workers = resolve_workers(args.workers)
    t0 = time.time()
    rows: List[Dict[str, Any]] = []
    detailed: List[Dict[str, Any]] = []
    checkpoint_path = os.path.join(args.out_dir, "vieta_full_field_checkpoint_audited.json")
    if args.resume:
        rows, detailed = load_checkpoint(checkpoint_path)
        if rows:
            print(f"[resume] loaded {len(rows)} completed row(s) from {checkpoint_path}", flush=True)

    protocol = {
        "protocol": "FASE-G1a-v23 Viète audited full killing field + PySR control",
        "patches": [
            "discovered_vs_oracle_separation",
            "prediction_sanity_filter_with_stack_pruning",
            "scale_aware_protected_division_and_domain_rejection",
            "adaptive_SVD_ridge",
            "candidate_distribution_shift_rejection",
            "separate_unstable_prediction_and_coordinate_recovery_failures",
        ],
        "matrix": args.matrix,
        "seeds": seeds,
        "sample_sizes": sample_sizes,
        "noise_levels": noise_levels,
        "pysr_scope": args.pysr_scope,
        "pysr_regimes": args.pysr_regimes,
        "r2_threshold": args.r2_threshold,
        "spine": "X -> Z(0) -> Z(1) -> ... -> Z(k)",
        "target_law": "y = p*sqrt(2+a)/2 + noise",
        "destroyer_clause": "Oracle/post-hoc abstraction cannot close the gate; closure requires actual selected-expression radical coordinate and radical-product discovery, stable predictions, terminal span recovery, and OOF R2 threshold.",
        "robust_config": asdict(robust_cfg),
        "parallel": {
            "requested": args.workers,
            "resolved_workers": workers,
            "apple_silicon": _IS_APPLE_ARM,
            "detected_performance_cores": detect_performance_cores(),
            "process_start_method": "spawn" if workers > 1 else None,
            "blas_threads_per_process": 1 if _IS_APPLE_ARM else "environment/default",
        },
        "environment": {
            "python": sys.version,
            "platform": platform.platform(),
            "machine": platform.machine(),
            "numpy": np.__version__,
            "numpy_config": capture_numpy_config(),
        },
    }

    completed_keys = {case_key(r) for r in rows if not r.get("error")}

    conditions = [(i, seed, n, noise, args.matrix, asdict(robust_cfg))
                  for i, (seed, n, noise) in enumerate(
                      (s_n_no for s_n_no in ((s, n0, no) for s in seeds for n0 in sample_sizes for no in noise_levels))
                  )]

    if not args.skip_fase:
        print(f"[FASE] {len(conditions)} conditions, {workers} worker(s)", flush=True)
        results: Dict[int, List[Tuple[str, Dict[str, Any]]]] = {}
        if workers == 1:
            for payload in conditions:
                idx, cases = worker_fase_condition(payload)
                results[idx] = cases
                print(f"  completed condition {idx + 1}/{len(conditions)}", flush=True)
        else:
            ctx = mp.get_context("spawn")
            with ProcessPoolExecutor(max_workers=workers, mp_context=ctx) as ex:
                future_map = {ex.submit(worker_fase_condition, p): p[0] for p in conditions}
                done = 0
                for fut in as_completed(future_map):
                    idx = future_map[fut]
                    try:
                        out_idx, cases = fut.result()
                        results[out_idx] = cases
                    except Exception as e:
                        protocol.setdefault("parallel_worker_errors", []).append({
                            "condition_index": idx, "error": repr(e), "traceback": traceback.format_exc()
                        })
                    done += 1
                    print(f"  completed condition {done}/{len(conditions)}", flush=True)

        for idx in sorted(results):
            for label, case in results[idx]:
                flat = flatten_fase_case(case, label)
                rows.append(flat)
                detailed.append({"kind": "FASE_case", "row": flat, "case": case})

    # PySR controls are intentionally serial here. Julia startup/search parallelism
    # should not be nested inside the FASE process pool.
    pysr_available = False
    if args.pysr_scope != "none" and args.pysr_regimes != "none":
        try:
            _ = legacy_runner.get_pysr_regressor()
            pysr_available = True
        except Exception as e:
            protocol["pysr_import_error"] = repr(e)

    if pysr_available:
        for seed in seeds:
            for n in sample_sizes:
                for noise in noise_levels:
                    if not legacy_runner.pysr_condition_filter(args.pysr_scope, n, noise):
                        continue
                    regimes: List[bool] = []
                    if args.pysr_regimes in ("no_sqrt", "both"):
                        regimes.append(False)
                    if args.pysr_regimes in ("with_sqrt", "both"):
                        regimes.append(True)
                    for with_sqrt in regimes:
                        label = "PySR_with_sqrt" if with_sqrt else "PySR_no_sqrt"
                        key = ("PySR", label, seed, n, noise, with_sqrt)
                        if args.resume and key in completed_keys:
                            print(f"[resume] skipping {label} seed={seed} n={n} noise={noise}", flush=True)
                            continue
                        try:
                            pr = legacy_runner.run_pysr_oof(seed, n, noise, with_sqrt=with_sqrt, args=args)
                            # Rename ambiguous fields to the audited schema.
                            pr["best_discovered_expression"] = pr.pop("best_expression_or_abstraction", None)
                            pr["best_oracle_posthoc_abstraction"] = None
                            pr["unstable_prediction_detected"] = False
                            pr["unstable_prediction_fail"] = False
                            pr["coordinate_recovery_fail"] = None
                            pr = json_safe(pr)
                            rows.append(pr)
                            detailed.append({"kind": "PySR_case", "row": {k: v for k, v in pr.items() if k != "folds"}, "folds": pr.get("folds", [])})
                            completed_keys.add(key)
                            print(f"[PySR] {label} seed={seed} n={n} noise={noise} R2={pr['R2_oof']:.6f}", flush=True)
                            completed_pysr = sum(1 for r in rows if r.get("method") == "PySR")
                            if args.checkpoint_every > 0 and completed_pysr % args.checkpoint_every == 0:
                                write_checkpoint(checkpoint_path, protocol, rows, detailed, t0)
                        except Exception as e:
                            err = {
                                "method": "PySR", "label": label,
                                "primitive_regime": "with sqrt" if with_sqrt else "no sqrt",
                                "include_sqrt": with_sqrt, "seed": seed, "n": n, "noise": noise,
                                "error": repr(e), "traceback": traceback.format_exc(),
                            }
                            err = json_safe(err)
                            rows.append(err)
                            detailed.append({"kind": "PySR_error", "row": err})
                            write_checkpoint(checkpoint_path, protocol, rows, detailed, t0)

    report = {
        "protocol": protocol,
        "rows": rows,
        "aggregates": aggregate_rows(rows),
        "detailed": detailed,
        "elapsed_sec_total": float(time.time() - t0),
    }
    json_path = os.path.join(args.out_dir, "vieta_full_field_report_audited.json")
    csv_path = os.path.join(args.out_dir, "vieta_full_field_rows_audited.csv")
    atomic_json_dump(json_path, report)
    write_csv(csv_path, json_safe(rows))
    # Keep a final checkpoint as a resumable audit trail.
    write_checkpoint(checkpoint_path, protocol, rows, detailed, t0)
    print(f"\nWROTE {json_path}")
    print(f"WROTE {csv_path}")
    print(f"WROTE {checkpoint_path}")


if __name__ == "__main__":
    main()
