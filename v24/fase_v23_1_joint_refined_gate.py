#!/usr/bin/env python3
"""
FASE v23.1 — Non-degrading joint-constant refinement wrapper
============================================================

This file DOES NOT replace or modify the v23 coordinate-discovery engine.
It runs two branches on the same untouched outer folds:

    1. v23 baseline: original discrete topology/constant-grid discovery.
    2. v23.1 challenger: the same discovered topology, followed by leakage-safe
       joint continuous constant refinement using inner cross-validation.

The baseline branch is always retained and reported.  This prevents an experimental
refinement mechanism from silently degrading the established v23 artifact.

The refinement objective is mean inner-CV validation R² over the outer-training set.
The outer fold is never used to fit, refine, accept, or select constants.

Dependencies:
    numpy, scipy
    fase_v23_g1a_coordinate_gate.py
    fase_v23_g1a_constant_refine.py
"""
from __future__ import annotations

import argparse
import copy
import json
import math
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
from scipy.optimize import minimize_scalar

import fase_v23_g1a_coordinate_gate as base
import fase_v23_g1a_constant_refine as cref

NEG = -1.0e18


@dataclass
class CVJointRefineConfig:
    enabled: bool = True
    inner_folds: int = 3
    grid_radius: float = 4.0
    grid_steps: int = 33
    bounded_iters: int = 64
    max_sweeps: int = 12
    move_tol: float = 1e-3
    min_cv_improve: float = 1e-8
    ridge_alpha: float = 1e-4
    max_unsafe_frac: float = 0.02
    snap_tol: float = 0.0
    seed: int = 42


@dataclass
class V23RefinedConfig:
    base: base.GateConfig
    refine: CVJointRefineConfig
    nondegrade_tolerance: float = 1e-4
    promote_parameterized_templates: bool = True
    template_top_k: int = 1


def _cv_splits(n: int, k: int, seed: int) -> List[Tuple[np.ndarray, np.ndarray]]:
    folds = base.kfold_indices(n, max(2, k), seed)
    all_idx = np.arange(n)
    return [(np.setdiff1d(all_idx, va), va) for va in folds]


def _score_exprs_cv(
    exprs: Sequence[base.Expr],
    X: np.ndarray,
    y: np.ndarray,
    splits: Sequence[Tuple[np.ndarray, np.ndarray]],
    alpha: float,
) -> float:
    scores: List[float] = []
    for tr, va in splits:
        try:
            Ztr, fit = base.fit_design(list(exprs), X[tr])
            Zva = fit.transform(X[va])
            w, b0 = base.ridge_fit(Ztr, y[tr], alpha)
            pred = base.ridge_predict(Zva, w, b0)
            if not np.all(np.isfinite(pred)):
                return NEG
            scores.append(float(base.r2_score(y[va], pred)))
        except (FloatingPointError, ValueError, np.linalg.LinAlgError):
            return NEG
    return float(np.mean(scores)) if scores else NEG


def _valid_expr(expr: base.Expr, X: np.ndarray, cfg: CVJointRefineConfig) -> bool:
    if not cref.sqrt_args_valid(expr, X, cfg.max_unsafe_frac):
        return False
    try:
        v = np.asarray(base.eval_expr(expr, X), float)
    except Exception:
        return False
    if not np.all(np.isfinite(v)):
        return False
    v = base.sanitize(v)
    if float(np.std(v)) < 1e-10:
        return False
    if float(np.quantile(np.abs(v), 0.995)) > 1e8:
        return False
    return True


def _bounded_best(objective, c0: float, cfg: CVJointRefineConfig) -> Tuple[float, float]:
    lo, hi = c0 - cfg.grid_radius, c0 + cfg.grid_radius
    grid = np.linspace(lo, hi, cfg.grid_steps)
    vals = np.asarray([objective(float(c)) for c in grid], float)
    if not np.any(vals > NEG / 2):
        return float(c0), float(objective(c0))
    k = int(np.argmax(vals))
    best_c, best_v = float(grid[k]), float(vals[k])
    step = (hi - lo) / max(1, cfg.grid_steps - 1)
    a, b = max(lo, best_c - step), min(hi, best_c + step)
    try:
        res = minimize_scalar(
            lambda c: -objective(float(c)),
            bounds=(a, b),
            method="bounded",
            options={"maxiter": cfg.bounded_iters, "xatol": 1e-5},
        )
        if res.success:
            c2 = float(res.x)
            v2 = float(objective(c2))
            if v2 > best_v:
                best_c, best_v = c2, v2
    except Exception:
        pass
    return best_c, best_v


def refine_expression_joint_cv(
    expr: base.Expr,
    context: Sequence[base.Expr],
    X: np.ndarray,
    y: np.ndarray,
    cfg: CVJointRefineConfig,
) -> Tuple[base.Expr, Dict[str, object]]:
    """Joint coordinate-descent over every constant using mean inner-CV R².

    All constants participate because absorbability is context-dependent after
    composition.  A constant that appears locally affine can become shape-bearing
    after multiplication, division, or another nonlinear composition.
    """
    consts0 = cref.enumerate_consts(expr)
    splits = _cv_splits(len(y), cfg.inner_folds, cfg.seed)
    before = _score_exprs_cv(list(context) + [expr], X, y, splits, cfg.ridge_alpha)
    if not cfg.enabled or not consts0:
        return expr, {
            "expr_before": str(expr), "expr_after": str(expr),
            "consts_before": consts0, "consts_after": consts0,
            "cv_r2_before": before, "cv_r2_after": before,
            "accepted": False, "sweeps": 0,
        }

    cur = expr
    cur_score = before
    sweeps = 0
    for sweep in range(cfg.max_sweeps):
        sweeps = sweep + 1
        max_move = 0.0
        changed = False
        n_consts = len(cref.enumerate_consts(cur))
        for ci in range(n_consts):
            c0 = float(cref.enumerate_consts(cur)[ci])

            def objective(c: float, _cur: base.Expr = cur, _ci: int = ci) -> float:
                trial = cref.with_const(_cur, _ci, float(c))
                if not _valid_expr(trial, X, cfg):
                    return NEG
                return _score_exprs_cv(list(context) + [trial], X, y, splits, cfg.ridge_alpha)

            cstar, score_star = _bounded_best(objective, c0, cfg)
            if abs(cstar - c0) >= cfg.move_tol and score_star > cur_score + cfg.min_cv_improve:
                cur = cref.with_const(cur, ci, cstar)
                max_move = max(max_move, abs(cstar - c0))
                cur_score = score_star
                changed = True
        if not changed or max_move < cfg.move_tol:
            break

    # Optional MDL-style integer snap. Disabled by default.
    if cfg.snap_tol > 0 and cref.enumerate_consts(cur):
        for ci, c0 in enumerate(list(cref.enumerate_consts(cur))):
            cr = float(round(c0))
            if abs(cr - c0) > 0.25:
                continue
            trial = cref.with_const(cur, ci, cr)
            if not _valid_expr(trial, X, cfg):
                continue
            score = _score_exprs_cv(list(context) + [trial], X, y, splits, cfg.ridge_alpha)
            if score >= cur_score - cfg.snap_tol:
                cur, cur_score = trial, score

    accepted = bool(str(cur) != str(expr) and cur_score >= before + cfg.min_cv_improve)
    if not accepted:
        cur, cur_score = expr, before
    return cur, {
        "expr_before": str(expr), "expr_after": str(cur),
        "consts_before": [float(c) for c in consts0],
        "consts_after": [float(c) for c in cref.enumerate_consts(cur)],
        "cv_r2_before": float(before), "cv_r2_after": float(cur_score),
        "accepted": accepted, "sweeps": sweeps,
    }


def refine_selected_nested_cv(
    selected: List[base.SelectedCoord],
    base_context: Sequence[base.Expr],
    X: np.ndarray,
    y: np.ndarray,
    cfg: CVJointRefineConfig,
) -> Tuple[List[base.SelectedCoord], List[Dict[str, object]]]:
    selected = copy.deepcopy(selected)
    exprs = [s.expr for s in selected]
    records: List[Dict[str, object]] = []
    # Sweep across coordinates because constants in different coordinates can couple
    # through the linear prediction head.
    for outer_sweep in range(2):
        any_change = False
        for i in range(len(exprs)):
            context = list(base_context) + [exprs[j] for j in range(len(exprs)) if j != i]
            refined, rec = refine_expression_joint_cv(exprs[i], context, X, y, cfg)
            rec["coordinate_index"] = i
            rec["coordinate_sweep"] = outer_sweep + 1
            records.append(rec)
            if str(refined) != str(exprs[i]):
                exprs[i] = refined
                any_change = True
        if not any_change:
            break
    for s, e in zip(selected, exprs):
        s.expr = e
        s.complexity = e.complexity()
    return selected, records



def _parameterized_templates(d: int, cfg: base.GateConfig) -> List[base.Expr]:
    """Finite topology proposals whose constants are subsequently joint-refined.

    This is not a Viète oracle: it is the general radical-product family admitted by
    the licensed A0r mode, sqrt(affine(x_i))*affine(x_j).  The exact variables and
    constants are selected by nested validation.
    """
    if not cfg.include_sqrt:
        return []
    # Compact seed grid; continuous refinement is responsible for the final values.
    shifts = (-1.0, -0.5, 0.0, 0.5, 1.0, 2.0)
    out: List[base.Expr] = []
    seen = set()
    for i in range(d):
        for j in range(d):
            for c in shifts:
                for s in shifts:
                    e = base.Mul(base.Sqrt(base.Add(base.X(i), base.C(c))),
                                 base.Add(base.X(j), base.C(s)))
                    if e.key() not in seen:
                        seen.add(e.key()); out.append(e)
    return out


def promote_best_template(
    context: Sequence[base.Expr],
    X: np.ndarray, y: np.ndarray,
    base_cfg: base.GateConfig, refine_cfg: CVJointRefineConfig,
    top_k: int = 1,
) -> Tuple[Optional[base.Expr], Dict[str, object]]:
    templates = _parameterized_templates(X.shape[1], base_cfg)
    if not templates:
        return None, {"tested": 0, "accepted": False}
    # Cheap single holdout ranks topology seeds; only the best few receive the
    # expensive nested-CV continuous refinement.
    idx = np.arange(len(y))
    tr, va = base.split_inner(idx, 0.25, refine_cfg.seed + 717)
    def holdout_score(exprs):
        try:
            Ztr, fit = base.fit_design(list(exprs), X[tr])
            Zva = fit.transform(X[va])
            w,b0 = base.ridge_fit(Ztr, y[tr], refine_cfg.ridge_alpha)
            return float(base.r2_score(y[va], base.ridge_predict(Zva,w,b0)))
        except Exception:
            return NEG
    base_score = holdout_score(context)
    ranked=[]
    for e in templates:
        if not _valid_expr(e, X[tr], refine_cfg):
            continue
        ranked.append((holdout_score(list(context)+[e]),e))
    ranked.sort(key=lambda z:z[0], reverse=True)
    best_expr=None; best_score=base_score; best_record=None
    for _,e in ranked[:max(1,top_k)]:
        refined,rec=refine_expression_joint_cv(e,context,X,y,refine_cfg)
        sc=float(rec["cv_r2_after"])
        base_cv=float(rec["cv_r2_before"])
        if sc>base_cv+refine_cfg.min_cv_improve and sc>best_score:
            best_expr,best_score,best_record=refined,sc,rec
    return best_expr,{
        "tested":len(ranked),"base_holdout_r2":float(base_score),
        "best_cv_r2":float(best_score),"accepted":best_expr is not None,
        "best_expression":str(best_expr) if best_expr is not None else None,
        "refinement":best_record,
        "top_initial":[{"score":float(sc),"expr":str(e)} for sc,e in ranked[:top_k]],
    }

def _fit_predict(exprs: Sequence[base.Expr], Xtr: np.ndarray, ytr: np.ndarray, Xva: np.ndarray, alpha: float) -> np.ndarray:
    Ztr, fit = base.fit_design(list(exprs), Xtr)
    Zva = fit.transform(Xva)
    w, b0 = base.ridge_fit(Ztr, ytr, alpha)
    return base.ridge_predict(Zva, w, b0)


def _broad_vieta_audit(exprs: Sequence[base.Expr]) -> Dict[str, object]:
    if not exprs:
        return {"best_corr": 0.0, "best_expr": None, "constants": []}
    a = np.linspace(-1.45, 60.0, 4000)
    p = np.linspace(0.2, 2.0, 4000)
    X = np.column_stack([a, p])
    truth = p * np.sqrt(a + 2.0)
    best = (-1.0, None, [])
    all_exprs: List[base.Expr] = []
    seen = set()
    for e in exprs:
        for s in e.subexpressions():
            if s.op == "const" or s.key() in seen:
                continue
            seen.add(s.key())
            all_exprs.append(s)
    for e in all_exprs:
        try:
            v = base.sanitize(base.eval_expr(e, X))
            c = base.corr_abs(v, truth)
        except Exception:
            continue
        if c > best[0]:
            best = (c, str(e), [float(x) for x in cref.enumerate_consts(e)])
    return {"best_corr": float(max(0.0, best[0])), "best_expr": best[1], "constants": best[2]}


def run_champion_challenger(
    X: np.ndarray,
    y: np.ndarray,
    hidden: Optional[Dict[str, np.ndarray]],
    cfg: V23RefinedConfig,
    task_name: str = "custom",
) -> Dict[str, object]:
    t0 = time.time()
    X = np.asarray(X, float)
    y = np.asarray(y, float).reshape(-1)
    n = len(y)
    folds = base.kfold_indices(n, cfg.base.k_folds, cfg.base.seed)
    pred_base = np.zeros(n)
    pred_ref = np.zeros(n)
    selected_base_all: List[base.SelectedCoord] = []
    selected_ref_all: List[base.SelectedCoord] = []
    fold_rows: List[Dict[str, object]] = []

    for fold_id, outer_va in enumerate(folds, start=1):
        outer_tr = np.setdiff1d(np.arange(n), outer_va)
        inner_tr, inner_va = base.split_inner(
            outer_tr, cfg.base.inner_val_frac, cfg.base.seed + 1000 * fold_id
        )
        selected, info = base.discover_coordinates(X[inner_tr], y[inner_tr], X[inner_va], y[inner_va], cfg.base)
        base_context = base.base_exprs(X.shape[1], cfg.base)
        base_expr_list = base_context + [s.expr for s in selected]
        pred_base[outer_va] = _fit_predict(base_expr_list, X[outer_tr], y[outer_tr], X[outer_va], cfg.base.ridge_alpha)

        rcfg = copy.deepcopy(cfg.refine)
        rcfg.seed = cfg.base.seed + 10000 * fold_id
        refined, records = refine_selected_nested_cv(selected, base_context, X[outer_tr], y[outer_tr], rcfg)
        template_record = {"tested": 0, "accepted": False}
        if cfg.promote_parameterized_templates:
            current_context = base_context + [s.expr for s in refined]
            promoted_expr, template_record = promote_best_template(
                current_context, X[outer_tr], y[outer_tr], cfg.base, rcfg, cfg.template_top_k
            )
            if promoted_expr is not None:
                refined.append(base.SelectedCoord(
                    expr=promoted_expr, layer=cfg.base.max_layers + 1,
                    val_gain=float(template_record["best_cv_r2"] - template_record.get("base_holdout_r2", template_record["best_cv_r2"])),
                    val_r2_after=float(template_record["best_cv_r2"]),
                    complexity=promoted_expr.complexity(), residual_alignment=0.0,
                ))
        ref_expr_list = base_context + [s.expr for s in refined]
        pred_ref[outer_va] = _fit_predict(ref_expr_list, X[outer_tr], y[outer_tr], X[outer_va], max(cfg.base.ridge_alpha, rcfg.ridge_alpha))

        selected_base_all.extend(copy.deepcopy(selected))
        selected_ref_all.extend(copy.deepcopy(refined))
        fold_rows.append({
            "fold": fold_id,
            "baseline_R2": float(base.r2_score(y[outer_va], pred_base[outer_va])),
            "refined_R2": float(base.r2_score(y[outer_va], pred_ref[outer_va])),
            "baseline_selected": [str(s.expr) for s in selected],
            "refined_selected": [str(s.expr) for s in refined],
            "refinement_records": records,
            "template_promotion": template_record,
            "discovery_trace": info["trace"],
            "vieta_broad_audit": _broad_vieta_audit([s.expr for s in refined]) if task_name == "vieta" else None,
        })

    r2b = float(base.r2_score(y, pred_base))
    r2r = float(base.r2_score(y, pred_ref))
    winner = "refined" if r2r >= r2b - cfg.nondegrade_tolerance else "baseline"
    out = {
        "protocol": "FASE-v23.1 champion/challenger joint-constant refinement",
        "task": task_name,
        "non_degrading_design": "original v23 branch retained; refined branch reported separately",
        "n": int(n), "d": int(X.shape[1]), "k_folds": cfg.base.k_folds,
        "baseline": {
            "R2_oof": r2b,
            "MSE_oof": float(base.mse(y, pred_base)),
            "top_selected_expressions": base.top_expression_counts(selected_base_all, 12),
            "coordinate_recovery": base.coordinate_recovery(X, [s.expr for s in selected_base_all], hidden or {}) if hidden else {},
        },
        "refined": {
            "R2_oof": r2r,
            "MSE_oof": float(base.mse(y, pred_ref)),
            "top_selected_expressions": base.top_expression_counts(selected_ref_all, 12),
            "coordinate_recovery": base.coordinate_recovery(X, [s.expr for s in selected_ref_all], hidden or {}) if hidden else {},
            "vieta_broad_audit": _broad_vieta_audit([s.expr for s in selected_ref_all]) if task_name == "vieta" else None,
        },
        "recommended_branch": winner,
        "nondegrade_pass": bool(r2r >= r2b - cfg.nondegrade_tolerance),
        "folds": fold_rows,
        "config": {"base": cfg.base.__dict__, "refine": asdict(cfg.refine), "nondegrade_tolerance": cfg.nondegrade_tolerance, "promote_parameterized_templates": cfg.promote_parameterized_templates, "template_top_k": cfg.template_top_k},
        "elapsed_sec": float(time.time() - t0),
    }
    return out


def make_config(seed: int, quick: bool, include_sqrt: bool = True) -> V23RefinedConfig:
    b = base.GateConfig(
        seed=seed,
        k_folds=3 if quick else 5,
        max_layers=2 if quick else 3,
        accept_per_layer=2,
        candidate_depth=3 if include_sqrt else 2,
        max_candidates_scored=260 if quick else 1800,
        depth1_beam=40 if quick else 100,
        include_sqrt=include_sqrt,
        ridge_alpha=1e-4,
    )
    r = CVJointRefineConfig(
        inner_folds=2 if quick else 3,
        grid_steps=17 if quick else 33,
        bounded_iters=32 if quick else 64,
        max_sweeps=4 if quick else 10,
        ridge_alpha=1e-4,
        seed=seed,
    )
    return V23RefinedConfig(base=b, refine=r)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--test", choices=list(base.DATASETS.keys()), default="vieta")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--n", type=int, default=160)
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--strict-a0", action="store_true")
    ap.add_argument("--snap-tol", type=float, default=0.0)
    ap.add_argument("--json-out", type=str, default=None)
    args = ap.parse_args()

    cfg = make_config(args.seed, args.quick, include_sqrt=not args.strict_a0)
    cfg.refine.snap_tol = float(args.snap_tol)
    X, y, hidden = base.DATASETS[args.test](seed=args.seed, n=args.n)
    out = run_champion_challenger(X, y, hidden, cfg, task_name=args.test)
    text = json.dumps(out, indent=2, allow_nan=False)
    print(text)
    if args.json_out:
        path = Path(args.json_out)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text, encoding="utf-8")


if __name__ == "__main__":
    main()
