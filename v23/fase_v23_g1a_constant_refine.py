#!/usr/bin/env python3
"""
FASE-G1a-v23 Constant Refinement
================================

Additive module (does not replace the base gate). It closes the diagnosed gap on
the Viete killing field: FASE-A0r recovers the *form* of the radical coordinate
(`sqrt(x0 + c) * affine(x1)`) but selects the wrong internal constant `c` off the
fixed grid {-2,-1,-0.5,0.5,1,2}, because over the bounded training domain the whole
family {sqrt(x0+c)} is ~0.99 mutually correlated and the residual-greedy linear-head
selector cannot resolve which c is correct. PySR closes the same gap because its
inner loop tunes real-valued constants by local optimization.

This module gives FASE the missing step: after a coordinate is accepted, run a 1-D
continuous search on each *non-absorbable* constant (one constant at a time,
coordinate-descent style), holding every other feature fixed.

Leakage discipline
------------------
Refinement is performed at the same nesting level as candidate selection:
    - the constant is OPTIMIZED on the inner-train split (X_tr, y_tr);
    - the refinement is ACCEPTED only if it does not hurt the inner-val split
      (X_va, y_va), mirroring how the discrete candidate was accepted.
The outer OOF fold is never touched here, so R2_oof remains leakage-free. The
inner-val gate is also what bounds the extra capacity of a continuous constant
versus a grid constant: refine on train, keep only if val agrees.

Which constants are refined
---------------------------
A constant is refinable iff perturbing it changes the *shape* of the standardized
feature (correlation with the unperturbed feature drops below 1). Pure scaling /
top-level shift constants are absorbed by feature standardization + the linear head,
so they are skipped. A constant inside `sqrt(...)` is the canonical refinable case.
This refinability test is itself a small perturbation probe, which is the same
identifiability logic the domain-shift audit uses, applied locally.

Dependencies: numpy + the base module fase_v23_g1a_coordinate_gate.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np

import fase_v23_g1a_coordinate_gate as base

NEG_INF = -1e18


@dataclass
class RefineConfig:
    enabled: bool = True
    grid_radius: float = 4.0        # search c in [c0 - R, c0 + R], clipped by sqrt-domain validity
    grid_steps: int = 41            # coarse grid resolution (global-ish bracket)
    golden_iters: int = 40          # golden-section iterations (precision, ~1e-4)
    max_unsafe_frac: float = 0.02   # max fraction of clipped sqrt args allowed (domain guard)
    min_val_improve: float = 0.0    # keep refinement only if inner-val R2 >= baseline + this
    n_sweeps: int = 2               # 1-D coordinate-descent passes over refinable constants
    snap_tol: float = 0.0           # if >0, snap c* to nearest integer when val-R2 within tol (MDL prior, off by default)
    probe_corr_thresh: float = 0.99995  # below this corr under a probe => constant is shape-bearing (refinable)
    ridge_alpha: float = 1e-6
    # --- joint refinement (coupled constants): converge, don't run a fixed tiny count ---
    joint_max_sweeps: int = 12      # cap on joint coordinate-descent sweeps
    joint_move_tol: float = 1e-3    # stop sweeping once the largest per-constant move drops below this;
                                    #   also the per-constant acceptance threshold, so the search freezes
                                    #   on the flat val-R2 plateau instead of noise-chasing past the optimum


# --------------------------------------------------------------------------------------
# Constant enumeration / replacement on the frozen Expr tree (pre-order indexing)
# --------------------------------------------------------------------------------------

def enumerate_consts(expr: base.Expr) -> List[float]:
    out: List[float] = []

    def walk(e: Optional[base.Expr]) -> None:
        if e is None:
            return
        if e.op == "const":
            out.append(float(e.const))
            return
        walk(e.left)
        walk(e.right)

    walk(expr)
    return out


def with_const(expr: base.Expr, target_index: int, new_value: float) -> base.Expr:
    counter = {"i": 0}

    def rebuild(e: Optional[base.Expr]) -> Optional[base.Expr]:
        if e is None:
            return None
        if e.op == "const":
            i = counter["i"]
            counter["i"] += 1
            if i == target_index:
                return base.Expr("const", const=float(new_value))
            return e
        nl = rebuild(e.left)
        nr = rebuild(e.right)
        return base.Expr(e.op, left=nl, right=nr, idx=e.idx, const=e.const, weights=e.weights)

    return rebuild(expr)


def shielded_const_indices(expr: base.Expr) -> List[int]:
    """Pre-order indices of constants whose effect a linear head CANNOT absorb.

    A constant is shielded iff its path to the root passes through a `sqrt`, or it
    lies inside the denominator of a `div`. Top-level additive/multiplicative
    constants are absorbed by feature standardization + the linear head, so refining
    them is at best a no-op and at worst noise-chasing; they are excluded here.
    """
    idxs: List[int] = []
    counter = {"i": 0}

    def walk(e: Optional[base.Expr], shielded: bool) -> None:
        if e is None:
            return
        if e.op == "const":
            i = counter["i"]
            counter["i"] += 1
            if shielded:
                idxs.append(i)
            return
        if e.op == "sqrt":
            walk(e.left, True)
            return
        if e.op == "div":
            walk(e.left, shielded)      # numerator keeps current shield status
            walk(e.right, True)         # denominator is nonlinear -> shielded
            return
        walk(e.left, shielded)
        walk(e.right, shielded)

    walk(expr, False)
    return idxs


def _standardize(v: np.ndarray) -> np.ndarray:
    v = np.asarray(v, float)
    sd = v.std()
    if sd < 1e-12:
        return v - v.mean()
    return (v - v.mean()) / sd


def _residualize(v: np.ndarray, Zc: np.ndarray) -> np.ndarray:
    """Remove the part of v explainable by the context design Zc (mean-centered)."""
    vc = v - v.mean()
    if Zc.shape[1] == 0:
        return vc
    coef, _, _, _ = np.linalg.lstsq(Zc, vc, rcond=1e-10)
    return vc - Zc @ coef


def fwl_contribution(Zc: np.ndarray, resid_y: np.ndarray, v: np.ndarray) -> float:
    """Marginal R2 contribution of feature v beyond the context design (FWL)."""
    rf = _residualize(np.asarray(v, float), Zc)
    return base.corr_abs(resid_y, rf) ** 2


def sqrt_args_valid(expr: base.Expr, X: np.ndarray, max_unsafe_frac: float) -> bool:
    ok = [True]

    def walk(e: Optional[base.Expr]) -> None:
        if e is None or not ok[0]:
            return
        if e.op == "sqrt":
            arg = np.asarray(base.eval_expr(e.left, X), float)
            if float(np.mean(arg < 0.0)) > max_unsafe_frac:
                ok[0] = False
            walk(e.left)
            return
        walk(e.left)
        walk(e.right)

    walk(expr)
    return ok[0]


# --------------------------------------------------------------------------------------
# Objective helpers (linear head re-fit, same normalization discipline as the base gate)
# --------------------------------------------------------------------------------------

def _val_r2_of(exprs: List[base.Expr], X_tr: np.ndarray, y_tr: np.ndarray,
               X_va: np.ndarray, y_va: np.ndarray, alpha: float) -> float:
    if not exprs:
        b0 = float(np.mean(y_tr))
        return base.r2_score(y_va, np.full(len(y_va), b0))
    Ztr, fit = base.fit_design(exprs, X_tr)
    Zva = fit.transform(X_va)
    w, b0 = base.ridge_fit(Ztr, y_tr, alpha)
    return base.r2_score(y_va, base.ridge_predict(Zva, w, b0))


def _train_r2_of(exprs: List[base.Expr], X_tr: np.ndarray, y_tr: np.ndarray, alpha: float) -> float:
    if not exprs:
        return 0.0
    Ztr, fit = base.fit_design(exprs, X_tr)
    w, b0 = base.ridge_fit(Ztr, y_tr, alpha)
    return base.r2_score(y_tr, base.ridge_predict(Ztr, w, b0))


def is_refinable(expr: base.Expr, ci: int, X: np.ndarray, rcfg: RefineConfig) -> bool:
    """A constant is refinable iff a finite probe changes the standardized feature's shape.

    Pure scale/shift constants are absorbed by standardization + the linear head and
    leave the feature affine-equivalent (corr == 1), so they are skipped.
    """
    consts = enumerate_consts(expr)
    c0 = consts[ci]
    f0 = base.sanitize(base.eval_expr(expr, X))
    if f0.std() < 1e-10:
        return False
    delta = max(0.5, 0.25 * (1.0 + abs(c0)))
    for d in (delta, -delta):
        trial = with_const(expr, ci, c0 + d)
        if not sqrt_args_valid(trial, X, max(0.05, rcfg.max_unsafe_frac)):
            continue
        f1 = base.sanitize(base.eval_expr(trial, X))
        if f1.std() < 1e-10:
            return True
        if base.corr_abs(f0, f1) < rcfg.probe_corr_thresh:
            return True
    return False


# --------------------------------------------------------------------------------------
# 1-D search: coarse grid bracket + golden-section refinement
# --------------------------------------------------------------------------------------

def _golden_max(f, a: float, b: float, iters: int) -> float:
    gr = (math.sqrt(5.0) - 1.0) / 2.0
    c = b - gr * (b - a)
    d = a + gr * (b - a)
    fc, fd = f(c), f(d)
    for _ in range(iters):
        if fc < fd:
            a, c, fc = c, d, fd
            d = a + gr * (b - a)
            fd = f(d)
        else:
            b, d, fd = d, c, fc
            c = b - gr * (b - a)
            fc = f(c)
    return 0.5 * (a + b)


def refine_one_constant(coord: base.Expr, ci: int, objective, c0: float, rcfg: RefineConfig) -> float:
    """1-D search (coarse grid bracket + golden section) maximizing `objective(c)`.

    `objective` already encapsulates validity + the FWL marginal contribution, and
    must return NEG_INF for invalid c. This keeps the search well-conditioned and
    measures only the part of the constant's effect the linear head cannot absorb.
    """
    lo, hi = c0 - rcfg.grid_radius, c0 + rcfg.grid_radius
    grid = np.linspace(lo, hi, rcfg.grid_steps)
    vals = np.array([objective(float(c)) for c in grid])
    if not np.any(vals > NEG_INF / 2):
        return c0
    k = int(np.argmax(vals))
    cbest = float(grid[k])
    step = (hi - lo) / (rcfg.grid_steps - 1)
    a, b = max(lo, cbest - step), min(hi, cbest + step)
    cstar = _golden_max(objective, a, b, rcfg.golden_iters)
    if objective(cstar) < objective(cbest):
        cstar = cbest
    if rcfg.snap_tol > 0.0:
        cr = float(round(cstar))
        if abs(cr - cstar) < 0.25 and objective(cr) >= objective(cstar) - rcfg.snap_tol:
            cstar = cr
    return float(cstar)


# --------------------------------------------------------------------------------------
# Entry point: refine all selected coordinates (val-gated, leakage-safe)
# --------------------------------------------------------------------------------------

def refine_coordinate_joint(
    coord: base.Expr,
    context: List[base.Expr],
    X_tr: np.ndarray, y_tr: np.ndarray,
    X_va: np.ndarray, y_va: np.ndarray,
    rcfg: "RefineConfig",
) -> Tuple[base.Expr, float, float]:
    """Jointly refine ALL constants in `coord` by coordinate descent on held-out val R2.

    Unlike the 1-D step, this disentangles coupled constants (e.g. the radical shift
    and the x1 shift in the Viete product coordinate `sqrt(x0+c)*(x1-s)`), which a
    one-at-a-time search bends against each other. The objective is the held-out
    inner-val R2 of the full design, which is what actually identifies the true
    structure; train-residual alignment does not, because the target is a product.

    Returns (refined_coord, val_before, val_after). The caller applies the accept gate.
    """
    n_consts = len(enumerate_consts(coord))
    val_before = _val_r2_of(context + [coord], X_tr, y_tr, X_va, y_va, rcfg.ridge_alpha)
    if n_consts == 0:
        return coord, val_before, val_before
    cur = coord
    max_sweeps = max(1, getattr(rcfg, "joint_max_sweeps", rcfg.n_sweeps))
    move_tol = getattr(rcfg, "joint_move_tol", 1e-3)
    for _ in range(max_sweeps):
        max_move = 0.0
        for ci in range(n_consts):
            c0 = enumerate_consts(cur)[ci]

            def objective(c: float, _ci: int = ci, _cur: base.Expr = cur) -> float:
                trial = with_const(_cur, _ci, c)
                if not sqrt_args_valid(trial, X_tr, rcfg.max_unsafe_frac):
                    return NEG_INF
                vtr = base.sanitize(base.eval_expr(trial, X_tr))
                if vtr.std() < 1e-10 or not np.all(np.isfinite(vtr)):
                    return NEG_INF
                return _val_r2_of(context + [trial], X_tr, y_tr, X_va, y_va, rcfg.ridge_alpha)

            cstar = refine_one_constant(cur, ci, objective, c0, rcfg)
            # Accept only moves above the movement tolerance. This both (a) lets the
            # descent walk the curved c-by-c ridge over successive sweeps, and (b)
            # freezes the search on the flat val-R2 plateau near the optimum instead
            # of noise-chasing tiny accepted steps past the true constant.
            if abs(cstar - c0) >= move_tol and objective(cstar) > objective(c0):
                cur = with_const(cur, ci, cstar)
                max_move = max(max_move, abs(cstar - c0))
        if max_move < move_tol:
            break
    val_after = _val_r2_of(context + [cur], X_tr, y_tr, X_va, y_va, rcfg.ridge_alpha)
    return cur, float(val_before), float(val_after)


def refine_selected_constants_joint(
    selected: List["base.SelectedCoord"],
    context_base_exprs: List[base.Expr],
    X_tr: np.ndarray, y_tr: np.ndarray,
    X_va: np.ndarray, y_va: np.ndarray,
    rcfg: Optional["RefineConfig"] = None,
) -> Tuple[List["base.SelectedCoord"], List[Dict[str, object]]]:
    """Joint-refine every accepted coordinate, accept per-coordinate iff inner-val improves.

    Leakage discipline is identical to the 1-D entry point: constants are tuned and
    accepted using only this fold's inner train/val; the outer fold is untouched.
    Recommended global commit rule (not enforced here): only adopt a refined constant
    across the run if it is stable across folds (cross-fold stability selection), which
    bounds the extra capacity of continuous constants at small n / high noise.
    """
    if rcfg is None:
        rcfg = RefineConfig()
    records: List[Dict[str, object]] = []
    if not rcfg.enabled or not selected:
        return selected, records
    exprs = [s.expr for s in selected]
    base_list = list(context_base_exprs)
    for k in range(len(exprs)):
        coord = exprs[k]
        context = base_list + [exprs[j] for j in range(len(exprs)) if j != k]
        refined, v0, v1 = refine_coordinate_joint(coord, context, X_tr, y_tr, X_va, y_va, rcfg)
        if str(refined) != str(coord) and v1 >= v0 + rcfg.min_val_improve:
            records.append({
                "coord_before": str(coord), "coord_after": str(refined),
                "consts_before": enumerate_consts(coord), "consts_after": enumerate_consts(refined),
                "val_r2_before": v0, "val_r2_after": v1,
            })
            exprs[k] = refined
    refined_sel: List["base.SelectedCoord"] = []
    for k, s in enumerate(selected):
        coord = exprs[k]
        context = base_list + [exprs[j] for j in range(len(exprs)) if j != k]
        val_after = _val_r2_of(context + [coord], X_tr, y_tr, X_va, y_va, rcfg.ridge_alpha)
        val_ctx = _val_r2_of(context, X_tr, y_tr, X_va, y_va, rcfg.ridge_alpha)
        s.expr = coord
        s.val_r2_after = float(val_after)
        s.val_gain = float(val_after - val_ctx)
        s.complexity = coord.complexity()
        refined_sel.append(s)
    return refined_sel, records


def refine_selected_constants(
    selected: List["base.SelectedCoord"],
    context_base_exprs: List[base.Expr],
    X_tr: np.ndarray,
    y_tr: np.ndarray,
    X_va: np.ndarray,
    y_va: np.ndarray,
    rcfg: Optional[RefineConfig] = None,
) -> Tuple[List["base.SelectedCoord"], List[Dict[str, object]]]:
    if rcfg is None:
        rcfg = RefineConfig()
    records: List[Dict[str, object]] = []
    if not rcfg.enabled or not selected:
        return selected, records

    exprs = [s.expr for s in selected]
    base_list = list(context_base_exprs)

    for sweep in range(rcfg.n_sweeps):
        changed = False
        for k in range(len(exprs)):
            coord = exprs[k]
            shielded = shielded_const_indices(coord)
            if not shielded:
                continue
            context = base_list + [exprs[j] for j in range(len(exprs)) if j != k]
            Zc, _ = base.fit_design(context, X_tr)
            resid_y = _residualize(np.asarray(y_tr, float), Zc)
            base_val = _val_r2_of(context + [coord], X_tr, y_tr, X_va, y_va, rcfg.ridge_alpha)
            new_coord = coord
            for ci in shielded:
                c0 = enumerate_consts(new_coord)[ci]

                def objective(c: float, _ci: int = ci, _coord: base.Expr = new_coord) -> float:
                    trial = with_const(_coord, _ci, c)
                    if not sqrt_args_valid(trial, X_tr, rcfg.max_unsafe_frac):
                        return NEG_INF
                    v = base.sanitize(base.eval_expr(trial, X_tr))
                    if v.std() < 1e-10 or not np.all(np.isfinite(v)):
                        return NEG_INF
                    return fwl_contribution(Zc, resid_y, v)

                cstar = refine_one_constant(new_coord, ci, objective, c0, rcfg)
                if abs(cstar - c0) < 1e-6:
                    continue
                trial_coord = with_const(new_coord, ci, cstar)
                trial_val = _val_r2_of(context + [trial_coord], X_tr, y_tr, X_va, y_va, rcfg.ridge_alpha)
                if trial_val >= base_val + rcfg.min_val_improve:
                    records.append({
                        "coord_before": str(new_coord),
                        "coord_after": str(trial_coord),
                        "const_index": ci,
                        "c_from": float(c0),
                        "c_to": float(cstar),
                        "val_r2_before": float(base_val),
                        "val_r2_after": float(trial_val),
                        "sweep": sweep + 1,
                    })
                    new_coord = trial_coord
                    base_val = trial_val
                    changed = True
            exprs[k] = new_coord
        if not changed:
            break

    refined: List["base.SelectedCoord"] = []
    for k, s in enumerate(selected):
        coord = exprs[k]
        context = base_list + [exprs[j] for j in range(len(exprs)) if j != k]
        val_after = _val_r2_of(context + [coord], X_tr, y_tr, X_va, y_va, rcfg.ridge_alpha)
        val_ctx = _val_r2_of(context, X_tr, y_tr, X_va, y_va, rcfg.ridge_alpha)
        s.expr = coord
        s.val_r2_after = float(val_after)
        s.val_gain = float(val_after - val_ctx)
        s.complexity = coord.complexity()
        refined.append(s)
    return refined, records
