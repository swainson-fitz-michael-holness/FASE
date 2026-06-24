#!/usr/bin/env python3
"""
FASE-G1a-v23 Hierarchy Gate: Primitive recursion + radical primitive + abstraction audit
======================================================================================

This file extends the first Gate-0.5 artifact rather than replacing it.
It tests Swainson's updated hypothesis:

  1. Low hierarchy: expanded primitive coordinates such as
       x0*x2 - x0*x3 + x1*x2 - x1*x3
     may be acceptable even when they do not expose the intended factored form.

  2. Higher hierarchy: a second audit should derive coordinate operators by
     abstracting the expanded primitive span into reusable factors, e.g.
       (x0 + x1)*(x2 - x3).

  3. Radicals are promoted from optional extension to a first-class primitive
     in this gate, because nested radicals/irrational coordinates cannot be
     obtained exactly from field operations alone.

The point is not to beat PySR. The point is to ask:
  Can primitive recursion discover a useful low-level span, and can a hierarchy
  audit infer higher-level coordinate operators from that span?

Dependencies: numpy only, plus the sibling file fase_v23_g1a_coordinate_gate.py.
"""

from __future__ import annotations

import argparse
import itertools
import json
import math
import time
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

import fase_v23_g1a_coordinate_gate as base

EPS = 1e-9


@dataclass
class HierarchyConfig:
    seed: int = 42
    k_folds: int = 5
    n: int = 700
    # Important change: radicals are now primitive by default.
    include_sqrt: bool = True
    # Need depth 3 for p*sqrt(2+a): sqrt(2+a) has depth 2, product makes depth 3.
    candidate_depth: int = 3
    max_layers: int = 3
    accept_per_layer: int = 3
    max_candidates_scored: int = 2400
    depth1_beam: int = 120
    inner_val_frac: float = 0.25
    ridge_alpha: float = 1e-6
    complexity_penalty: float = 1e-4
    min_gain: float = 1e-4
    redundancy_corr: float = 0.995
    n_random_projects: int = 0


def to_base_config(cfg: HierarchyConfig) -> base.GateConfig:
    return base.GateConfig(
        seed=cfg.seed,
        k_folds=cfg.k_folds,
        inner_val_frac=cfg.inner_val_frac,
        max_layers=cfg.max_layers,
        accept_per_layer=cfg.accept_per_layer,
        min_gain=cfg.min_gain,
        complexity_penalty=cfg.complexity_penalty,
        redundancy_corr=cfg.redundancy_corr,
        candidate_depth=cfg.candidate_depth,
        max_candidates_scored=cfg.max_candidates_scored,
        depth1_beam=cfg.depth1_beam,
        n_random_projects=cfg.n_random_projects,
        include_sqrt=cfg.include_sqrt,
        # Keep 2.0 because sqrt(2+a) is a necessary Viète coordinate.
        constants=(-2.0, -1.0, -0.5, 0.5, 1.0, 2.0),
        ridge_alpha=cfg.ridge_alpha,
    )


def unique_exprs(exprs: Sequence[base.Expr]) -> List[base.Expr]:
    out = []
    seen = set()
    for e in exprs:
        k = e.key()
        if k not in seen:
            seen.add(k)
            out.append(e)
    return out


def span_recovery(X: np.ndarray, exprs: List[base.Expr], hidden: Dict[str, np.ndarray], alpha: float = 1e-6) -> Dict[str, Dict[str, object]]:
    """Recover hidden coordinates from the linear span of discovered expressions.

    This is deliberately different from single-coordinate recovery. It asks whether
    the primitive span contains enough information to reconstruct a hidden coordinate,
    even if the coordinate is represented in expanded form.
    """
    exprs = unique_exprs(exprs)
    if not exprs:
        return {k: {"span_R2": 0.0, "span_corr": 0.0, "n_span_exprs": 0} for k in hidden}
    Z, fit = base.fit_design(exprs, X)
    out: Dict[str, Dict[str, object]] = {}
    for name, z in hidden.items():
        w, b0 = base.ridge_fit(Z, z, alpha)
        zhat = base.ridge_predict(Z, w, b0)
        out[name] = {
            "span_R2": float(base.r2_score(z, zhat)),
            "span_corr": float(base.corr_abs(z, zhat)),
            "n_span_exprs": len(exprs),
        }
    return out


def polynomial_pair_design(X: np.ndarray, include_linear: bool = True, include_square: bool = False) -> Tuple[np.ndarray, List[Tuple[int, int]]]:
    """Pairwise polynomial design. Returns design and monomial keys.

    key (i, j) with i == j means x_i^2 if include_square=True.
    key (-1, i) means linear x_i.
    """
    n, d = X.shape
    cols = []
    keys: List[Tuple[int, int]] = []
    if include_linear:
        for i in range(d):
            cols.append(X[:, i])
            keys.append((-1, i))
    for i in range(d):
        j0 = i if include_square else i + 1
        for j in range(j0, d):
            cols.append(X[:, i] * X[:, j])
            keys.append((i, j))
    return np.column_stack(cols) if cols else np.zeros((n, 0)), keys


def fit_pairwise_coefficients(X: np.ndarray, target: np.ndarray, alpha: float = 1e-8) -> Tuple[Dict[Tuple[int, int], float], float, float]:
    """Fit a low-degree primitive polynomial explanation to a target vector."""
    Phi, keys = polynomial_pair_design(X, include_linear=True, include_square=False)
    # Standardize for numerical solve, then convert coefficients back to raw columns approximately.
    mu = Phi.mean(axis=0, keepdims=True)
    sd = Phi.std(axis=0, keepdims=True)
    sd = np.where(sd < 1e-12, 1.0, sd)
    Phis = (Phi - mu) / sd
    w, b0s = base.ridge_fit(Phis, target, alpha)
    pred = base.ridge_predict(Phis, w, b0s)
    # Raw coefficient for Phi column = standardized coefficient / sd.
    raw_w = w / sd.reshape(-1)
    raw_b0 = float(b0s - (mu.reshape(-1) / sd.reshape(-1) @ w))
    coeffs = {k: float(v) for k, v in zip(keys, raw_w)}
    return coeffs, raw_b0, float(base.r2_score(target, pred))


def coefficient_matrix_from_pairs(coeffs: Dict[Tuple[int, int], float], d: int) -> np.ndarray:
    B = np.zeros((d, d), dtype=float)
    for (i, j), v in coeffs.items():
        if i < 0:
            continue
        if i == j:
            B[i, j] += v
        else:
            # Store off-diagonal as directional but symmetric for scanning. For a cross block,
            # B[A,B] will see the coefficient once.
            B[i, j] += v
            B[j, i] += v
    return B


def format_linear_factor(indices: Sequence[int], coeff: np.ndarray, thr: float = 1e-6) -> str:
    terms = []
    for idx, c in zip(indices, coeff):
        if abs(c) < thr:
            continue
        sign = "+" if c >= 0 else "-"
        mag = abs(c)
        if abs(mag - 1.0) < 5e-2:
            body = f"x{idx}"
        else:
            body = f"{mag:.3g}*x{idx}"
        terms.append((sign, body))
    if not terms:
        return "0"
    s0, b0 = terms[0]
    out = ("-" if s0 == "-" else "") + b0
    for s, b in terms[1:]:
        out += f" {s} {b}"
    return out


def derive_rank1_bilinear_abstractions(
    X: np.ndarray,
    target: np.ndarray,
    hidden: Optional[Dict[str, np.ndarray]] = None,
    alpha: float = 1e-8,
    max_rows: int = 8,
) -> Dict[str, object]:
    """Derive higher-level product coordinates from an expanded bilinear footprint.

    We fit pairwise primitive products to the low-level model's OOF prediction,
    then search variable bipartitions whose cross-coefficient block is near rank-1.
    A rank-1 block corresponds to a factored coordinate:
        (u^T x_A) * (v^T x_B).
    """
    X = np.asarray(X, float)
    n, d = X.shape
    coeffs, intercept, footprint_r2 = fit_pairwise_coefficients(X, target, alpha=alpha)
    B = coefficient_matrix_from_pairs(coeffs, d)
    rows = []
    all_indices = list(range(d))
    # For d up to ~8 this is fine. Avoid duplicate complements by requiring mask < complement.
    for r in range(1, d):
        for A_tuple in itertools.combinations(all_indices, r):
            A = list(A_tuple)
            Bidx = [i for i in all_indices if i not in A]
            if not Bidx:
                continue
            mask = sum(1 << i for i in A)
            comp = sum(1 << i for i in Bidx)
            if mask > comp:
                continue
            Cblock = B[np.ix_(A, Bidx)]
            energy = float(np.sum(Cblock ** 2))
            if energy < 1e-12:
                continue
            U, S, Vt = np.linalg.svd(Cblock, full_matrices=False)
            rank1_ratio = float((S[0] ** 2) / (np.sum(S ** 2) + EPS))
            # Candidate coordinate; scaling is arbitrary, but this preserves coefficient magnitude.
            u = U[:, 0] * math.sqrt(S[0])
            v = Vt[0, :] * math.sqrt(S[0])
            z = (X[:, A] @ u) * (X[:, Bidx] @ v)
            # Use candidate alone as explanation; sign/scale handled by linear fit.
            Z = z.reshape(-1, 1)
            w, b0 = base.ridge_fit(Z, target, alpha=alpha)
            pred = base.ridge_predict(Z, w, b0)
            cand_r2 = float(base.r2_score(target, pred))
            hidden_scores = {}
            if hidden:
                for name, h in hidden.items():
                    hidden_scores[name] = float(base.corr_abs(z, h))
            rows.append({
                "A": A,
                "B": Bidx,
                "rank1_ratio": rank1_ratio,
                "cross_energy": energy,
                "candidate_R2_vs_target": cand_r2,
                "candidate_expr": f"({format_linear_factor(A, u)})*({format_linear_factor(Bidx, v)})",
                "left_coeffs": [float(x) for x in u],
                "right_coeffs": [float(x) for x in v],
                "hidden_abs_corr": hidden_scores,
            })
    rows.sort(key=lambda r: (r["candidate_R2_vs_target"], r["rank1_ratio"], r["cross_energy"]), reverse=True)
    return {
        "footprint_type": "pairwise_primitive_polynomial_from_low_level_prediction",
        "footprint_R2_vs_target": footprint_r2,
        "intercept": intercept,
        "top_rank1_bilinear_abstractions": rows[:max_rows],
    }


def run_hierarchy_gate(X: np.ndarray, y: np.ndarray, hidden: Dict[str, np.ndarray], cfg: HierarchyConfig) -> Dict[str, object]:
    t0 = time.time()
    bcfg = to_base_config(cfg)
    n = len(y)
    folds = base.kfold_indices(n, bcfg.k_folds, bcfg.seed)
    yhat = np.zeros(n)
    all_selected: List[base.SelectedCoord] = []
    fold_summaries = []

    for fold_id, val_idx in enumerate(folds, start=1):
        train_idx = np.setdiff1d(np.arange(n), val_idx)
        inner_train_idx, inner_val_idx = base.split_inner(train_idx, bcfg.inner_val_frac, bcfg.seed + 1000 * fold_id)
        selected, info = base.discover_coordinates(
            X[inner_train_idx], y[inner_train_idx],
            X[inner_val_idx], y[inner_val_idx],
            bcfg,
        )
        # Continuous constant refinement (additive, leakage-safe).
        # Optimizes each coordinate's constants on this fold's inner-train and keeps
        # the result only if this fold's inner-val agrees. The outer val_idx is never
        # touched here, so R2_oof stays leakage-free.
        #
        # JOINT is the default: the Viete product coordinate sqrt(x0+c)*(x1-s) has two
        # coupled constants that a one-at-a-time (1-D) search bends against each other,
        # so 1-D stalls at the wrong radical shift. Joint coordinate-descent on held-out
        # inner-val R2 disentangles them and recovers the law to the precision the data
        # supports. Set cfg.refine_mode="1d" to use the shielded one-at-a-time step.
        refine_records = []
        if getattr(cfg, "refine_constants", True):
            import fase_v23_g1a_constant_refine as refine
            rcfg = refine.RefineConfig(
                enabled=True,
                ridge_alpha=bcfg.ridge_alpha,
                min_val_improve=getattr(cfg, "refine_min_val_improve", 0.0),
                snap_tol=getattr(cfg, "refine_snap_tol", 0.0),
            )
            refine_fn = (refine.refine_selected_constants
                         if getattr(cfg, "refine_mode", "joint") == "1d"
                         else refine.refine_selected_constants_joint)
            selected, refine_records = refine_fn(
                selected, base.base_exprs(X.shape[1], bcfg),
                X[inner_train_idx], y[inner_train_idx],
                X[inner_val_idx], y[inner_val_idx],
                rcfg,
            )
        selected_exprs = base.base_exprs(X.shape[1], bcfg) + [s.expr for s in selected]
        Ztr, fit = base.fit_design(selected_exprs, X[train_idx])
        Zva = fit.transform(X[val_idx])
        w, b0 = base.ridge_fit(Ztr, y[train_idx], bcfg.ridge_alpha)
        yhat[val_idx] = base.ridge_predict(Zva, w, b0)
        all_selected.extend(selected)
        fold_summaries.append({
            "fold": fold_id,
            "val_R2": float(base.r2_score(y[val_idx], yhat[val_idx])),
            "selected": [str(s.expr) for s in selected],
            "trace": info["trace"],
            "constant_refinements": refine_records,
        })

    selected_exprs_only = unique_exprs([s.expr for s in all_selected])
    selected_plus_subexprs = unique_exprs([se for e in selected_exprs_only for se in e.subexpressions() if se.op != "const"])
    single = base.coordinate_recovery(X, selected_exprs_only, hidden)
    selected_span = span_recovery(X, selected_exprs_only, hidden, alpha=bcfg.ridge_alpha)
    subexpr_span = span_recovery(X, selected_plus_subexprs, hidden, alpha=bcfg.ridge_alpha)
    abstraction = derive_rank1_bilinear_abstractions(X, yhat, hidden=hidden)
    result = {
        "protocol": "FASE-G1a-v23 hierarchy gate / primitive recursion + radical primitive + abstraction audit",
        "primitive_algebra": ["+", "-", "*", "/", "sqrt_pos", "compose", "normalize", "project"],
        "n": int(n),
        "d": int(X.shape[1]),
        "R2_oof": float(base.r2_score(y, yhat)),
        "MSE_oof": float(base.mse(y, yhat)),
        "elapsed_sec": float(time.time() - t0),
        "top_selected_expressions": base.top_expression_counts(all_selected, 16),
        "single_coordinate_recovery": single,
        "selected_span_recovery": selected_span,
        "subexpression_span_recovery": subexpr_span,
        "higher_hierarchy_abstraction": abstraction,
        "folds": fold_summaries,
        "config": cfg.__dict__,
    }
    return result


def summarize_hierarchy(result: Dict[str, object], r2_thr: float = 0.95, span_thr: float = 0.95, abstract_thr: float = 0.95) -> Dict[str, object]:
    selected_span = result.get("selected_span_recovery", {})

    # Two notions are separated deliberately:
    #   full_span_pass: every named hidden coordinate is in the selected primitive span.
    #   terminal_span_pass: only the task-level coordinate/law is in the span.
    # For the composed smoke test, z_sum and z_sub are the human-preferred factors;
    # an expanded primitive solution may validly recover z_product without recovering
    # z_sum and z_sub as explicit coordinates. Factor discovery is then delegated to
    # the higher-hierarchy abstraction audit.
    def row_pass(row: Dict[str, object]) -> bool:
        return row.get("span_corr", 0.0) >= span_thr or row.get("span_R2", 0.0) >= span_thr

    full_span_pass = all(row_pass(row) for row in selected_span.values()) if selected_span else True
    terminal_markers = ("product", "p_next", "next", "mul", "ratio")
    terminal_names = [name for name in selected_span if any(m in name for m in terminal_markers)]
    if not terminal_names and selected_span:
        # fallback: use the last hidden coordinate by insertion order, which is how
        # the smoke generators report the final task coordinate.
        terminal_names = [list(selected_span.keys())[-1]]
    terminal_span_pass = all(row_pass(selected_span[name]) for name in terminal_names) if terminal_names else True

    abstractions = result.get("higher_hierarchy_abstraction", {}).get("top_rank1_bilinear_abstractions", [])
    best_abs = abstractions[0] if abstractions else None
    higher_abstraction_pass = bool(best_abs and best_abs.get("candidate_R2_vs_target", 0.0) >= abstract_thr and best_abs.get("rank1_ratio", 0.0) >= 0.98)
    return {
        "R2_oof": result["R2_oof"],
        "MSE_oof": result["MSE_oof"],
        "prediction_pass": bool(result["R2_oof"] >= r2_thr),
        "terminal_coordinate_names": terminal_names,
        "terminal_span_pass": bool(terminal_span_pass),
        "full_named_coordinate_span_pass": bool(full_span_pass),
        "higher_abstraction_pass": higher_abstraction_pass,
        "gate_pass_low_hierarchy": bool(result["R2_oof"] >= r2_thr and terminal_span_pass),
        "gate_pass_with_bilinear_abstraction": bool(result["R2_oof"] >= r2_thr and terminal_span_pass and higher_abstraction_pass),
        "best_higher_abstraction": best_abs,
        "top_selected_expressions": result.get("top_selected_expressions", [])[:8],
        "single_coordinate_recovery": result.get("single_coordinate_recovery", {}),
        "selected_span_recovery": selected_span,
        "elapsed_sec": result["elapsed_sec"],
    }


def run_named(test: str, cfg: HierarchyConfig) -> Dict[str, object]:
    X, y, hidden = base.DATASETS[test](seed=cfg.seed, n=cfg.n)
    result = run_hierarchy_gate(X, y, hidden, cfg)
    return {"result": result, "summary": summarize_hierarchy(result)}


def run_demo(cfg: HierarchyConfig) -> Dict[str, object]:
    out = {}
    for name in ["mul", "ratio", "composed", "vieta"]:
        out[name] = run_named(name, cfg)["summary"]
    out["vieta_trajectory_reference"] = base.vieta_trajectory(10)
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--test", choices=list(base.DATASETS.keys()) + ["demo"], default="demo")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--n", type=int, default=700)
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--json-out", type=str, default=None)
    args = ap.parse_args()
    cfg = HierarchyConfig(
        seed=args.seed,
        n=220 if args.quick else args.n,
        k_folds=3 if args.quick else 5,
        max_layers=2 if args.quick else 3,
        max_candidates_scored=500 if args.quick else 2400,
        depth1_beam=50 if args.quick else 120,
    )
    if args.test == "demo":
        out = run_demo(cfg)
    else:
        out = run_named(args.test, cfg)
    text = json.dumps(out, indent=2)
    print(text)
    if args.json_out:
        with open(args.json_out, "w", encoding="utf-8") as f:
            f.write(text)


if __name__ == "__main__":
    main()
