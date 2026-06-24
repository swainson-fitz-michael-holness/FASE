#!/usr/bin/env python3
"""
FASE-G1a-v23 Viète Killing Field
================================

Purpose
-------
Stress-test the proposed FASE v23 spine:

    X -> Z(0) -> Z(1) -> ... -> Z(k)

on a Viète-inspired radical transition law. This is not a PySR benchmark and
not the full FASE-G1-v23 gate. It is a micro-killing-field for one question:

    Can primitive recursive coordinate discovery recover the radical-product
    coordinate required by Viète's formula, and under what rough sample/noise
    conditions does it fail?

Primitive regimes
-----------------
A0 strict:
    {+, -, *, /, compose, normalize, project}

A0r radical extension:
    A0 + {sqrt_pos}

Viète transition target
-----------------------
Viète's nested radical recurrence may be written as:

    a_{k+1} = sqrt(2 + a_k)
    p_{k+1} = p_k * a_{k+1}/2

The synthetic task trains on random states (a, p) and predicts:

    y = p * sqrt(2 + a)/2 + noise

The hidden coordinates are:

    z_rad  = sqrt(2 + a)
    z_next = p * sqrt(2 + a)/2

Outputs
-------
A JSON report with:
    - strict A0 vs A0r comparison;
    - noise sweep;
    - sample-size sweep;
    - radical-product abstraction audit;
    - gate status.

Dependencies
------------
Requires the sibling files:
    fase_v23_g1a_coordinate_gate.py
    fase_v23_g1a_hierarchy_gate.py
"""

from __future__ import annotations

import argparse
import json
import math
import time
from typing import Dict, List, Tuple

import numpy as np

import fase_v23_g1a_coordinate_gate as base
import fase_v23_g1a_hierarchy_gate as hierarchy

EPS = 1e-9


def make_vieta(seed: int, n: int, noise: float = 0.001,
               a_low: float = -1.5, a_high: float = 8.0,
               p_low: float = 0.2, p_high: float = 2.0) -> Tuple[np.ndarray, np.ndarray, Dict[str, np.ndarray]]:
    rng = np.random.default_rng(seed)
    a = rng.uniform(a_low, a_high, size=n)
    p = rng.uniform(p_low, p_high, size=n)
    X = np.column_stack([a, p])
    z_rad = np.sqrt(2.0 + a)
    z_next = p * z_rad / 2.0
    y = z_next + rng.normal(0.0, noise, size=n)
    return X, y, {"z_sqrt_2_plus_a": z_rad, "z_p_next": z_next}


def cfg_for(seed: int, n: int, include_sqrt: bool, quick: bool) -> hierarchy.HierarchyConfig:
    # Moderate defaults: strong enough to recover Viète at n≈160-320, but fast enough
    # for repeated smoke runs. These are intentionally not tuned per condition.
    return hierarchy.HierarchyConfig(
        seed=seed,
        n=n,
        include_sqrt=include_sqrt,
        k_folds=2 if quick else 5,
        candidate_depth=3,
        max_layers=2 if quick else 3,
        accept_per_layer=3,
        max_candidates_scored=180 if quick else 1800,
        depth1_beam=40 if quick else 110,
        min_gain=1e-4,
        complexity_penalty=1e-4,
        n_random_projects=0,
        ridge_alpha=1e-6,
    )


def run_once(seed: int, n: int, noise: float, include_sqrt: bool, quick: bool) -> Dict[str, object]:
    X, y, hidden = make_vieta(seed=seed, n=n, noise=noise)
    cfg = cfg_for(seed=seed, n=n, include_sqrt=include_sqrt, quick=quick)
    result = hierarchy.run_hierarchy_gate(X, y, hidden, cfg)
    summary = hierarchy.summarize_hierarchy(result)
    radical_audit = radical_product_audit(X, y, hidden)
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
        "radical_product_abstraction_audit": radical_audit,
        "top_selected_expressions": result.get("top_selected_expressions", [])[:12],
        "selected_span_recovery": result.get("selected_span_recovery", {}),
        "subexpression_span_recovery": result.get("subexpression_span_recovery", {}),
        "single_coordinate_recovery": result.get("single_coordinate_recovery", {}),
        "elapsed_sec": result.get("elapsed_sec"),
    }


def fit_1d_r2(z: np.ndarray, y: np.ndarray) -> Tuple[float, float, float]:
    Z = np.asarray(z, float).reshape(-1, 1)
    w, b0 = base.ridge_fit(Z, y, alpha=1e-8)
    pred = base.ridge_predict(Z, w, b0)
    return float(base.r2_score(y, pred)), float(w[0] if len(w) else 0.0), float(b0)


def radical_product_audit(X: np.ndarray, y: np.ndarray, hidden: Dict[str, np.ndarray]) -> Dict[str, object]:
    """Search a tiny radical-product abstraction class.

    This is deliberately not the same as giving the model an arbitrary function bank.
    It is a hierarchy audit: after low-level recursion, can a compact law in the
    radical-product family explain the target?

    Candidate family:
        sqrt_pos(c + s*x_i)                 radical coordinate
        x_j * sqrt_pos(c + s*x_i)           radical-product coordinate

    A linear head absorbs constant scale, so p*sqrt(2+a)/2 is equivalent to
    x1*sqrt(2+x0) up to coefficient 1/2.
    """
    X = np.asarray(X, float)
    y = np.asarray(y, float)
    n, d = X.shape
    constants = [-4.0, -2.0, -1.0, -0.5, 0.0, 0.5, 1.0, 2.0, 3.0, 4.0, 8.0]
    signs = [1.0, -1.0]
    rows: List[Dict[str, object]] = []
    for i in range(d):
        for c in constants:
            for s in signs:
                rad_arg = c + s * X[:, i]
                valid_frac = float(np.mean(rad_arg >= 0.0))
                if valid_frac < 0.98:
                    continue
                rad = np.sqrt(np.clip(rad_arg, 0.0, None))
                rad_r2, rad_w, rad_b = fit_1d_r2(rad, y)
                rows.append({
                    "kind": "radical",
                    "expr": f"sqrt_pos({c:g} {'+' if s >= 0 else '-'} x{i})",
                    "R2_vs_y": rad_r2,
                    "linear_weight": rad_w,
                    "linear_intercept": rad_b,
                    "valid_frac": valid_frac,
                    "corr_z_rad": float(base.corr_abs(rad, hidden.get("z_sqrt_2_plus_a", rad))),
                    "corr_z_next": float(base.corr_abs(rad, hidden.get("z_p_next", y))),
                })
                for j in range(d):
                    prod = X[:, j] * rad
                    prod_r2, prod_w, prod_b = fit_1d_r2(prod, y)
                    rows.append({
                        "kind": "radical_product",
                        "expr": f"x{j}*sqrt_pos({c:g} {'+' if s >= 0 else '-'} x{i})",
                        "R2_vs_y": prod_r2,
                        "linear_weight": prod_w,
                        "linear_intercept": prod_b,
                        "valid_frac": valid_frac,
                        "corr_z_rad": float(base.corr_abs(rad, hidden.get("z_sqrt_2_plus_a", rad))),
                        "corr_z_next": float(base.corr_abs(prod, hidden.get("z_p_next", y))),
                    })
    rows.sort(key=lambda r: (r["R2_vs_y"], r.get("corr_z_next", 0.0), r.get("corr_z_rad", 0.0)), reverse=True)
    best = rows[0] if rows else None
    return {
        "audit_family": "sqrt_pos(c +/- x_i), x_j*sqrt_pos(c +/- x_i)",
        "best_candidate": best,
        "top_candidates": rows[:8],
        "pass_radical_product": bool(best and best["kind"] == "radical_product" and best["R2_vs_y"] >= 0.95 and best.get("corr_z_next", 0.0) >= 0.95),
    }


def gate_verdict(case: Dict[str, object]) -> Dict[str, object]:
    summary = case["summary"]
    condition = case.get("condition", {})
    include_sqrt = bool(condition.get("include_sqrt", False))
    single = case.get("single_coordinate_recovery", {})
    selected_span = case.get("selected_span_recovery", {})
    sub_span = case.get("subexpression_span_recovery", {})
    rad_audit = case.get("radical_product_abstraction_audit", {})

    pred = bool(summary.get("prediction_pass", False))
    # For Viète, direct selected-span may be too strict because p*sqrt(2+a) can
    # first appear as a subexpression footprint. Track both.
    next_selected_span = float(selected_span.get("z_p_next", {}).get("span_corr", 0.0))
    next_sub_span = float(sub_span.get("z_p_next", {}).get("span_corr", 0.0))
    terminal_span = bool(max(next_selected_span, next_sub_span) >= 0.95)

    # Radical coordinate should be present in the FASE-discovered expression tree.
    # The external radical-product audit is not allowed to rescue strict A0.
    rad_single = float(single.get("z_sqrt_2_plus_a", {}).get("best_abs_corr", 0.0))
    rad_sub = float(sub_span.get("z_sqrt_2_plus_a", {}).get("span_corr", 0.0))
    radical_present_in_discovery = bool(max(rad_single, rad_sub) >= 0.95)
    radical_product = bool(rad_audit.get("pass_radical_product", False))

    # Strict A0 can be informative, but it cannot close a Viète radical gate.
    gate = bool(include_sqrt and pred and terminal_span and radical_present_in_discovery and radical_product)
    return {
        "prediction_pass": pred,
        "terminal_span_pass": terminal_span,
        "terminal_coordinate_selected_span_corr": next_selected_span,
        "terminal_coordinate_subexpression_span_corr": next_sub_span,
        "radical_coordinate_present_in_discovery": radical_present_in_discovery,
        "radical_coordinate_single_corr": rad_single,
        "radical_coordinate_subexpr_span_corr": rad_sub,
        "radical_product_abstraction_pass": radical_product,
        "strict_A0_not_eligible_for_radical_gate": not include_sqrt,
        "vieta_gate_pass": gate,
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--n", type=int, default=220)
    ap.add_argument("--noise", type=float, default=0.001)
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--json-out", type=str, default="/mnt/data/fase_v23_g1a_vieta_killing_field_report.json")
    args = ap.parse_args()

    t0 = time.time()
    report: Dict[str, object] = {
        "protocol": "FASE-G1a-v23 Viète micro-killing-field",
        "spine": "X -> Z(0) -> Z(1) -> ... -> Z(k)",
        "claim": "Primitive recursion with radical primitive can recover Viète's radical-product coordinate under nested OOF discipline.",
        "destroyer_clause": "High prediction alone is insufficient; require terminal span recovery, radical coordinate presence, and radical-product abstraction.",
        "target_law": "y = p*sqrt(2+a)/2 + noise",
        "seed": args.seed,
        "quick": args.quick,
    }

    # 1. Strict A0 vs radical extension at the same condition.
    strict = run_once(args.seed, args.n, args.noise, include_sqrt=False, quick=args.quick)
    radical = run_once(args.seed, args.n, args.noise, include_sqrt=True, quick=args.quick)
    strict["gate_verdict"] = gate_verdict(strict)
    radical["gate_verdict"] = gate_verdict(radical)
    report["A0_vs_A0r"] = {"strict_A0": strict, "A0_plus_sqrt": radical}

    # 2. Noise sweep in radical regime.
    noise_levels = [0.0, 0.001, 0.01, 0.05, 0.10] if not args.quick else [0.0, 0.05]
    noise_sweep = []
    for noise in noise_levels:
        case = run_once(args.seed, args.n, noise, include_sqrt=True, quick=args.quick)
        case["gate_verdict"] = gate_verdict(case)
        noise_sweep.append(case)
    report["noise_sweep_A0r"] = noise_sweep

    # 3. Sample sweep in radical regime.
    sample_sizes = [80, 160, 320, 640] if not args.quick else [80, 160]
    sample_sweep = []
    for n in sample_sizes:
        case = run_once(args.seed, n, args.noise, include_sqrt=True, quick=args.quick)
        case["gate_verdict"] = gate_verdict(case)
        sample_sweep.append(case)
    report["sample_sweep_A0r"] = sample_sweep
    report["vieta_reference_trajectory"] = base.vieta_trajectory(10)
    report["elapsed_sec_total"] = float(time.time() - t0)

    # Compact summary table.
    def compact(case: Dict[str, object]) -> Dict[str, object]:
        return {
            "regime": case["condition"]["primitive_regime"],
            "n": case["condition"]["n"],
            "noise": case["condition"]["noise"],
            "R2_oof": case["summary"].get("R2_oof"),
            "terminal_selected_span_corr": case["gate_verdict"].get("terminal_coordinate_selected_span_corr"),
            "terminal_subexpr_span_corr": case["gate_verdict"].get("terminal_coordinate_subexpression_span_corr"),
            "radical_coord_in_discovery": case["gate_verdict"].get("radical_coordinate_present_in_discovery"),
            "radical_product": case["gate_verdict"].get("radical_product_abstraction_pass"),
            "gate": case["gate_verdict"].get("vieta_gate_pass"),
            "best_abstraction": case["radical_product_abstraction_audit"].get("best_candidate", {}).get("expr") if case.get("radical_product_abstraction_audit") else None,
        }
    report["compact_tables"] = {
        "A0_vs_A0r": [compact(strict), compact(radical)],
        "noise_sweep_A0r": [compact(c) for c in noise_sweep],
        "sample_sweep_A0r": [compact(c) for c in sample_sweep],
    }

    text = json.dumps(report, indent=2)
    with open(args.json_out, "w", encoding="utf-8") as f:
        f.write(text)
    print(text)


if __name__ == "__main__":
    main()
