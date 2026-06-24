#!/usr/bin/env python3
"""
FASE-G1a-v23 Viète Full Killing Field + PySR Control
=====================================================

Purpose
-------
Run the Viète radical-product killing field at field-test scale on a local
machine, with optional PySR controls on the identical synthetic data and outer
folds.

The scientific question is not merely prediction:

    Can the FASE spine X -> Z(0) -> ... -> Z(k) recover the radical coordinate
    and radical-product law required by Viète, and under what sample/noise
    conditions does it fail?

Targets
-------
Synthetic Viète transition state:

    X = [a, p]
    z_rad  = sqrt(2 + a)
    z_next = p * sqrt(2 + a) / 2
    y      = z_next + noise

Primitive regimes
-----------------
FASE strict A0:
    {+, -, *, /, compose, normalize, project}

FASE A0r:
    A0 + {sqrt_pos}

PySR controls:
    raw X, with either no sqrt or sqrt enabled in the unary operator set.

Gate rule
---------
Strict A0 can be scientifically informative, but it is not eligible to close
Viète's radical gate. Radical-gate closure requires A0r plus:

    - OOF R2 >= threshold;
    - terminal coordinate span/subexpression recovery;
    - radical coordinate present in FASE discovery;
    - radical-product abstraction pass.

Outputs
-------
Writes:
    <out_dir>/vieta_full_field_report.json
    <out_dir>/vieta_full_field_rows.csv

Dependencies
------------
Required sibling files:
    fase_v23_g1a_coordinate_gate.py
    fase_v23_g1a_hierarchy_gate.py
    fase_v23_g1a_vieta_killing_field.py

Optional:
    pysr

Example commands
----------------
Fast sanity check:
    python fase_v23_g1a_vieta_full_field_with_pysr.py --quick --pysr-scope anchor

Full FASE field, anchor PySR controls:
    python fase_v23_g1a_vieta_full_field_with_pysr.py --matrix full --pysr-scope anchor

Full FASE field, full PySR controls over the same grid:
    python fase_v23_g1a_vieta_full_field_with_pysr.py --matrix full --pysr-scope full \
      --pysr-iterations 1000 --pysr-timeout 120
"""

from __future__ import annotations

import argparse
import csv
import inspect
import json
import math
import os
import platform
import re
import sys
import time
import traceback
from collections import defaultdict
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np

import fase_v23_g1a_coordinate_gate as base
import fase_v23_g1a_vieta_killing_field as vkf

EPS = 1e-9


def boolstr(x: Any) -> str:
    return "true" if bool(x) else "false"


def condition_grid(matrix: str) -> Tuple[List[int], List[int], List[float]]:
    if matrix == "quick":
        return [42], [80, 160], [0.001, 0.05]
    if matrix == "medium":
        return [42, 1337], [80, 160, 320], [0.0, 0.001, 0.01, 0.05, 0.10]
    if matrix == "full":
        return [42, 1337, 2025, 9001], [50, 80, 160, 320, 640], [0.0, 0.001, 0.01, 0.05, 0.10, 0.20]
    raise ValueError(f"unknown matrix {matrix!r}")


def pysr_condition_filter(scope: str, n: int, noise: float) -> bool:
    """Decide whether to run PySR on a grid point."""
    if scope == "none":
        return False
    if scope == "full":
        return True
    if scope == "anchor":
        # Compact but useful: low noise / moderate noise at sample sizes where
        # FASE should be strong if the spine is real.
        return n in (160, 320) and noise in (0.001, 0.05)
    if scope == "minimal":
        return n == 160 and noise == 0.001
    raise ValueError(f"unknown pysr scope {scope!r}")


def flatten_fase_case(case: Dict[str, Any], label: str) -> Dict[str, Any]:
    cond = case.get("condition", {})
    gv = case.get("gate_verdict", {})
    summary = case.get("summary", {})
    best_abs = None
    audit = case.get("radical_product_abstraction_audit", {})
    if isinstance(audit, dict):
        best = audit.get("best_candidate") or {}
        best_abs = best.get("expr")
    rows = case.get("top_selected_expressions", []) or []
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
        "terminal_selected_span_corr": gv.get("terminal_coordinate_selected_span_corr"),
        "terminal_subexpr_span_corr": gv.get("terminal_coordinate_subexpression_span_corr"),
        "radical_in_discovery": gv.get("radical_coordinate_present_in_discovery"),
        "radical_single_corr": gv.get("radical_coordinate_single_corr"),
        "radical_subexpr_span_corr": gv.get("radical_coordinate_subexpr_span_corr"),
        "radical_product_pass": gv.get("radical_product_abstraction_pass"),
        "vieta_gate_pass": gv.get("vieta_gate_pass"),
        "strict_A0_not_eligible": gv.get("strict_A0_not_eligible_for_radical_gate"),
        "best_expression_or_abstraction": best_abs,
        "top_expressions_json": json.dumps(rows[:8]),
        "error": None,
    }


def run_fase_condition(seed: int, n: int, noise: float, include_sqrt: bool, quick: bool) -> Dict[str, Any]:
    case = vkf.run_once(seed=seed, n=n, noise=noise, include_sqrt=include_sqrt, quick=quick)
    case["gate_verdict"] = vkf.gate_verdict(case)
    return case


# -----------------------------
# PySR control
# -----------------------------

def get_pysr_regressor():
    from pysr import PySRRegressor  # type: ignore
    return PySRRegressor


def filter_pysr_kwargs(PySRRegressor, kwargs: Dict[str, Any]) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """Drop kwargs unsupported by the installed PySR version when possible."""
    try:
        sig = inspect.signature(PySRRegressor.__init__)
        params = sig.parameters
        accepts_kwargs = any(p.kind == inspect.Parameter.VAR_KEYWORD for p in params.values())
        if accepts_kwargs:
            return dict(kwargs), {}
        allowed = {k for k in params if k != "self"}
        used = {k: v for k, v in kwargs.items() if k in allowed}
        dropped = {k: v for k, v in kwargs.items() if k not in allowed}
        return used, dropped
    except Exception:
        return dict(kwargs), {"signature_introspection_failed": True}


def build_pysr_kwargs(seed: int, with_sqrt: bool, args: argparse.Namespace) -> Dict[str, Any]:
    unary_ops: List[str] = ["sqrt"] if with_sqrt else []
    if args.pysr_extra_unary:
        unary_ops.extend([u.strip() for u in args.pysr_extra_unary.split(",") if u.strip()])
    return {
        "niterations": args.pysr_iterations,
        "binary_operators": ["+", "-", "*", "/"],
        "unary_operators": unary_ops,
        "maxsize": args.pysr_maxsize,
        "model_selection": "best",
        "deterministic": True,
        "parallelism": "serial",
        "procs": 0,
        "batching": False,
        "timeout_in_seconds": None if args.pysr_timeout <= 0 else args.pysr_timeout,
        "random_state": seed,
        "verbosity": 0,
    }


def safe_pysr_expression(model: Any) -> Dict[str, Any]:
    out: Dict[str, Any] = {"expression": None, "best_row": None, "equations": None}
    try:
        if hasattr(model, "sympy"):
            out["expression"] = str(model.sympy())
    except Exception as e:
        out["sympy_error"] = repr(e)
    try:
        if hasattr(model, "get_best"):
            best = model.get_best()
            if hasattr(best, "to_dict"):
                out["best_row"] = best.to_dict()
            else:
                out["best_row"] = str(best)
            if out["expression"] is None:
                if isinstance(out["best_row"], dict):
                    out["expression"] = str(out["best_row"].get("equation"))
                else:
                    out["expression"] = str(out["best_row"])
    except Exception as e:
        out["get_best_error"] = repr(e)
    try:
        eq = getattr(model, "equations_", None)
        if eq is not None:
            out["equations"] = str(eq.tail(8) if hasattr(eq, "tail") else eq)
    except Exception as e:
        out["equations_error"] = repr(e)
    return out


def run_pysr_oof(seed: int, n: int, noise: float, with_sqrt: bool, args: argparse.Namespace) -> Dict[str, Any]:
    t0 = time.time()
    X, y, hidden = vkf.make_vieta(seed=seed, n=n, noise=noise)
    k_folds = 2 if args.quick else args.k_folds
    folds = base.kfold_indices(n, k_folds, seed)
    yhat = np.zeros(n)
    fold_rows: List[Dict[str, Any]] = []
    kwargs_dropped_total: Dict[str, Any] = {}
    PySRRegressor = get_pysr_regressor()

    for fold_id, val_idx in enumerate(folds, start=1):
        train_idx = np.setdiff1d(np.arange(n), val_idx)
        kwargs_raw = build_pysr_kwargs(seed=seed + 1000 * fold_id, with_sqrt=with_sqrt, args=args)
        kwargs_used, kwargs_dropped = filter_pysr_kwargs(PySRRegressor, kwargs_raw)
        kwargs_dropped_total.update(kwargs_dropped)
        ft0 = time.time()
        try:
            model = PySRRegressor(**kwargs_used)
            model.fit(X[train_idx], y[train_idx])
            pred = np.asarray(model.predict(X[val_idx]), float).reshape(-1)
            pred = np.nan_to_num(pred, nan=float(np.nanmean(y[train_idx])), posinf=1e9, neginf=-1e9)
            yhat[val_idx] = pred
            expr_info = safe_pysr_expression(model)
            fold_rows.append({
                "fold": fold_id,
                "val_R2": float(base.r2_score(y[val_idx], yhat[val_idx])),
                "val_MSE": float(base.mse(y[val_idx], yhat[val_idx])),
                "elapsed_sec": float(time.time() - ft0),
                "expression": expr_info.get("expression"),
                "expr_info": expr_info,
                "kwargs_used": kwargs_used,
                "kwargs_dropped_by_version": kwargs_dropped,
            })
        except Exception as e:
            fold_rows.append({
                "fold": fold_id,
                "error": repr(e),
                "traceback": traceback.format_exc(),
                "elapsed_sec": float(time.time() - ft0),
                "kwargs_used": kwargs_used,
                "kwargs_dropped_by_version": kwargs_dropped,
            })
            yhat[val_idx] = float(np.mean(y[train_idx]))

    R2 = float(base.r2_score(y, yhat))
    MSE = float(base.mse(y, yhat))
    expressions = [fr.get("expression") for fr in fold_rows if fr.get("expression")]
    contains_sqrt = any("sqrt" in str(e).lower() for e in expressions)
    # PySR can close a radical-control gate only if sqrt was available, it used/recovered
    # a sqrt expression, and it predicts well. This is a control, not FASE gate closure.
    radical_control_pass = bool(with_sqrt and R2 >= args.r2_threshold and contains_sqrt)
    return {
        "method": "PySR",
        "label": "PySR_with_sqrt" if with_sqrt else "PySR_no_sqrt",
        "primitive_regime": "raw_X + PySR operators with sqrt" if with_sqrt else "raw_X + PySR operators no sqrt",
        "include_sqrt": with_sqrt,
        "seed": seed,
        "n": n,
        "noise": noise,
        "k_folds": k_folds,
        "R2_oof": R2,
        "MSE_oof": MSE,
        "elapsed_sec": float(time.time() - t0),
        "prediction_pass": bool(R2 >= args.r2_threshold),
        "terminal_span_pass": None,
        "terminal_selected_span_corr": None,
        "terminal_subexpr_span_corr": None,
        "radical_in_discovery": contains_sqrt,
        "radical_single_corr": None,
        "radical_subexpr_span_corr": None,
        "radical_product_pass": radical_control_pass,
        "vieta_gate_pass": radical_control_pass,
        "strict_A0_not_eligible": not with_sqrt,
        "best_expression_or_abstraction": expressions[0] if expressions else None,
        "top_expressions_json": json.dumps(expressions[:8]),
        "folds": fold_rows,
        "kwargs_dropped_by_version": kwargs_dropped_total,
        "error": None,
    }


def aggregate_rows(rows: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    groups: Dict[Tuple[Any, ...], List[Dict[str, Any]]] = defaultdict(list)
    for r in rows:
        key = (r.get("method"), r.get("label"), r.get("primitive_regime"), r.get("n"), r.get("noise"))
        groups[key].append(r)
    out = []
    for key, rs in sorted(groups.items(), key=lambda kv: tuple(str(x) for x in kv[0])):
        vals = [float(r["R2_oof"]) for r in rs if r.get("R2_oof") is not None and math.isfinite(float(r["R2_oof"]))]
        gates = [bool(r.get("vieta_gate_pass")) for r in rs if r.get("vieta_gate_pass") is not None]
        elapsed = [float(r["elapsed_sec"]) for r in rs if r.get("elapsed_sec") is not None and math.isfinite(float(r["elapsed_sec"]))]
        out.append({
            "method": key[0],
            "label": key[1],
            "primitive_regime": key[2],
            "n": key[3],
            "noise": key[4],
            "runs": len(rs),
            "R2_mean": float(np.mean(vals)) if vals else None,
            "R2_min": float(np.min(vals)) if vals else None,
            "R2_median": float(np.median(vals)) if vals else None,
            "R2_max": float(np.max(vals)) if vals else None,
            "gate_pass_rate": float(np.mean(gates)) if gates else None,
            "elapsed_total_sec": float(np.sum(elapsed)) if elapsed else None,
        })
    return {"by_method_n_noise": out}


def write_csv(path: str, rows: Sequence[Dict[str, Any]]) -> None:
    fieldnames = [
        "method", "label", "primitive_regime", "include_sqrt", "seed", "n", "noise", "k_folds",
        "R2_oof", "MSE_oof", "elapsed_sec", "prediction_pass", "terminal_span_pass",
        "terminal_selected_span_corr", "terminal_subexpr_span_corr", "radical_in_discovery",
        "radical_single_corr", "radical_subexpr_span_corr", "radical_product_pass", "vieta_gate_pass",
        "strict_A0_not_eligible", "best_expression_or_abstraction", "error",
    ]
    with open(path, "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        w.writeheader()
        for r in rows:
            w.writerow(r)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--matrix", choices=["quick", "medium", "full"], default="full")
    ap.add_argument("--quick", action="store_true", help="Alias for --matrix quick and 2-fold internal runs.")
    ap.add_argument("--out-dir", default="./vieta_field_out")
    ap.add_argument("--k-folds", type=int, default=5)
    ap.add_argument("--r2-threshold", type=float, default=0.95)
    ap.add_argument("--pysr-scope", choices=["none", "minimal", "anchor", "full"], default="anchor")
    ap.add_argument("--pysr-regimes", choices=["none", "no_sqrt", "with_sqrt", "both"], default="both")
    ap.add_argument("--pysr-iterations", type=int, default=1000)
    ap.add_argument("--pysr-maxsize", type=int, default=30)
    ap.add_argument("--pysr-timeout", type=int, default=120, help="Per-fold PySR timeout seconds; <=0 disables.")
    ap.add_argument("--pysr-extra-unary", default="", help="Comma-separated extra PySR unary ops, e.g. abs,log,exp. Leave empty for clean radical control.")
    ap.add_argument("--skip-fase", action="store_true")
    args = ap.parse_args()
    if args.quick:
        args.matrix = "quick"

    os.makedirs(args.out_dir, exist_ok=True)
    seeds, sample_sizes, noise_levels = condition_grid(args.matrix)
    t0 = time.time()
    rows: List[Dict[str, Any]] = []
    detailed: List[Dict[str, Any]] = []

    protocol = {
        "protocol": "FASE-G1a-v23 Viète full killing field + PySR control",
        "matrix": args.matrix,
        "seeds": seeds,
        "sample_sizes": sample_sizes,
        "noise_levels": noise_levels,
        "pysr_scope": args.pysr_scope,
        "pysr_regimes": args.pysr_regimes,
        "r2_threshold": args.r2_threshold,
        "spine": "X -> Z(0) -> Z(1) -> ... -> Z(k)",
        "target_law": "y = p*sqrt(2+a)/2 + noise",
        "destroyer_clause": "High prediction alone is insufficient for FASE gate closure; require radical coordinate discovery and radical-product abstraction.",
        "environment": {
            "python": sys.version,
            "platform": platform.platform(),
            "numpy": np.__version__,
        },
    }

    # Optional PySR import check once.
    pysr_available = False
    if args.pysr_scope != "none" and args.pysr_regimes != "none":
        try:
            _ = get_pysr_regressor()
            pysr_available = True
        except Exception as e:
            protocol["pysr_import_error"] = repr(e)
            pysr_available = False

    total_conditions = len(seeds) * len(sample_sizes) * len(noise_levels)
    count = 0
    for seed in seeds:
        for n in sample_sizes:
            for noise in noise_levels:
                count += 1
                print(f"[condition {count}/{total_conditions}] seed={seed} n={n} noise={noise}", flush=True)

                if not args.skip_fase:
                    for include_sqrt, label in [(False, "FASE_strict_A0"), (True, "FASE_A0r")]:
                        try:
                            case = run_fase_condition(seed, n, noise, include_sqrt=include_sqrt, quick=(args.matrix == "quick"))
                            flat = flatten_fase_case(case, label=label)
                            rows.append(flat)
                            detailed.append({"kind": "FASE_case", "row": flat, "case": case})
                            print(f"  {label}: R2={flat['R2_oof']:.6f} gate={flat['vieta_gate_pass']}", flush=True)
                        except Exception as e:
                            err = {
                                "method": "FASE", "label": label, "primitive_regime": "A0r" if include_sqrt else "strict A0",
                                "include_sqrt": include_sqrt, "seed": seed, "n": n, "noise": noise,
                                "error": repr(e), "traceback": traceback.format_exc(),
                            }
                            rows.append(err)
                            detailed.append({"kind": "FASE_error", "row": err})
                            print(f"  {label}: ERROR {e!r}", flush=True)

                if pysr_available and pysr_condition_filter(args.pysr_scope, n, noise):
                    regimes: List[bool] = []
                    if args.pysr_regimes in ("no_sqrt", "both"):
                        regimes.append(False)
                    if args.pysr_regimes in ("with_sqrt", "both"):
                        regimes.append(True)
                    for with_sqrt in regimes:
                        label = "PySR_with_sqrt" if with_sqrt else "PySR_no_sqrt"
                        try:
                            pr = run_pysr_oof(seed, n, noise, with_sqrt=with_sqrt, args=args)
                            rows.append(pr)
                            detailed.append({"kind": "PySR_case", "row": {k: v for k, v in pr.items() if k != "folds"}, "folds": pr.get("folds", [])})
                            print(f"  {label}: R2={pr['R2_oof']:.6f} expr={pr.get('best_expression_or_abstraction')}", flush=True)
                        except Exception as e:
                            err = {
                                "method": "PySR", "label": label, "primitive_regime": "with sqrt" if with_sqrt else "no sqrt",
                                "include_sqrt": with_sqrt, "seed": seed, "n": n, "noise": noise,
                                "error": repr(e), "traceback": traceback.format_exc(),
                            }
                            rows.append(err)
                            detailed.append({"kind": "PySR_error", "row": err})
                            print(f"  {label}: ERROR {e!r}", flush=True)

    report = {
        "protocol": protocol,
        "rows": rows,
        "aggregates": aggregate_rows(rows),
        "detailed": detailed,
        "elapsed_sec_total": float(time.time() - t0),
    }
    json_path = os.path.join(args.out_dir, "vieta_full_field_report.json")
    csv_path = os.path.join(args.out_dir, "vieta_full_field_rows.csv")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)
    write_csv(csv_path, rows)
    print(f"\nWROTE {json_path}")
    print(f"WROTE {csv_path}")


if __name__ == "__main__":
    main()
