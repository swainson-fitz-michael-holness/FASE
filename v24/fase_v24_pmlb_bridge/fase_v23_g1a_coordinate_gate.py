#!/usr/bin/env python3
"""
FASE-G1a-v23: Primitive Recursive Coordinate Discovery Gate
===========================================================

A small, auditable artifact for testing whether primitive recursion can discover
useful coordinates without the v22 nonlinear function bank.

Default primitive algebra A0:
    { +, -, *, /, compose, normalize, project }

Optional radical extension A0sqrt:
    A0 + { sqrt_pos }

Why sqrt is optional:
    Strict field operations {+,-,*,/} cannot exactly express Viète-style nested
    radicals.  The Viète smoke test is therefore deliberately run twice:
      1. strict A0: should usually fail or underperform;
      2. A0sqrt: should recover the transition coordinate sqrt(2 + a).

The artifact prioritizes:
    - nested / OOF discipline,
    - explicit expression traces,
    - coordinate recovery metrics,
    - small search budgets,
    - no sklearn / PySR dependency.

This is not a full FASE rewrite. It is the spine test.
"""

from __future__ import annotations

import argparse
import json
import math
import random
import time
from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np

EPS = 1e-9

# -----------------------------
# Metrics and linear head
# -----------------------------

def r2_score(y: np.ndarray, yhat: np.ndarray) -> float:
    y = np.asarray(y, float).reshape(-1)
    yhat = np.asarray(yhat, float).reshape(-1)
    ss_res = float(np.sum((y - yhat) ** 2))
    ss_tot = float(np.sum((y - y.mean()) ** 2)) + EPS
    return 1.0 - ss_res / ss_tot


def mse(y: np.ndarray, yhat: np.ndarray) -> float:
    y = np.asarray(y, float).reshape(-1)
    yhat = np.asarray(yhat, float).reshape(-1)
    return float(np.mean((y - yhat) ** 2))


def ridge_fit(Z: np.ndarray, y: np.ndarray, alpha: float = 1e-6) -> Tuple[np.ndarray, float]:
    """Ridge with an unregularized intercept."""
    Z = np.asarray(Z, float)
    y = np.asarray(y, float).reshape(-1)
    n = len(y)
    if Z.ndim == 1:
        Z = Z.reshape(-1, 1)
    if Z.shape[1] == 0:
        return np.zeros(0), float(y.mean())
    muZ = Z.mean(axis=0, keepdims=True)
    muy = float(y.mean())
    Zc = Z - muZ
    yc = y - muy
    A = Zc.T @ Zc + alpha * np.eye(Z.shape[1])
    b = Zc.T @ yc
    try:
        w = np.linalg.solve(A, b)
    except np.linalg.LinAlgError:
        w = np.linalg.pinv(A, rcond=1e-10) @ b
    b0 = muy - (muZ @ w.reshape(-1, 1)).item()
    return w, b0


def ridge_predict(Z: np.ndarray, w: np.ndarray, b0: float) -> np.ndarray:
    Z = np.asarray(Z, float)
    if Z.ndim == 1:
        Z = Z.reshape(-1, 1)
    if w.size == 0:
        return np.full(Z.shape[0], b0)
    return Z @ w + b0


# -----------------------------
# Expression system
# -----------------------------

@dataclass(frozen=True)
class Expr:
    op: str
    left: Optional["Expr"] = None
    right: Optional["Expr"] = None
    idx: Optional[int] = None
    const: Optional[float] = None
    weights: Optional[Tuple[float, ...]] = None

    def complexity(self) -> int:
        if self.op == "x":
            return 1
        if self.op == "const":
            return 0
        if self.op == "proj":
            return 2
        if self.op == "sqrt":
            return 2 + (self.left.complexity() if self.left else 0)
        c = 1
        if self.left is not None:
            c += self.left.complexity()
        if self.right is not None:
            c += self.right.complexity()
        return c

    def depth(self) -> int:
        if self.op in {"x", "const", "proj"}:
            return 0
        return 1 + max(self.left.depth() if self.left else 0,
                       self.right.depth() if self.right else 0)

    def key(self) -> str:
        return str(self)

    def subexpressions(self) -> List["Expr"]:
        out = [self]
        if self.left is not None:
            out.extend(self.left.subexpressions())
        if self.right is not None:
            out.extend(self.right.subexpressions())
        # stable unique order
        seen = set()
        unique = []
        for e in out:
            k = e.key()
            if k not in seen:
                seen.add(k)
                unique.append(e)
        return unique

    def __str__(self) -> str:
        if self.op == "x":
            return f"x{self.idx}"
        if self.op == "const":
            if self.const is None:
                return "c?"
            if abs(self.const - round(self.const)) < 1e-12:
                return str(int(round(self.const)))
            return f"{self.const:.6g}"
        if self.op == "proj":
            # hash-like compact summary of weights
            if self.weights is None:
                return "proj(?)"
            arr = np.array(self.weights)
            sig = ",".join(f"{v:.2f}" for v in arr[:4])
            if len(arr) > 4:
                sig += ",..."
            return f"proj[{sig}]"
        if self.op == "add":
            return f"({self.left}+{self.right})"
        if self.op == "sub":
            return f"({self.left}-{self.right})"
        if self.op == "mul":
            return f"({self.left}*{self.right})"
        if self.op == "div":
            return f"({self.left}/{self.right})"
        if self.op == "sqrt":
            return f"sqrt_pos({self.left})"
        return f"{self.op}(?)"


def X(i: int) -> Expr:
    return Expr("x", idx=i)


def C(v: float) -> Expr:
    return Expr("const", const=float(v))


def P(w: Sequence[float]) -> Expr:
    return Expr("proj", weights=tuple(float(x) for x in w))


def Add(a: Expr, b: Expr) -> Expr:
    return Expr("add", left=a, right=b)


def Sub(a: Expr, b: Expr) -> Expr:
    return Expr("sub", left=a, right=b)


def Mul(a: Expr, b: Expr) -> Expr:
    return Expr("mul", left=a, right=b)


def Div(a: Expr, b: Expr) -> Expr:
    return Expr("div", left=a, right=b)


def Sqrt(a: Expr) -> Expr:
    return Expr("sqrt", left=a)


def safe_div(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    b = np.asarray(b, float)
    denom = np.where(np.abs(b) < EPS, np.where(b >= 0, EPS, -EPS), b)
    return np.asarray(a, float) / denom


def eval_expr(expr: Expr, Xmat: np.ndarray) -> np.ndarray:
    Xmat = np.asarray(Xmat, float)
    n = Xmat.shape[0]
    if expr.op == "x":
        return Xmat[:, int(expr.idx)]
    if expr.op == "const":
        return np.full(n, float(expr.const))
    if expr.op == "proj":
        w = np.asarray(expr.weights, float)
        return Xmat @ w
    if expr.op == "add":
        return eval_expr(expr.left, Xmat) + eval_expr(expr.right, Xmat)
    if expr.op == "sub":
        return eval_expr(expr.left, Xmat) - eval_expr(expr.right, Xmat)
    if expr.op == "mul":
        return eval_expr(expr.left, Xmat) * eval_expr(expr.right, Xmat)
    if expr.op == "div":
        return safe_div(eval_expr(expr.left, Xmat), eval_expr(expr.right, Xmat))
    if expr.op == "sqrt":
        # Radical extension: positive-domain sqrt. Negative arguments are clipped,
        # and clipping is implicitly punished by validation performance/domain checks.
        return np.sqrt(np.clip(eval_expr(expr.left, Xmat), 0.0, None))
    raise ValueError(f"Unknown op: {expr.op}")


def sanitize(v: np.ndarray, clip: float = 1e6) -> np.ndarray:
    return np.nan_to_num(v, nan=0.0, posinf=clip, neginf=-clip)


@dataclass
class DesignFit:
    exprs: List[Expr]
    mu: np.ndarray
    sd: np.ndarray

    def transform(self, Xmat: np.ndarray) -> np.ndarray:
        if not self.exprs:
            return np.zeros((Xmat.shape[0], 0))
        raw = np.column_stack([sanitize(eval_expr(e, Xmat)) for e in self.exprs])
        return (raw - self.mu) / self.sd


def fit_design(exprs: List[Expr], Xmat: np.ndarray) -> Tuple[np.ndarray, DesignFit]:
    if not exprs:
        fit = DesignFit([], np.zeros((1, 0)), np.ones((1, 0)))
        return np.zeros((Xmat.shape[0], 0)), fit
    raw = np.column_stack([sanitize(eval_expr(e, Xmat)) for e in exprs])
    mu = raw.mean(axis=0, keepdims=True)
    sd = raw.std(axis=0, keepdims=True)
    sd = np.where(sd < 1e-10, 1.0, sd)
    return (raw - mu) / sd, DesignFit(list(exprs), mu, sd)


def corr_abs(a: np.ndarray, b: np.ndarray) -> float:
    a = np.asarray(a, float).reshape(-1)
    b = np.asarray(b, float).reshape(-1)
    if a.std() < 1e-12 or b.std() < 1e-12:
        return 0.0
    c = np.corrcoef(a, b)[0, 1]
    if not np.isfinite(c):
        return 0.0
    return float(abs(c))


# -----------------------------
# Candidate generation
# -----------------------------

@dataclass
class GateConfig:
    seed: int = 42
    k_folds: int = 5
    inner_val_frac: float = 0.25
    max_layers: int = 3
    accept_per_layer: int = 2
    min_gain: float = 1e-4
    complexity_penalty: float = 1e-4
    redundancy_corr: float = 0.995
    candidate_depth: int = 2
    max_candidates_scored: int = 2500
    depth1_beam: int = 80
    n_random_projects: int = 0
    include_sqrt: bool = False
    constants: Tuple[float, ...] = (-2.0, -1.0, -0.5, 0.5, 1.0, 2.0)
    ridge_alpha: float = 1e-6


def base_exprs(d: int, cfg: GateConfig) -> List[Expr]:
    exprs = [X(i) for i in range(d)]
    rng = np.random.default_rng(cfg.seed)
    for _ in range(cfg.n_random_projects):
        w = rng.normal(size=d)
        norm = np.linalg.norm(w) + EPS
        exprs.append(P(w / norm))
    return exprs


def add_if_new(out: List[Expr], seen: set, e: Expr, max_depth: int) -> None:
    if e.depth() > max_depth:
        return
    k = e.key()
    if k not in seen:
        seen.add(k)
        out.append(e)


def primitive_one_step(exprs: List[Expr], cfg: GateConfig, max_depth: int) -> List[Expr]:
    """Generate one-step primitive compositions from existing coordinates."""
    out: List[Expr] = []
    seen = set()
    m = len(exprs)
    for i in range(m):
        a = exprs[i]
        for j in range(i, m):
            b = exprs[j]
            if i != j:
                add_if_new(out, seen, Add(a, b), max_depth)
                add_if_new(out, seen, Sub(a, b), max_depth)
                add_if_new(out, seen, Sub(b, a), max_depth)
            add_if_new(out, seen, Mul(a, b), max_depth)
            if i != j:
                add_if_new(out, seen, Div(a, b), max_depth)
                add_if_new(out, seen, Div(b, a), max_depth)
        # constant interactions: needed for affine shifts and radical forms.
        for c in cfg.constants:
            ce = C(c)
            add_if_new(out, seen, Add(a, ce), max_depth)
            add_if_new(out, seen, Sub(a, ce), max_depth)
            add_if_new(out, seen, Mul(a, ce), max_depth)
            add_if_new(out, seen, Div(a, ce), max_depth)
            if abs(c) > EPS:
                add_if_new(out, seen, Div(ce, a), max_depth)
            if cfg.include_sqrt:
                add_if_new(out, seen, Sqrt(Add(a, ce)), max_depth)
                add_if_new(out, seen, Sqrt(Sub(a, ce)), max_depth)
    if cfg.include_sqrt:
        for a in exprs:
            add_if_new(out, seen, Sqrt(a), max_depth)
    return out


def proxy_rank(cands: List[Expr], Xtr: np.ndarray, resid: np.ndarray, limit: int) -> List[Expr]:
    """Rank by residual alignment and validity; used only to control explosion."""
    scored = []
    for e in cands:
        v = sanitize(eval_expr(e, Xtr))
        if v.std() < 1e-10:
            continue
        if np.mean(np.isfinite(v)) < 1.0:
            continue
        # high clipping/saturation is suspicious; let it survive only if not extreme
        if np.quantile(np.abs(v), 0.99) > 1e5:
            continue
        score = corr_abs(v, resid) - 1e-4 * e.complexity()
        scored.append((score, e))
    scored.sort(key=lambda t: t[0], reverse=True)
    return [e for _, e in scored[:limit]]


def generate_candidates(current: List[Expr], Xtr: np.ndarray, resid: np.ndarray, cfg: GateConfig) -> List[Expr]:
    """Generate candidate coordinate expressions with small local beam search."""
    seen = {e.key() for e in current}
    all_cands: List[Expr] = []

    # depth-1 around current coordinates
    d1 = primitive_one_step(current, cfg, max_depth=max(1, cfg.candidate_depth))
    d1 = [e for e in d1 if e.key() not in seen]
    d1_ranked = proxy_rank(d1, Xtr, resid, cfg.depth1_beam)
    # Radical coordinates may be necessary as parents even when they are weak
    # univariate predictors. Preserve a small structural slice so lookahead can
    # form products such as p * sqrt(2 + a).
    if cfg.include_sqrt:
        sqrt_structural = [e for e in d1 if "sqrt_pos" in str(e)]
        # prefer simple radicals first
        sqrt_structural.sort(key=lambda e: (e.depth(), e.complexity(), str(e)))
        merged = []
        local_seen = set()
        for e in sqrt_structural[: max(20, cfg.depth1_beam)] + d1_ranked:
            if e.key() not in local_seen:
                local_seen.add(e.key())
                merged.append(e)
        d1_ranked = merged[: max(cfg.depth1_beam, 40)]
    for e in d1_ranked:
        add_if_new(all_cands, seen, e, cfg.candidate_depth)

    # depth-2 compositions: this is the crucial lookahead that lets the gate
    # discover coordinates such as (x0+x1)*(x2-x3) without requiring the partial
    # sums to improve y independently.
    if cfg.candidate_depth >= 2:
        pool = current + d1_ranked
        d2 = primitive_one_step(pool, cfg, max_depth=cfg.candidate_depth)
        d2 = [e for e in d2 if e.key() not in seen]
        d2_ranked = proxy_rank(d2, Xtr, resid, cfg.max_candidates_scored)
        for e in d2_ranked:
            add_if_new(all_cands, seen, e, cfg.candidate_depth)

    # final hard cap by proxy rank to keep turnaround quick
    return proxy_rank(all_cands, Xtr, resid, cfg.max_candidates_scored)


# -----------------------------
# Discovery loop
# -----------------------------

@dataclass
class SelectedCoord:
    expr: Expr
    layer: int
    val_gain: float
    val_r2_after: float
    complexity: int
    residual_alignment: float


def max_redundancy(candidate: Expr, selected: List[Expr], Xtr: np.ndarray) -> float:
    if not selected:
        return 0.0
    vc = sanitize(eval_expr(candidate, Xtr))
    return max(corr_abs(vc, sanitize(eval_expr(e, Xtr))) for e in selected)


def discover_coordinates(
    X_select: np.ndarray,
    y_select: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    cfg: GateConfig,
) -> Tuple[List[SelectedCoord], Dict[str, object]]:
    selected_exprs = base_exprs(X_select.shape[1], cfg)
    discovered: List[SelectedCoord] = []
    trace: List[Dict[str, object]] = []

    for layer in range(cfg.max_layers):
        Ztr, fit = fit_design(selected_exprs, X_select)
        Zva = fit.transform(X_val)
        w, b0 = ridge_fit(Ztr, y_select, cfg.ridge_alpha)
        pred_tr = ridge_predict(Ztr, w, b0)
        pred_val = ridge_predict(Zva, w, b0)
        base_r2 = r2_score(y_val, pred_val)
        resid = y_select - pred_tr

        cands = generate_candidates(selected_exprs, X_select, resid, cfg)
        scored: List[Tuple[float, SelectedCoord]] = []
        # Evaluate each candidate by adding exactly one standardized coordinate
        # to the already-computed design matrix. This keeps the gate fast and
        # preserves the same normalization discipline as fit_design().
        existing_raw = [sanitize(eval_expr(e0, X_select)) for e0 in selected_exprs]
        for e in cands:
            vtr_raw = sanitize(eval_expr(e, X_select))
            if vtr_raw.std() < 1e-10:
                continue
            if existing_raw:
                if max(corr_abs(vtr_raw, v0) for v0 in existing_raw) >= cfg.redundancy_corr:
                    continue
            mu = float(vtr_raw.mean())
            sd = float(vtr_raw.std())
            if sd < 1e-10:
                continue
            ztr = ((vtr_raw - mu) / sd).reshape(-1, 1)
            vva_raw = sanitize(eval_expr(e, X_val))
            zva = ((vva_raw - mu) / sd).reshape(-1, 1)
            Ztr_c = np.hstack([Ztr, ztr])
            Zva_c = np.hstack([Zva, zva])
            wc, b0c = ridge_fit(Ztr_c, y_select, cfg.ridge_alpha)
            pred_c = ridge_predict(Zva_c, wc, b0c)
            r2_c = r2_score(y_val, pred_c)
            gain = r2_c - base_r2
            align = corr_abs(vtr_raw, resid)
            objective = gain - cfg.complexity_penalty * e.complexity() + 0.01 * align
            if gain > cfg.min_gain:
                scored.append((objective, SelectedCoord(
                    expr=e,
                    layer=layer + 1,
                    val_gain=float(gain),
                    val_r2_after=float(r2_c),
                    complexity=e.complexity(),
                    residual_alignment=float(align),
                )))
        scored.sort(key=lambda t: t[0], reverse=True)
        accepted = []
        # Sequential acceptance prevents same-layer duplicates such as c, c+1, c-1.
        # A shifted/scaled coordinate is equivalent after normalization, so it must
        # not consume another coordinate slot.
        for _, s in scored:
            trial_existing = selected_exprs + [a.expr for a in accepted]
            if max_redundancy(s.expr, trial_existing, X_select) >= cfg.redundancy_corr:
                continue
            selected_exprs.append(s.expr)
            discovered.append(s)
            accepted.append(s)
            if len(accepted) >= cfg.accept_per_layer:
                break

        trace.append({
            "layer": layer + 1,
            "base_inner_val_r2": float(base_r2),
            "n_candidates": len(cands),
            "accepted": [
                {
                    "expr": str(s.expr),
                    "gain": s.val_gain,
                    "r2_after": s.val_r2_after,
                    "complexity": s.complexity,
                    "residual_alignment": s.residual_alignment,
                }
                for s in accepted
            ],
        })
        if not accepted:
            break
    return discovered, {"trace": trace}


# -----------------------------
# OOF harness
# -----------------------------

def kfold_indices(n: int, k: int, seed: int) -> List[np.ndarray]:
    rng = np.random.default_rng(seed)
    idx = rng.permutation(n)
    return [arr.astype(int) for arr in np.array_split(idx, k)]


def split_inner(idx: np.ndarray, frac: float, seed: int) -> Tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    idx = rng.permutation(idx)
    n_val = max(1, int(round(len(idx) * frac)))
    return idx[n_val:], idx[:n_val]


def coordinate_recovery(
    Xmat: np.ndarray,
    exprs: List[Expr],
    hidden: Dict[str, np.ndarray],
) -> Dict[str, Dict[str, object]]:
    # Include all subexpressions of accepted expressions. This matters because
    # a discovered final law may contain the hidden coordinate internally.
    all_exprs: List[Expr] = []
    seen = set()
    for e in exprs:
        for s in e.subexpressions():
            k = s.key()
            if k not in seen and s.op != "const":
                seen.add(k)
                all_exprs.append(s)
    out: Dict[str, Dict[str, object]] = {}
    for name, z in hidden.items():
        best_corr = -1.0
        best_expr = None
        for e in all_exprs:
            c = corr_abs(sanitize(eval_expr(e, Xmat)), z)
            if c > best_corr:
                best_corr = c
                best_expr = str(e)
        out[name] = {
            "best_abs_corr": float(best_corr if best_corr >= 0 else 0.0),
            "best_expr": best_expr,
        }
    return out


def run_oof_gate(
    Xmat: np.ndarray,
    y: np.ndarray,
    hidden: Optional[Dict[str, np.ndarray]],
    cfg: GateConfig,
) -> Dict[str, object]:
    t0 = time.time()
    Xmat = np.asarray(Xmat, float)
    y = np.asarray(y, float).reshape(-1)
    n = len(y)
    folds = kfold_indices(n, cfg.k_folds, cfg.seed)
    yhat = np.zeros(n)
    all_selected: List[SelectedCoord] = []
    fold_summaries = []

    for fold_id, val_idx in enumerate(folds, start=1):
        train_idx = np.setdiff1d(np.arange(n), val_idx)
        inner_train_idx, inner_val_idx = split_inner(train_idx, cfg.inner_val_frac, cfg.seed + 1000 * fold_id)

        selected, info = discover_coordinates(
            Xmat[inner_train_idx], y[inner_train_idx],
            Xmat[inner_val_idx], y[inner_val_idx],
            cfg,
        )
        selected_exprs = base_exprs(Xmat.shape[1], cfg) + [s.expr for s in selected]
        Ztr, fit = fit_design(selected_exprs, Xmat[train_idx])
        Zva = fit.transform(Xmat[val_idx])
        w, b0 = ridge_fit(Ztr, y[train_idx], cfg.ridge_alpha)
        yhat[val_idx] = ridge_predict(Zva, w, b0)
        all_selected.extend(selected)
        fold_summaries.append({
            "fold": fold_id,
            "n_selected": len(selected),
            "val_r2": r2_score(y[val_idx], yhat[val_idx]),
            "selected": [str(s.expr) for s in selected],
            "trace": info["trace"],
        })

    selected_exprs_only = [s.expr for s in all_selected]
    recovery = coordinate_recovery(Xmat, selected_exprs_only, hidden or {}) if hidden else {}
    result = {
        "protocol": "FASE-G1a-v23 primitive recursive coordinate discovery / nested OOF",
        "include_sqrt": cfg.include_sqrt,
        "A0": ["+", "-", "*", "/", "compose", "normalize", "project"],
        "radical_extension": ["sqrt_pos"] if cfg.include_sqrt else [],
        "n": int(n),
        "d": int(Xmat.shape[1]),
        "k_folds": cfg.k_folds,
        "R2_oof": float(r2_score(y, yhat)),
        "MSE_oof": float(mse(y, yhat)),
        "elapsed_sec": float(time.time() - t0),
        "n_total_selected": len(all_selected),
        "top_selected_expressions": top_expression_counts(all_selected, 12),
        "coordinate_recovery": recovery,
        "folds": fold_summaries,
        "config": cfg.__dict__,
    }
    return result


def top_expression_counts(selected: List[SelectedCoord], k: int) -> List[Dict[str, object]]:
    counts: Dict[str, int] = {}
    gains: Dict[str, List[float]] = {}
    for s in selected:
        key = str(s.expr)
        counts[key] = counts.get(key, 0) + 1
        gains.setdefault(key, []).append(s.val_gain)
    rows = []
    for key, c in counts.items():
        rows.append({"expr": key, "count": c, "mean_inner_gain": float(np.mean(gains[key]))})
    rows.sort(key=lambda r: (r["count"], r["mean_inner_gain"]), reverse=True)
    return rows[:k]


# -----------------------------
# Synthetic smoke datasets
# -----------------------------

def make_mul(seed: int = 42, n: int = 600) -> Tuple[np.ndarray, np.ndarray, Dict[str, np.ndarray]]:
    rng = np.random.default_rng(seed)
    Xmat = rng.uniform(-2, 2, size=(n, 3))
    z = Xmat[:, 0] * Xmat[:, 1]
    y = 2.0 * z + 0.1 * Xmat[:, 2] + rng.normal(0, 0.02, size=n)
    return Xmat, y, {"z_mul_x0_x1": z}


def make_composed(seed: int = 42, n: int = 700) -> Tuple[np.ndarray, np.ndarray, Dict[str, np.ndarray]]:
    rng = np.random.default_rng(seed)
    Xmat = rng.uniform(-2, 2, size=(n, 4))
    z1 = Xmat[:, 0] + Xmat[:, 1]
    z2 = Xmat[:, 2] - Xmat[:, 3]
    z3 = z1 * z2
    y = z3 + rng.normal(0, 0.02, size=n)
    return Xmat, y, {"z_sum_x0_x1": z1, "z_sub_x2_x3": z2, "z_product": z3}


def make_ratio(seed: int = 42, n: int = 600) -> Tuple[np.ndarray, np.ndarray, Dict[str, np.ndarray]]:
    rng = np.random.default_rng(seed)
    x0 = rng.uniform(-2, 2, size=n)
    # keep denominator away from zero so division is a legitimate primitive, not a numerical trap
    x1 = rng.choice([-1, 1], size=n) * rng.uniform(0.4, 2.0, size=n)
    x2 = rng.uniform(-1, 1, size=n)
    Xmat = np.column_stack([x0, x1, x2])
    z = x0 / x1
    y = 1.5 * z + 0.05 * x2 + rng.normal(0, 0.01, size=n)
    return Xmat, y, {"z_ratio_x0_x1": z}


def make_projection_product(seed: int = 42, n: int = 800, d: int = 5) -> Tuple[np.ndarray, np.ndarray, Dict[str, np.ndarray]]:
    rng = np.random.default_rng(seed)
    Xmat = rng.normal(size=(n, d))
    w1 = rng.normal(size=d); w1 /= np.linalg.norm(w1) + EPS
    w2 = rng.normal(size=d); w2 -= w1 * (w1 @ w2); w2 /= np.linalg.norm(w2) + EPS
    z1 = Xmat @ w1
    z2 = Xmat @ w2
    z3 = z1 * z2
    y = z3 + rng.normal(0, 0.02, size=n)
    return Xmat, y, {"z_project_1": z1, "z_project_2": z2, "z_project_product": z3}


def make_vieta_transition(seed: int = 42, n: int = 700) -> Tuple[np.ndarray, np.ndarray, Dict[str, np.ndarray]]:
    """Viète-inspired state transition law.

    Viète nested radical recurrence can be written as:
        a_0 = 0
        a_{k+1} = sqrt(2 + a_k)
        p_{k+1} = p_k * a_{k+1}/2
        pi_k ≈ 2/p_k

    We train on random states (a, p) in an extended positive-radical domain, not just the tiny
    finite trajectory, so the target is the transition law. The extended domain prevents the radical from being nearly linear on [0,2]:
        y = p_next = p * sqrt(2 + a)/2.
    """
    rng = np.random.default_rng(seed)
    a = rng.uniform(-1.5, 8.0, size=n)
    p = rng.uniform(0.2, 2.0, size=n)
    Xmat = np.column_stack([a, p])
    z_rad = np.sqrt(2.0 + a)
    z_next_p = p * z_rad / 2.0
    y = z_next_p + rng.normal(0, 0.001, size=n)
    return Xmat, y, {"z_sqrt_2_plus_a": z_rad, "z_p_next": z_next_p}


def vieta_trajectory(n_terms: int = 12) -> Dict[str, List[float]]:
    a = 0.0
    p = 1.0
    rows = []
    for k in range(1, n_terms + 1):
        a = math.sqrt(2.0 + a)
        p *= a / 2.0
        rows.append({"k": k, "a_k": a, "p_k": p, "pi_est": 2.0 / p, "abs_err": abs(math.pi - 2.0 / p)})
    return {"rows": rows, "pi": math.pi}


DATASETS = {
    "mul": make_mul,
    "composed": make_composed,
    "ratio": make_ratio,
    "projection_product": make_projection_product,
    "vieta": make_vieta_transition,
}


def summarize_gate(result: Dict[str, object], r2_thr: float = 0.95, recovery_thr: float = 0.95) -> Dict[str, object]:
    rec = result.get("coordinate_recovery", {}) or {}
    recovery_pass = True
    for _, item in rec.items():
        if item.get("best_abs_corr", 0.0) < recovery_thr:
            recovery_pass = False
            break
    return {
        "R2_oof": result["R2_oof"],
        "MSE_oof": result["MSE_oof"],
        "r2_pass": bool(result["R2_oof"] >= r2_thr),
        "coordinate_recovery_pass": bool(recovery_pass if rec else True),
        "gate_pass": bool(result["R2_oof"] >= r2_thr and (recovery_pass if rec else True)),
        "top_selected_expressions": result["top_selected_expressions"],
        "coordinate_recovery": rec,
        "elapsed_sec": result["elapsed_sec"],
    }


def run_demo(seed: int, quick: bool = False) -> Dict[str, object]:
    cfg = GateConfig(
        seed=seed,
        k_folds=3 if quick else 5,
        max_layers=2 if quick else 3,
        accept_per_layer=2,
        candidate_depth=2,
        max_candidates_scored=160 if quick else 1500,
        depth1_beam=22 if quick else 70,
        n_random_projects=0,
        include_sqrt=False,
    )
    outputs = {}
    for name in ["mul", "composed", "ratio"]:
        Xmat, y, hidden = DATASETS[name](seed=seed, n=180 if quick else 650)
        res = run_oof_gate(Xmat, y, hidden, cfg)
        outputs[name + "__A0"] = summarize_gate(res)

    # Viète strict A0: intentionally lacks sqrt. This should reveal the limit.
    Xv, yv, hv = make_vieta_transition(seed=seed, n=180 if quick else 650)
    res_v_strict = run_oof_gate(Xv, yv, hv, cfg)
    outputs["vieta__strict_A0"] = summarize_gate(res_v_strict)

    # Viète radical extension.
    cfg_sqrt = GateConfig(**{**cfg.__dict__, "include_sqrt": True})
    res_v_sqrt = run_oof_gate(Xv, yv, hv, cfg_sqrt)
    outputs["vieta__A0sqrt"] = summarize_gate(res_v_sqrt)
    outputs["vieta_trajectory_reference"] = vieta_trajectory(10)
    return outputs


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--test", choices=list(DATASETS.keys()) + ["demo"], default="demo")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--n", type=int, default=650)
    ap.add_argument("--include-sqrt", action="store_true", help="Enable radical extension A0 + sqrt_pos.")
    ap.add_argument("--quick", action="store_true", help="Smaller, faster smoke run.")
    ap.add_argument("--json-out", type=str, default=None)
    args = ap.parse_args()

    if args.test == "demo":
        out = run_demo(args.seed, quick=args.quick)
    else:
        cfg = GateConfig(
            seed=args.seed,
            k_folds=3 if args.quick else 5,
            max_layers=2 if args.quick else 3,
            candidate_depth=2,
            max_candidates_scored=160 if args.quick else 1500,
            depth1_beam=22 if args.quick else 70,
            include_sqrt=args.include_sqrt,
        )
        Xmat, y, hidden = DATASETS[args.test](seed=args.seed, n=args.n)
        res = run_oof_gate(Xmat, y, hidden, cfg)
        out = {"result": res, "summary": summarize_gate(res)}
        if args.test == "vieta":
            out["vieta_trajectory_reference"] = vieta_trajectory(12)

    text = json.dumps(out, indent=2)
    print(text)
    if args.json_out:
        with open(args.json_out, "w", encoding="utf-8") as f:
            f.write(text)


if __name__ == "__main__":
    main()
