# ogset.py
from __future__ import annotations
from dataclasses import dataclass
from typing import List, Optional, Dict, Tuple
import numpy as np
import random

# ---------- dataclasses ----------
@dataclass
class OgsetOp:
    name: str
    idx: int
    coeff: float
    stderr: float
    delta_bic_bits: float
    sel_rank: int
    freq: float = 1.0

@dataclass
class OgsetResult:
    intercept: float
    ops: List[OgsetOp]
    yhat: np.ndarray
    resid: np.ndarray
    gauge_scales: np.ndarray
    W_info: Dict[str, float]          # cond(W), min_eig, max_eig, cond(XtWX)
    selected_idx: List[int]
    bits_path: List[float]
    rss_path: List[float]
    r2_train_path: List[float]
    og_r2_train: float
    og_r2_val: Optional[float]

# ---------- gauge helpers ----------
def _as_weight_matrix(Sigma: Optional[np.ndarray], n: int) -> np.ndarray:
    if Sigma is None:
        return np.eye(n)
    Sig = np.asarray(Sigma, float)
    if Sig.ndim == 1:
        inv = np.where(Sig > 0, 1.0 / np.clip(Sig, 1e-12, None), 0.0)
        return np.diag(inv)
    S = 0.5 * (Sig + Sig.T)
    return np.linalg.pinv(S, rcond=1e-12)

def _whitener_from_W(W: np.ndarray) -> np.ndarray:
    try:
        # L^T L = W
        L = np.linalg.cholesky(W)
        return L.T
    except np.linalg.LinAlgError:
        evals, evecs = np.linalg.eigh(W)
        evals = np.clip(evals, 0.0, None)
        print("[Σ] using eig-sqrt whitener (Cholesky failed)",
              f"min_eig={float(evals.min()):.3e}",
              f"max_eig={float(evals.max()):.3e}")
        return (evecs @ np.diag(np.sqrt(evals)) @ evecs.T)

# ---------- model selection ----------
def _ebic_bits(RSS: float, n: int, k: int, p: int, gamma: float) -> float:
    # EBIC in bits; log C(p,k) ≈ k log p (conservative and fast)
    ebic = n * np.log(max(RSS / n, 1e-300)) + k * np.log(n)
    if gamma > 0 and k > 0 and p > 1:
        ebic += 2.0 * gamma * k * np.log(p)
    return -ebic / np.log(2.0)

def build_ogset(
    y: np.ndarray,
    Phi: np.ndarray,
    atom_names: List[str],
    Sigma: Optional[np.ndarray] = None,
    max_ops: Optional[int] = None,
    bic_bits_threshold: Optional[float] = None,
    ebic_gamma: Optional[float] = None,
    max_corr: float = 0.98,
    bag_boots: int = 0,
    bag_frac: float = 0.8,
    min_freq: float = 0.0,
    rng: Optional[random.Random] = None,
    include_intercept: bool = True,
    # optional validation to report OG-only R²
    y_val: Optional[np.ndarray] = None,
    Phi_val: Optional[np.ndarray] = None,
) -> OgsetResult:

    y = np.asarray(y).reshape(-1)
    n, p = Phi.shape
    if rng is None:
        rng = random.Random(1337)

    # n-aware defaults
    if bic_bits_threshold is None:
        bic_bits_threshold = 1.0 if n <= 500 else 0.7
    if max_ops is None:
        max_ops = 10 if n <= 500 else 16
    if ebic_gamma is None:
        ebic_gamma = 0.25 if n <= 500 else 0.0

    W = _as_weight_matrix(Sigma, n)
    L = _whitener_from_W(W)
    evals = np.linalg.eigvalsh(W)
    min_e, max_e = float(np.clip(evals.min(), 0.0, None)), float(evals.max())
    condW = float(max_e / max(min_e, 1e-16))

    Pw = L @ Phi
    yw = L @ y

    # unit-gauge columns
    scales = np.linalg.norm(Pw, axis=0)
    scales_safe = np.where(scales > 0, scales, 1.0)
    Pw_unit = Pw / scales_safe

    # intercept
    if include_intercept:
        if np.allclose(W, np.diag(np.diag(W))):
            mu = float((np.diag(W) @ y).sum() / np.diag(W).sum())
        else:
            mu = float(np.linalg.lstsq(L @ np.ones((n,1)), yw, rcond=None)[0])
    else:
        mu = 0.0
    y0 = y - mu
    yw0 = L @ y0

    RSS0 = float(yw0 @ yw0)
    k0 = 1 if include_intercept else 0
    best_bits = _ebic_bits(RSS0, n, k0, p, ebic_gamma)

    selected: List[int] = []
    ops: List[OgsetOp] = []
    bits_path, rss_path, r2_train_path = [best_bits], [RSS0], [1.0 - RSS0 / (yw @ yw + 1e-300)]

    available = set(range(p))
    while len(selected) < max_ops and available:
        best_j, best_gain_bits, best_sol = -1, 0.0, None
        for j in available:
            Xw = Pw_unit[:, [*selected, j]]
            beta_u, *_ = np.linalg.lstsq(Xw, yw0, rcond=None)
            resid = yw0 - Xw @ beta_u
            RSS = float(resid @ resid)
            bits = _ebic_bits(RSS, n, k0 + len(beta_u), p, ebic_gamma)
            gain = bits - best_bits
            if selected:
                cw = Pw_unit[:, j]
                corr = float(np.max(np.abs(Pw_unit[:, selected].T @ cw)))
                if corr > max_corr:
                    continue
            if gain > best_gain_bits:
                best_j, best_gain_bits, best_sol = j, gain, (beta_u, RSS)
        if best_j < 0 or best_gain_bits < bic_bits_threshold:
            break

        selected.append(best_j)
        available.remove(best_j)
        beta_u, RSS = best_sol
        best_bits += best_gain_bits

        # diagnostics at this step
        Xw = Pw_unit[:, selected]
        beta_u, *_ = np.linalg.lstsq(Xw, yw0, rcond=None)
        resid_w = yw0 - Xw @ beta_u
        sigma2 = float(resid_w @ resid_w / max(n - (k0 + len(selected)), 1))
        XtX_inv = np.linalg.pinv(Xw.T @ Xw, rcond=1e-12)
        se_u = np.sqrt(np.clip(np.diag(XtX_inv) * sigma2, 0.0, None))
        bits_path.append(best_bits)
        rss_path.append(float(resid_w @ resid_w))
        r2_train_path.append(1.0 - rss_path[-1] / (yw @ yw + 1e-300))

        beta = beta_u / scales_safe[selected]
        se = se_u / scales_safe[selected]
        j = selected[-1]
        ops.append(OgsetOp(
            name=atom_names[j], idx=int(j),
            coeff=float(beta[-1]), stderr=float(se[-1]),
            delta_bic_bits=float(best_gain_bits), sel_rank=len(selected)
        ))

    # final GLS on selected columns
    if selected:
        X = Phi[:, selected]
        XtW = X.T @ W
        G = XtW @ X
        beta = np.linalg.lstsq(G, XtW @ y0, rcond=None)[0]
        resid = y0 - X @ beta
        sigma2 = float(resid.T @ W @ resid / max(n - (k0 + len(selected)), 1))
        cov = np.linalg.pinv(G, rcond=1e-12) * sigma2
        se = np.sqrt(np.clip(np.diag(cov), 0.0, None))
        for r, (cj, sej) in enumerate(zip(beta, se)):
            ops[r].coeff = float(cj)
            ops[r].stderr = float(sej)
        yhat = mu + X @ beta
        cond_XtWX = float(np.linalg.cond(G))
    else:
        yhat = np.full_like(y, mu, dtype=float)
        resid = y - yhat
        cond_XtWX = float("nan")

    # optional bagging selection frequency (diagnostic)
    if bag_boots and selected:
        counts = {j: 0 for j in selected}
        m = max(4, int(bag_frac * n))
        for _ in range(bag_boots):
            idx = np.array(sorted(rng.sample(range(n), m)))
            sub = build_ogset(y[idx], Phi[idx], atom_names, Sigma=None,
                              max_ops=len(selected), bic_bits_threshold=bic_bits_threshold,
                              ebic_gamma=ebic_gamma, max_corr=max_corr,
                              bag_boots=0, include_intercept=include_intercept,
                              min_freq=0.0)
            for j in sub.selected_idx:
                if j in counts:
                    counts[j] += 1
        for op in ops:
            op.freq = counts.get(op.idx, 0) / float(bag_boots)
        if min_freq > 0:
            keep = [i for i, op in enumerate(ops) if op.freq >= min_freq]
            selected = [selected[i] for i in keep]
            ops = [ops[i] for i in keep]

    og_r2_train = 1.0 - float((resid.T @ W @ resid) / max(y0.T @ W @ y0, 1e-300))
    og_r2_val = None
    if (y_val is not None) and (Phi_val is not None) and selected:
        yv = np.asarray(y_val).reshape(-1)
        Xv = Phi_val[:, selected]
        yhat_v = mu + Xv @ beta
        ss_res = float(np.sum((yv - yhat_v) ** 2))
        ss_tot = float(np.sum((yv - np.mean(yv)) ** 2))
        og_r2_val = 1.0 - (ss_res / max(ss_tot, 1e-300))

    return OgsetResult(
        intercept=float(mu), ops=ops,
        yhat=np.asarray(yhat), resid=np.asarray(resid),
        gauge_scales=scales_safe,
        W_info={"cond": condW, "min_eig": min_e, "max_eig": max_e, "cond_XtWX": cond_XtWX},
        selected_idx=selected,
        bits_path=bits_path, rss_path=rss_path, r2_train_path=r2_train_path,
        og_r2_train=og_r2_train, og_r2_val=og_r2_val
    )

# ---------- export W-orthonormal super-ops ----------
def export_superfeatures_orthonormal(Phi_sel: np.ndarray, L: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Return (X_train, T) with columns W-orthonormal on TRAIN:
      A = L @ Phi_sel; A = Q R (QR); T = R^{-1}; X_train = Phi_sel @ T
      So L @ X_train = Q (orthonormal). Reuse T to get X_val = Phi_val @ T.
    """
    A = L @ Phi_sel
    Q, R = np.linalg.qr(A, mode="reduced")
    T = np.linalg.inv(R)
    X_train = Phi_sel @ T
    return X_train, T

def apply_superfeatures(Phi_sel: np.ndarray, T: np.ndarray) -> np.ndarray:
    return Phi_sel @ T

# legacy export if you ever need raw selected columns
def make_ogset_superfeatures(Phi: np.ndarray, selected_idx: List[int]) -> np.ndarray:
    return Phi[:, selected_idx]