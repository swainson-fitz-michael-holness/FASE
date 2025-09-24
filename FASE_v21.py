# FASE_v21_kfold.py — Σ‑aware OG‑SET + k‑fold, RDMP(GLS), stability selection
# Single-file drop‑in built from your v20.1 with the improvements discussed.
# Optional deps: pmlb (datasets), pysr (baseline). No sklearn required.
#
# Usage (demo):
#   python FASE_v21_kfold.py
#
# Key changes vs v20.1:
#  - K‑fold CV driver (no external deps), OOF metrics, operator stability.
#  - Σ‑weighted residualization (RDMP) everywhere (fit_residualization_gls).
#  - OG‑SET bagging now passes fold Σ (GLS‑consistent stability freq).
#  - Robust W‑orthonormal export (QR + triangular solve, SVD fallback).
#  - Clean subset_sigma helper; no explicit Σ⁻¹ inverses exposed to user code.
#  - Clear evidence in bits via ΔBIC → bits for all accepted blocks.

from __future__ import annotations

import numpy as np, math, random, copy, hashlib, warnings, itertools, time, json, logging
from dataclasses import dataclass, field
from typing import List, Dict, Any, Tuple, Optional, Callable
from collections import Counter

# =========================
# Config
# =========================
CONFIG = {
    "LOG_LEVEL": "INFO",
    "SEEDS": [42, 1337],
    "K_FOLDS": 5,
    "DATASETS": [
        # PMLB demo list (comment out to run your own data)
        "579_fri_c0_250_5",
        "581_fri_c3_500_25",
        # "582_fri_c1_500_25",
        # "593_fri_c1_1000_10",
        # "596_fri_c2_250_5"
    ],
    "LINEAR_LOCK_THR": 0.95,
    "MAX_ATOMS": 36,
    "MAX_GRAMMAR": 12,
    "ALPHA_RIDGE": 1e-3,
    "RDMP_ALPHA": 1e-4,
    "STAGE1": {"CRITERION":"MDL","MIN_MSE_GAIN":1e-4,"DUP_CORR_THR":0.995},
    "USE_RULIAD_STAGE25": True,
    "RULIAD": {
        "depth": 6, "K_per_parent": 25, "frontier_size": 24, "random_seed": 42,
        "keep_outputs_topk": 24, "energy": {"lam":1.0,"mu":0.05,"nu":0.05,"mdl_scale":1.0,"xi":0.02,"sign_flip_indices":()},
        "use_param_opt": True
    },
    "MDL_COSTS": {
        "type_bits":{"bilinear":10.0,"relu_proj":8.0,"sinproj":8.0,"fct":11.0,"perm_invar":7.0,
                     "dihedral_invar":8.0,"group_invar":10.0,"combo_vec":10.0,"hsml_unary":5.5,
                     "hsml_binary":7.5,"atomic":4.0,"ruliad":12.0},
        "per_col_bits":0.75,"per_real_bits":1.5,"bit_weight":1.25
    },
    # ---------- OG-SET knobs ----------
    "OGSET": {
        "enable": True,
        "max_ops": None,                 # auto: 10 if n<=500 else 16
        "bic_bits_threshold": None,      # auto: 0.8 (small n) / 0.6 (large n)
        "ebic_gamma": None,              # auto: 0.25 (small n) / 0.0 (large n)
        "max_corr": 0.98,                # block near-duplicate atoms
        "bag_boots": 8,                  # 0 disables selection-stability bagging
        "bag_frac": 0.8,
        "min_freq": 0.60,                # drop ops selected in <60% of bags
        "orthonormal_export": True,      # export W-orthonormal OG features (for baselines)
        "augment_pysr": True,
        # Final inclusion rule after K folds (consensus model)
        "final_min_freq": 0.60,
        "final_min_sign_stab": 0.90,
        "final_min_bits": 6.0,
    },
    # ---------- PySR baseline ----------
    "COMPARE_WITH_PYSR": True,
    "PYSR": {
        "niterations": 250, "maxsize": 18, "maxdepth": 8,
        "binary_operators": ["+","-","*","/"], "unary_operators": ["sin","cos","tanh","exp","log","abs"],
        "batching": True, "deterministic": True, "parallelism": "serial"
    }
}
EPS = 1e-9

logging.basicConfig(level=getattr(logging, CONFIG.get("LOG_LEVEL", "INFO")))
logger = logging.getLogger("FASE")

import builtins as _builtins
_builtins.print = lambda *a, **k: logger.info(" ".join(str(x) for x in a))

# =========================
# Utils: Σ handling, whitening, criteria
# =========================
def subset_sigma(Sigma: Optional[np.ndarray], idx: np.ndarray) -> Optional[np.ndarray]:
    if Sigma is None:
        return None
    Sigma = np.asarray(Sigma)
    if Sigma.ndim == 1:
        return Sigma[idx]
    return Sigma[np.ix_(idx, idx)]

def chol_whiten(Sigma: np.ndarray):
    """Diag-only Cholesky; full Σ delegates to robust whitener path."""
    Sig = np.asarray(Sigma, float)
    if Sig.ndim == 1:
        w = np.clip(Sig, 1e-12, None)
        L = np.diag(np.sqrt(w)); Linv = np.diag(1.0/np.sqrt(w))
        return L, Linv
    W = _as_weight_matrix(Sig, Sig.shape[0])
    C = _whitener_from_W(W)
    L = np.linalg.inv(C.T)
    Linv = C
    return L, Linv

def _as_weight_matrix(Sigma: Optional[np.ndarray], n: int) -> np.ndarray:
    if Sigma is None:
        return np.eye(n)
    Sig = np.asarray(Sigma, float)
    if Sig.ndim == 1:
        inv = np.where(Sig>0, 1.0/np.clip(Sig,1e-12,None), 0.0)
        return np.diag(inv)
    S = 0.5*(Sig + Sig.T)
    return np.linalg.pinv(S, rcond=1e-12)

def _whitener_from_W(W: np.ndarray) -> np.ndarray:
    try:
        C = np.linalg.cholesky(W)
        return C.T
    except np.linalg.LinAlgError:
        evals, evecs = np.linalg.eigh(W)
        evals = np.clip(evals, 0.0, None)
        print("[Σ] using eig-sqrt whitener (Cholesky failed)",
              f"min_eig={float(evals.min()):.3e}",
              f"max_eig={float(evals.max()):.3e}")
        return evecs @ np.diag(np.sqrt(evals)) @ evecs.T

def _cached_whitener(Sigma: np.ndarray, n: int) -> Tuple[np.ndarray, np.ndarray]:
    W = _as_weight_matrix(Sigma, n)
    C = _whitener_from_W(W)
    return W, C

def gls_chi2(residuals: np.ndarray, Sigma: np.ndarray) -> float:
    r = np.asarray(residuals, float).reshape(-1)
    if Sigma is None:
        return (r @ r).item()
    W, _ = _cached_whitener(Sigma, len(r))
    return (r @ (W @ r)).item()

# Criteria in GLS geometry

def r2_score(y, yhat):
    y = np.asarray(y).ravel(); yhat = np.asarray(yhat).ravel()
    ss_res = np.sum((y - yhat)**2); ss_tot = np.sum((y - y.mean())**2) + 1e-12
    return 1.0 - ss_res/ss_tot

def gls_r2(y, yhat, Sigma):
    y = np.asarray(y).ravel(); yhat = np.asarray(yhat).ravel()
    n = len(y)
    if Sigma is None:
        return r2_score(y, yhat)
    W = _as_weight_matrix(Sigma, n)
    one = np.ones((n,1))
    denom = (one.T @ W @ one).item() + 1e-12
    ybar = ((one.T @ W @ y.reshape(-1,1)) / denom).item()
    r = y - yhat
    num = (r.T @ W @ r).item()
    den = ((y - ybar).T @ W @ (y - ybar)).item() + 1e-12
    return 1.0 - num/den

def aicc_from_residuals(residuals, n, k):
    rss = np.sum(residuals**2)
    if n <= k + 1: return np.inf
    sigma2 = rss / n
    AIC = n*np.log(sigma2 + EPS) + 2*k
    corr = (2*k*(k+1)) / (n - k - 1)
    return AIC + corr

def bic_from_residuals(residuals, n, k):
    rss = np.sum(residuals**2); sigma2 = rss / n
    return n*np.log(sigma2 + EPS) + k*np.log(n + EPS)

def crit_from_residuals(residuals, n, k, criterion="AICc", mdl_bits=0.0, bit_weight=1.0, Sigma: Optional[np.ndarray]=None):
    crit = criterion.upper()
    if crit == "AICc":
        if Sigma is not None:
            chi2 = gls_chi2(residuals, Sigma)
            AIC = n*np.log((chi2/max(1,n)) + EPS) + 2*k
            corr = (2*k*(k+1))/(n - k - 1) if n>k+1 else np.inf
            return AIC + corr
        return aicc_from_residuals(residuals, n, k)
    if crit == "BIC":
        if Sigma is not None:
            chi2 = gls_chi2(residuals, Sigma)
            return n*np.log((chi2/max(1,n)) + EPS) + k*np.log(n + EPS)
        return bic_from_residuals(residuals, n, k)
    if crit == "MDL":
        if Sigma is not None:
            chi2 = gls_chi2(residuals, Sigma)
            return n*np.log((chi2/max(1,n)) + EPS) + bit_weight*mdl_bits
        rss = np.sum(residuals**2); sigma2 = rss / max(1, n)
        return n*np.log(sigma2 + EPS) + bit_weight*mdl_bits
    raise ValueError("Unknown criterion")

def bits_from_delta_bic(delta_bic: float) -> float:
    return -delta_bic/(2.0*math.log(2.0))

def w_mean_std(V: np.ndarray, Sigma: Optional[np.ndarray]):
    V = np.asarray(V, float)
    if Sigma is None:
        mu = V.mean(axis=0, keepdims=True)
        sd = V.std(axis=0, keepdims=True)
        sd[sd<1e-12]=1.0
        return mu, sd
    n = V.shape[0]
    W = _as_weight_matrix(Sigma, n)
    one = np.ones((n,1))
    denom = (one.T @ W @ one).item() + 1e-12
    mu = (one.T @ W @ V) / denom
    C = V - mu
    sd = np.sqrt(np.maximum(np.sum(C * (W @ C), axis=0, keepdims=True) / denom, 1e-12))
    return mu, sd

# =========================
# Ridge (GLS-aware)
# =========================

def ridge_with_intercept(X, y, alpha, Sigma: Optional[np.ndarray]=None):
    X = np.asarray(X, float); y = np.asarray(y, float).ravel()
    n, d = X.shape
    if d == 0:
        if Sigma is None:
            mu_y = float(y.mean())
        else:
            W = _as_weight_matrix(Sigma, n)
            one = np.ones((n,1))
            denom = (one.T @ W @ one).item() + 1e-12
            mu_y = ((one.T @ W @ y.reshape(-1,1)) / denom).item()
        return np.zeros(0), mu_y, {"muX": np.zeros((1,0)), "muy": mu_y, "Sigma": Sigma}

    W = _as_weight_matrix(Sigma, n) if Sigma is not None else None
    if W is None:
        muX = X.mean(axis=0, keepdims=True); muy = float(y.mean())
        Xc = X - muX; yc = y - muy
        A = Xc.T @ Xc + alpha * np.eye(d)
        b = Xc.T @ yc
        w = np.linalg.solve(A, b)
        b0 = muy - (muX @ w.reshape(-1,1)).item()
        return w, b0, {"muX": muX, "muy": muy, "Sigma": None}
    else:
        one = np.ones((n,1))
        denom = (one.T @ W @ one).item() + 1e-12
        muX = (one.T @ W @ X) / denom
        muy = ((one.T @ W @ y.reshape(-1,1)) / denom).item()
        Xc = X - muX
        yc = y - muy
        A = Xc.T @ W @ Xc + alpha * np.eye(d)
        b = Xc.T @ W @ yc
        w = np.linalg.solve(A, b)
        b0 = muy - (muX @ w.reshape(-1,1)).item()
        return w, b0, {"muX": muX, "muy": muy, "Sigma": Sigma}

def predict_with_intercept(X, w, b0):
    return (X @ w + b0) if w.size else np.full(X.shape[0], b0)

# =========================
# Safe corr & scaling
# =========================

def corrcoef_safe(X: np.ndarray) -> np.ndarray:
    X = np.asarray(X, float)
    if X.ndim == 1: X = X[:, None]
    n, d = X.shape
    if d == 0: return np.zeros((0, 0))
    if d == 1: return np.array([[1.0]])
    Xc = X - np.mean(X, axis=0, keepdims=True)
    std = Xc.std(axis=0, ddof=0)
    good = std > 1e-12
    inv = np.zeros_like(std)
    inv[good] = 1.0 / std[good]
    C = (Xc.T @ Xc) / max(n - 1, 1)
    R = C * (inv[:, None] * inv[None, :])
    R[~np.isfinite(R)] = 0.0
    for i in range(d):
        R[i, i] = 1.0 if good[i] else 0.0
    return R

def safe_num(a): return np.nan_to_num(a, nan=0.0, posinf=1e6, neginf=-1e6)

def col_scale_fit(v):
    v = np.asarray(v).reshape(-1,1); mu = float(v.mean()); sd = float(v.std())
    if sd < 1e-12: sd = 1.0
    return mu, sd

def col_scale_apply(v, mu, sd): return (np.asarray(v).reshape(-1,1) - mu) / sd

# =========================
# Σ-weighted residualization (RDMP-GLS)
# =========================

def fit_residualization_gls(Fb_tr, Fbase_tr, Sigma_tr, alpha=1e-4):
    Fb_tr = np.asarray(Fb_tr, float); Fbase_tr = np.asarray(Fbase_tr, float)
    if Fb_tr.size == 0 or Fbase_tr.size == 0:
        return np.zeros((Fbase_tr.shape[1] if Fbase_tr.ndim==2 else 0,
                         Fb_tr.shape[1] if Fb_tr.ndim==2 else 0))

    if Sigma_tr is None:
        Fw, Zw = Fbase_tr, Fb_tr
    else:
        n = Fb_tr.shape[0]
        _, C = _cached_whitener(Sigma_tr, n)
        Fw, Zw = C @ Fbase_tr, C @ Fb_tr

    alpha_i = float(alpha)
    for _ in range(3):
        G = Fw.T @ Fw + alpha_i * np.eye(Fw.shape[1])
        R = Fw.T @ Zw
        try:
            return np.linalg.solve(G, R)
        except np.linalg.LinAlgError:
            alpha_i *= 10.0
    return np.linalg.pinv(G, rcond=1e-12) @ R

def apply_residualization(Fb, Fbase, Gamma):
    return Fb if Gamma.size == 0 else (Fb - Fbase @ Gamma)

# =========================
# Group structures (L4)
# =========================
@dataclass
class GroupSpec:
    sign_groups: List[List[int]] = None
    perm_groups: List[List[int]] = None
    rot2d_pairs: List[Tuple[int,int]] = None
    scale_groups: List[List[int]] = None
    def __post_init__(self):
        self.sign_groups = self.sign_groups or []
        self.perm_groups = self.perm_groups or []
        self.rot2d_pairs = self.rot2d_pairs or []
        self.scale_groups = self.scale_groups or []

class GroupInvariantFeatures:
    def __init__(self, spec: GroupSpec, include_raw=False, degree=2, eps=1e-12):
        self.spec = spec; self.include_raw = include_raw; self.degree = degree; self.eps = eps
    def fit(self, X, y=None): return self
    def transform(self, X):
        X = np.asarray(X, float); N, D = X.shape
        feats = []
        if self.include_raw: feats.append(X)
        for idxs in self.spec.sign_groups:
            V = X[:, idxs]
            feats += [np.abs(V), V**2,
                      np.sum(np.abs(V),1,keepdims=True),
                      np.sum(V**2,1,keepdims=True)]
        for idxs in self.spec.perm_groups:
            V = X[:, idxs]
            s1 = np.sum(V,1,keepdims=True); s2 = np.sum(V**2,1,keepdims=True)
            pair_sum = 0.5*(s1**2 - s2)
            feats += [s1, s2, pair_sum]
        for (i,j) in self.spec.rot2d_pairs:
            xi, xj = X[:,i], X[:,j]; r2 = (xi*xi + xj*xj)[:,None]
            feats.append(r2)
        for idxs in self.spec.scale_groups:
            V = X[:, idxs]; nrm = np.sqrt(np.sum(V**2,1,keepdims=True)) + self.eps
            feats += [V/nrm, np.log(nrm + self.eps)]
        return np.hstack(feats) if feats else np.zeros((N,0))

# =========================
# HSML atomic candidate bank
# =========================

def make_unary_bank():
    def id(x): return x
    def sq(x): return x*x
    def cube(x): return x*x*x
    def absx(x): return np.abs(x)
    def sgnx(x): return np.sign(x)*x
    def sgn_sqrt(x): return np.sign(x)*np.sqrt(np.abs(x)+EPS)
    def sqrtabs(x): return np.sqrt(np.abs(x)+EPS)
    def log1pabs(x): return np.log1p(np.abs(x))
    def recp1(x): return x/(1.0+np.abs(x))
    def recp2(x): return x/(1.0+x*x)
    def invp(x): return 1.0/(np.abs(x)+1.0)
    def softsign(x): return x/(1.0+np.abs(x))
    def tanh(x): return np.tanh(x)
    def sigmoid(x): return 1.0/(1.0+np.exp(-np.clip(x, -20, 20)))
    def atan(x): return np.arctan(x)
    def sin(x): return np.sin(x)
    def cos(x): return np.cos(x)
    def sinh_sat(x): return np.sinh(np.clip(x, -3, 3))
    def relu(x): return np.maximum(0.0, x)
    def lrelu(x): return np.where(x>=0, x, 0.1*x)
    def elu(x): return np.where(x>0, x, np.exp(np.clip(x,-20,0))-1)
    def gelu(x): return 0.5*x*(1.0+np.tanh(np.sqrt(2/np.pi)*(x+0.044715*x**3)))
    def swish(x): return x*sigmoid(x)
    def hat(x): return np.clip(np.abs(x), 0, 3.0)
    return {
        "x": id, "x^2": sq, "x^3": cube, "|x|": absx, "sgn*x": sgnx,
        "sgn*sqrt|x|": sgn_sqrt, "sqrt|x|": sqrtabs, "log1p|x|": log1pabs,
        "x/(1+|x|)": recp1, "x/(1+x^2)": recp2, "1/(1+|x|)": invp, "softsign": softsign,
        "tanh": tanh, "sigmoid": sigmoid, "atan": atan, "sin": sin, "cos": cos,
        "sinh_sat": sinh_sat, "relu": relu, "lrelu": lrelu, "elu": elu, "gelu": gelu, "swish": swish, "hat": hat,
    }
UNARY_FUNCS = make_unary_bank()

def projection_defs(X, rng, n_proj=24):
    N, D = X.shape; defs = []
    if D == 1:
        defs.append(("proj[e0]", np.array([1.0]))); return defs
    C = corrcoef_safe(X) if D>1 else np.zeros((D,D))
    if D>1: np.fill_diagonal(C, 0.0)
    if D>1:
        i, j = divmod(np.argmax(np.abs(C)), D)
        e_i = np.zeros(D); e_i[i]=1.0
        e_j = np.zeros(D); e_j[j]=1.0
        defs += [(f"proj[ei:{i}]", e_i), (f"proj[ej:{j}]", e_j)]
    while len(defs) < n_proj:
        w = rng.normal(size=D); w /= (np.linalg.norm(w)+EPS)
        defs.append((f"proj[rand:{len(defs)}]", w))
    return defs

# Build atomic candidates (train/val transforms kept separate but with train-fitted scaling)

def build_stage1_candidates(Xtr, Xva, rng, raw_unary=True, n_proj=24, n_pairs=24):
    Ntr, D = Xtr.shape; bank = []; names = []
    if raw_unary:
        for fname, fn in UNARY_FUNCS.items():
            Ztr = safe_num(fn(Xtr)); Zva = safe_num(fn(Xva))
            if Ztr.ndim == 1: Ztr = Ztr[:,None]
            if Zva.ndim == 1: Zva = Zva[:,None]
            if Ztr.shape[1] != D:  # only 1-to-1
                continue
            for j in range(D):
                vtr0 = Ztr[:, j:j+1]; vva0 = Zva[:, j:j+1]
                mu, sd = col_scale_fit(vtr0)
                vtr = col_scale_apply(vtr0, mu, sd); vva = col_scale_apply(vva0, mu, sd)
                if float(vtr.std()) < 1e-8: continue
                def make_apply(fn=fn, j=j, mu=mu, sd=sd):
                    return lambda X, _cache=None: col_scale_apply(safe_num(fn(X[:, j:j+1])), mu, sd)
                spec = {"kind":"unary","fname":fname,"idx":int(j),"mu":float(mu),"sd":float(sd)}
                bank.append({"name": f"{fname}[x{j}]", "ztr": vtr, "zva": vva, "apply": make_apply(), "spec": spec})
                names.append(f"{fname}[x{j}]")
    pdefs = projection_defs(Xtr, rng, n_proj=n_proj)
    for pnm, w in pdefs:
        ztrb = (Xtr @ w).reshape(-1,1); zvab = (Xva @ w).reshape(-1,1)
        for fname, fn in UNARY_FUNCS.items():
            vtr0 = safe_num(fn(ztrb)); vva0 = safe_num(fn(zvab))
            mu, sd = col_scale_fit(vtr0)
            vtr = col_scale_apply(vtr0, mu, sd); vva = col_scale_apply(vva0, mu, sd)
            if float(vtr.std()) < 1e-8: continue
            def make_apply(fn=fn, w=w.copy(), mu=mu, sd=sd):
                return lambda X, _cache=None: col_scale_apply(safe_num(fn((X @ w).reshape(-1,1))), mu, sd)
            spec = {"kind":"hsml_unary","fname":fname,"w":w.copy().tolist(),"mu":float(mu),"sd":float(sd)}
            bank.append({"name": f"hsml:{fname}({pnm})", "ztr": vtr, "zva": vva, "apply": make_apply(), "spec": spec})
            names.append(f"hsml:{fname}({pnm})")
    pair_idxs = list(itertools.combinations(range(len(pdefs)), 2))
    rng.shuffle(pair_idxs); pair_idxs = pair_idxs[:n_pairs]
    for a,b in pair_idxs:
        p1_nm, w1 = pdefs[a]; p2_nm, w2 = pdefs[b]
        z1_tr = (Xtr @ w1).reshape(-1,1); z1_va = (Xva @ w1).reshape(-1,1)
        z2_tr = (Xtr @ w2).reshape(-1,1); z2_va = (Xva @ w2).reshape(-1,1)
        ops = [
            ("+", z1_tr+z2_tr, z1_va+z2_va, lambda X, w1=w1, w2=w2: (X@w1).reshape(-1,1)+(X@w2).reshape(-1,1)),
            ("-", z1_tr-z2_tr, z1_va-z2_va, lambda X, w1=w1, w2=w2: (X@w1).reshape(-1,1)-(X@w2).reshape(-1,1)),
            ("*", z1_tr*z2_tr, z1_va*z2_va, lambda X, w1=w1, w2=w2: ((X@w1).reshape(-1,1))*((X@w2).reshape(-1,1))),
            ("/", z1_tr/(np.abs(z2_tr)+1e-3), z1_va/(np.abs(z2_va)+1e-3),
                  lambda X, w1=w1, w2=w2: (X@w1).reshape(-1,1)/(np.abs((X@w2).reshape(-1,1))+1e-3)),
        ]
        for op_nm, vtr0, vva0, raw_fn in ops:
            vtr0 = safe_num(vtr0); vva0 = safe_num(vva0)
            mu, sd = col_scale_fit(vtr0); vtr = col_scale_apply(vtr0, mu, sd); vva = col_scale_apply(vva0, mu, sd)
            if float(vtr.std()) < 1e-8: continue
            def make_apply(raw_fn=raw_fn, mu=mu, sd=sd):
                return lambda X, _cache=None: col_scale_apply(safe_num(raw_fn(X)), mu, sd)
            spec = {"kind":"hsml_binary","op":op_nm,"w1":w1.copy().tolist(),"w2":w2.copy().tolist(),"mu":float(mu),"sd":float(sd)}
            bank.append({"name": f"hsml2:({p1_nm}{op_nm}{p2_nm})", "ztr": vtr, "zva": vva, "apply": make_apply(), "spec": spec})
            names.append(f"hsml2:({p1_nm}{op_nm}{p2_nm})")
    return bank, names

# =========================
# Stage-1 forward selection (GLS-aware scoring)
# =========================

def mdl_bits_for(kind: str, params: Dict, n_cols: int, mdl_costs: Dict):
    tb = mdl_costs["type_bits"].get(kind, 8.0)
    per_col = mdl_costs["per_col_bits"] * max(0, int(n_cols))
    pcount = 0
    for v in params.values():
        if isinstance(v, (float, int)): pcount += 1
        elif isinstance(v, (list, tuple, np.ndarray)): pcount += int(np.size(v))
    per_real = mdl_costs["per_real_bits"] * pcount
    return float(tb + per_col + per_real)

def forward_atomic_search(Xtr, ytr, Xva, yva, rng, max_atoms, log_prefix=" ", Sigma_tr=None, Sigma_va=None):
    print(log_prefix + "--- Stage 1: Atomic Alphabet ---")
    cand, cand_names = build_stage1_candidates(Xtr, Xva, rng, raw_unary=True, n_proj=24, n_pairs=24)
    print(log_prefix + f" (built {len(cand)} candidates)")
    criterion = CONFIG["STAGE1"]["CRITERION"]
    bit_weight = CONFIG["MDL_COSTS"]["bit_weight"]

    Ftr = np.zeros((Xtr.shape[0],0)); Fva = np.zeros((Xva.shape[0],0))
    chosen = []; chosen_specs = []

    # baseline
    beta, b0, _ = ridge_with_intercept(Ftr, ytr, CONFIG["ALPHA_RIDGE"], Sigma=Sigma_tr)
    res_va = yva - predict_with_intercept(Fva, beta, b0)
    base_bits = mdl_bits_for("atomic", {}, 0, CONFIG["MDL_COSTS"])
    best_score = crit_from_residuals(res_va, len(yva), Ftr.shape[1]+1,
                                     criterion=criterion, mdl_bits=base_bits, bit_weight=bit_weight, Sigma=Sigma_va)
    base_mse = float(np.mean(res_va**2))
    used = set(); step = 0
    W_tr = _as_weight_matrix(Sigma_tr, len(ytr)) if Sigma_tr is not None else None
    while step < max_atoms:
        step += 1
        best_idx = -1; best_add_score = None; best_pack = None
        for i, c in enumerate(cand):
            if i in used: continue
            Gamma = fit_residualization_gls(c["ztr"], Ftr, Sigma_tr, alpha=CONFIG["RDMP_ALPHA"]) if Ftr.size else np.zeros((0, c["ztr"].shape[1]))
            R_tr = apply_residualization(c["ztr"], Ftr, Gamma) if Ftr.size else c["ztr"]
            R_va = apply_residualization(c["zva"], Fva, Gamma) if Fva.size else c["zva"]
            if Ftr.size:
                if W_tr is not None:
                    diff = c["ztr"] - (Ftr @ Gamma)
                    num = (diff.T @ W_tr @ diff).item() ** 0.5
                    den = (c["ztr"].T @ W_tr @ c["ztr"]).item() ** 0.5 + 1e-12
                else:
                    num = np.linalg.norm(c["ztr"] - (Ftr @ Gamma))
                    den = np.linalg.norm(c["ztr"]) + 1e-12
                if den>0 and (1.0 - num/den) > CONFIG["STAGE1"]["DUP_CORR_THR"]:
                    continue
            mu, sd = w_mean_std(R_tr, Sigma_tr)
            R_trs = (R_tr - mu)/sd; R_vas = (R_va - mu)/sd
            if float(R_trs.std()) < 1e-8: continue
            Ftr2 = np.hstack([Ftr, R_trs]) if Ftr.size else R_trs
            Fva2 = np.hstack([Fva, R_vas]) if Fva.size else R_vas
            k2 = Ftr2.shape[1] + 1
            beta2, b02, _ = ridge_with_intercept(Ftr2, ytr, CONFIG["ALPHA_RIDGE"], Sigma=Sigma_tr)
            res2 = yva - predict_with_intercept(Fva2, beta2, b02)
            bits2 = mdl_bits_for("atomic", {}, Ftr2.shape[1], CONFIG["MDL_COSTS"])
            score2 = crit_from_residuals(res2, len(yva), k2, criterion=criterion,
                                         mdl_bits=bits2, bit_weight=bit_weight, Sigma=Sigma_va)
            if (best_add_score is None) or (score2 < best_add_score):
                best_idx, best_add_score = i, score2
                best_pack = (Gamma, mu, sd, R_trs, R_vas)
        if best_idx < 0: break
        beta_tmp, b0_tmp, _ = ridge_with_intercept(
            np.hstack([Ftr, best_pack[3]]) if Ftr.size else best_pack[3],
            ytr, CONFIG["ALPHA_RIDGE"], Sigma=Sigma_tr)
        res_tmp = yva - predict_with_intercept(
            np.hstack([Fva, best_pack[4]]) if Fva.size else best_pack[4],
            beta_tmp, b0_tmp)
        mse_tmp = float(np.mean(res_tmp**2))
        if (best_add_score < best_score - 1e-8) or (mse_tmp < base_mse - CONFIG["STAGE1"]["MIN_MSE_GAIN"]):
            used.add(best_idx)
            Ftr = np.hstack([Ftr, best_pack[3]]) if Ftr.size else best_pack[3]
            Fva = np.hstack([Fva, best_pack[4]]) if Fva.size else best_pack[4]
            chosen.append(cand[best_idx]["name"])
            chosen_specs.append(dict(spec=cand[best_idx]["spec"], Gamma=best_pack[0], mu=best_pack[1], sd=best_pack[2]))
            best_score = best_add_score
            base_mse = mse_tmp
            print(log_prefix + f" [A{len(chosen)}] + {cand[best_idx]['name']} (MDL★ ↓ → {best_score:.2f})")
        else:
            break
    if chosen:
        preview = chosen[:8]; print(log_prefix + f" --- Atomic Alphabet: {preview}{'...' if len(chosen)>8 else ''} ---")
    else:
        print(log_prefix + " (no gain; empty atomic set)")
    return dict(Ftr=Ftr, Fva=Fva, atoms=chosen, specs=chosen_specs,
                best_score=best_score, cand=cand, cand_names=cand_names)

# =========================
# Stage-2 grammar (kept from v20.1; Γ now GLS; criteria GLS)
# =========================

def block_bilinear(X, i, j): return (X[:, i:i+1] * X[:, j:j+1])

def block_relu_proj(X, w, t): return np.maximum(0.0, (X @ w) - t)[:,None]

def block_sinproj(X, w, b): return np.sin(X @ w + b)[:,None]

def block_fct(X, w, b0, kappa):
    z = X @ w + b0; return (np.cos(z) + (1.0/kappa)*np.cos(kappa*z))[:,None]

def block_perm_invar(X, idxs):
    V = X[:, idxs]; s1 = np.sum(V,1,keepdims=True); s2 = np.sum(V**2,1,keepdims=True)
    p11 = 0.5*(s1**2 - s2)
    return np.hstack([s1, s2, p11])

def block_dihedral_invar(X, i, j, use_r4=False):
    xi, xj = X[:, i], X[:, j]; r2 = (xi*xi + xj*xj)[:,None]
    return np.hstack([r2, r2**2]) if use_r4 else r2

def block_group_invar(X, spec):
    return GroupInvariantFeatures(spec, include_raw=False, degree=2).transform(X)

def block_combo_vector(X, w1, w2):
    z1 = (X @ w1).reshape(-1,1); z2 = (X @ w2).reshape(-1,1)
    return np.hstack([z1+z2, z1-z2, z1*z2, z1/(np.abs(z2)+1e-3)])

def stage2_grammar_search(Xtr, ytr, Xva, yva, base_info, criterion, mdl_costs,
                          use_rdmp=True, seed=1337, max_grammar=8,
                          group_spec: Optional[GroupSpec]=None, log_prefix=" ",
                          Sigma_tr=None, Sigma_va=None):
    rng = np.random.default_rng(seed)
    Fbase_tr = base_info["Ftr"].copy(); Fbase_va = base_info["Fva"].copy()

    beta_b, b0_b, _ = ridge_with_intercept(Fbase_tr, ytr, CONFIG["ALPHA_RIDGE"], Sigma=Sigma_tr)
    res_va = yva - predict_with_intercept(Fbase_va, beta_b, b0_b)
    base_bits = mdl_bits_for("atomic", {}, Fbase_tr.shape[1], mdl_costs)
    base_mse = float(np.mean(res_va**2))
    k_base = Fbase_tr.shape[1] + 1
    base_score= crit_from_residuals(res_va, len(yva), k_base, criterion=criterion,
                                    mdl_bits=base_bits, bit_weight=mdl_costs["bit_weight"], Sigma=Sigma_va)
    accepted = []
    d = Xtr.shape[1]
    if group_spec is None:
        group_spec = GroupSpec(sign_groups=[list(range(d))])

    def fit_block(kind, Fb_tr, Fb_va, params):
        nonlocal Fbase_tr, Fbase_va, base_score, base_bits, accepted, base_mse
        if Fb_tr is None or Fb_tr.size == 0: return False
        Gamma = fit_residualization_gls(Fb_tr, Fbase_tr, Sigma_tr, alpha=CONFIG["RDMP_ALPHA"]) if use_rdmp else np.zeros((Fbase_tr.shape[1], Fb_tr.shape[1]))
        R_tr = apply_residualization(Fb_tr, Fbase_tr, Gamma) if use_rdmp else Fb_tr
        R_va = apply_residualization(Fb_va, Fbase_va, Gamma) if use_rdmp else Fb_va
        mus, sds = w_mean_std(R_tr, Sigma_tr)
        R_trs = (R_tr - mus)/sds; R_vas = (R_va - mus)/sds
        Ftr = np.hstack([Fbase_tr, R_trs]) if Fbase_tr.size else R_trs
        Fva = np.hstack([Fbase_va, R_vas]) if Fbase_va.size else R_vas
        k = Ftr.shape[1] + 1
        if k >= len(ytr) - 1: return False
        beta, b0, _ = ridge_with_intercept(Ftr, ytr, CONFIG["ALPHA_RIDGE"], Sigma=Sigma_tr)
        res2 = yva - predict_with_intercept(Fva, beta, b0)
        bits = mdl_bits_for(kind, params, R_trs.shape[1], mdl_costs) + mdl_bits_for("atomic", {}, Fbase_tr.shape[1], mdl_costs)
        score = crit_from_residuals(res2, len(yva), k, criterion=criterion, mdl_bits=bits, bit_weight=mdl_costs["bit_weight"], Sigma=Sigma_va)
        gain = (score < base_score - 1e-6) and (np.mean(res2**2) <= base_mse + 1e-4)
        if gain:
            print(log_prefix + f" [+] {kind} ({criterion}★ ↓ → {score:.2f} | val_mse {base_mse:.4f}→{np.mean(res2**2):.4f})")
            def apply_block(X, kind=kind, params=copy.deepcopy(params), Gamma=Gamma, mus=mus.copy(), sds=sds.copy()):
                Fb = None
                if kind == "relu_proj": Fb = block_relu_proj(X, params["w"], params["t"])
                elif kind == "sinproj": Fb = block_sinproj(X, params["w"], params["b"])
                elif kind == "fct": Fb = block_fct(X, params["w"], params["b0"], params["kappa"])
                elif kind == "bilinear": Fb = block_bilinear(X, params["i"], params["j"])
                elif kind == "dihedral_invar": Fb = block_dihedral_invar(X, params["i"], params["j"], use_r4=False)
                elif kind == "perm_invar": Fb = block_perm_invar(X, params["group_idx"])
                elif kind == "group_invar": Fb = block_group_invar(X, params["spec"])
                elif kind == "combo_vec": Fb = block_combo_vector(X, params["w1"], params["w2"])
                elif kind == "ruliad": Fb = params["state_features"](X)
                else: Fb = None
                return Gamma, mus, sds, Fb
            accepted.append(dict(kind=kind, params=params, Gamma=Gamma, mus=mus, sds=sds, apply=apply_block))
            Fbase_tr, Fbase_va = Ftr, Fva
            base_bits = bits; base_mse = float(np.mean(res2**2)); base_score = score
            return True
        return False

    print(log_prefix + "--- Stage 2: Grammar & RDMP/L4 ---")
    for t in range(max_grammar):
        rng_local = np.random.default_rng(seed+t)
        w = rng_local.normal(size=d); w /= (np.linalg.norm(w)+EPS)
        wb = rng_local.normal(size=d); wb /= (np.linalg.norm(wb)+EPS)
        wf = rng_local.normal(size=d); wf /= (np.linalg.norm(wf)+EPS)
        t0 = np.median(Xtr @ w) if d>0 else 0.0
        b = float(rng_local.uniform(-np.pi, np.pi))
        b0 = float(rng_local.uniform(-1.5, 1.5)); kappa = float(rng_local.uniform(0.7, 1.8))

        if d > 1:
            C = corrcoef_safe(Xtr); np.fill_diagonal(C, 0.0)
            i_best, j_best = divmod(int(np.argmax(np.abs(C))), d)
        else:
            i_best = j_best = 0

        fit_block("relu_proj", block_relu_proj(Xtr,w,t0), block_relu_proj(Xva,w,t0), dict(w=w,t=t0))
        fit_block("sinproj", block_sinproj(Xtr,wb,b), block_sinproj(Xva,wb,b), dict(w=wb,b=b))
        fit_block("fct", block_fct(Xtr,wf,b0,kappa), block_fct(Xva,wf,b0,kappa), dict(w=wf,b0=b0,kappa=kappa))
        if d>1:
            fit_block("bilinear", block_bilinear(Xtr,i_best,j_best), block_bilinear(Xva,i_best,j_best), dict(i=i_best,j=j_best))
            fit_block("dihedral_invar", block_dihedral_invar(Xtr,i_best,j_best,False), block_dihedral_invar(Xva,i_best,j_best,False), dict(i=i_best,j=j_best))
            fit_block("group_invar", block_group_invar(Xtr, group_spec), block_group_invar(Xva, group_spec), dict(spec=group_spec))
            w1 = rng_local.normal(size=d); w1/= (np.linalg.norm(w1)+EPS)
            w2 = rng_local.normal(size=d); w2/= (np.linalg.norm(w2)+EPS)
            fit_block("combo_vec", block_combo_vector(Xtr, w1, w2), block_combo_vector(Xva, w1, w2), dict(w1=w1,w2=w2))

    kept = "atomic" if len(accepted)==0 else "full"
    return dict(accepted=accepted, kept=kept, Ftr=Fbase_tr, Fva=Fbase_va, score=base_score)

# =========================
# Stage-2.5 Mini-Ruliad (unchanged core; Γ now GLS; MDL/Σ scoring)
# =========================
class Op:
    def __init__(self, name:str, fn:Callable[...,np.ndarray], arity:int,
                 commutative:bool=False, param_names:Tuple[str,...]=(),
                 param_init:Callable[[],Dict[str,float]]=lambda:{}, mdl_cost_bits:float=8.0):
        self.name=name; self.fn=fn; self.arity=arity; self.commutative=commutative
        self.param_names=param_names; self.param_init=param_init; self.mdl_cost_bits=mdl_cost_bits
class OpRegistry:
    def __init__(self):
        self.ops: Dict[str,Op]={}; self._register_defaults()
    def add(self, op:Op): self.ops[op.name]=op
    def get(self, name:str)->Op: return self.ops[name]
    def _register_defaults(self):
        eps=1e-8
        def _un(fn): return lambda x: fn(x)
        def _bi(fn): return lambda a,b: fn(a,b)
        self.add(Op("id", _un(lambda x: x), 1, mdl_cost_bits=1.0))
        self.add(Op("sin", _un(np.sin), 1)); self.add(Op("cos", _un(np.cos), 1))
        self.add(Op("tanh", _un(np.tanh), 1)); self.add(Op("abs", _un(np.abs), 1))
        self.add(Op("square", _un(lambda x: x*x), 1)); self.add(Op("signed_sqrt", _un(lambda x: np.sign(x)*np.sqrt(np.abs(x)+eps)), 1))
        self.add(Op("log1p_abs", _un(lambda x: np.log1p(np.abs(x))), 1))
        self.add(Op("x_over_one_plus_beta_abs", lambda x,beta: x/(1.0+beta*np.abs(x)+eps), 1,
                    param_names=("beta",), param_init=lambda: {"beta":1.0}))
        self.add(Op("sin_affine", lambda x,omega,phi: np.sin(omega*x+phi), 1,
                    param_names=("omega","phi"), param_init=lambda: {"omega":1.0,"phi":0.0}))
        self.add(Op("add", _bi(lambda a,b: a+b), 2, commutative=True))
        self.add(Op("sub", _bi(lambda a,b: a-b), 2))
        self.add(Op("mul", _bi(lambda a,b: a*b), 2, commutative=True))
        self.add(Op("safe_div", _bi(lambda a,b: a/(np.sign(b)*np.maximum(np.abs(b),eps))), 2))
        self.add(Op("max", _bi(np.maximum), 2, commutative=True)); self.add(Op("min", _bi(np.minimum), 2, commutative=True))

@dataclass
class Node:
    id:int; op:str; parents:Tuple[int,...]; params:Dict[str,float]=field(default_factory=dict); is_input:bool=False

@dataclass
class HypergraphState:
    registry: OpRegistry
    nodes: Dict[int,Node]
    output_ids: List[int]
    input_ids: List[int]
    rule_history: List[str]=field(default_factory=list)
    def _topo_order(self)->List[int]:
        visited, order=set(),[]
        def dfs(u):
            if u in visited: return
            visited.add(u)
            for p in self.nodes[u].parents:
                if p in self.nodes: dfs(p)
            order.append(u)
        for nid in self.nodes: dfs(nid)
        return order
    def features(self, X: np.ndarray)->np.ndarray:
        X = np.asarray(X, float)
        cache={nid: X[:,[nid]] for nid in self.input_ids}
        N=X.shape[0]
        for nid in self._topo_order():
            node=self.nodes[nid]
            if node.is_input: continue
            op=self.registry.get(node.op)
            vals=[cache[p] for p in node.parents]
            if op.arity==1:
                if op.param_names:
                    args=[vals[0]]+[node.params.get(p, op.param_init()[p]) for p in op.param_names]
                    out=op.fn(*args)
                else:
                    out=op.fn(vals[0])
            elif op.arity==2: out=op.fn(vals[0],vals[1])
            else: raise NotImplementedError
            cache[nid]=out
        feats=[cache[i] for i in self.output_ids]
        return np.hstack(feats) if len(feats)>1 else (feats[0] if feats else np.zeros((N,0)))
    def mdl_bits(self, param_precision_bits:int=12)->float:
        cost=0.0
        for node in self.nodes.values():
            if node.is_input: cost+=1.0; continue
            op=self.registry.get(node.op); cost+=op.mdl_cost_bits
            cost += 0.5*sum(max(1.0, math.log2(1+p)) for p in node.parents)
            cost += param_precision_bits*len(node.params)
            cost += 2.0*len(self.output_ids)
        return float(cost)
    def redundancy(self, X: np.ndarray, sample_cap:int=512)->float:
        F=self.features(X)
        if F.ndim==1: F=F.reshape(-1,1)
        if F.shape[1]<=1: return 0.0
        N=F.shape[0]; idx=np.random.choice(N, size=min(N,sample_cap), replace=False)
        Fs=F[idx];
        if Fs.shape[1]<=1: return 0.0
        Fs=(Fs-Fs.mean(0,keepdims=True))/(Fs.std(0,keepdims=True)+1e-8)
        C=corrcoef_safe(Fs); off=np.abs(C-np.eye(C.shape[0]))
        return float(off.sum()/(off.shape[0]*(off.shape[1]-1)))
    def canonical_hash(self)->str:
        t=[]
        for nid in sorted(self.nodes):
            node=self.nodes[nid]; op=self.registry.get(node.op)
            parents=list(node.parents)
            if op.commutative: parents=sorted(parents)
            qparams=tuple((k, round(v,6)) for k,v in sorted(node.params.items()))
            t.append((nid,node.op,tuple(parents),qparams,node.is_input))
        return hashlib.sha256(repr(tuple(t)).encode()).hexdigest()
    def clone(self)->"HypergraphState": return copy.deepcopy(self)

def seed_from_inputs(X: np.ndarray, registry: Optional[OpRegistry]=None)->HypergraphState:
    reg=registry or OpRegistry()
    nodes={}; D=X.shape[1]; inputs=[]
    for i in range(D):
        nodes[i]=Node(id=i, op="id", parents=(i,), params={}, is_input=True); inputs.append(i)
    next_id=D; outputs=[]
    for i in range(D):
        for opn in ("sin","cos","tanh","square","x_over_one_plus_beta_abs"):
            op=reg.get(opn); params=op.param_init() if op.param_names else {}
            nodes[next_id]=Node(id=next_id, op=opn, parents=(i,), params=params)
            outputs.append(next_id); next_id+=1
    return HypergraphState(registry=reg, nodes=nodes, output_ids=outputs, input_ids=inputs)

@dataclass
class Rule:
    name:str; apply:Callable[[HypergraphState], Optional[HypergraphState]]

def rule_commute_and_canon()->Rule:
    def _apply(state:HypergraphState):
        s=state.clone(); changed=False
        for node in s.nodes.values():
            if node.is_input: continue
            op=s.registry.get(node.op)
            if op.commutative:
                cp=tuple(sorted(node.parents))
                if cp!=node.parents: node.parents=cp; changed=True
        if not changed: return None
        s.rule_history.append("canon"); return s
    return Rule("canon", _apply)

def rule_eliminate_double_abs()->Rule:
    def _apply(state:HypergraphState):
        s=state.clone()
        for nid,node in s.nodes.items():
            if node.op=="abs":
                p=s.nodes[node.parents[0]]
                if p.op=="abs":
                    node.parents=(p.parents[0],); s.rule_history.append("abs_abs→abs"); return s
        return None
    return Rule("abs_chain_simplify", _apply)

def rule_factor_distribute()->Rule:
    def _apply(state:HypergraphState):
        s=state.clone()
        adds=[n for n in s.nodes.values() if (not n.is_input) and n.op=="add"]
        random.shuffle(adds)
        for add in adds:
            if len(add.parents)!=2: continue
            p1,p2=add.parents; n1,n2=s.nodes[p1], s.nodes[p2]
            if n1.op=="mul" and n2.op=="mul":
                a1,b1=n1.parents; a2,b2=n2.parents
                common=None
                if a1==a2: common,r1,r2=a1,b1,b2
                elif a1==b2: common,r1,r2=a1,b1,a2
                elif b1==a2: common,r1,r2=b1,a1,b2
                elif b1==b2: common,r1,r2=b1,a1,a2
                if common is not None:
                    new_add_id=max(s.nodes)+1
                    s.nodes[new_add_id]=Node(id=new_add_id, op="add", parents=(r1,r2))
                    add.op="mul"; add.parents=(common,new_add_id)
                    s.rule_history.append("factor"); return s
        muls=[n for n in s.nodes.values() if (not n.is_input) and n.op=="mul"]
        random.shuffle(muls)
        for mul in muls:
            if len(mul.parents)!=2: continue
            a,b=mul.parents; nb=s.nodes[b]
            if nb.op=="add" and len(nb.parents)==2:
                r1,r2=nb.parents
                m1=max(s.nodes)+1; m2=m1+1; add_id=m2+1
                s.nodes[m1]=Node(id=m1, op="mul", parents=(a,r1))
                s.nodes[m2]=Node(id=m2, op="mul", parents=(a,r2))
                s.nodes[add_id]=Node(id=add_id, op="add", parents=(m1,m2))
                for node in s.nodes.values():
                    node.parents=tuple(add_id if p==mul.id else p for p in node.parents)
                s.rule_history.append("distribute"); return s
        return None
    return Rule("factor_or_distribute", _apply)

def rule_param_opt(steps:int=8, lr:float=0.2, Xv:np.ndarray=None, yv:np.ndarray=None)->Rule:
    assert Xv is not None and yv is not None
    def _apply(state:HypergraphState):
        t=state.clone()
        cand=[n for n in t.nodes.values() if (not n.is_input) and t.registry.get(n.op).param_names]
        if not cand: return None
        node=random.choice(cand); op=t.registry.get(node.op)
        cur={k: node.params.get(k, op.param_init()[k]) for k in op.param_names}
        best_loss=None; best=cur.copy(); step_lr=lr
        for _ in range(steps):
            improved=False
            for pname in op.param_names:
                v=cur[pname]
                for delta in (+step_lr, -step_lr):
                    trial=cur.copy(); trial[pname]=v+delta
                    node.params.update(trial)
                    F=t.features(Xv)
                    mu=F.mean(0,keepdims=True); sd=F.std(0,keepdims=True)+1e-8
                    Z=(F-mu)/sd
                    A = Z.T @ Z + CONFIG["ALPHA_RIDGE"] * np.eye(Z.shape[1]); b = Z.T @ yv.ravel()
                    w=np.linalg.solve(A,b); yhat=Z@w
                    loss=float(np.mean((yv.ravel()-yhat.ravel())**2))
                    if (best_loss is None) or (loss<best_loss): best_loss=loss; best=trial.copy(); improved=True
                cur[pname]=best[pname]
            step_lr*=0.5
            if not improved: break
        node.params.update(best)
        t.rule_history.append(f"opt:{node.op}"); return t
    return Rule("param_opt", _apply)

def rule_toggle_product()->Rule:
    def _apply(state:HypergraphState):
        s=state.clone()
        ids=[nid for nid,n in s.nodes.items() if not n.is_input]
        if len(ids)<2: return None
        a,b=random.sample(ids,2); new_id=max(s.nodes)+1
        s.nodes[new_id]=Node(id=new_id, op="mul", parents=(a,b))
        if new_id not in s.output_ids: s.output_ids.append(new_id)
        s.rule_history.append("add_mul"); return s
    return Rule("add_mul_feature", _apply)

@dataclass
class EnergyConfig:
    lam: float=1.0; mu: float=0.05; nu: float=0.05; mdl_scale: float=1.0; xi: float=0.02
    sign_flip_indices: Tuple[int,...]=()

def _fit_and_predict_h(s:HypergraphState, X:np.ndarray, y:np.ndarray)->Tuple[np.ndarray,np.ndarray]:
    F=s.features(X);
    if F.ndim==1: F=F.reshape(-1,1)
    mu=F.mean(0,keepdims=True); sd=F.std(0,keepdims=True)+1e-8; Z=(F-mu)/sd
    A = Z.T @ Z + CONFIG["ALPHA_RIDGE"] * np.eye(Z.shape[1]); b = Z.T @ y.ravel()
    w=np.linalg.solve(A,b); yhat=Z@w; return yhat, w

def order_invariance_penalty(state:HypergraphState, X:np.ndarray, y:np.ndarray)->float:
    s0=state.clone(); y0,_=_fit_and_predict_h(s0,X,y)
    s1=rule_commute_and_canon().apply(state) or state
    y1,_=_fit_and_predict_h(s1,X,y)
    return float(np.mean(np.abs(y0-y1))/(np.std(y)+1e-8))

def sign_flip_penalty(state:HypergraphState, X:np.ndarray, y:np.ndarray, idxs:Tuple[int,...])->float:
    if not idxs: return 0.0
    s=state.clone(); y0,_=_fit_and_predict_h(s,X,y)
    Xf=X.copy()
    for i in idxs: Xf[:,i]*=-1.0
    y1,_=_fit_and_predict_h(s,Xf,y)
    return float(np.mean(np.abs(y0-y1))/(np.std(y)+1e-8))

def energy_h(state:HypergraphState, Xv:np.ndarray, yv:np.ndarray, cfg:EnergyConfig, Sigma_v: Optional[np.ndarray]=None)->float:
    yhat,_=_fit_and_predict_h(state,Xv,yv)
    # Σ-aware loss (falls back to MSE if Sigma_v is None)
    if Sigma_v is None:
        loss=float(np.mean((yv.ravel()-yhat.ravel())**2))
    else:
        r=(yv.ravel()-yhat.ravel())
        loss=float(gls_chi2(r, Sigma_v)/max(len(yv),1))
    inv=order_invariance_penalty(state,Xv,yv); red=state.redundancy(Xv)
    mdl=state.mdl_bits(); sf=sign_flip_penalty(state,Xv,yv,cfg.sign_flip_indices)
    return float(cfg.lam*loss + cfg.mu*inv + cfg.nu*red + cfg.mdl_scale*(mdl/1e4) + cfg.xi*sf)

def _ngrams(seq: List[str], n:int=2)->set:
    if len(seq)<n: return set()
    return set(tuple(seq[i:i+n]) for i in range(len(seq)-n+1))

def branchial_distance(a:HypergraphState, b:HypergraphState, n:int=2)->float:
    A=_ngrams(a.rule_history,n); B=_ngrams(b.rule_history,n)
    if not A and not B: return 0.0
    return 1.0 - len(A & B)/max(1, len(A|B))

def select_top_k_diverse(cands: List[Tuple[HypergraphState,float]], k:int)->List[Tuple[HypergraphState,float]]:
    if len(cands)<=k: return sorted(cands, key=lambda t:t[1])
    cands=sorted(cands, key=lambda t:t[1])
    chosen=[cands[0]]; rem=cands[1:]
    while len(chosen)<k and rem:
        best_i=0; best_d=-1.0
        for i,(s,sc) in enumerate(rem):
            mind=min(branchial_distance(s,c[0],n=2) for c in chosen)
            if mind>best_d: best_i=i; best_d=mind
        chosen.append(rem.pop(best_i))
    return chosen

class RuliadRefiner:
    def __init__(self, registry:Optional[OpRegistry]=None, energy_cfg:EnergyConfig=EnergyConfig(),
                 depth:int=3, K_per_parent:int=20, frontier_size:int=20,
                 random_seed:int=42, use_param_opt:bool=True, sigma_val: Optional[np.ndarray]=None):
        self.registry=registry or OpRegistry(); self.energy_cfg=energy_cfg
        self.depth=depth; self.K=K_per_parent; self.frontier_size=frontier_size
        random.seed(random_state_to_py(random_seed)); np.random.seed(random_state_to_py(random_seed)); self.use_param_opt=use_param_opt
        self.Sigma_val = sigma_val
    def _rules(self, Xv, yv)->List[Rule]:
        rules=[rule_commute_and_canon(), rule_eliminate_double_abs(), rule_factor_distribute(), rule_toggle_product()]
        if self.use_param_opt: rules.append(rule_param_opt(steps=8, lr=0.2, Xv=Xv, yv=yv))
        return rules
    def refine(self, Xtr:np.ndarray, ytr:np.ndarray, Xv:np.ndarray, yv:np.ndarray,
               seed_state:Optional[HypergraphState]=None)->HypergraphState:
        state0=seed_state or seed_from_inputs(Xtr, self.registry)
        _=energy_h(state0, Xv, yv, self.energy_cfg, self.Sigma_val)
        rules=self._rules(Xv,yv); frontier=[(state0, energy_h(state0,Xv,yv,self.energy_cfg,self.Sigma_val))]
        archive={state0.canonical_hash()}; best_state,best_score=frontier[0]; stagnation=0
        for d in range(self.depth):
            props=[]
            for s,_ in frontier:
                for _ in range(self.K):
                    rule=random.choice(rules); s2=rule.apply(s)
                    if s2 is None: continue
                    h=s2.canonical_hash()
                    if h in archive: continue
                    sc=energy_h(s2,Xv,yv,self.energy_cfg,self.Sigma_val)
                    props.append((s2,sc)); archive.add(h)
            if not props: break
            frontier = select_top_k_diverse(props, self.frontier_size)
            cand_best_state, cand_best_score = min(frontier, key=lambda t:t[1])
            if cand_best_score < best_score - 1e-6:
                best_state,best_score=cand_best_state,cand_best_score; stagnation=0
            else:
                stagnation+=1
                if stagnation>=1:
                    self.frontier_size=min(self.frontier_size+4, 40)
                    self.K=min(self.K+5, 40)
        return best_state

def random_state_to_py(seed: int) -> int:
    # tiny helper to keep seeding stable even if numpy semantics change
    return int(seed) & 0x7fffffff

# Stage 2.5 wrapper

def stage2p5_ruliad_search(Xtr, ytr, Xva, yva, base_info, mdl_costs, cfg_dict, log_prefix=" ",
                           Sigma_tr=None, Sigma_va=None):
    print(log_prefix + "--- Stage 2.5: Mini-Ruliad Refiner ---")
    reg=OpRegistry()
    ref=RuliadRefiner(
        registry=reg,
        energy_cfg=EnergyConfig(**cfg_dict["energy"]),
        depth=cfg_dict["depth"], K_per_parent=cfg_dict["K_per_parent"], frontier_size=cfg_dict["frontier_size"],
        random_seed=cfg_dict["random_seed"], use_param_opt=cfg_dict.get("use_param_opt", True),
        sigma_val=Sigma_va
    )
    t0=time.time()
    best=ref.refine(Xtr,ytr,Xva,yva,seed_state=None)
    t1=time.time()

    F_r_tr = best.features(Xtr); F_r_va = best.features(Xva)
    if F_r_tr.ndim==1: F_r_tr=F_r_tr.reshape(-1,1)
    if F_r_va.ndim==1: F_r_va=F_r_va.reshape(-1,1)
    if F_r_tr.shape[1] == 0:
        print(log_prefix+f" [×] ruliad rejected (no features generated; time {t1-t0:.2f}s)")
        return dict(accepted=[], kept="full", Ftr=base_info["Ftr"], Fva=base_info["Fva"], score=base_info.get("score", np.inf))

    k_keep = int(cfg_dict.get("keep_outputs_topk", min(24, F_r_tr.shape[1])))
    if F_r_tr.shape[1] > k_keep:
        yc = (yva - yva.mean()).ravel()
        Z = (F_r_va - F_r_va.mean(0, keepdims=True)) / (F_r_va.std(0, keepdims=True) + 1e-8)
        zden = np.sqrt((Z**2).sum(0)) + 1e-12
        yden = np.sqrt((yc**2).sum()) + 1e-12
        corrs = np.abs(Z.T @ yc) / (zden * yden)
        idxs = np.argsort(-corrs)[:k_keep]
        F_r_tr = F_r_tr[:, idxs]; F_r_va = F_r_va[:, idxs]
        def state_features(X, idxs=idxs, state=best):
            F = state.features(X)
            if F.ndim==1: F=F.reshape(-1,1)
            return F[:, idxs]
    else:
        def state_features(X, state=best):
            F = state.features(X)
            return F if F.ndim==2 else F.reshape(-1,1)

    Fbase_tr = base_info["Ftr"].copy(); Fbase_va = base_info["Fva"].copy()
    Gamma = fit_residualization_gls(F_r_tr, Fbase_tr, Sigma_tr, alpha=CONFIG["RDMP_ALPHA"]) if Fbase_tr.size else np.zeros((0, F_r_tr.shape[1]))
    R_tr = apply_residualization(F_r_tr, Fbase_tr, Gamma) if Fbase_tr.size else F_r_tr
    R_va = apply_residualization(F_r_va, Fbase_va, Gamma) if Fbase_va.size else F_r_va
    mus, sds = w_mean_std(R_tr, Sigma_tr)
    R_trs = (R_tr - mus)/sds; R_vas = (R_va - mus)/sds

    Ftr2 = np.hstack([Fbase_tr, R_trs]) if Fbase_tr.size else R_trs
    Fva2 = np.hstack([Fbase_va, R_vas]) if Fbase_va.size else R_vas

    beta_b, b0_b, _ = ridge_with_intercept(Fbase_tr, ytr, CONFIG["ALPHA_RIDGE"], Sigma=Sigma_tr)
    res_b = yva - predict_with_intercept(Fbase_va, beta_b, b0_b)
    base_bits = mdl_bits_for("atomic", {}, Fbase_tr.shape[1], mdl_costs)
    base_score= crit_from_residuals(res_b, len(yva), Fbase_tr.shape[1]+1, criterion="MDL",
                                    mdl_bits=base_bits, bit_weight=mdl_costs["bit_weight"], Sigma=Sigma_va)
    base_mse = float(np.mean(res_b**2))

    beta2, b02, _ = ridge_with_intercept(Ftr2, ytr, CONFIG["ALPHA_RIDGE"], Sigma=Sigma_tr)
    res2 = yva - predict_with_intercept(Fva2, beta2, b02)
    bits2 = (mdl_bits_for("ruliad", {"param_count":0}, R_trs.shape[1], mdl_costs)
             + mdl_bits_for("atomic", {}, Fbase_tr.shape[1], mdl_costs))
    score2= crit_from_residuals(res2, len(yva), Ftr2.shape[1]+1, criterion="MDL",
                                mdl_bits=bits2, bit_weight=mdl_costs["bit_weight"], Sigma=Sigma_va)

    gain = (score2 < base_score - 1e-6) or (float(np.mean(res2**2)) < base_mse - 1e-3)
    if gain:
        print(log_prefix+f" [+] ruliad (MDL★ ↓ → {score2:.2f} | val_mse {base_mse:.4f}→{np.mean(res2**2):.4f} | #features={R_trs.shape[1]} | nodes={len(best.nodes)} | {t1-t0:.2f}s)")
        def apply_block(X, Gamma=Gamma, mus=mus.copy(), sds=sds.copy(), f=state_features):
            Fb = f(X); return Gamma, mus, sds, Fb
        accepted = dict(kind="ruliad",
                        params={"state_features": state_features,
                                "nodes": len(best.nodes),
                                "rule_history_tail": best.rule_history[-8:]},
                        Gamma=Gamma, mus=mus, sds=sds, apply=apply_block)
        return dict(accepted=[accepted], kept="full+ruliad", Ftr=Ftr2, Fva=Fva2, score=score2)
    else:
        print(log_prefix+f" [×] ruliad rejected (no MDL/val-MSE gain; time {t1-t0:.2f}s)")
        return dict(accepted=[], kept="full", Ftr=Fbase_tr, Fva=Fbase_va, score=base_score)

# =========================
# OG-SET (inline, improved bagging & export)
# =========================
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
    W_info: Dict[str, float]
    selected_idx: List[int]
    bits_path: List[float]
    rss_path: List[float]
    r2_train_path: List[float]
    og_r2_train: float
    og_r2_val: Optional[float]

def _ebic_bits(RSS: float, n: int, k: int, p: int, gamma: float) -> float:
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
        bic_bits_threshold = 0.8 if n <= 500 else 0.6
    if max_ops is None:
        max_ops = 10 if n <= 500 else 16
    if ebic_gamma is None:
        ebic_gamma = 0.25 if n <= 500 else 0.0

    W = _as_weight_matrix(Sigma, n)
    L = _whitener_from_W(W)
    evals = np.linalg.eigvalsh((W+W.T)/2.0)
    min_e, max_e = float(np.clip(evals.min(), 0.0, None)), float(evals.max())
    condW = float(max_e / max(min_e, 1e-16)) if max_e>0 else float("inf")

    Pw = L @ Phi
    yw = L @ y

    # unit-gauge columns
    scales = np.linalg.norm(Pw, axis=0)
    scales_safe = np.where(scales > 0, scales, 1.0)
    Pw_unit = Pw / scales_safe

    # intercept
    if include_intercept:
        if np.allclose(W, np.diag(np.diag(W))):
            wvec = np.diag(W)
            mu = float((wvec * y).sum() / max(wvec.sum(), 1e-12))
        else:
            mu = np.linalg.lstsq(L @ np.ones((n,1)), yw, rcond=None)[0].item()
    else:
        mu = 0.0
    y0 = y - mu
    yw0 = L @ y0

    RSS0 = (yw0 @ yw0).item()
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
            RSS = (resid @ resid).item()
            bits = _ebic_bits(RSS, n, k0 + len(beta_u), p, ebic_gamma)
            gain = bits - best_bits
            if selected:
                cw = Pw_unit[:, j]
                corr = np.max(np.abs(Pw_unit[:, selected].T @ cw)).item()
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
        sigma2 = (resid_w @ resid_w / max(n - (k0 + len(selected)), 1)).item()
        XtX_inv = np.linalg.pinv(Xw.T @ Xw, rcond=1e-12)
        se_u = np.sqrt(np.clip(np.diag(XtX_inv) * sigma2, 0.0, None))
        bits_path.append(best_bits)
        rss_path.append((resid_w @ resid_w).item())
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
        sigma2 = (resid.T @ W @ resid / max(n - (k0 + len(selected)), 1)).item()
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

    # bagging with Σ passed through (GLS-consistent)
    if bag_boots and selected:
        counts = {j: 0 for j in selected}
        m = max(4, int(bag_frac * n))
        Sig_arr = np.asarray(Sigma) if Sigma is not None else None
        for _ in range(bag_boots):
            idx = np.array(sorted(rng.sample(range(n), m)))
            Sigma_sub = None
            if Sig_arr is not None:
                Sigma_sub = Sig_arr[np.ix_(idx, idx)] if Sig_arr.ndim==2 else Sig_arr[idx]
            sub = build_ogset(y[idx], Phi[idx], atom_names, Sigma=Sigma_sub,
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
            if selected:
                X = Phi[:, selected]
                XtW = X.T @ W
                G = XtW @ X
                beta = np.linalg.lstsq(G, XtW @ y0, rcond=None)[0]
                resid = y0 - X @ beta
                sigma2 = (resid.T @ W @ resid / max(n - (k0 + len(selected)), 1)).item()
                cov = np.linalg.pinv(G, rcond=1e-12) * sigma2
                se = np.sqrt(np.clip(np.diag(cov), 0.0, None))
                for r, (cj, sej) in enumerate(zip(beta, se)):
                    ops[r].coeff = float(cj)
                    ops[r].stderr = float(sej)
                yhat = mu + X @ beta
            else:
                yhat = np.full_like(y, mu, dtype=float)
                resid = y - yhat

    og_r2_train = 1.0 - ((resid.T @ W @ resid) / max(y0.T @ W @ y0, 1e-300)).item()
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

# Export W-orthonormal super-ops (robust)

def export_superfeatures_orthonormal(Phi_sel: np.ndarray, L: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Return (X_train, T) with columns W-orthonormal on TRAIN:
      A = L @ Phi_sel; A = Q R (QR); T solves R T = I; X_train = Phi_sel @ T
      Reuse T to get X_val = Phi_val @ T.
    """
    A = L @ Phi_sel
    Q, R = np.linalg.qr(A, mode="reduced")
    # Triangular solve instead of inv(R)
    I = np.eye(R.shape[0])
    try:
        T = np.linalg.solve(R, I)
    except np.linalg.LinAlgError:
        U, S, Vt = np.linalg.svd(A, full_matrices=False)
        T = (Vt.T / np.maximum(S, 1e-12))
    X_train = Phi_sel @ T
    return X_train, T

def apply_superfeatures(Phi_sel: np.ndarray, T: np.ndarray) -> np.ndarray:
    return Phi_sel @ T

# =========================
# Logging
# =========================

def log_ogset_summary(res: OgsetResult):
    print(" --- OG-SET (GLS, gauge-normalized) ---")
    line = "  GLS line: y ≈ {:.6g}".format(res.intercept)
    for op in res.ops:
        line += " {:+.6g}·[{}]".format(op.coeff, op.name)
    print(line)
    print("  Operators:")
    for r, op in enumerate(res.ops, 1):
        print(f"   [{r:02d}] {op.name:>30s}  a={op.coeff:+.6g}  SE={op.stderr:.3g}  ΔBIC(bits)={op.delta_bic_bits:+.3f}  freq={op.freq:>4.0%}")
    wi = res.W_info
    print(f"  Gauge/Σ checks: cond(W)={wi.get('cond', float('nan')):.3g}, min_eig={wi.get('min_eig', float('nan')):.3g}, max_eig={wi.get('max_eig', float('nan')):.3g}, cond(XᵀWX)={wi.get('cond_XtWX', float('nan')):.3g}")

def log_ogset_diagnostics(res: OgsetResult):
    if not res.ops:
        print("  (no OG-SET operators selected)")
        return
    print(f"  OG-only R² (train)={res.og_r2_train:.4f}" + (f" | (val)={res.og_r2_val:.4f}" if res.og_r2_val is not None else ""))
    print("  Path: k  bits    r2_train")
    last_k = len(res.ops)
    for k, (b, r2) in enumerate(zip(res.bits_path, res.r2_train_path)):
        star = "*" if k == last_k else " "
        print(f"   {k:>2d}{star} {b:>8.2f}  {r2:>8.4f}")

# =========================
# Group discovery helper
# =========================

def discover_groups(X, corr_thr_perm=0.98, var_tol_rot=0.1, max_rot_pairs=12):
    X = np.asarray(X, float); N, D = X.shape
    spec = GroupSpec(sign_groups=[list(range(D))], perm_groups=[], rot2d_pairs=[], scale_groups=[])
    if D < 2: return spec, []
    Z = (X - X.mean(0,keepdims=True)) / (X.std(0,keepdims=True) + 1e-12)
    C = corrcoef_safe(Z); np.fill_diagonal(C, 0.0)
    used = set()
    for i in range(D):
        if i in used: continue
        group = [i]
        for j in range(i+1, D):
            if j in used: continue
            if abs(C[i,j]) >= corr_thr_perm:
                group.append(j); used.add(j)
        if len(group) >= 3: spec.perm_groups.append(group)
    vars_ = X.var(0); pairs = []
    for i in range(D):
        for j in range(i+1, D):
            if abs(C[i,j]) < 0.1:
                vm = 0.5*(vars_[i]+vars_[j]);
                if vm==0: continue
                if abs(vars_[i]-vars_[j])/vm < var_tol_rot:
                    pairs.append((i,j))
    pairs = pairs[:max_rot_pairs]; spec.rot2d_pairs = pairs
    return spec, []

# =========================
# Single-split runner — with OG-SET wired in
# =========================

def run_fase_given_split(Xtr, ytr, Xva, yva, seed=1337, name="", Sigma_tr=None, Sigma_va=None, og_keep: Optional[set]=None):
    rng = np.random.default_rng(seed)

    # Stage 1
    s1 = forward_atomic_search(
        Xtr, ytr, Xva, yva, rng,
        max_atoms=CONFIG["MAX_ATOMS"], log_prefix=" ",
        Sigma_tr=Sigma_tr, Sigma_va=Sigma_va
    )
    Ftr, Fva = s1["Ftr"], s1["Fva"]
    stage1_specs = list(s1["specs"])

    # ---------- OG-SET over the atomic alphabet ----------
    og_train = og_val = None
    og_names = []
    og_selected_info = []

    if CONFIG["OGSET"]["enable"] and s1["cand"]:
        Phi_tr = np.hstack([c["ztr"] for c in s1["cand"]])
        Phi_va = np.hstack([c["zva"] for c in s1["cand"]])
        atom_names = s1["cand_names"]

        ogcfg = CONFIG["OGSET"]
        sel_idx = []
        if og_keep is not None:
            sel_idx = [j for j, nm in enumerate(atom_names) if f"og:{nm}" in og_keep]
        else:
            og = build_ogset(
                y=ytr, Phi=Phi_tr, atom_names=atom_names,
                Sigma=Sigma_tr,
                max_ops=ogcfg.get("max_ops"),
                bic_bits_threshold=ogcfg.get("bic_bits_threshold"),
                ebic_gamma=ogcfg.get("ebic_gamma"),
                max_corr=ogcfg.get("max_corr", 0.98),
                bag_boots=ogcfg.get("bag_boots", 0),
                bag_frac=ogcfg.get("bag_frac", 0.8),
                min_freq=ogcfg.get("min_freq", 0.0),
                y_val=yva, Phi_val=Phi_va,
            )
            log_ogset_summary(og)
            log_ogset_diagnostics(og)
            sel_idx = getattr(og, "selected_idx", [])
            for op in getattr(og, "ops", []):
                nm = f"og:{op.name}"
                sign = 0
                if op.coeff > 0:
                    sign = 1
                elif op.coeff < 0:
                    sign = -1
                og_selected_info.append({"name": nm, "sign": int(sign), "bits": float(op.delta_bic_bits)})

        if sel_idx:
            og_train_raw = Phi_tr[:, sel_idx]
            og_val_raw   = Phi_va[:, sel_idx]
            Ftr = np.hstack([Ftr, og_train_raw]) if Ftr.size else og_train_raw
            Fva = np.hstack([Fva, og_val_raw])   if Fva.size else og_val_raw
            og_names = [f"og:{atom_names[j]}" for j in sel_idx]
            for j in sel_idx:
                stage1_specs.append(dict(spec=s1["cand"][j]["spec"], Gamma=np.zeros((0,1)), mu=np.zeros(1), sd=np.ones(1)))

            og_ortho_tr = og_ortho_va = None
            if ogcfg.get("orthonormal_export", True):
                W = _as_weight_matrix(Sigma_tr, len(ytr))
                L = _whitener_from_W(W)
                og_ortho_tr, T = export_superfeatures_orthonormal(Phi_tr[:, sel_idx], L)
                og_ortho_va = apply_superfeatures(Phi_va[:, sel_idx], T)

            og_train = og_ortho_tr if og_ortho_tr is not None else og_train_raw
            og_val   = og_ortho_va if og_ortho_va is not None else og_val_raw

            print(f" [+] OG-SET injected {len(sel_idx)} super-ops.")
        else:
            print(" [ ] OG-SET selected no operators (no ΔBIC gain).")

    # ===== Fit linear base and decide on Stage-2 / Stage-2.5 =====
    w, b0, _ = ridge_with_intercept(Ftr, ytr, CONFIG["ALPHA_RIDGE"], Sigma=Sigma_tr)
    yhat = predict_with_intercept(Fva, w, b0)
    val_r2 = r2_score(yva, yhat)
    val_r2_gls = gls_r2(yva, yhat, Sigma_va)
    lock = (val_r2 >= CONFIG["LINEAR_LOCK_THR"])
    kept = "atomic+ogset" if og_names else "atomic"
    accepted_blocks = []

    if not lock and CONFIG["MAX_GRAMMAR"] > 0:
        spec, _ = discover_groups(Xtr)
        s2 = stage2_grammar_search(
            Xtr, ytr, Xva, yva,
            dict(Ftr=Ftr, Fva=Fva),
            criterion="MDL", mdl_costs=CONFIG["MDL_COSTS"],
            use_rdmp=True, seed=seed, max_grammar=CONFIG["MAX_GRAMMAR"],
            group_spec=spec, log_prefix=" ",
            Sigma_tr=Sigma_tr, Sigma_va=Sigma_va
        )
        kept = ("full" if not og_names else "full+ogset")
        Ftr, Fva = s2["Ftr"], s2["Fva"]
        accepted_blocks = s2["accepted"]
        w, b0, _ = ridge_with_intercept(Ftr, ytr, CONFIG["ALPHA_RIDGE"], Sigma=Sigma_tr)
        yhat = predict_with_intercept(Fva, w, b0)
        val_r2 = r2_score(yva, yhat)
        val_r2_gls = gls_r2(yva, yhat, Sigma_va)

    if CONFIG.get("USE_RULIAD_STAGE25", False) and not lock:
        s25 = stage2p5_ruliad_search(
            Xtr, ytr, Xva, yva,
            dict(Ftr=Ftr, Fva=Fva), CONFIG["MDL_COSTS"], CONFIG["RULIAD"], log_prefix=" ",
            Sigma_tr=Sigma_tr, Sigma_va=Sigma_va
        )
        if len(s25["accepted"]) > 0:
            kept = s25["kept"] + ("+ogset" if og_names else "")
            Ftr, Fva = s25["Ftr"], s25["Fva"]
            accepted_blocks.extend(s25["accepted"])
            w, b0, _ = ridge_with_intercept(Ftr, ytr, CONFIG["ALPHA_RIDGE"], Sigma=Sigma_tr)
            yhat = predict_with_intercept(Fva, w, b0)
            val_r2 = r2_score(yva, yhat)
            val_r2_gls = gls_r2(yva, yhat, Sigma_va)

    val_mse = float(np.mean((yva - yhat)**2))
    if lock:
        print(" (Linear-first lock) Validation R² = %.4f ≥ %.2f → skip Stage 2/2.5." % (val_r2, CONFIG["LINEAR_LOCK_THR"]))
    print("✅ Final R²=%.4f | GLS-R²=%.4f | MSE=%8.4f | kept=%s" % (val_r2, val_r2_gls, val_mse, kept))

    model = FASEModel(stage1_specs=stage1_specs, stage2_blocks=accepted_blocks, w=w, b0=b0)
    return dict(
        R2=val_r2, R2_gls=val_r2_gls, MSE=val_mse, kept=kept, blocks=accepted_blocks, model=model,
        og_train=og_train, og_val=og_val, og_names=og_names, og_selected=og_selected_info
    )

# =========================
# Model wrapper
# =========================

def apply_stage1_spec(spec, X, _cache):
    base = spec["spec"]
    kind = base["kind"]
    if kind == "unary":
        fn = UNARY_FUNCS[base["fname"]]
        z0 = safe_num(fn(X[:, [base["idx"]]]))
    elif kind == "hsml_unary":
        fn = UNARY_FUNCS[base["fname"]]
        w = np.asarray(base["w"])
        z0 = safe_num(fn((X @ w).reshape(-1,1)))
    elif kind == "hsml_binary":
        w1 = np.asarray(base["w1"]); w2 = np.asarray(base["w2"])
        z1 = (X @ w1).reshape(-1,1); z2 = (X @ w2).reshape(-1,1)
        op = base["op"]
        if op == "+": z0 = z1 + z2
        elif op == "-": z0 = z1 - z2
        elif op == "*": z0 = z1 * z2
        else: z0 = z1 / (np.abs(z2)+1e-3)
        z0 = safe_num(z0)
    else:
        raise ValueError(f"unknown stage1 spec {kind}")
    mu0 = np.asarray(base.get("mu",0.0)); sd0 = np.asarray(base.get("sd",1.0))
    z = col_scale_apply(z0, mu0, sd0)
    F_prev = _cache.get("F")
    Gamma = spec["Gamma"]
    if F_prev is None or F_prev.size==0 or Gamma.size==0:
        R = z
    else:
        R = z - F_prev @ Gamma
    mu = spec["mu"]; sd = spec["sd"]
    R = (R - mu)/sd
    _cache["F"] = np.hstack([F_prev, R]) if (F_prev is not None and F_prev.size) else R
    return R
class FASEModel:
    def __init__(self, stage1_specs, stage2_blocks, w, b0):
        self.stage1_specs = stage1_specs
        self.stage2_blocks = stage2_blocks
        self.w = w; self.b0 = b0

    def _build_features(self, X):
        s1_cache = {"F": None}
        for spec in self.stage1_specs:
            apply_stage1_spec(spec, X, s1_cache)
        F = s1_cache.get("F")
        for blk in self.stage2_blocks:
            Gamma, mus, sds, Fb = blk["apply"](X)
            R = apply_residualization(Fb, F, Gamma) if F is not None and F.size > 0 else Fb
            R = (R - mus)/sds
            F = np.hstack([F, R]) if F is not None and F.size > 0 else R
        return F if F is not None else np.zeros((X.shape[0],0))

    def predict(self, X):
        F = self._build_features(X)
        return predict_with_intercept(F, self.w, self.b0)

    def to_dict(self):
        def ser_stage1(sp):
            base = {k:(v.tolist() if isinstance(v,np.ndarray) else v) for k,v in sp["spec"].items()}
            return {
                "spec": base,
                "Gamma": sp["Gamma"].tolist(),
                "mu": sp["mu"].tolist(),
                "sd": sp["sd"].tolist()
            }
        def ser_block(blk):
            if blk["kind"] == "ruliad":
                raise ValueError("Serialization of ruliad blocks not supported")
            params = {k:(v.tolist() if isinstance(v,np.ndarray) else v) for k,v in blk["params"].items()}
            return {"kind": blk["kind"], "params": params,
                    "Gamma": blk["Gamma"].tolist(), "mus": blk["mus"].tolist(), "sds": blk["sds"].tolist()}
        return {
            "w": self.w.tolist(),
            "b0": float(self.b0),
            "stage1": [ser_stage1(s) for s in self.stage1_specs],
            "stage2": [ser_block(b) for b in self.stage2_blocks]
        }

    @classmethod
    def from_dict(cls, d):
        def deser_stage1(s):
            base = {k:(np.asarray(v) if isinstance(v,list) else v) for k,v in s["spec"].items()}
            return {"spec": base,
                    "Gamma": np.asarray(s["Gamma"]),
                    "mu": np.asarray(s["mu"]),
                    "sd": np.asarray(s["sd"])}
        def make_block_apply(kind, params, Gamma, mus, sds):
            def _apply(X, kind=kind, params=params, Gamma=Gamma, mus=mus, sds=sds):
                Fb = None
                if kind == "relu_proj": Fb = block_relu_proj(X, np.asarray(params["w"]), params["t"])
                elif kind == "sinproj": Fb = block_sinproj(X, np.asarray(params["w"]), params["b"])
                elif kind == "fct": Fb = block_fct(X, np.asarray(params["w"]), params["b0"], params["kappa"])
                elif kind == "bilinear": Fb = block_bilinear(X, params["i"], params["j"])
                elif kind == "dihedral_invar": Fb = block_dihedral_invar(X, params["i"], params["j"], use_r4=False)
                elif kind == "perm_invar": Fb = block_perm_invar(X, params["group_idx"])
                elif kind == "group_invar": Fb = block_group_invar(X, params["spec"])
                elif kind == "combo_vec": Fb = block_combo_vector(X, np.asarray(params["w1"]), np.asarray(params["w2"]))
                else:
                    Fb = None
                return Gamma, mus, sds, Fb
            return _apply
        def deser_block(b):
            params = {k:(np.asarray(v) if isinstance(v,list) else v) for k,v in b["params"].items()}
            Gamma = np.asarray(b["Gamma"]); mus = np.asarray(b["mus"]); sds = np.asarray(b["sds"])
            apply = make_block_apply(b["kind"], params, Gamma, mus, sds)
            return {"kind": b["kind"], "params": params, "Gamma": Gamma, "mus": mus, "sds": sds, "apply": apply}
        stage1 = [deser_stage1(s) for s in d.get("stage1", [])]
        stage2 = [deser_block(b) for b in d.get("stage2", [])]
        w = np.asarray(d["w"])
        b0 = float(d["b0"])
        return cls(stage1_specs=stage1, stage2_blocks=stage2, w=w, b0=b0)

    def save(self, path):
        with open(path, "w") as f:
            json.dump(self.to_dict(), f)

    @staticmethod
    def load(path):
        with open(path, "r") as f:
            data = json.load(f)
        return FASEModel.from_dict(data)

# =========================
# Simple K-fold splitter (no sklearn)
# =========================

def kfold_indices(n: int, K: int, seed: int=1337, shuffle: bool=True):
    idx = np.arange(n)
    if shuffle:
        rng = np.random.default_rng(seed)
        rng.shuffle(idx)
    folds = []
    fold_sizes = [(n + i) // K for i in range(K)]
    start = 0
    parts = []
    for fs in fold_sizes:
        parts.append(idx[start:start+fs])
        start += fs
    for k in range(K):
        va = parts[k]
        tr = np.concatenate([parts[i] for i in range(K) if i != k]) if K>1 else idx
        folds.append((tr, va))
    return folds

# =========================
# K-fold driver with OOF & stability
# =========================

def run_fase_kfold(X, y, Sigma=None, K=5, seed=1337, config=CONFIG):
    folds = kfold_indices(len(y), K, seed=seed, shuffle=True)
    oof_yhat = np.zeros_like(y, dtype=float)
    fold_reports = []
    all_selected_ops = []
    sign_tallies: Dict[str, List[int]] = {}
    bits_tallies: Dict[str, List[float]] = {}

    for k, (tr, va) in enumerate(folds, 1):
        print(f"\n==================== Fold {k}/{K} ====================")
        Xtr, ytr = X[tr], y[tr]; Xva, yva = X[va], y[va]
        Sig_tr = subset_sigma(Sigma, tr); Sig_va = subset_sigma(Sigma, va)

        out = run_fase_given_split(
            Xtr, ytr, Xva, yva, seed=seed+k,
            name=f"fold{k}", Sigma_tr=Sig_tr, Sigma_va=Sig_va
        )
        yhat_va = out["model"].predict(Xva).ravel()
        oof_yhat[va] = yhat_va

        fold_reports.append(dict(
            R2=out["R2"], R2_gls=out["R2_gls"], MSE=out["MSE"], kept=out["kept"],
            og_names=out["og_names"], blocks=[b["kind"] for b in out["blocks"]],
        ))
        all_selected_ops.extend(out["og_names"])  # names are stable keys
        for info in out.get("og_selected", []):
            nm = info["name"]; s = info["sign"]; b = info.get("bits", 0.0)
            sign_tallies.setdefault(nm, []).append(int(s))
            bits_tallies.setdefault(nm, []).append(float(b))

    R2_oof = r2_score(y, oof_yhat)
    R2_oof_gls = gls_r2(y, oof_yhat, Sigma)
    MSE_oof = float(np.mean((y - oof_yhat)**2))

    counts = Counter(all_selected_ops)
    stab = {nm: counts[nm]/K for nm in counts}
    # Sign-stability = fraction agreeing with the majority sign over appearances
    sign_stab = {}
    for nm, signs in sign_tallies.items():
        pos = sum(1 for s in signs if s>0); neg = sum(1 for s in signs if s<0)
        total = len(signs)
        sign_stab[nm] = (max(pos, neg)/float(total)) if total else 0.0
    # Per-op evidence floor across folds (min ΔBIC bits)
    og_min_bits = {nm: float(np.min(v)) for nm, v in bits_tallies.items() if len(v)}

    print("\n==================== OOF Metrics ====================")
    print(f"OOF R²={R2_oof:.4f} | OOF GLS-R²={R2_oof_gls:.4f} | OOF MSE={MSE_oof:.6f}")
    print("Operator stability (freq over folds):")
    for nm, fr in sorted(stab.items(), key=lambda t:-t[1]):
        print(f"  {nm:>40s}  {fr:>5.1%}")
    if sign_stab:
        print("Operator sign-stability (agreement with majority sign):")
        for nm, ss in sorted(sign_stab.items(), key=lambda t:-t[1]):
            print(f"  {nm:>40s}  {ss:>5.1%}")

    report = dict(R2_oof=R2_oof, R2_oof_gls=R2_oof_gls, MSE_oof=MSE_oof,
                  folds=fold_reports, og_stability=stab, og_stability_signs=sign_stab,
                  og_min_bits=og_min_bits, oof_pred=oof_yhat)
    report["model"] = fit_consensus_model(X, y, Sigma, report, config=config)
    return report

def fit_consensus_model(X, y, Sigma, kfold_report, config=CONFIG):
    ogcfg = config.get("OGSET", {})
    thr_f = ogcfg.get("final_min_freq", 0.0)
    thr_s = ogcfg.get("final_min_sign_stab", 0.0)
    thr_b = ogcfg.get("final_min_bits", 0.0)
    stab = kfold_report.get("og_stability", {})
    sign = kfold_report.get("og_stability_signs", {})
    bits = kfold_report.get("og_min_bits", {})
    keep = {nm for nm, fr in stab.items()
            if fr >= thr_f and sign.get(nm, 0.0) >= thr_s and bits.get(nm, 0.0) >= thr_b}
    if keep:
        print(f"[Consensus] using {len(keep)} OG ops")
    else:
        print("[Consensus] no OG ops passed thresholds")
    seed = config.get("SEEDS", [1337])[0]
    out = run_fase_given_split(X, y, X, y, seed=seed, name="consensus",
                               Sigma_tr=Sigma, Sigma_va=Sigma, og_keep=keep)
    return out["model"]

# =========================
# Optional: PySR baseline wrapper
# =========================

def run_pysr_baseline(Xtr, ytr, Xva, yva, seed=1337, pysr_cfg=None,
                      extra_features_train=None, extra_features_val=None, extra_feature_names=None):
    try:
        from pysr import PySRRegressor
    except Exception as e:
        print(f"[PySR] Not available. Install with: pip install pysr  (error: {e})")
        return {"R2": float("nan"), "expr": None}

    cfg = pysr_cfg or {}
    Xtr2 = np.asarray(Xtr); Xva2 = np.asarray(Xva)
    if extra_features_train is not None and extra_features_val is not None and extra_features_train.size and extra_features_val.size:
        Xtr2 = np.hstack([Xtr2, np.asarray(extra_features_train)])
        Xva2 = np.hstack([Xva2, np.asarray(extra_features_val)])

    try:
        model = PySRRegressor(
            niterations=cfg.get("niterations", 600),
            binary_operators=cfg.get("binary_operators", ["+","-","*","/"]),
            unary_operators=cfg.get("unary_operators", ["sin","cos","tanh","exp","log","abs"]),
            procs=0, random_state=seed, progress=True,
            batching=cfg.get("batching", True), batch_size=min(2048, Xtr2.shape[0]),
            maxsize=cfg.get("maxsize", 20), maxdepth=cfg.get("maxdepth", 10),
            model_selection="best", verbosity=0
        )
        model.fit(Xtr2, ytr)
        yhat = model.predict(Xva2)
        R2 = r2_score(yva, yhat)
        expr = None
        try: expr = str(model.sympy())
        except Exception: expr = None
        return {"R2": R2, "expr": expr}
    except Exception as e:
        print(f"[PySR] Failed: {e}")
        return {"R2": float("nan"), "expr": None}

# =========================
# Simple split helper (no sklearn)
# =========================

def train_val_split_indices(n: int, val_frac: float = 0.2, seed: int = 1337, shuffle: bool = True):
    idx = np.arange(n)
    if shuffle:
        rng = np.random.default_rng(seed)
        rng.shuffle(idx)
    n_val = max(1, int(round(val_frac * n)))
    va = idx[:n_val]
    tr = idx[n_val:]
    return tr, va

# =========================
# Data loaders (PMLB optional) + synthetic fallback
# =========================

def load_pmlb_dataset(name: str):
    try:
        # pmlb >=1.0
        from pmlb import fetch_data
        X, y = fetch_data(name, return_X_y=True)
        X = np.asarray(X, float)
        y = np.asarray(y, float).reshape(-1)
        return X, y
    except Exception as e:
        print(f"[PMLB] Could not load '{name}': {e}")
        return None, None

def make_synthetic(seed: int = 42, n: int = 800, d: int = 10, rho_feat: float = 0.5, gls_noise: bool = True):
    rng = np.random.default_rng(seed)
    # correlated Gaussian features via random SPD covariance
    A = rng.normal(size=(d, d))
    SigX = A @ A.T
    X = rng.multivariate_normal(mean=np.zeros(d), cov=SigX, size=n)

    # true signal with hetero nonlinear structure
    z1 = X[:, 0]
    z2 = X[:, 1]
    z3 = X[:, 2]
    z4 = X[:, 3]
    z5 = X[:, 4] if d >= 5 else 0.0

    f = (
        1.3 * np.sin(z1 + 0.25)
        + 0.8 * np.maximum(0.0, z2 - 0.1)
        - 0.9 * (z3 * z4)
        + 0.55 / (1.0 + np.abs(z5))
    )

    # GLS-style noise: AR(1) covariance or heteroskedastic diag
    if gls_noise:
        t = np.arange(n)
        rho = 0.85
        # Toeplitz covariance (AR1)
        C = rho ** np.abs(np.subtract.outer(t, t))
        # scale so marginal var ~ 0.7^2
        C *= (0.7 ** 2)
        L = np.linalg.cholesky(C + 1e-12 * np.eye(n))
        eps = L @ rng.normal(size=n)
        Sigma = C
    else:
        s = 0.6 + 0.2 * np.abs(rng.normal(size=n))
        eps = rng.normal(scale=s)
        Sigma = s  # pass as diag vector (supported)

    y = f + eps
    # center and rescale y a bit for numerical niceness
    y = (y - y.mean()) / (y.std() + 1e-9)
    return X, y, Sigma

# =========================
# Pretty report helpers
# =========================

def print_fold_report(fr):
    for i, r in enumerate(fr, 1):
        print(f"  Fold {i:02d}:  R²={r['R2']:.4f}  MSE={r['MSE']:.6f}  kept={r['kept']}")
        if r["og_names"]:
            preview = ", ".join([n.replace('og:', '') for n in r["og_names"][:5]])
            if len(r["og_names"]) > 5:
                preview += ", …"
            print(f"           OG ops: {preview}")
        if r["blocks"]:
            print(f"           Blocks: {', '.join(r['blocks'])}")

# =========================
# Demo runner for one dataset
# =========================

def run_demo_on_dataset(name: str, X: Optional[np.ndarray]=None, y: Optional[np.ndarray]=None,
                        Sigma: Optional[np.ndarray]=None, seed: int = 1337):
    print("\n====================================================")
    print(f"Dataset: {name}")
    print("====================================================")

    if X is None or y is None:
        Xp, yp = load_pmlb_dataset(name)
        if Xp is None:
            print("[demo] Falling back to synthetic data (GLS noise).")
            Xp, yp, Sigma = make_synthetic(seed=seed, n=900, d=10, gls_noise=True)
        else:
            Xp = np.asarray(Xp, float); yp = np.asarray(yp, float).reshape(-1)
            # No Σ from PMLB — you can leave Σ=None or synthesize a heteroskedastic diag
            if Sigma is None:
                # mild heteroskedastic weights as a demo
                w = 0.8 + 0.4 * np.abs(np.sin(np.linspace(0, 10, len(yp))))
                Sigma = w  # vector form means diag(W)
        X, y = Xp, yp

    # K-fold with OOF/stability
    K = CONFIG.get("K_FOLDS", 5)
    seed0 = CONFIG["SEEDS"][0] if CONFIG.get("SEEDS") else seed
    kres = run_fase_kfold(X, y, Sigma=Sigma, K=K, seed=seed0, config=CONFIG)
    print_fold_report(kres["folds"])

    # Optional: single-split + PySR baseline with OG-SET augmentation
    if CONFIG.get("COMPARE_WITH_PYSR", False):
        print("\n-------------------- PySR (baseline) --------------------")
        n = len(y)
        tr, va = train_val_split_indices(n, val_frac=0.2, seed=seed0, shuffle=True)
        Sig_tr = subset_sigma(Sigma, tr); Sig_va = subset_sigma(Sigma, va)
        out = run_fase_given_split(
            X[tr], y[tr], X[va], y[va],
            seed=seed0+1, name="pysr_split", Sigma_tr=Sig_tr, Sigma_va=Sig_va
        )
        og_tr, og_va, og_names = out["og_train"], out["og_val"], out["og_names"]
        pysr = run_pysr_baseline(
            X[tr], y[tr], X[va], y[va],
            seed=seed0+2, pysr_cfg=CONFIG.get("PYSR", {}),
            extra_features_train=og_tr, extra_features_val=og_va, extra_feature_names=og_names
        )
        print(f"[PySR] R² (val) = {pysr.get('R2', float('nan')):.4f}")
        if pysr.get("expr"):
            print(f"[PySR] best expression: {pysr['expr']}")
    print("====================================================\n")
    return kres

# =========================
# __main__
# =========================

if __name__ == "__main__":
    np.set_printoptions(precision=4, suppress=True)
    random.seed(random_state_to_py(CONFIG["SEEDS"][0] if CONFIG.get("SEEDS") else 1337))
    np.random.seed(random_state_to_py(CONFIG["SEEDS"][0] if CONFIG.get("SEEDS") else 1337))

    datasets = CONFIG.get("DATASETS", [])
    if not datasets:
        datasets = ["synthetic_demo"]

    results = {}
    for ds in datasets:
        if ds == "synthetic_demo":
            X, y, Sigma = make_synthetic(
                seed=CONFIG["SEEDS"][0] if CONFIG.get("SEEDS") else 1337,
                n=900, d=10, gls_noise=True
            )
            res = run_demo_on_dataset(ds, X=X, y=y, Sigma=Sigma, seed=CONFIG["SEEDS"][0])
        else:
            res = run_demo_on_dataset(ds, seed=CONFIG["SEEDS"][0])
        results[ds] = dict(R2_OOF=res["R2_oof"], MSE_OOF=res["MSE_oof"], og_stability=res["og_stability"])

    print("\n==================== Summary ====================")
    for ds, r in results.items():
        print(f"{ds:>20s}  |  OOF R²={r['R2_OOF']:.4f}  MSE={r['MSE_OOF']:.6f}  |  #stable ops={len(r['og_stability'])}")    