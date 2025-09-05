# FASE v20.0 — HSML+L4+Mini-Ruliad + OG-SET/GLS (AICc/BIC/MDL via χ²), single cell
# If needed: pip install pmlb pysr

from __future__ import annotations
import numpy as np, json, math, random, copy, hashlib, warnings, itertools, time
from dataclasses import dataclass, field
from typing import List, Dict, Any, Tuple, Optional, Callable
from collections import Counter, defaultdict
from sklearn.metrics import mean_squared_error

warnings.filterwarnings("ignore")

# Optional: comment these two lines out if you don't want PMLB/PySR bits
import pmlb
from pmlb import dataset_names, fetch_data, classification_dataset_names, regression_dataset_names

print("PMLB datasets available:", len(dataset_names))

# =========================
# Config
# =========================
CONFIG = {
    "SEEDS": [42, 1337, 2027],
    "K_FOLDS": 5,
    "DATASETS": [
        "579_fri_c0_250_5","581_fri_c3_500_25","582_fri_c1_500_25","583_fri_c1_1000_50",
        "584_fri_c4_500_25","586_fri_c3_1000_25","588_fri_c4_1000_100","589_fri_c2_1000_25",
        "590_fri_c0_1000_50","591_fri_c1_100_10","592_fri_c4_1000_25","593_fri_c1_1000_10",
        "594_fri_c2_100_5","595_fri_c0_1000_10","596_fri_c2_250_5","597_fri_c2_500_5",
        "598_fri_c0_1000_25","599_fri_c2_1000_5","601_fri_c1_250_5"
    ],
    "LINEAR_LOCK_THR": 0.95,
    "MAX_ATOMS": 32,
    "MAX_GRAMMAR": 12,
    "ALPHA_RIDGE": 2e-3,
    "RDMP_ALPHA": 1e-4,
    "DO_CV_STABILITY": False,
    "STAGE1": {"CRITERION": "MDL", "MIN_MSE_GAIN": 1e-4, "DUP_CORR_THR": 0.995},
    "USE_RULIAD_STAGE25": True,
    "RULIAD": {
        "depth": 6, "K_per_parent": 25, "frontier_size": 24, "random_seed": 42,
        "keep_outputs_topk": 24, "energy": {"lam":1.0,"mu":0.05,"nu":0.05,"mdl_scale":1.0,"xi":0.02,"sign_flip_indices":()},
        "use_param_opt": True
    },
    "MDL_COSTS": {
        "type_bits":{"bilinear":8.0,"relu_proj":8.0,"sinproj":8.0,"fct":11.0,"perm_invar":7.0,
                     "dihedral_invar":8.0,"group_invar":10.0,"combo_vec":9.0,"hsml_unary":5.5,
                     "hsml_binary":6.5,"atomic":4.0,"ruliad":12.0},
        "per_col_bits":0.75, "per_real_bits":1.5, "bit_weight":1.0
    },
    "COMPARE_WITH_PYSR": True,
    "PYSR": {"niterations":250,"maxsize":18,"maxdepth":8,"binary_operators":["+","-","*","/"],
             "unary_operators":["sin","cos","tanh","exp","log","abs"],"batching":True},
    # --- Optional OG-SET driver: supply paths to run GLS on your basis Yb & Σ ---
    # CSVs must be numeric (no header). y_csv is a single column (N,).
    # Sigma JSON must be a nested list or {"Sigma":[[...],[...],...]}.
    "OGSET": {
        "Yb_csv": "",    # e.g., "/content/Yb.csv" (N x K)
        "y_csv":  "",    # e.g., "/content/y.csv"  (N,)
        "Sigma_json": "" # e.g., "/content/Sigma.json"
    }
}
CONFIG["PYSR"].update({"deterministic":True,"parallelism":"serial","constraints":{"^":(-1,1)}})

EPS = 1e-9

# =========================
# OG-SET / GLS utilities
# =========================
def chol_whiten(Sigma: np.ndarray):
    """Whiten helper kept for backward compatibility.
    For 1-D diagonal Σ it performs a safe sqrt; for full Σ it delegates
    to the robust weight-matrix/eig path so callers don't Cholesky raw Σ."""
    Sig = np.asarray(Sigma, float)
    if Sig.ndim == 1:  # diagonal variances safe
        w = np.clip(Sig, 1e-12, None)
        L = np.diag(np.sqrt(w)); Linv = np.diag(1.0/np.sqrt(w))
        return L, Linv
    W = _as_weight_matrix(Sig, Sig.shape[0])
    C = _whitener_from_W(W)
    L = np.linalg.inv(C.T)   # so that L @ L.T ≈ Σ (best-effort)
    Linv = C
    return L, Linv

def _as_weight_matrix(Sigma: Optional[np.ndarray], n: int) -> np.ndarray:
    if Sigma is None:
        return np.eye(n)
    Sig = np.asarray(Sigma, float)
    if Sig.ndim == 1:
        inv = np.where(Sig>0, 1.0/np.clip(Sig,1e-12,None), 0.0)
        return np.diag(inv)
    S = 0.5*(Sig + Sig.T)  # neutralize tiny asymmetries
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

def gls_chi2(residuals: np.ndarray, Sigma: np.ndarray) -> float:
    r = np.asarray(residuals, float).reshape(-1)
    if Sigma is None:
        return float(r @ r)
    W = _as_weight_matrix(Sigma, len(r))
    return float(r @ (W @ r))

# =========================
# Metrics & Criteria (AICc, BIC, MDL; GLS-aware)
# =========================
def r2_score(y, yhat):
    y = np.asarray(y).ravel(); yhat = np.asarray(yhat).ravel()
    ss_res = np.sum((y - yhat)**2); ss_tot = np.sum((y - y.mean())**2) + 1e-12
    return 1.0 - ss_res/ss_tot

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

def mdl_bits_for(kind: str, params: Dict, n_cols: int, mdl_costs: Dict):
    tb = mdl_costs["type_bits"].get(kind, 8.0)
    per_col = mdl_costs["per_col_bits"] * max(0, int(n_cols))
    pcount = 0
    for v in params.values():
        if isinstance(v, (float, int)): pcount += 1
        elif isinstance(v, (list, tuple, np.ndarray)): pcount += int(np.size(v))
    per_real = mdl_costs["per_real_bits"] * pcount
    return float(tb + per_col + per_real)

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

# =========================
# Robust correlation util
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

# =========================
# Ridge (with intercept), GLS-aware
# =========================
def ridge_with_intercept(X, y, alpha, Sigma: Optional[np.ndarray]=None):
    X = np.asarray(X, float); y = np.asarray(y, float).ravel()
    n, d = X.shape
    if d == 0:
        return np.zeros(0), float(y.mean()), {"muX": np.zeros((1,0)), "muy": float(y.mean()), "Sigma": Sigma}

    if Sigma is not None:
        W = _as_weight_matrix(Sigma, n)      # robust Σ^{-1} (pinv for full Σ)
        C = _whitener_from_W(W)              # C^T C = W (chol or eig sqrt)
        Xw = C @ X
        yw = (C @ y.reshape(-1,1)).reshape(-1)
    else:
        Xw, yw = X, y

    muX = Xw.mean(axis=0, keepdims=True); muy = float(yw.mean())
    Xc = Xw - muX; yc = yw - muy
    A = Xc.T @ Xc + alpha*np.eye(d); b = Xc.T @ yc
    w = np.linalg.solve(A, b)
    b0 = muy - (muX @ w.reshape(-1,1)).item()
    return w, b0, {"muX": muX, "muy": muy, "Sigma": Sigma}

def predict_with_intercept(X, w, b0):
    return (X @ w + b0) if w.size else np.full(X.shape[0], b0)

# =========================
# L4 group spec & transforms
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
# Atomic space (HSML)
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
def safe_num(a): return np.nan_to_num(a, nan=0.0, posinf=1e6, neginf=-1e6)

def col_scale_fit(v):
    v = np.asarray(v).reshape(-1,1); mu = float(v.mean()); sd = float(v.std())
    if sd < 1e-12: sd = 1.0
    return mu, sd

def col_scale_apply(v, mu, sd): return (np.asarray(v).reshape(-1,1) - mu) / sd

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

def build_stage1_candidates(Xtr, Xva, rng, raw_unary=True, n_proj=24, n_pairs=24):
    Ntr, D = Xtr.shape; bank = []
    if raw_unary:
        for fname, fn in UNARY_FUNCS.items():
            Ztr = safe_num(fn(Xtr)); Zva = safe_num(fn(Xva))
            if Ztr.ndim == 1: Ztr = Ztr[:,None]
            if Zva.ndim == 1: Zva = Zva[:,None]
            if Ztr.shape[1] != D:  # only 1-to-1 functions
                continue
            for j in range(D):
                vtr0 = Ztr[:, j:j+1]; vva0 = Zva[:, j:j+1]
                mu, sd = col_scale_fit(vtr0)
                vtr = col_scale_apply(vtr0, mu, sd); vva = col_scale_apply(vva0, mu, sd)
                if float(vtr.std()) < 1e-8: continue
                def make_apply(fn=fn, j=j, mu=mu, sd=sd):
                    return lambda X, _cache=None: col_scale_apply(safe_num(fn(X[:, j:j+1])), mu, sd)
                bank.append({"name": f"{fname}[x{j}]", "ztr": vtr, "zva": vva, "apply": make_apply()})
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
            bank.append({"name": f"hsml:{fname}({pnm})", "ztr": vtr, "zva": vva, "apply": make_apply()})
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
            bank.append({"name": f"hsml2:({p1_nm}{op_nm}{p2_nm})", "ztr": vtr, "zva": vva, "apply": make_apply()})
    return bank

# RDMP helpers
def fit_residualization(Fb_tr, Fbase_tr, alpha=1e-4):
    if Fb_tr.size == 0 or Fbase_tr.size == 0: return np.zeros((0, Fb_tr.shape[1]))
    A = Fbase_tr.T @ Fbase_tr + alpha*np.eye(Fbase_tr.shape[1]); B = Fbase_tr.T @ Fb_tr
    return np.linalg.solve(A, B)

def apply_residualization(Fb, Fbase, Gamma):
    return Fb if Gamma.size == 0 else (Fb - Fbase @ Gamma)

# =========================
# Stage-1 forward selection (GLS-aware)
# =========================
def forward_atomic_search(Xtr, ytr, Xva, yva, rng, max_atoms, log_prefix=" ", Sigma_tr=None, Sigma_va=None):
    print(log_prefix + "--- Stage 1: Atomic Alphabet ---")
    cand = build_stage1_candidates(Xtr, Xva, rng, raw_unary=True, n_proj=24, n_pairs=24)
    print(log_prefix + f" (built {len(cand)} candidates)")
    criterion = CONFIG["STAGE1"]["CRITERION"]
    bit_weight = CONFIG["MDL_COSTS"]["bit_weight"]

    Ftr = np.zeros((Xtr.shape[0],0)); Fva = np.zeros((Xva.shape[0],0))
    chosen = []; chosen_apply = []

    # baseline (intercept-only)
    beta, b0, _ = ridge_with_intercept(Ftr, ytr, CONFIG["ALPHA_RIDGE"], Sigma=Sigma_tr)
    res_va = yva - predict_with_intercept(Fva, beta, b0)
    base_bits = mdl_bits_for("atomic", {}, 0, CONFIG["MDL_COSTS"])
    best_score = crit_from_residuals(res_va, len(yva), Ftr.shape[1]+1,
                                     criterion=criterion, mdl_bits=base_bits, bit_weight=bit_weight, Sigma=Sigma_va)
    base_mse = float(np.mean(res_va**2))
    used = set(); step = 0
    while step < max_atoms:
        step += 1
        best_idx = -1; best_add_score = None; best_pack = None
        for i, c in enumerate(cand):
            if i in used: continue
            Gamma = fit_residualization(c["ztr"], Ftr, alpha=CONFIG["RDMP_ALPHA"]) if Ftr.size else np.zeros((0, c["ztr"].shape[1]))
            R_tr = apply_residualization(c["ztr"], Ftr, Gamma) if Ftr.size else c["ztr"]
            R_va = apply_residualization(c["zva"], Fva, Gamma) if Fva.size else c["zva"]
            if Ftr.size:
                num = np.linalg.norm(c["ztr"] - (Ftr @ Gamma))
                den = np.linalg.norm(c["ztr"]) + 1e-12
                if den>0 and (1.0 - num/den) > CONFIG["STAGE1"]["DUP_CORR_THR"]:
                    continue
            mu = R_tr.mean(axis=0, keepdims=True); sd = R_tr.std(axis=0, keepdims=True); sd[sd<1e-12]=1.0
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
            def make_stage1_apply(apply_cand=cand[best_idx]["apply"], Gamma=best_pack[0], mu=best_pack[1].copy(), sd=best_pack[2].copy()):
                def _apply(X, _cache={"F": None}):
                    z = apply_cand(X, _cache=_cache)
                    F_prev = _cache.get("F")
                    if F_prev is None or F_prev.size==0 or Gamma.size==0:
                        R = z
                    else:
                        R = z - F_prev @ Gamma
                    R = (R - mu)/sd
                    _cache["F"] = np.hstack([F_prev, R]) if (F_prev is not None and F_prev.size) else R
                    return R
                return _apply
            chosen_apply.append(make_stage1_apply())
            best_score = best_add_score
            base_mse = mse_tmp
            print(log_prefix + f" [A{len(chosen)}] + {cand[best_idx]['name']} (MDL★ ↓ → {best_score:.2f})")
        else:
            break
    if not chosen:
        print(log_prefix + " (no gain; empty atomic set)")
    else:
        preview = chosen[:8]; print(log_prefix + f" --- Atomic Alphabet: {preview}{'...' if len(chosen)>8 else ''} ---")
    return dict(Ftr=Ftr, Fva=Fva, atoms=chosen, apply_fns=chosen_apply, best_score=best_score)

# =========================
# Grammar blocks (Stage-2), GLS-aware
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
        Gamma = fit_residualization(Fb_tr, Fbase_tr, alpha=CONFIG["RDMP_ALPHA"]) if use_rdmp else np.zeros((Fbase_tr.shape[1], Fb_tr.shape[1]))
        R_tr = apply_residualization(Fb_tr, Fbase_tr, Gamma) if use_rdmp else Fb_tr
        R_va = apply_residualization(Fb_va, Fbase_va, Gamma) if use_rdmp else Fb_va
        mus = R_tr.mean(axis=0, keepdims=True); sds = R_tr.std(axis=0, keepdims=True); sds[sds<1e-12]=1.0
        R_trs = (R_tr - mus)/sds; R_vas = (R_va - mus)/sds
        Ftr = np.hstack([Fbase_tr, R_trs]) if Fbase_tr.size else R_trs
        Fva = np.hstack([Fbase_va, R_vas]) if Fbase_va.size else R_vas
        k = Ftr.shape[1] + 1
        if k >= len(ytr) - 1: return False
        beta, b0, _ = ridge_with_intercept(Ftr, ytr, CONFIG["ALPHA_RIDGE"], Sigma=Sigma_tr)
        res2 = yva - predict_with_intercept(Fva, beta, b0)
        bits = mdl_bits_for(kind, params, R_trs.shape[1], mdl_costs) + mdl_bits_for("atomic", {}, Fbase_tr.shape[1], mdl_costs)
        score = crit_from_residuals(res2, len(yva), k, criterion=criterion, mdl_bits=bits, bit_weight=mdl_costs["bit_weight"], Sigma=Sigma_va)
        gain = (score < base_score - 1e-6) or (float(np.mean(res2**2)) < base_mse - 1e-3)
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
            if group_spec.perm_groups:
                g = max(group_spec.perm_groups, key=len)
                fit_block("perm_invar", block_perm_invar(Xtr, g), block_perm_invar(Xva, g), dict(group_idx=g))
            fit_block("group_invar", block_group_invar(Xtr, group_spec), block_group_invar(Xva, group_spec), dict(spec=group_spec))
            w1 = rng_local.normal(size=d); w1/= (np.linalg.norm(w1)+EPS)
            w2 = rng_local.normal(size=d); w2/= (np.linalg.norm(w2)+EPS)
            fit_block("combo_vec", block_combo_vector(Xtr, w1, w2), block_combo_vector(Xva, w1, w2), dict(w1=w1,w2=w2))

    kept = "atomic" if len(accepted)==0 else "full"
    return dict(accepted=accepted, kept=kept, Ftr=Fbase_tr, Fva=Fbase_va, score=base_score)

# =========================
# Stage-2.5 Mini-Ruliad (same as v19, GLS-aware scoring)
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
        # unary
        self.add(Op("id", _un(lambda x: x), 1, mdl_cost_bits=1.0))
        self.add(Op("sin", _un(np.sin), 1)); self.add(Op("cos", _un(np.cos), 1))
        self.add(Op("tanh", _un(np.tanh), 1)); self.add(Op("abs", _un(np.abs), 1))
        self.add(Op("square", _un(lambda x: x*x), 1)); self.add(Op("signed_sqrt", _un(lambda x: np.sign(x)*np.sqrt(np.abs(x)+eps)), 1))
        self.add(Op("log1p_abs", _un(lambda x: np.log1p(np.abs(x))), 1))
        self.add(Op("x_over_one_plus_beta_abs", lambda x,beta: x/(1.0+beta*np.abs(x)+eps), 1,
                    param_names=("beta",), param_init=lambda: {"beta":1.0}))
        self.add(Op("sin_affine", lambda x,omega,phi: np.sin(omega*x+phi), 1,
                    param_names=("omega","phi"), param_init=lambda: {"omega":1.0,"phi":0.0}))
        # binary
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
    weights: Optional[np.ndarray]=None
    norm_mu: Optional[np.ndarray]=None
    norm_sigma: Optional[np.ndarray]=None
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
        C=corrcoef_safe(Fs)
        off=np.abs(C-np.eye(C.shape[0]))
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
                    A=Z.T@Z + 1e-3*np.eye(Z.shape[1]); b=Z.T@yv.ravel()
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
    A=Z.T@Z + 1e-3*np.eye(Z.shape[1]); b=Z.T@y.ravel()
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

def energy_h(state:HypergraphState, Xv:np.ndarray, yv:np.ndarray, cfg:EnergyConfig)->float:
    yhat,_=_fit_and_predict_h(state,Xv,yv); loss=float(np.mean((yv.ravel()-yhat.ravel())**2))
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
                 random_seed:int=42, use_param_opt:bool=True):
        self.registry=registry or OpRegistry(); self.energy_cfg=energy_cfg
        self.depth=depth; self.K=K_per_parent; self.frontier_size=frontier_size
        random.seed(random_seed); np.random.seed(random_seed); self.use_param_opt=use_param_opt
    def _rules(self, Xv, yv)->List[Rule]:
        rules=[rule_commute_and_canon(), rule_eliminate_double_abs(), rule_factor_distribute(), rule_toggle_product()]
        if self.use_param_opt: rules.append(rule_param_opt(steps=8, lr=0.2, Xv=Xv, yv=yv))
        return rules
    def refine(self, Xtr:np.ndarray, ytr:np.ndarray, Xv:np.ndarray, yv:np.ndarray,
               seed_state:Optional[HypergraphState]=None)->HypergraphState:
        state0=seed_state or seed_from_inputs(Xtr, self.registry)
        _=energy_h(state0, Xv, yv, self.energy_cfg)
        rules=self._rules(Xv,yv); frontier=[(state0, energy_h(state0,Xv,yv,self.energy_cfg))]
        archive={state0.canonical_hash()}; best_state,best_score=frontier[0]; stagnation=0
        for d in range(self.depth):
            props=[]
            for s,_ in frontier:
                for _ in range(self.K):
                    rule=random.choice(rules); s2=rule.apply(s)
                    if s2 is None: continue
                    h=s2.canonical_hash()
                    if h in archive: continue
                    sc=energy_h(s2,Xv,yv,self.energy_cfg)
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

def stage2p5_ruliad_search(Xtr, ytr, Xva, yva, base_info, mdl_costs, cfg_dict, log_prefix=" ",
                           Sigma_tr=None, Sigma_va=None):
    print(log_prefix + "--- Stage 2.5: Mini-Ruliad Refiner ---")
    reg=OpRegistry()
    ref=RuliadRefiner(
        registry=reg,
        energy_cfg=EnergyConfig(**cfg_dict["energy"]),
        depth=cfg_dict["depth"], K_per_parent=cfg_dict["K_per_parent"], frontier_size=cfg_dict["frontier_size"],
        random_seed=cfg_dict["random_seed"], use_param_opt=cfg_dict.get("use_param_opt", True)
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
    Gamma = fit_residualization(F_r_tr, Fbase_tr, alpha=CONFIG["RDMP_ALPHA"]) if Fbase_tr.size else np.zeros((0, F_r_tr.shape[1]))
    R_tr = apply_residualization(F_r_tr, Fbase_tr, Gamma) if Fbase_tr.size else F_r_tr
    R_va = apply_residualization(F_r_va, Fbase_va, Gamma) if Fbase_va.size else F_r_va
    mus = R_tr.mean(axis=0, keepdims=True); sds = R_tr.std(axis=0, keepdims=True); sds[sds<1e-12]=1.0
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
    param_count = 0
    bits2 = (mdl_bits_for("ruliad", {"param_count":param_count}, R_trs.shape[1], mdl_costs)
             + mdl_bits_for("atomic", {}, Fbase_tr.shape[1], mdl_costs))
    score2= crit_from_residuals(res2, len(yva), Ftr2.shape[1]+1, criterion="MDL",
                                mdl_bits=bits2, bit_weight=mdl_costs["bit_weight"], Sigma=Sigma_va)

    gain = (score2 < base_score - 1e-6) or (float(np.mean(res2**2)) < base_mse - 1e-3)
    if gain:
        print(log_prefix+f" [+] ruliad (MDL★ ↓ → {score2:.2f} | val_mse {base_mse:.4f}→{np.mean(res2**2):.4f} | "
              f"#features={R_trs.shape[1]} | nodes={len(best.nodes)} | {t1-t0:.2f}s)")
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
# Model wrapper
# =========================
class FASEModel:
    def __init__(self, stage1_applies, stage2_blocks, w, b0):
        self.stage1_applies = stage1_applies
        self.stage2_blocks = stage2_blocks
        self.w = w; self.b0 = b0
    def _build_features(self, X):
        s1_cache = {"F": None}
        for fn in self.stage1_applies:
            _ = fn(X, _cache=s1_cache)
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

# =========================
# Auto-discover L4 groups
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
    sams = [make_sign_flip_sampler(list(range(D)))]
    for g in spec.perm_groups: sams.append(make_perm_sampler(g))
    for (i,j) in spec.rot2d_pairs: sams.append(make_rot2d_sampler(i,j))
    return spec, sams

def make_sign_flip_sampler(idxs: List[int], p: float = 0.5):
    def _sample(X):
        Xg = X.copy(); mask = (np.random.rand(X.shape[0],1) < p).astype(float)
        Xg[:, idxs] *= (1 - 2*mask); return Xg
    return _sample

def make_perm_sampler(idxs: List[int]):
    idxs = list(idxs)
    def _sample(X):
        Xg = X.copy(); perm = np.random.permutation(idxs)
        Xg[:, idxs] = Xg[:, perm]; return Xg
    return _sample

def make_rot2d_sampler(i: int, j: int):
    def _sample(X):
        Xg = X.copy()
        th = np.random.uniform(0, 2*np.pi); c,s = np.cos(th), np.sin(th)
        xi, xj = X[:,i].copy(), X[:,j].copy()
        Xg[:,i] = c*xi - s*xj; Xg[:,j] = s*xi + c*xj; return Xg
    return _sample

def make_scale_sampler(idxs: List[int], log_sigma: float = 0.2):
    def _sample(X):
        Xg = X.copy(); scale = np.exp(np.random.normal(0, log_sigma))
        Xg[:, idxs] *= scale; return Xg
    return _sample

def group_probe_score(predict_fn: Callable[[np.ndarray], np.ndarray],
                      X: np.ndarray,
                      samplers: List[Callable[[np.ndarray], np.ndarray]],
                      n_draws: int = 16) -> Dict[str, float]:
    y0 = predict_fn(X).reshape(-1); scores = []; out = {}
    for k, s in enumerate(samplers):
        diffs = []
        for _ in range(n_draws):
            Xg = s(X); yg = predict_fn(Xg).reshape(-1)
            diffs.append(np.mean(np.abs(y0 - yg)))
        out[f"sampler_{k}"] = float(np.median(diffs)); scores.append(out[f"sampler_{k}"])
    out["median"] = float(np.median(scores)) if scores else np.nan
    return out

# =========================
# Single split runner (GLS-aware)
# =========================
def run_fase_given_split(Xtr, ytr, Xva, yva, seed=1337, name="", Sigma_tr=None, Sigma_va=None):
    rng = np.random.default_rng(seed)
    # Stage 1
    s1 = forward_atomic_search(Xtr, ytr, Xva, yva, rng,
                               max_atoms=CONFIG["MAX_ATOMS"], log_prefix=" ",
                               Sigma_tr=Sigma_tr, Sigma_va=Sigma_va)
    Ftr, Fva = s1["Ftr"], s1["Fva"]
    w, b0, _ = ridge_with_intercept(Ftr, ytr, CONFIG["ALPHA_RIDGE"], Sigma=Sigma_tr)
    yhat = predict_with_intercept(Fva, w, b0)
    val_r2 = r2_score(yva, yhat)
    lock = (val_r2 >= CONFIG["LINEAR_LOCK_THR"])
    kept = "atomic"; accepted_blocks = []

    # Stage 2
    if not lock and CONFIG["MAX_GRAMMAR"] > 0:
        spec, _ = discover_groups(Xtr)
        s2 = stage2_grammar_search(
            Xtr, ytr, Xva, yva,
            dict(Ftr=Ftr, Fva=Fva, atoms=s1["atoms"]),
            criterion="MDL", mdl_costs=CONFIG["MDL_COSTS"],
            use_rdmp=True, seed=seed, max_grammar=CONFIG["MAX_GRAMMAR"],
            group_spec=spec, log_prefix=" ",
            Sigma_tr=Sigma_tr, Sigma_va=Sigma_va
        )
        kept = s2["kept"]; Ftr, Fva = s2["Ftr"], s2["Fva"]; accepted_blocks = s2["accepted"]
        w, b0, _ = ridge_with_intercept(Ftr, ytr, CONFIG["ALPHA_RIDGE"], Sigma=Sigma_tr)
        yhat = predict_with_intercept(Fva, w, b0); val_r2 = r2_score(yva, yhat)

    # Stage 2.5
    if CONFIG.get("USE_RULIAD_STAGE25", False) and not lock:
        s25 = stage2p5_ruliad_search(
            Xtr, ytr, Xva, yva,
            dict(Ftr=Ftr, Fva=Fva), CONFIG["MDL_COSTS"], CONFIG["RULIAD"], log_prefix=" ",
            Sigma_tr=Sigma_tr, Sigma_va=Sigma_va
        )
        if len(s25["accepted"])>0:
            kept = s25["kept"]
            Ftr, Fva = s25["Ftr"], s25["Fva"]
            accepted_blocks.extend(s25["accepted"])
            w, b0, _ = ridge_with_intercept(Ftr, ytr, CONFIG["ALPHA_RIDGE"], Sigma=Sigma_tr)
            yhat = predict_with_intercept(Fva, w, b0); val_r2 = r2_score(yva, yhat)

    val_mse = float(np.mean((yva - yhat)**2))
    if lock:
        print(" (Linear-first lock) Validation R² = %.4f ≥ %.2f → skip Stage 2/2.5." % (val_r2, CONFIG["LINEAR_LOCK_THR"]))
    print("✅ Final R²=%.4f | MSE=%8.4f | atoms=%s | kept=%s" % (
        val_r2, val_mse, str(s1["atoms"][:6]) + ("..." if len(s1["atoms"])>6 else ""), kept))
    model = FASEModel(stage1_applies=s1["apply_fns"], stage2_blocks=accepted_blocks, w=w, b0=b0)
    return dict(R2=val_r2, MSE=val_mse, atoms=s1["atoms"], kept=kept, blocks=accepted_blocks,
                model=model, yva=yva, yhat=yhat, Sigma_va=Sigma_va, residuals=(yva - yhat))

# =========================
# CV Stability (unchanged)
# =========================
def kfold_indices(n, k, seed):
    rng = np.random.default_rng(seed); idx = np.arange(n); rng.shuffle(idx)
    folds = np.array_split(idx, k)
    for i in range(k):
        va = folds[i]
        tr = np.hstack([folds[j] for j in range(k) if j!=i])
        yield tr, va

def run_cv_stability_dataset(name, seeds, k_folds):
    X, y = load_dataset(name)
    print(f"\n==================== FASE v20 (CV) — {name} ====================")
    r2s, mses = [], []
    atom_counts = Counter(); block_counts = Counter()
    inv_scores = []
    for seed in seeds:
        for fold, (tr, va) in enumerate(kfold_indices(X.shape[0], k_folds, seed)):
            print(f"\n-- seed={seed} fold={fold+1}/{k_folds} --")
            Xtr, ytr = X[tr], y[tr]; Xva, yva = X[va], y[va]
            out = run_fase_given_split(Xtr, ytr, Xva, yva, seed=seed, name=name)
            r2s.append(out["R2"]); mses.append(out["MSE"])
            for a in out["atoms"]: atom_counts[a] += 1
            for b in out["blocks"]: block_counts[b["kind"]] += 1
            spec, sams = discover_groups(Xtr)
            probe = group_probe_score(out["model"].predict, Xva, sams, n_draws=8)
            inv_scores.append(probe["median"])
            print(f" InvarianceProbe median |Δ| = {probe['median']:.4g}")
    mean_r2, sd_r2 = float(np.mean(r2s)), float(np.std(r2s))
    mean_mse, sd_mse = float(np.mean(mses)), float(np.std(mses))
    mean_inv = float(np.nanmean(inv_scores)) if len(inv_scores) else float("nan")
    print("\nAblation-free CV Summary")
    print(f" R² = {mean_r2:.4f} ± {sd_r2:.4f} | MSE = {mean_mse:.5f} ± {sd_mse:.5f}")
    print(f" InvarianceProbe median |Δ| (↓ better): {mean_inv:.4g}")
    print("\nTop atoms by inclusion:")
    for (a, c) in atom_counts.most_common(12): print(f" {a:60s} ×{c}")
    if block_counts:
        print("\nBlock kind inclusion:")
        for (k, c) in block_counts.most_common(): print(f" {k:16s} ×{c}")
    return dict(r2s=r2s, mses=mses, atom_counts=atom_counts, block_counts=block_counts, inv_scores=inv_scores)

# =========================
# PySR baseline (same split)
# =========================
def run_pysr_baseline(Xtr, ytr, Xva, yva, seed=1337, pysr_cfg=None):
    try:
        from pysr import PySRRegressor
    except Exception as e:
        print(f"[PySR] Not available. Install with: pip install pysr  (error: {e})")
        return {"R2": float("nan"), "expr": None}

    cfg = pysr_cfg or {}
    try:
        model = PySRRegressor(
            niterations=cfg.get("niterations", 600),
            binary_operators=cfg.get("binary_operators", ["+", "-", "*", "/", "pow"]),
            unary_operators=cfg.get("unary_operators", ["sin","cos","tanh","exp","log","abs"]),
            extra_sympy_mappings={"pow": lambda a,b: a**b},
            procs=0, random_state=seed, progress=True,
            batching=cfg.get("batching", True), batch_size=min(2048, Xtr.shape[0]),
            maxsize=cfg.get("maxsize", 20), maxdepth=cfg.get("maxdepth", 10),
            model_selection="best", verbosity=0
        )
        model.fit(Xtr, ytr)
        yhat = model.predict(Xva)
        R2 = r2_score(yva, yhat)
        expr = None
        try:
            expr = str(model.sympy())
        except Exception:
            expr = None
        return {"R2": R2, "expr": expr}
    except Exception as e:
        print(f"[PySR] Failed during run: {e}")
        return {"R2": float("nan"), "expr": None}

# =========================
# PMLB loader
# =========================
def load_dataset(name: str) -> Tuple[np.ndarray, np.ndarray]:
    print(f"Fetching PMLB dataset: {name}...")
    X, y = pmlb.fetch_data(name, return_X_y=True)
    X = np.asarray(X, float)
    y = np.asarray(y).reshape(-1).astype(float)
    return X, y

# =========================
# OG-SET Driver (GLS fit on external basis Yb)
# =========================
def _print_table(rows: List[List[Any]], headers: List[str]):
    widths = [max(len(str(h)), max(len(str(r[i])) for r in rows) if rows else 0) for i,h in enumerate(headers)]
    fmt = " | ".join("{:%d}"%w for w in widths)
    sep = "-+-".join("-"*w for w in widths)
    print(fmt.format(*headers)); print(sep)
    for r in rows: print(fmt.format(*[str(x) for x in r]))

def run_ogset_gls(Yb: np.ndarray, y: np.ndarray, Sigma: np.ndarray, alpha: float = 2e-3, names: Optional[List[str]]=None):
    N, K = Yb.shape
    names = names or [f"op_{j}" for j in range(K)]
    # Fit baseline (intercept only) under GLS
    w0, b0, _ = ridge_with_intercept(np.zeros((N,0)), y, alpha, Sigma=Sigma)
    yhat0 = np.full(N, b0); r0 = y - yhat0
    # Fit full basis under GLS
    w, b, _ = ridge_with_intercept(Yb, y, alpha, Sigma=Sigma)
    yhat = predict_with_intercept(Yb, w, b); r = y - yhat
    # Criteria (GLS)
    k0 = 1; k = K + 1
    AICc0 = crit_from_residuals(r0, N, k0, "AICc", Sigma=Sigma)
    BIC0  = crit_from_residuals(r0, N, k0, "BIC",  Sigma=Sigma)
    MDL0  = crit_from_residuals(r0, N, k0, "MDL",  Sigma=Sigma)
    AICc1 = crit_from_residuals(r,  N, k,  "AICc", Sigma=Sigma)
    BIC1  = crit_from_residuals(r,  N, k,  "BIC",  Sigma=Sigma)
    MDL1  = crit_from_residuals(r,  N, k,  "MDL",  Sigma=Sigma)
    # ΔBIC in bits
    LN2 = math.log(2.0)
    dBIC_bits = (BIC0 - BIC1) / LN2
    dMDL_bits = (MDL0 - MDL1) / LN2
    # Gauge/Σ checks
    try:
        L, Linv = chol_whiten(Sigma)
        fro_rel = float(np.linalg.norm(L@L.T - Sigma) / (np.linalg.norm(Sigma)+1e-12))
        condS = float(np.linalg.cond(Sigma))
        rw = Linv @ r
        rw_std = float(rw.std())
    except Exception as e:
        L = Linv = None
        fro_rel, condS, rw_std = float("nan"), float("inf"), float("nan")
    # Print GLS line (model selection deltas)
    print("\n=== OG-SET / GLS model selection (natural logs; Δ in bits) ===")
    rows = [
        ["Intercept", k0, f"{AICc0:.3f}", f"{BIC0:.3f}", f"{MDL0:.3f}", "—", "—"],
        ["Basis+Int", k,  f"{AICc1:.3f}", f"{BIC1:.3f}", f"{MDL1:.3f}", f"{dBIC_bits:.2f}", f"{dMDL_bits:.2f}"],
    ]
    _print_table(rows, ["Model","k","AICc_GLS","BIC_GLS","MDL_GLS","ΔBIC (bits)","ΔMDL (bits)"])
    # Operator amplitudes
    print("\n=== Operator amplitudes (GLS ridge) ===")
    amps = [(names[j], float(w[j])) for j in range(K)]
    amps = sorted(amps, key=lambda t: -abs(t[1]))
    _print_table([[nm, f"{val:+.6f}"] for nm,val in amps], ["operator","amplitude"])
    # Gauge/Σ diagnostics
    print("\n=== Gauge / Σ Diagnostics ===")
    diag = [
        ["SPD(Σ) via Cholesky", "OK" if L is not None else "FAIL"],
        ["Relative Fro error ||LLᵀ-Σ||/||Σ||", f"{fro_rel:.3e}"],
        ["cond(Σ)", f"{condS:.3e}"],
        ["std( L^{-1} r )", f"{rw_std:.3f} (≈1 if model & Σ calibrated)"]
    ]
    _print_table(diag, ["check","value"])
    return {"w": w, "b": b, "dBIC_bits": dBIC_bits, "dMDL_bits": dMDL_bits}

# =========================
# Main runner
# =========================
def run_main(config):
    # If OG-SET paths are present, run GLS basis evaluation first
    og = config.get("OGSET", {})
    if og and og.get("Yb_csv") and og.get("y_csv") and og.get("Sigma_json"):
        try:
            Yb = np.loadtxt(og["Yb_csv"], delimiter=",", dtype=float)
            y  = np.loadtxt(og["y_csv"],  delimiter=",", dtype=float).reshape(-1)
            with open(og["Sigma_json"], "r") as f:
                raw = json.load(f)
            Sigma = np.array(raw["Sigma"] if isinstance(raw, dict) and "Sigma" in raw else raw, dtype=float)
            if Yb.shape[0] != y.shape[0] or Sigma.shape[0] != y.shape[0] or Sigma.shape[1] != y.shape[0]:
                raise ValueError(f"Shape mismatch: Yb={Yb.shape}, y={y.shape}, Sigma={Sigma.shape}")
            print("\n==================== OG-SET / GLS evaluation ====================")
            _ = run_ogset_gls(Yb, y, Sigma, alpha=config["ALPHA_RIDGE"])
        except Exception as e:
            print(f"[OG-SET] Skipped due to error: {e}")

    print("\n==================== FASE v20 — HSML+L4+Mini-Ruliad (GLS-ready) ====================")
    seeds = config["SEEDS"]; datasets = config["DATASETS"]
    if config.get("DO_CV_STABILITY", False):
        for ds in datasets: _ = run_cv_stability_dataset(ds, seeds, config["K_FOLDS"])
    else:
        for ds in datasets:
            print(f"\n==================== Processing Dataset: {ds} ====================")
            try:
                X, y = load_dataset(ds)
                n = X.shape[0]; va = max(1, int(0.2*n))
                idx = np.arange(n); np.random.default_rng(1337).shuffle(idx)
                Xtr, ytr = X[idx[va:]], y[idx[va:]]
                Xva, yva = X[idx[:va]], y[idx[:va]]
                # No Σ for PMLB (pass None) — pipeline remains GLS-ready
                out = run_fase_given_split(Xtr, ytr, Xva, yva, seed=1337, name=ds, Sigma_tr=None, Sigma_va=None)
                if CONFIG.get("COMPARE_WITH_PYSR", False):
                    if ds in regression_dataset_names:
                        ps = run_pysr_baseline(Xtr, ytr, Xva, yva, seed=1337, pysr_cfg=CONFIG.get("PYSR", {}))
                        print(f"[PySR]  Validation R² = {ps['R2']:.4f}")
                        if ps["expr"] is not None:
                            print(f"[PySR]  Best expression: {ps['expr']}")
                    else:
                        print("[PySR]  Skipped (classification dataset).")
            except Exception as e:
                print(f"Could not process dataset {ds}. Error: {e}")

# =========================
# Go
# =========================
if __name__ == '__main__':
    run_main(CONFIG)
