#!/usr/bin/env python3
"""
Judgment-Universalizer (Gate Closure Edition)

What’s new vs the refactor:
- Strong selectivity control: squared coverage penalty around a target band (default center=0.25, halfwidth=0.05)
- Softer sparsity (defaults) to avoid always-ignite
- Simplified calibrator (temperature + affine on v) with optional near-identity constraint if "full" matrix is selected
- Diversity & informativeness regularizers:
  * Argmax-histogram entropy (encourage diversity of chosen α-modes across the batch)
  * α-entropy logging (global & |ignite) to prove non-trivial concentration
  * v-variance target (global & |ignite)
  * Optional v–E correlation (weak) to ensure v tracks invariant energy a bit
- Symmetric metrics: we report A->B and B->A conditional δ_v and D_α and F1, plus the averaged versions
- Cleaner CSV logging for reviewer-readiness

Usage examples:
  # π decimal (unlearnable), digits file needed
  python judgment_universalizer_cli_gate.py --dataset pi --pi-file /path/to/pi_digits.txt --epochs 15 --L 128

  # Pseudo-π (positive control)
  python judgment_universalizer_cli_gate.py --dataset pseudo --epochs 12 --L 128

  # BBP hex-π
  python judgment_universalizer_cli_gate.py --dataset bbp --epochs 12 --L 128
"""
import argparse, os, csv, math, time, sys, random
from dataclasses import dataclass
from typing import Tuple, Dict, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    import pywt
    HAVE_PYWT = True
except Exception:
    HAVE_PYWT = False

def set_seed(seed: int = 1337):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)

def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)

def mad_numpy(x: np.ndarray) -> float:
    med = np.median(x); return float(np.median(np.abs(x - med)) + 1e-12)

def schedule_linear(ep: int, ep_total: int, start: float, end: float) -> float:
    if ep_total <= 1: return end
    t = min(max(ep, 0), ep_total); return start + (end - start) * (t / float(ep_total))

def entropy_rows(p: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    return -(p.clamp_min(eps).log() * p.clamp_min(eps)).sum(dim=1)

def hist_entropy_argmax(p: torch.Tensor, K: int) -> float:
    with torch.no_grad():
        counts = torch.bincount(torch.argmax(p, dim=1), minlength=K).float()
        freq = counts / counts.sum().clamp_min(1.0)
        H = -(freq.clamp_min(1e-12) * freq.clamp_min(1e-12).log()).sum().item()
        return H

def _tofloat(x):
    return float(x.item()) if torch.is_tensor(x) else float(x)

def load_pi_digits_from_file(path: str, N: Optional[int] = None) -> np.ndarray:
    with open(path, "r") as f: s = f.read()
    digs = [int(ch) for ch in s if ch.isdigit()]
    if N is not None: digs = digs[:N]
    return np.array(digs, dtype=np.int64)

def champernowne_digits(N: int) -> np.ndarray:
    buf = []; k = 1
    while len(buf) < N:
        buf.extend(list(str(k))); k += 1
    digs = [int(ch) for ch in buf[:N]]
    return np.array(digs, dtype=np.int64)

def _series(j: int, n: int) -> float:
    s = 0.0
    for k in range(n + 1):
        s = (s + pow(16, n - k, 16 * (8 * k + j)) / (8 * k + j)) % 1.0
    return s

def _tail(j: int, n: int) -> float:
    t = 0.0; k = n + 1; p = 1.0 / pow(16.0, k)
    for _ in range(1000):
        t += p / (8.0 * k + j); p /= 16.0; k += 1
    return t

def bbp_hex_pi_digits(N: int, start: int = 0) -> np.ndarray:
    digs = []
    for n in range(start, start + N):
        x = (4.0 * (_series(1, n) + _tail(1, n))
             - 2.0 * (_series(4, n) + _tail(4, n))
             - 1.0 * (_series(5, n) + _tail(5, n))
             - 1.0 * (_series(6, n) + _tail(6, n)))
        x = x % 1.0; d = int(x * 16.0); digs.append(d)
    return np.array(digs, dtype=np.int64)

def make_windows(digs: np.ndarray, L: int, stride: int = 1) -> Tuple[np.ndarray, np.ndarray]:
    X, y = [], []
    for i in range(0, len(digs) - L - 1, stride):
        X.append(digs[i:i + L]); y.append(digs[i + L])
    X = np.stack(X).astype(np.float32); y = np.array(y, dtype=np.int64)
    vmax = int(max(9, int(digs.max()))); X = X / float(max(1, vmax))
    return X, y

class WorldA(nn.Module):
    def __init__(self, L: int, K: int = 10, hidden: int = 128):
        super().__init__()
        self.feat_dim = L; self.K = K
        self.net = nn.Sequential(nn.Linear(L, hidden), nn.GELU(), nn.Linear(hidden, K))
    def phi(self, x): return x
    def fit(self, x, y):
        logits = self.net(x); logp = F.log_softmax(logits, dim=-1)
        nll = F.nll_loss(logp, y, reduction='none').unsqueeze(1)
        W = torch.ones_like(nll); dF = -nll; curv = torch.zeros_like(nll)
        eps = F.one_hot(y, num_classes=self.K).float() - torch.exp(logp)
        return logp, eps, {"W": W, "dF": dF, "curv": curv, "E": (eps**2).sum(dim=1, keepdim=True)}

class WorldB(nn.Module):
    def __init__(self, L: int, K: int = 10, hidden: int = 128, wave: str = "db4"):
        super().__init__()
        self.L=L; self.K=K; self.wave=wave; self.feat_dim=L
        self.net = nn.Sequential(nn.Linear(self.feat_dim, hidden), nn.GELU(), nn.Linear(hidden, K))
    def _dwt_row(self, row: np.ndarray) -> np.ndarray:
        if HAVE_PYWT:
            coeffs = pywt.wavedec(row, self.wave, mode='periodization')
            vec = np.concatenate([c.ravel() for c in coeffs], axis=0)
            if len(vec) < self.L: vec = np.pad(vec, (0, self.L - len(vec)))
            else: vec = vec[:self.L]
            return vec.astype(np.float32)
        else:
            return row.astype(np.float32)
    def phi(self, x):
        xp = x.detach().cpu().numpy()
        xb = np.stack([self._dwt_row(row) for row in xp], axis=0)
        if not HAVE_PYWT: print("[WARN] pywt not found; WorldB uses identity features.", file=sys.stderr)
        return torch.from_numpy(xb).to(x.device)
    def fit(self, xB, y):
        logits = self.net(xB); logp = F.log_softmax(logits, dim=-1)
        nll = F.nll_loss(logp, y, reduction='none').unsqueeze(1)
        W = torch.ones_like(nll); dF = -nll; curv = torch.zeros_like(nll)
        eps = F.one_hot(y, num_classes=self.K).float() - torch.exp(logp)
        return logp, eps, {"W": W, "dF": dF, "curv": curv, "E": (eps**2).sum(dim=1, keepdim=True)}

def gini(p: torch.Tensor) -> torch.Tensor:
    b, K = p.shape; s, _ = torch.sort(p, dim=1)
    i = torch.arange(1, K + 1, device=p.device).float().unsqueeze(0)
    return ((2 * i - K - 1) * s).sum(dim=1) / (K - 1)

def pack_invariants(eps: torch.Tensor, W: torch.Tensor, dF: torch.Tensor, curv: torch.Tensor) -> torch.Tensor:
    E = W * (eps ** 2).sum(dim=1, keepdim=True); return torch.cat([E, dF, curv], dim=1)

class Translator(nn.Module):
    def __init__(self, d_in: int, d_out: int):
        super().__init__(); self.lin = nn.Linear(d_in, d_out, bias=True)
    def forward(self, x): return self.lin(x)

class CalibratorSimple(nn.Module):
    def __init__(self, K: int):
        super().__init__()
        self.scale = nn.Parameter(torch.ones(1)); self.shift = nn.Parameter(torch.zeros(1))
        self.log_tau = nn.Parameter(torch.zeros(1))
    def forward(self, v: torch.Tensor, a: torch.Tensor):
        v2 = self.scale * v + self.shift
        tau = torch.exp(self.log_tau).clamp_min(1e-3)
        a2 = F.softmax((a + 1e-8).log() / tau, dim=-1)
        return v2, a2

class CalibratorFull(nn.Module):
    def __init__(self, K: int, lam_identity: float = 1e-2):
        super().__init__()
        self.scale = nn.Parameter(torch.ones(1)); self.shift = nn.Parameter(torch.zeros(1))
        self.log_tau = nn.Parameter(torch.zeros(1)); self.M = nn.Parameter(torch.eye(K))
        self.lam_identity = lam_identity
    def forward(self, v, a):
        v2 = self.scale * v + self.shift
        tau = torch.exp(self.log_tau).clamp_min(1e-3)
        logits = (a + 1e-8).log() @ self.M
        a2 = F.softmax(logits / tau, dim=-1); return v2, a2
    def identity_reg(self):
        return self.lam_identity * ((self.M - torch.eye(self.M.shape[0], device=self.M.device))**2).mean()

class JudgmentHead(nn.Module):
    def __init__(self, d_feat: int, K: int = 8, hidden: int = 64):
        super().__init__(); self.K=K
        self.v = nn.Sequential(nn.Linear(d_feat, hidden), nn.GELU(), nn.Linear(hidden, 1))
        self.a = nn.Sequential(nn.Linear(d_feat, hidden), nn.GELU(), nn.Linear(hidden, K))
    def forward(self, feat: torch.Tensor):
        v = self.v(feat).squeeze(-1); a = F.softmax(self.a(feat), dim=-1); return v, a

def ignition_mask(alpha: torch.Tensor, thr_gini: float = 0.8) -> torch.Tensor:
    return (gini(alpha) >= thr_gini).float()

def ignition_f1(m1: torch.Tensor, m2: torch.Tensor) -> float:
    tp = ((m1 == 1) & (m2 == 1)).sum().item()
    fp = ((m1 == 0) & (m2 == 1)).sum().item()
    fn = ((m1 == 1) & (m2 == 0)).sum().item()
    prec = tp / (tp + fp + 1e-9); rec  = tp / (tp + fn + 1e-9)
    if (prec + rec) == 0: return 0.0
    return 2 * prec * rec / (prec + rec)

@dataclass
class Lambdas:
    nat: float; gauge: float; cyc: float; hinge: float; sparsity: float; coverage: float
    diversity: float; vvar: float; mi: float; tau_reg: float

@dataclass
class CoverageBand:
    center: float; halfwidth: float

def compute_losses_and_metrics(
    xb, yb, worldA, worldB, judgA, judgB, tauAB, tauBA, calAB, calBA,
    lambdas: Lambdas, cov: CoverageBand, ignite_thr: float, logK: float, hinge_eps: float,
    calibrator_type: str = "simple"
):
    xA = worldA.phi(xb); logpA, epsA, invA = worldA.fit(xA, yb)
    xiA = pack_invariants(epsA, invA["W"], invA["dF"], invA["curv"]); vA, aA = judgA(xiA)

    xB = worldB.phi(xb); logpB, epsB, invB = worldB.fit(xB, yb)
    xiB = pack_invariants(epsB, invB["W"], invB["dF"], invB["curv"]); vB, aB = judgB(xiB)

    xA2B = tauAB(xA); xB2A = tauBA(xB)
    logpAB, epsAB, invAB = worldB.fit(xA2B, yb); xiAB = pack_invariants(epsAB, invAB["W"], invAB["dF"], invAB["curv"]); vAB, aAB = judgB(xiAB)
    logpBA, epsBA, invBA = worldA.fit(xB2A, yb); xiBA = pack_invariants(epsBA, invBA["W"], invBA["dF"], invBA["curv"]); vBA, aBA = judgA(xiBA)

    vAcal, aAcal = calAB(vA, aA); vBcal, aBcal = calBA(vB, aB)

    nllA = F.nll_loss(logpA, yb); nllB = F.nll_loss(logpB, yb); L_task = nllA + nllB
    hingeA = F.relu(torch.abs(nllA - logK) - hinge_eps); hingeB = F.relu(torch.abs(nllB - logK) - hinge_eps)
    L_hinge = lambdas.hinge * (hingeA + hingeB)

    L_nat_A2B = F.mse_loss(vAcal, vAB) + F.kl_div((aAcal + 1e-8).log(), aAB, reduction="batchmean")
    L_nat_B2A = F.mse_loss(vBcal, vBA) + F.kl_div((aBcal + 1e-8).log(), aBA, reduction="batchmean")
    L_nat = L_nat_A2B + L_nat_B2A

    sA = torch.empty_like(epsA).uniform_(0.8, 1.25); sB = torch.empty_like(epsB).uniform_(0.8, 1.25)
    xiA_g = pack_invariants(epsA * sA, invA["W"] / (sA.pow(2).mean(dim=1, keepdim=True)), invA["dF"], invA["curv"])
    xiB_g = pack_invariants(epsB * sB, invB["W"] / (sB.pow(2).mean(dim=1, keepdim=True)), invB["dF"], invB["curv"])
    vAg, aAg = judgA(xiA_g); vBg, aBg = judgB(xiB_g)
    L_gauge = F.mse_loss(vAg, vA) + F.kl_div((aAg + 1e-8).log(), aA, reduction="batchmean") \
            + F.mse_loss(vBg, vB) + F.kl_div((aBg + 1e-8).log(), aB, reduction="batchmean")

    xABA = tauBA(xA2B); xBAB = tauAB(xB2A); L_cyc = F.mse_loss(xABA, xA) + F.mse_loss(xBAB, xB)

    ginA = gini(aA); ginAB = gini(aAB); ginB = gini(aB); ginBA = gini(aBA)
    igniteA  = (ginA  >= ignite_thr).float(); igniteAB = (ginAB >= ignite_thr).float()
    igniteB  = (ginB  >= ignite_thr).float(); igniteBA = (ginBA >= ignite_thr).float()

    p_ignite_A = 0.5 * (igniteA.mean() + igniteAB.mean())
    p_ignite_B = 0.5 * (igniteB.mean() + igniteBA.mean())
    p_ignite   = 0.5 * (p_ignite_A + p_ignite_B)

    L_sparsity = -lambdas.sparsity * (ginA.mean() + ginAB.mean() + ginB.mean() + ginBA.mean())
    def cov_penalty(p):
        d = torch.abs(p - cov.center) - cov.halfwidth; d = torch.relu(d); return d * d
    L_coverage = lambdas.coverage * (cov_penalty(p_ignite_A) + cov_penalty(p_ignite_B))

    H_arg_A = hist_entropy_argmax(aA, judgA.K);  H_arg_AB = hist_entropy_argmax(aAB, judgB.K)
    H_arg_B = hist_entropy_argmax(aB, judgB.K);  H_arg_BA = hist_entropy_argmax(aBA, judgA.K)
    KJ = judgA.K; Hmax = math.log(KJ)
    L_diversity = lambdas.diversity * ((Hmax - H_arg_A) + (Hmax - H_arg_AB) + (Hmax - H_arg_B) + (Hmax - H_arg_BA))

    def var1(x): return x.var(unbiased=False)
    target_var = 0.25
    Var_v = 0.25 * (var1(vA) + var1(vAB) + var1(vB) + var1(vBA))
    L_vvar = lambdas.vvar * (Var_v - target_var)**2

    def corr2(x, y):
        x = x - x.mean(); y = y - y.mean()
        vx = (x * x).mean() + 1e-12; vy = (y * y).mean() + 1e-12
        return ((x * y).mean() ** 2) / (vx * vy)
    E_A = invA["E"].squeeze(1); E_B = invB["E"].squeeze(1)
    L_mi = -lambdas.mi * (corr2(vA, E_A) + corr2(vB, E_B))

    tau_reg = 0.0
    if hasattr(calAB, "log_tau"): tau_reg += (calAB.log_tau ** 2).mean().item()
    if hasattr(calBA, "log_tau"): tau_reg += (calBA.log_tau ** 2).mean().item()
    L_tau = lambdas.tau_reg * tau_reg

    L_cal_id = 0.0
    if calibrator_type == "full":
        L_cal_id = getattr(calAB, "identity_reg", lambda: 0.0)() + getattr(calBA, "identity_reg", lambda: 0.0)()

    L = L_task + lambdas.nat * L_nat + lambdas.gauge * L_gauge + lambdas.cyc * L_cyc \
        + L_hinge + L_sparsity + L_coverage + L_diversity + L_vvar + L_mi + L_tau + L_cal_id

    ign_f1_A2B = float(ignition_f1(igniteA.cpu(), igniteAB.cpu()))
    ign_f1_B2A = float(ignition_f1(igniteB.cpu(), igniteBA.cpu()))
    ign_f1_avg = 0.5 * (ign_f1_A2B + ign_f1_B2A)

    dv_A2B = (vAcal - vAB).detach().cpu().numpy(); dvmad_A2B = mad_numpy(dv_A2B)
    Dalpha_A2B = float(F.kl_div((aAcal + 1e-8).log(), aAB, reduction="batchmean").item())
    dv_B2A = (vBcal - vBA).detach().cpu().numpy(); dvmad_B2A = mad_numpy(dv_B2A)
    Dalpha_B2A = float(F.kl_div((aBcal + 1e-8).log(), aBA, reduction="batchmean").item())

    with torch.no_grad():
        union_A2B = ((igniteA + igniteAB) > 0).cpu().numpy().astype(bool)
        if union_A2B.any():
            dvmad_A2B_c = mad_numpy(dv_A2B[union_A2B])
            Dalpha_A2B_c = float(F.kl_div((aAcal[union_A2B] + 1e-8).log(), aAB[union_A2B], reduction="batchmean").item())
        else:
            dvmad_A2B_c, Dalpha_A2B_c = float('nan'), float('nan')
        union_B2A = ((igniteB + igniteBA) > 0).cpu().numpy().astype(bool)
        if union_B2A.any():
            dvmad_B2A_c = mad_numpy(dv_B2A[union_B2A])
            Dalpha_B2A_c = float(F.kl_div((aBcal[union_B2A] + 1e-8).log(), aBA[union_B2A], reduction="batchmean").item())
        else:
            dvmad_B2A_c, Dalpha_B2A_c = float('nan'), float('nan')

    Ha = entropy_rows(aA).mean().item();    Ha_AB = entropy_rows(aAB).mean().item()
    Hb = entropy_rows(aB).mean().item();    Hb_BA = entropy_rows(aBA).mean().item()
    def mean_on(mask, t):
        if mask.sum() == 0: return float('nan')
        return t[mask.bool()].mean().item()
    Ha_c  = mean_on((igniteA>0), entropy_rows(aA));   HaAB_c= mean_on((igniteAB>0), entropy_rows(aAB))
    Hb_c  = mean_on((igniteB>0), entropy_rows(aB));   HbBA_c= mean_on((igniteBA>0), entropy_rows(aBA))

    var_v = 0.25 * (Var_v).item()
    var_v_c = 0.0
    for vv, mm in [(vA, igniteA), (vAB, igniteAB), (vB, igniteB), (vBA, igniteBA)]:
        if mm.sum() > 0: var_v_c += (vv[mm>0].var(unbiased=False)).item()
    var_v_c *= 0.25

    metrics = dict(
        L=_tofloat(L), L_task=_tofloat(L_task), L_nat=_tofloat(L_nat),
        L_gauge=_tofloat(L_gauge), L_cyc=_tofloat(L_cyc),
        L_hinge=_tofloat(L_hinge), L_sparsity=_tofloat(L_sparsity),
        L_coverage=_tofloat(L_coverage), L_diversity=float(L_diversity),
        L_vvar=_tofloat(L_vvar), L_mi=_tofloat(L_mi), L_tau=float(L_tau),
        nllA=_tofloat(nllA), nllB=_tofloat(nllB),
        ign_f1_A2B=ign_f1_A2B, ign_f1_B2A=ign_f1_B2A, ign_f1_avg=ign_f1_avg,
        delta_v_mad_A2B=dvmad_A2B, D_alpha_A2B=Dalpha_A2B,
        delta_v_mad_B2A=dvmad_B2A, D_alpha_B2A=Dalpha_B2A,
        delta_v_mad_A2B_cond=dvmad_A2B_c, D_alpha_A2B_cond=Dalpha_A2B_c,
        delta_v_mad_B2A_cond=dvmad_B2A_c, D_alpha_B2A_cond=Dalpha_B2A_c,
        p_ignite=_tofloat(p_ignite), p_ignite_A=_tofloat(p_ignite_A), p_ignite_B=_tofloat(p_ignite_B),
        H_alpha_A=Ha, H_alpha_AB=Ha_AB, H_alpha_B=Hb, H_alpha_BA=Hb_BA,
        H_alpha_A_cond=Ha_c, H_alpha_AB_cond=HaAB_c, H_alpha_B_cond=Hb_c, H_alpha_BA_cond=HbBA_c,
        H_argmax_A=H_arg_A, H_argmax_AB=H_arg_AB, H_argmax_B=H_arg_B, H_argmax_BA=H_arg_BA,
        Var_v=var_v, Var_v_cond=var_v_c
    )
    return L, metrics

def run_epoch(dloader, worldA, worldB, judgA, judgB, tauAB, tauBA, calAB, calBA, optimizer,
              lambdas, cov, ignite_thr, logK, hinge_eps, device="cpu", train=True, calibrator_type="simple"):
    modules = [worldA, worldB, judgA, judgB, tauAB, tauBA, calAB, calBA]
    for m in modules: m.train() if train else m.eval()
    logs = []
    for xb, yb in dloader:
        xb = xb.to(device); yb = yb.to(device)
        L, m = compute_losses_and_metrics(
            xb, yb, worldA, worldB, judgA, judgB, tauAB, tauBA, calAB, calBA,
            lambdas, cov, ignite_thr, logK, hinge_eps, calibrator_type=calibrator_type
        )
        if train:
            optimizer.zero_grad(set_to_none=True); L.backward(); optimizer.step()
        logs.append(m)
    if not logs: return {}
    keys = list(logs[0].keys())
    agg = {k: float(np.nanmean([m[k] for m in logs])) for k in keys}
    return agg

def main():
    ap = argparse.ArgumentParser(description="Judgment-Universalizer Gate Closure (patched)")
    ap.add_argument("--dataset", choices=["pi", "pseudo", "bbp"], default="pseudo")
    ap.add_argument("--pi-file", type=str, default=None)
    ap.add_argument("--N", type=int, default=50000)
    ap.add_argument("--L", type=int, default=128)
    ap.add_argument("--stride", type=int, default=1)
    ap.add_argument("--epochs", type=int, default=12)
    ap.add_argument("--batch-size", type=int, default=256)
    ap.add_argument("--train-frac", type=float, default=0.8)
    ap.add_argument("--hidden", type=int, default=128)
    ap.add_argument("--K", type=int, default=None)
    ap.add_argument("--seed", type=int, default=1337)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--warmup-epochs", type=int, default=2)
    ap.add_argument("--hinge-eps", type=float, default=0.02)
    ap.add_argument("--lambda-hinge", type=float, default=0.1)
    ap.add_argument("--lambda-nat", type=float, default=1.0)
    ap.add_argument("--lambda-gauge", type=float, default=0.5)
    ap.add_argument("--lambda-cyc", type=float, default=1.0)
    ap.add_argument("--lambda-sparsity", type=float, default=0.05)
    ap.add_argument("--lambda-coverage", type=float, default=3.0)
    ap.add_argument("--cov-center", type=float, default=0.25)
    ap.add_argument("--cov-halfwidth", type=float, default=0.05)
    ap.add_argument("--lambda-diversity", type=float, default=0.05)
    ap.add_argument("--lambda-vvar", type=float, default=0.01)
    ap.add_argument("--lambda-mi", type=float, default=1e-3)
    ap.add_argument("--calibrator", choices=["simple", "full"], default="simple")
    ap.add_argument("--lambda-tau", type=float, default=1e-3)
    ap.add_argument("--lambda-cal-id", type=float, default=1e-2)
    ap.add_argument("--ignite-thr-start", type=float, default=0.55)
    ap.add_argument("--ignite-thr-end", type=float, default=0.85)
    ap.add_argument("--ignite-thr-epochs", type=int, default=20)
    ap.add_argument("--logdir", type=str, default="./ju_logs_gate")
    ap.add_argument("--no-plots", action="store_true")
    args = ap.parse_args()

    random.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed); torch.cuda.manual_seed_all(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"; os.makedirs(args.logdir, exist_ok=True)

    if args.dataset == "pi":
        if not args.pi_file or not os.path.exists(args.pi_file):
            print("[ERROR] --pi-file required for dataset=pi", file=sys.stderr); sys.exit(2)
        digs = load_pi_digits_from_file(args.pi_file, N=args.N); K = args.K or 10
    elif args.dataset == "pseudo":
        digs = champernowne_digits(args.N); K = args.K or 10
    elif args.dataset == "bbp":
        digs = bbp_hex_pi_digits(args.N, start=0); K = args.K or 16

    X, y = make_windows(digs, L=args.L, stride=args.stride)
    n = len(X); n_train = int(n * args.train_frac)
    Xtr, ytr = X[:n_train], y[:n_train]; Xva, yva = X[n_train:], y[n_train:]
    tr_ds = torch.utils.data.TensorDataset(torch.from_numpy(Xtr), torch.from_numpy(ytr))
    va_ds = torch.utils.data.TensorDataset(torch.from_numpy(Xva), torch.from_numpy(yva))
    tr_loader = torch.utils.data.DataLoader(tr_ds, batch_size=args.batch_size, shuffle=True, drop_last=True)
    va_loader = torch.utils.data.DataLoader(va_ds, batch_size=args.batch_size, shuffle=False, drop_last=False)

    worldA = WorldA(L=args.L, K=K, hidden=args.hidden).to(device)
    worldB = WorldB(L=args.L, K=K, hidden=args.hidden, wave="db4").to(device)
    judgA  = JudgmentHead(d_feat=3, K=8, hidden=64).to(device)
    judgB  = JudgmentHead(d_feat=3, K=8, hidden=64).to(device)
    tauAB  = Translator(d_in=worldA.feat_dim, d_out=worldB.feat_dim).to(device)
    tauBA  = Translator(d_in=worldB.feat_dim, d_out=worldA.feat_dim).to(device)
    if args.calibrator == "simple":
        calAB = CalibratorSimple(K=8).to(device); calBA = CalibratorSimple(K=8).to(device)
    else:
        calAB = CalibratorFull(K=8, lam_identity=args.lambda_cal_id).to(device)
        calBA = CalibratorFull(K=8, lam_identity=args.lambda_cal_id).to(device)

    params = list(worldA.parameters()) + list(worldB.parameters()) + \
             list(judgA.parameters()) + list(judgB.parameters()) + \
             list(tauAB.parameters()) + list(tauBA.parameters()) + \
             list(calAB.parameters()) + list(calBA.parameters())
    opt = torch.optim.Adam(params, lr=args.lr)

    lambdas = Lambdas(args.lambda_nat, args.lambda_gauge, args.lambda_cyc,
                      args.lambda_hinge, args.lambda_sparsity, args.lambda_coverage,
                      args.lambda_diversity, args.lambda_vvar, args.lambda_mi, args.lambda_tau)
    cov = CoverageBand(center=args.cov_center, halfwidth=args.cov_halfwidth); logK = math.log(K)

    csv_path = os.path.join(args.logdir, f"log_{args.dataset}_L{args.L}_gate.csv")
    with open(csv_path, "w", newline="") as f:
        w = csv.writer(f); w.writerow([
          "epoch","split","L","L_task","L_nat","L_gauge","L_cyc","L_hinge","L_sparsity","L_coverage","L_diversity","L_vvar","L_mi","L_tau",
          "nllA","nllB","ign_f1_A2B","ign_f1_B2A","ign_f1_avg",
          "delta_v_mad_A2B","D_alpha_A2B","delta_v_mad_B2A","D_alpha_B2A",
          "delta_v_mad_A2B_cond","D_alpha_A2B_cond","delta_v_mad_B2A_cond","D_alpha_B2A_cond",
          "p_ignite","p_ignite_A","p_ignite_B",
          "H_alpha_A","H_alpha_AB","H_alpha_B","H_alpha_BA",
          "H_alpha_A_cond","H_alpha_AB_cond","H_alpha_B_cond","H_alpha_BA_cond",
          "H_argmax_A","H_argmax_AB","H_argmax_B","H_argmax_BA",
          "Var_v","Var_v_cond","ignite_thr"
        ])
        for ep in range(1, args.epochs + 1):
            thr = schedule_linear(ep-1, max(1, args.ignite_thr_epochs-1), args.ignite_thr_start, args.ignite_thr_end)
            if ep == args.warmup_epochs + 1:
                for p in worldA.net.parameters(): p.requires_grad = False
                for p in worldB.net.parameters(): p.requires_grad = False
                print(f"[INFO] Froze base predictors at epoch {ep-1}", file=sys.stderr)

            tr = run_epoch(tr_loader, worldA, worldB, judgA, judgB, tauAB, tauBA, calAB, calBA, opt,
                           lambdas, cov, ignite_thr=thr, logK=logK, hinge_eps=args.hinge_eps,
                           device=device, train=True, calibrator_type=args.calibrator)
            va = run_epoch(va_loader, worldA, worldB, judgA, judgB, tauAB, tauBA, calAB, calBA, optimizer=None,
                           lambdas=lambdas, cov=cov, ignite_thr=thr, logK=logK, hinge_eps=args.hinge_eps,
                           device=device, train=False, calibrator_type=args.calibrator)

            for split, m in [("train", tr), ("val", va)]:
                if not m: continue
                w.writerow([
                    ep, split, m["L"], m["L_task"], m["L_nat"], m["L_gauge"], m["L_cyc"], m["L_hinge"], m["L_sparsity"], m["L_coverage"], m["L_diversity"], m["L_vvar"], m["L_mi"], m["L_tau"],
                    m["nllA"], m["nllB"], m["ign_f1_A2B"], m["ign_f1_B2A"], m["ign_f1_avg"],
                    m["delta_v_mad_A2B"], m["D_alpha_A2B"], m["delta_v_mad_B2A"], m["D_alpha_B2A"],
                    m["delta_v_mad_A2B_cond"], m["D_alpha_A2B_cond"], m["delta_v_mad_B2A_cond"], m["D_alpha_B2A_cond"],
                    m["p_ignite"], m["p_ignite_A"], m["p_ignite_B"],
                    m["H_alpha_A"], m["H_alpha_AB"], m["H_alpha_B"], m["H_alpha_BA"],
                    m["H_alpha_A_cond"], m["H_alpha_AB_cond"], m["H_alpha_B_cond"], m["H_alpha_BA_cond"],
                    m["H_argmax_A"], m["H_argmax_AB"], m["H_argmax_B"], m["H_argmax_BA"],
                    m["Var_v"], m["Var_v_cond"], thr
                ]); f.flush()
            print(f"[epoch {ep}] val: NLLA={va.get('nllA', float('nan')):.4f} "
                  f"p_ignite={va.get('p_ignite', float('nan')):.3f} "
                  f"F1(avg)={va.get('ign_f1_avg', float('nan')):.3f} "
                  f"δv|A2B={va.get('delta_v_mad_A2B_cond', float('nan')):.3g} "
                  f"Dα|A2B={va.get('D_alpha_A2B_cond', float('nan')):.3g} "
                  f"δv|B2A={va.get('delta_v_mad_B2A_cond', float('nan')):.3g} "
                  f"Dα|B2A={va.get('D_alpha_B2A_cond', float('nan')):.3g}")
    print(f"[DONE] Logs at: {csv_path}")
    print("[CRITERIA] For π: NLL≈logK; p_ignite in band; ign_F1_avg≥0.8; δv|ignite≤0.1; Dα|ignite≤0.05; α-entropy low on ignitions; argmax entropy high; Var(v|ignite)>0.1.")

if __name__ == "__main__":
    main()