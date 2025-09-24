#!/usr/bin/env python3
import argparse, os, csv, math, time, sys, random
from dataclasses import dataclass
from typing import Tuple, Dict, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

# Optional wavelet dependency
try:
    import pywt
    HAVE_PYWT = True
except Exception:
    HAVE_PYWT = False

# ----------------------------- Utilities -----------------------------

def set_seed(seed: int = 1337):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)

def mad_numpy(x: np.ndarray) -> float:
    med = np.median(x)
    return float(np.median(np.abs(x - med)) + 1e-12)

def schedule_linear(ep: int, ep_total: int, start: float, end: float) -> float:
    if ep_total <= 1: return end
    t = min(max(ep, 0), ep_total)
    return start + (end - start) * (t / float(ep_total))

# ----------------------------- Datasets -----------------------------

def load_pi_digits_from_file(path: str, N: Optional[int] = None) -> np.ndarray:
    with open(path, "r") as f:
        s = f.read()
    digs = [int(ch) for ch in s if ch.isdigit()]
    if N is not None:
        digs = digs[:N]
    if len(digs) < 1000:
        print(f"[WARN] π file had only {len(digs)} digits.", file=sys.stderr)
    return np.array(digs, dtype=np.int64)

def champernowne_digits(N: int) -> np.ndarray:
    buf = []
    k = 1
    while len(buf) < N:
        buf.extend(list(str(k)))
        k += 1
    digs = [int(ch) for ch in buf[:N]]
    return np.array(digs, dtype=np.int64)

# BBP hex π support (returns hex digits 0..15)
def _series(j: int, n: int) -> float:
    s = 0.0
    for k in range(n + 1):
        s = (s + pow(16, n - k, 16 * (8 * k + j)) / (8 * k + j)) % 1.0
    return s

def _tail(j: int, n: int) -> float:
    t = 0.0
    k = n + 1
    p = 1.0 / pow(16.0, k)
    # Truncate infinite tail to manageable steps (good enough for our use)
    for _ in range(1000):
        t += p / (8.0 * k + j)
        p /= 16.0
        k += 1
    return t

def bbp_hex_pi_digits(N: int, start: int = 0) -> np.ndarray:
    digs = []
    for n in range(start, start + N):
        x = (4.0 * (_series(1, n) + _tail(1, n))
             - 2.0 * (_series(4, n) + _tail(4, n))
             - 1.0 * (_series(5, n) + _tail(5, n))
             - 1.0 * (_series(6, n) + _tail(6, n)))
        x = x % 1.0
        d = int(x * 16.0)
        digs.append(d)
    return np.array(digs, dtype=np.int64)

def make_windows(digs: np.ndarray, L: int, stride: int = 1) -> Tuple[np.ndarray, np.ndarray]:
    X, y = [], []
    for i in range(0, len(digs) - L - 1, stride):
        X.append(digs[i:i + L])
        y.append(digs[i + L])
    X = np.stack(X).astype(np.float32)
    y = np.array(y, dtype=np.int64)
    vmax = int(max(9, int(digs.max())))
    X = X / float(max(1, vmax))
    return X, y

# ----------------------------- Worlds & Models -----------------------------

class WorldA(nn.Module):
    def __init__(self, L: int, K: int = 10, hidden: int = 128):
        super().__init__()
        self.feat_dim = L
        self.net = nn.Sequential(nn.Linear(L, hidden), nn.GELU(),
                                 nn.Linear(hidden, K))
        self.K = K

    def phi(self, x):  # identity
        return x

    def fit(self, x, y):
        logits = self.net(x)
        logp = F.log_softmax(logits, dim=-1)
        nll = F.nll_loss(logp, y, reduction='none').unsqueeze(1)
        W = torch.ones_like(nll)  # scalar precision per sample (extendable)
        dF = -nll                 # improvement proxy
        curv = torch.zeros_like(nll)  # placeholder
        eps = F.one_hot(y, num_classes=self.K).float() - torch.exp(logp)
        return logp, eps, {"W": W, "dF": dF, "curv": curv}

class WorldB(nn.Module):
    def __init__(self, L: int, K: int = 10, hidden: int = 128, wave: str = "db4"):
        super().__init__()
        self.L = L
        self.K = K
        self.wave = wave
        self.feat_dim = L  # pack to fixed length
        self.net = nn.Sequential(nn.Linear(self.feat_dim, hidden), nn.GELU(),
                                 nn.Linear(hidden, K))

    def _dwt_row(self, row: np.ndarray) -> np.ndarray:
        if HAVE_PYWT:
            coeffs = pywt.wavedec(row, self.wave, mode='periodization')
            vec = np.concatenate([c.ravel() for c in coeffs], axis=0)
            if len(vec) < self.L:
                vec = np.pad(vec, (0, self.L - len(vec)))
            else:
                vec = vec[:self.L]
            return vec.astype(np.float32)
        else:
            return row.astype(np.float32)

    def phi(self, x):
        xp = x.detach().cpu().numpy()
        xb = np.stack([self._dwt_row(row) for row in xp], axis=0)
        if not HAVE_PYWT:
            print("[WARN] pywt not found; WorldB uses identity features.", file=sys.stderr)
        return torch.from_numpy(xb).to(x.device)

    def fit(self, xB, y):
        logits = self.net(xB)
        logp = F.log_softmax(logits, dim=-1)
        nll = F.nll_loss(logp, y, reduction='none').unsqueeze(1)
        W = torch.ones_like(nll)
        dF = -nll
        curv = torch.zeros_like(nll)
        eps = F.one_hot(y, num_classes=self.K).float() - torch.exp(logp)
        return logp, eps, {"W": W, "dF": dF, "curv": curv}

def gini(p: torch.Tensor) -> torch.Tensor:
    b, K = p.shape
    s, _ = torch.sort(p, dim=1)
    i = torch.arange(1, K + 1, device=p.device).float().unsqueeze(0)
    return ((2 * i - K - 1) * s).sum(dim=1) / (K - 1)

def pack_invariants(eps: torch.Tensor, W: torch.Tensor, dF: torch.Tensor, curv: torch.Tensor) -> torch.Tensor:
    E = W * (eps ** 2).sum(dim=1, keepdim=True)  # precision-weighted error norm
    return torch.cat([E, dF, curv], dim=1)  # (b,3)

class Translator(nn.Module):
    def __init__(self, d_in: int, d_out: int):
        super().__init__()
        self.lin = nn.Linear(d_in, d_out, bias=True)
    def forward(self, x): return self.lin(x)

class Calibrator(nn.Module):
    def __init__(self, K: int):
        super().__init__()
        self.scale = nn.Parameter(torch.ones(1))
        self.shift = nn.Parameter(torch.zeros(1))
        self.log_tau = nn.Parameter(torch.zeros(1))
        self.perm = nn.Parameter(torch.eye(K))
    def forward(self, v: torch.Tensor, alpha: torch.Tensor):
        v2 = self.scale * v + self.shift
        tau = torch.exp(self.log_tau).clamp_min(1e-3)
        logits = (alpha + 1e-8).log() @ self.perm
        a2 = F.softmax(logits / tau, dim=-1)
        return v2, a2

class JudgmentHead(nn.Module):
    def __init__(self, d_feat: int, K: int = 8, hidden: int = 64):
        super().__init__()
        self.v = nn.Sequential(nn.Linear(d_feat, hidden), nn.GELU(), nn.Linear(hidden, 1))
        self.a = nn.Sequential(nn.Linear(d_feat, hidden), nn.GELU(), nn.Linear(hidden, K))
        self.K = K
    def forward(self, feat: torch.Tensor):
        v = self.v(feat).squeeze(-1)
        a = F.softmax(self.a(feat), dim=-1)
        return v, a

# ----------------------------- Metrics -----------------------------

def ignition_mask(alpha: torch.Tensor, thr_gini: float = 0.8) -> torch.Tensor:
    return (gini(alpha) >= thr_gini).float()

def ignition_f1(m1: torch.Tensor, m2: torch.Tensor) -> float:
    tp = ((m1 == 1) & (m2 == 1)).sum().item()
    fp = ((m1 == 0) & (m2 == 1)).sum().item()
    fn = ((m1 == 1) & (m2 == 0)).sum().item()
    prec = tp / (tp + fp + 1e-9)
    rec  = tp / (tp + fn + 1e-9)
    if (prec + rec) == 0:
        return 0.0
    return 2 * prec * rec / (prec + rec)

# ----------------------------- Train / Eval -----------------------------

@dataclass
class Lambdas:
    nat: float
    gauge: float
    cyc: float
    hinge: float
    sparsity: float
    coverage: float

def compute_losses_and_metrics(
    xb, yb, worldA, worldB, judgA, judgB, tauAB, tauBA, calAB, calBA,
    lambdas: Lambdas, ignite_thr: float, logK: float, hinge_eps: float,
    device: str = "cpu"
):
    # Native judgments
    xA = worldA.phi(xb)
    logpA, epsA, invA = worldA.fit(xA, yb)
    xiA = pack_invariants(epsA, invA["W"], invA["dF"], invA["curv"])
    vA, aA = judgA(xiA)

    xB = worldB.phi(xb)
    logpB, epsB, invB = worldB.fit(xB, yb)
    xiB = pack_invariants(epsB, invB["W"], invB["dF"], invB["curv"])
    vB, aB = judgB(xiB)

    # Translate features A->B and B->A
    xA2B = tauAB(xA)
    xB2A = tauBA(xB)

    # Judgments after translation (re-evaluate base predictors at translated features)
    logpAB, epsAB, invAB = worldB.fit(xA2B, yb)
    xiAB = pack_invariants(epsAB, invAB["W"], invAB["dF"], invAB["curv"])
    vAB, aAB = judgB(xiAB)

    logpBA, epsBA, invBA = worldA.fit(xB2A, yb)
    xiBA = pack_invariants(epsBA, invBA["W"], invBA["dF"], invBA["curv"])
    vBA, aBA = judgA(xiBA)

    # Calibrate judgments for commutation
    vAcal, aAcal = calAB(vA, aA)
    vBcal, aBcal = calBA(vB, aB)

    # Base task
    L_task = F.nll_loss(logpA, yb) + F.nll_loss(logpB, yb)

    # Hinge around chance
    nllA = F.nll_loss(logpA, yb)
    nllB = F.nll_loss(logpB, yb)
    hingeA = F.relu(torch.abs(nllA - logK) - hinge_eps)
    hingeB = F.relu(torch.abs(nllB - logK) - hinge_eps)
    L_hinge = lambdas.hinge * (hingeA + hingeB)

    # Naturality: scalar mse + KL on attention
    L_nat = F.mse_loss(vAcal, vAB) + F.kl_div((aAcal + 1e-8).log(), aAB, reduction="batchmean") \
          + F.mse_loss(vBcal, vBA) + F.kl_div((aBcal + 1e-8).log(), aBA, reduction="batchmean")

    # Gauge: scale residual channels (should not change judgment)
    s = torch.empty_like(epsA).uniform_(0.8, 1.25)
    xiA_g = pack_invariants(epsA * s, invA["W"] / (s.pow(2).mean(dim=1, keepdim=True)), invA["dF"], invA["curv"])
    vAg, aAg = judgA(xiA_g)
    L_gauge = F.mse_loss(vAg, vA) + F.kl_div((aAg + 1e-8).log(), aA, reduction="batchmean")

    # Cycle on features
    xABA = tauBA(xA2B); xBAB = tauAB(xB2A)
    L_cyc = F.mse_loss(xABA, xA) + F.mse_loss(xBAB, xB)

    # Sparsity (encourage ignition) and coverage (keep a band of rates)
    ginA = gini(aA); ginAB = gini(aAB)
    igniteA = (ginA >= ignite_thr).float()
    igniteAB = (ginAB >= ignite_thr).float()
    p_ignite = 0.5 * (igniteA.mean() + igniteAB.mean())

    L_sparsity = -lambdas.sparsity * (ginA.mean() + ginAB.mean())
    # Coverage band [0.15, 0.35] as default; could be args
    cov_lo, cov_hi = 0.15, 0.35
    L_coverage = lambdas.coverage * (F.relu(p_ignite - cov_hi) + F.relu(cov_lo - p_ignite))

    # Total loss
    L = L_task + lambdas.nat * L_nat + lambdas.gauge * L_gauge + lambdas.cyc * L_cyc + L_hinge + L_sparsity + L_coverage

    # Metrics (global)
    ign_f1 = float(ignition_f1(igniteA.cpu(), igniteAB.cpu()))
    # δ_v (MAD) global
    dv_raw = (vAcal - vAB).detach().cpu().numpy()
    delta_v_mad = mad_numpy(dv_raw)
    # D_alpha global
    D_alpha = float(F.kl_div((aAcal + 1e-8).log(), aAB, reduction="batchmean").item())

    # Conditional metrics (on union of ignitions)
    with torch.no_grad():
        union_mask = ((igniteA + igniteAB) > 0).cpu().numpy().astype(bool)
        if union_mask.any():
            dv_c = (vAcal - vAB).detach().cpu().numpy()[union_mask]
            delta_v_mad_c = mad_numpy(dv_c)
            aAcal_c = aAcal.detach()[union_mask]
            aAB_c   = aAB.detach()[union_mask]
            D_alpha_c = float(F.kl_div((aAcal_c + 1e-8).log(), aAB_c, reduction="batchmean").item())
        else:
            delta_v_mad_c = float('nan')
            D_alpha_c = float('nan')

    metrics = dict(
        L=float(L.item()), L_task=float(L_task.item()), L_nat=float(L_nat.item()),
        L_gauge=float(L_gauge.item()), L_cyc=float(L_cyc.item()),
        L_hinge=float(L_hinge.item()), L_sparsity=float(L_sparsity.item()), L_coverage=float(L_coverage.item()),
        nllA=float(nllA.item()), nllB=float(nllB.item()), ign_f1=ign_f1,
        delta_v_mad=delta_v_mad, D_alpha=D_alpha,
        delta_v_mad_cond=delta_v_mad_c, D_alpha_cond=D_alpha_c,
        p_ignite=float(p_ignite.item())
    )
    return L, metrics

def run_epoch(dloader, worldA, worldB, judgA, judgB, tauAB, tauBA, calAB, calBA, optimizer,
              lambdas: Lambdas, ignite_thr: float, logK: float, hinge_eps: float,
              device: str = "cpu", train: bool = True):
    if train:
        worldA.train(); worldB.train(); judgA.train(); judgB.train(); tauAB.train(); tauBA.train(); calAB.train(); calBA.train()
    else:
        worldA.eval(); worldB.eval(); judgA.eval(); judgB.eval(); tauAB.eval(); tauBA.eval(); calAB.eval(); calBA.eval()

    logs = []
    for xb, yb in dloader:
        xb = xb.to(device); yb = yb.to(device)

        L, m = compute_losses_and_metrics(
            xb, yb, worldA, worldB, judgA, judgB, tauAB, tauBA, calAB, calBA,
            lambdas, ignite_thr, logK, hinge_eps, device=device
        )
        if train:
            optimizer.zero_grad(set_to_none=True)
            L.backward()
            optimizer.step()
        logs.append(m)

    if not logs:
        return {}

    # Average metrics
    keys = list(logs[0].keys())
    agg = {k: float(np.nanmean([m[k] for m in logs])) for k in keys}
    return agg

# ----------------------------- Main CLI -----------------------------

def main():
    ap = argparse.ArgumentParser(description="Refactored Judgment-Universalizer with invariance on π / pseudo / BBP")
    ap.add_argument("--dataset", choices=["pi", "pseudo", "bbp"], default="pseudo")
    ap.add_argument("--pi-file", type=str, default=None, help="Path to decimal π digits file for --dataset pi")
    ap.add_argument("--N", type=int, default=50000, help="Total digits to load/generate")
    ap.add_argument("--L", type=int, default=128, help="Window length")
    ap.add_argument("--stride", type=int, default=1, help="Window stride")
    ap.add_argument("--epochs", type=int, default=10)
    ap.add_argument("--batch-size", type=int, default=256)
    ap.add_argument("--train-frac", type=float, default=0.8)
    ap.add_argument("--worldB", choices=["wavelet"], default="wavelet")
    ap.add_argument("--hidden", type=int, default=128)
    ap.add_argument("--K", type=int, default=None, help="Override classes; default 10 (decimal) or 16 (bbp)")
    ap.add_argument("--seed", type=int, default=1337)
    ap.add_argument("--lr", type=float, default=1e-3)

    # Phase & protection knobs
    ap.add_argument("--warmup-epochs", type=int, default=2, help="Epochs before freezing base predictors")
    ap.add_argument("--hinge-eps", type=float, default=0.02, help="Tolerance around log(K) for hinge")
    ap.add_argument("--lambda-hinge", type=float, default=0.1)

    # Invariance/gauge/cycle
    ap.add_argument("--lambda-nat", type=float, default=1.0)
    ap.add_argument("--lambda-gauge", type=float, default=0.5)
    ap.add_argument("--lambda-cyc", type=float, default=1.0)

    # Judgment shaping
    ap.add_argument("--lambda-sparsity", type=float, default=0.2, help="Encourage α concentration")
    ap.add_argument("--lambda-coverage", type=float, default=0.1, help="Keep ignition rate in a band")

    # Ignition threshold schedule
    ap.add_argument("--ignite-thr-start", type=float, default=0.6)
    ap.add_argument("--ignite-thr-end", type=float, default=0.8)
    ap.add_argument("--ignite-thr-epochs", type=int, default=10)

    ap.add_argument("--logdir", type=str, default="./ju_logs_refactor")
    ap.add_argument("--no-plots", action="store_true", help="Skip matplotlib plots")
    args = ap.parse_args()

    set_seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    ensure_dir(args.logdir)

    # Load digits
    if args.dataset == "pi":
        if not args.pi_file or not os.path.exists(args.pi_file):
            print("[ERROR] --pi-file path is required for dataset=pi", file=sys.stderr)
            sys.exit(2)
        digs = load_pi_digits_from_file(args.pi_file, N=args.N)
        K = args.K or 10
    elif args.dataset == "pseudo":
        digs = champernowne_digits(args.N)
        K = args.K or 10
    elif args.dataset == "bbp":
        digs = bbp_hex_pi_digits(args.N, start=0)
        K = args.K or 16
    else:
        raise ValueError("Unknown dataset")

    # Windows & split
    X, y = make_windows(digs, L=args.L, stride=args.stride)
    n = len(X)
    n_train = int(n * args.train_frac)
    Xtr, ytr = X[:n_train], y[:n_train]
    Xva, yva = X[n_train:], y[n_train:]

    tr_ds = torch.utils.data.TensorDataset(torch.from_numpy(Xtr), torch.from_numpy(ytr))
    va_ds = torch.utils.data.TensorDataset(torch.from_numpy(Xva), torch.from_numpy(yva))
    tr_loader = torch.utils.data.DataLoader(tr_ds, batch_size=args.batch_size, shuffle=True, drop_last=True)
    va_loader = torch.utils.data.DataLoader(va_ds, batch_size=args.batch_size, shuffle=False, drop_last=False)

    # Worlds & modules
    worldA = WorldA(L=args.L, K=K, hidden=args.hidden).to(device)
    worldB = WorldB(L=args.L, K=K, hidden=args.hidden, wave="db4").to(device)
    judgA  = JudgmentHead(d_feat=3, K=8, hidden=64).to(device)
    judgB  = JudgmentHead(d_feat=3, K=8, hidden=64).to(device)
    tauAB  = Translator(d_in=worldA.feat_dim, d_out=worldB.feat_dim).to(device)
    tauBA  = Translator(d_in=worldB.feat_dim, d_out=worldA.feat_dim).to(device)
    calAB  = Calibrator(K=8).to(device)
    calBA  = Calibrator(K=8).to(device)

    params = list(worldA.parameters()) + list(worldB.parameters()) + \
             list(judgA.parameters()) + list(judgB.parameters()) + \
             list(tauAB.parameters()) + list(tauBA.parameters()) + \
             list(calAB.parameters()) + list(calBA.parameters())
    opt = torch.optim.Adam(params, lr=args.lr)

    lambdas = Lambdas(
        nat=args.lambda_nat, gauge=args.lambda_gauge, cyc=args.lambda_cyc,
        hinge=args.lambda_hinge, sparsity=args.lambda_sparsity, coverage=args.lambda_coverage
    )

    logK = math.log(K)
    csv_path = os.path.join(args.logdir, f"log_{args.dataset}_L{args.L}_refactor.csv")
    with open(csv_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow([
            "epoch","split","L","L_task","L_nat","L_gauge","L_cyc","L_hinge","L_sparsity","L_coverage",
            "nllA","nllB","ign_f1","delta_v_mad","D_alpha","delta_v_mad_cond","D_alpha_cond","p_ignite","ignite_thr"
        ])

        for ep in range(1, args.epochs + 1):
            # Schedule ignition threshold
            thr = schedule_linear(ep-1, max(1, args.ignite_thr_epochs-1), args.ignite_thr_start, args.ignite_thr_end)

            # Warmup freeze to protect base predictors from collapse
            if ep == args.warmup_epochs + 1:
                for p in worldA.net.parameters(): p.requires_grad = False
                for p in worldB.net.parameters(): p.requires_grad = False
                print(f"[INFO] Froze base predictors at epoch {ep-1}", file=sys.stderr)

            tr = run_epoch(tr_loader, worldA, worldB, judgA, judgB, tauAB, tauBA, calAB, calBA, opt,
                           lambdas, ignite_thr=thr, logK=logK, hinge_eps=args.hinge_eps, device=device, train=True)
            va = run_epoch(va_loader, worldA, worldB, judgA, judgB, tauAB, tauBA, calAB, calBA, optimizer=None,
                           lambdas=lambdas, ignite_thr=thr, logK=logK, hinge_eps=args.hinge_eps, device=device, train=False)

            # Write CSV
            if tr:
                w.writerow([ep,"train", tr["L"], tr["L_task"], tr["L_nat"], tr["L_gauge"], tr["L_cyc"], tr["L_hinge"], tr["L_sparsity"], tr["L_coverage"],
                            tr["nllA"], tr["nllB"], tr["ign_f1"], tr["delta_v_mad"], tr["D_alpha"], tr["delta_v_mad_cond"], tr["D_alpha_cond"], tr["p_ignite"], thr])
            if va:
                w.writerow([ep,"val", va["L"], va["L_task"], va["L_nat"], va["L_gauge"], va["L_cyc"], va["L_hinge"], va["L_sparsity"], va["L_coverage"],
                            va["nllA"], va["nllB"], va["ign_f1"], va["delta_v_mad"], va["D_alpha"], va["delta_v_mad_cond"], va["D_alpha_cond"], va["p_ignite"], thr])
            f.flush()
            print(f"[epoch {ep}] val: nllA={va.get('nllA', float('nan')):.4f} ign_f1={va.get('ign_f1', float('nan')):.3f} "
                  f"dv={va.get('delta_v_mad', float('nan')):.4g} Dα={va.get('D_alpha', float('nan')):.4g} "
                  f"dv|ign={va.get('delta_v_mad_cond', float('nan')):.4g} Dα|ign={va.get('D_alpha_cond', float('nan')):.4g} "
                  f"p_ignite={va.get('p_ignite', float('nan')):.3f} thr={thr:.2f}")

    # Optional plotting
    if not args.no_plots:
        try:
            import matplotlib.pyplot as plt
            import pandas as pd
            df = pd.read_csv(csv_path)
            dfv = df[df["split"]=="val"]
            # Plot 1: NLL_A
            plt.figure()
            plt.plot(dfv["epoch"], dfv["nllA"])
            plt.xlabel("epoch"); plt.ylabel("NLL_A")
            plt.title("Validation NLL_A")
            plt.savefig(os.path.join(args.logdir, "plot_nllA.png")); plt.close()
            # Plot 2: delta_v_mad (global vs conditional)
            plt.figure()
            plt.plot(dfv["epoch"], dfv["delta_v_mad"])
            plt.xlabel("epoch"); plt.ylabel("delta_v_mad")
            plt.title("δ_v (MAD) global")
            plt.savefig(os.path.join(args.logdir, "plot_delta_v_mad.png")); plt.close()
            plt.figure()
            plt.plot(dfv["epoch"], dfv["delta_v_mad_cond"])
            plt.xlabel("epoch"); plt.ylabel("delta_v_mad|ignite")
            plt.title("δ_v (MAD) conditional on ignition")
            plt.savefig(os.path.join(args.logdir, "plot_delta_v_mad_cond.png")); plt.close()
            # Plot 3: D_alpha global vs conditional
            plt.figure()
            plt.plot(dfv["epoch"], dfv["D_alpha"])
            plt.xlabel("epoch"); plt.ylabel("D_alpha (nats)")
            plt.title("D_alpha global")
            plt.savefig(os.path.join(args.logdir, "plot_D_alpha.png")); plt.close()
            plt.figure()
            plt.plot(dfv["epoch"], dfv["D_alpha_cond"])
            plt.xlabel("epoch"); plt.ylabel("D_alpha|ignite (nats)")
            plt.title("D_alpha conditional")
            plt.savefig(os.path.join(args.logdir, "plot_D_alpha_cond.png")); plt.close()
            # Plot 4: ignition F1 and rate
            plt.figure()
            plt.plot(dfv["epoch"], dfv["ign_f1"])
            plt.xlabel("epoch"); plt.ylabel("Ignition F1")
            plt.title("Ignition agreement A→B")
            plt.savefig(os.path.join(args.logdir, "plot_ignition_f1.png")); plt.close()
            plt.figure()
            plt.plot(dfv["epoch"], dfv["p_ignite"])
            plt.xlabel("epoch"); plt.ylabel("p_ignite")
            plt.title("Ignition rate")
            plt.savefig(os.path.join(args.logdir, "plot_p_ignite.png")); plt.close()
        except Exception as e:
            print(f"[WARN] plotting failed: {e}", file=sys.stderr)

    print(f"[DONE] Logs at: {csv_path}")
    print("[INFO] For π decimal: expect NLL ≈ log(K). Universality should appear in conditional δ_v and D_alpha, with non-zero ignition F1.")
    print("[TIP] Use --dataset pseudo as a positive control; you should see task improve and invariance persist.")
    
if __name__ == "__main__":
    main()