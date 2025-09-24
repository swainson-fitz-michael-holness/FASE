#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Gauge-invariance experiment on digits of π across coordinate charts.

Charts:
  - cart2d: (x1, x2, x3)  [raw last-3-digit lift]
  - polar:  (r, theta/pi, x3)
  - cyl:    (rho, phi/pi, z=x3)
  - sph:    (rho, theta/pi, phi/pi)

Judges:
  - identical small MLPs per chart -> 10-class logits for next digit.

Calibrators:
  - per-chart affine (temperature + bias): y = a * logits + b
  - gauge consensus loss: MSE to consensus canonical logits
  - cycle loss: reconstruct original logits via inverse affine

Anti-collapse:
  - target coverage on ignition probability p_hat ~ p_target
  - variance regularizer on the “confidence” scalar v (margin)

Ignition:
  - v = max(logits) - mean(logits); s = sigmoid(v); ignite = s > thr
  - coverage enforced on mean(s) across charts and batch
  - honest F1 across chart pairs with safe handling of no-positives

Apple Silicon:
  - defaults to MPS if available
  - AMP optional (safe default is fp32). Use --amp to turn on.

CSV:
  - logs per epoch for train/val with per-chart NLL, ignition stats,
    gauge/cycle/coverage losses, Δv MAD, Dα, etc.

Author: you & me.
"""
import argparse
import math
import os
import random
from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader


# ----------------------------- Utils ---------------------------------

def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def device_pick(prefer_mps: bool = True) -> torch.device:
    if prefer_mps and torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def safe_mean(lst: List[float]) -> float:
    return float(sum(lst) / max(1, len(lst)))


def safe_item(x):
    if isinstance(x, torch.Tensor):
        return float(x.detach().cpu().item())
    return float(x)


def sigmoid(x: torch.Tensor) -> torch.Tensor:
    return torch.sigmoid(x)


def madd(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    # mean absolute deviation from mean
    return (a - a.mean(dim=0, keepdim=True)).abs().mean()


def f1_binary(y_true: torch.Tensor, y_pred: torch.Tensor) -> float:
    """
    Honest F1: if both positives are zero (no true or no pred), return 0.0
    y_true, y_pred are 0/1 tensors (N,)
    """
    y_true = y_true.view(-1).to(torch.int64)
    y_pred = y_pred.view(-1).to(torch.int64)
    tp = ((y_true == 1) & (y_pred == 1)).sum().item()
    fp = ((y_true == 0) & (y_pred == 1)).sum().item()
    fn = ((y_true == 1) & (y_pred == 0)).sum().item()
    if (tp + fp == 0) or (tp + fn == 0):
        return 0.0
    prec = tp / (tp + fp)
    rec = tp / (tp + fn)
    if prec + rec == 0:
        return 0.0
    return 2.0 * prec * rec / (prec + rec)


# ----------------------- Data: π digits -------------------------------

class PiDigits(Dataset):
    """
    Turns a long digit string ...d_{t-L} ... d_{t-1} -> target d_t
    Using last-3 digits (d_{t-3}, d_{t-2}, d_{t-1}) to build 3D base lift.
    """
    def __init__(self, pi_path: str, L: int = 128, split: str = "train", val_frac: float = 0.1):
        assert split in {"train", "val"}
        with open(pi_path, "r") as f:
            raw = f.read()
        digits = [int(ch) for ch in raw if ch.isdigit()]
        arr = np.array(digits, dtype=np.int64)
        # make windows (context length L) and next token
        N = len(arr)
        max_idx = N - (L + 1)
        if max_idx <= 0:
            raise ValueError(f"pi.txt too short for L={L}")
        X_ctx = []
        Y = []
        for i in range(max_idx):
            X_ctx.append(arr[i:i+L])
            Y.append(arr[i+L])
        X = np.stack(X_ctx, axis=0)  # (M, L)
        Y = np.array(Y, dtype=np.int64)  # (M,)

        # split
        M = X.shape[0]
        cut = int(M * (1.0 - val_frac))
        if split == "train":
            self.X = X[:cut]
            self.Y = Y[:cut]
        else:
            self.X = X[cut:]
            self.Y = Y[cut:]

        self.L = L

    def __len__(self):
        return self.X.shape[0]

    def __getitem__(self, idx):
        ctx = self.X[idx]  # (L,)
        target = self.Y[idx]  # scalar
        # base 3D lift: last three ctx digits normalized to [-1,1]
        d1, d2, d3 = ctx[-3], ctx[-2], ctx[-1]
        base = np.array([(d1 - 4.5)/5.0, (d2 - 4.5)/5.0, (d3 - 4.5)/5.0], dtype=np.float32)
        return base, target


# --------------- Coordinate charts and jitter -------------------------

def add_jitter(x: torch.Tensor, sigma: float, training: bool) -> torch.Tensor:
    if not training or sigma <= 0:
        return x
    return x + sigma * torch.randn_like(x)


def chart_cart2d(base3: torch.Tensor) -> torch.Tensor:
    # keep 3D as-is to keep dimensional parity
    return base3


def chart_polar(base3: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    x, y, z = base3[:, 0], base3[:, 1], base3[:, 2]
    r = torch.sqrt(x*x + y*y + eps)
    theta = torch.atan2(y, x) / math.pi  # normalize [-1,1]
    return torch.stack([r, theta, z], dim=1)


def chart_cyl(base3: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    x, y, z = base3[:, 0], base3[:, 1], base3[:, 2]
    rho = torch.sqrt(x*x + y*y + eps)
    phi = torch.atan2(y, x) / math.pi
    return torch.stack([rho, phi, z], dim=1)


def chart_sph(base3: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    x, y, z = base3[:, 0], base3[:, 1], base3[:, 2]
    rho = torch.sqrt(x*x + y*y + z*z + eps)
    theta = torch.atan2(y, x) / math.pi
    phi = torch.acos(torch.clamp(z / (rho + eps), -1.0, 1.0)) / math.pi  # [0,1]
    return torch.stack([rho, theta, phi], dim=1)


CHARTS = ("cart2d", "polar", "cyl", "sph")


# ------------------------- Models ------------------------------------

class MLPJudge(nn.Module):
    def __init__(self, in_dim: int = 3, hidden: int = 64, out_dim: int = 10):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.LayerNorm(hidden),
            nn.GELU(),
            nn.Linear(hidden, hidden),
            nn.GELU(),
            nn.Linear(hidden, out_dim),
        )

    def forward(self, x):
        return self.net(x)  # logits (N,10)


class AffineCalibrator(nn.Module):
    """
    Canonical affine calibrator per chart:
        y = a * logits + b
    with a = softplus(a_raw) + eps to keep positive scale (invertible).
    """
    def __init__(self, eps: float = 1e-4):
        super().__init__()
        self.a_raw = nn.Parameter(torch.tensor(0.0))
        self.b = nn.Parameter(torch.tensor(0.0))
        self.eps = eps

    def forward(self, logits):
        a = F.softplus(self.a_raw) + self.eps
        return a * logits + self.b

    def inverse(self, y):
        a = F.softplus(self.a_raw) + self.eps
        return (y - self.b) / a


# ----------------------- Training pieces ------------------------------

@dataclass
class LossWeights:
    lambda_gauge: float = 1.0
    lambda_cyc: float = 0.01
    lambda_coverage: float = 3.0
    lambda_vvar: float = 0.1


def ignite_schedule(epoch: int, total_epochs: int, t0: float, t1: float, warm: int = 0) -> float:
    if epoch < warm:
        return t0
    α = min(1.0, max(0.0, (epoch) / max(1, total_epochs)))
    return (1 - α) * t0 + α * t1


def compute_losses(
    batch_base3: torch.Tensor,
    targets: torch.Tensor,
    judges: Dict[str, MLPJudge],
    calibrators: Dict[str, AffineCalibrator],
    ignite_thr: float,
    lw: LossWeights,
    p_target: float,
    sigma_target: float,
    device: torch.device,
    jitter_sigma: float,
    training: bool,
):
    # Jitter before charting
    base3 = add_jitter(batch_base3, jitter_sigma, training)

    # Build chart features (N,3)
    feats = {
        "cart2d": chart_cart2d(base3),
        "polar":  chart_polar(base3),
        "cyl":    chart_cyl(base3),
        "sph":    chart_sph(base3),
    }

    # Judge logits per chart
    logits = {}
    nlls = {}
    ce = nn.CrossEntropyLoss(reduction="mean")
    for name in CHARTS:
        logit = judges[name](feats[name])  # (N,10)
        logits[name] = logit
        nlls[name] = ce(logit, targets)

    # Task loss: mean CE across charts
    L_task = torch.stack(list(nlls.values())).mean()

    # Calibrated logits to canonical
    cal_logits = {}
    for name in CHARTS:
        cal_logits[name] = calibrators[name](logits[name])

    # Consensus canonical logits
    cal_stack = torch.stack([cal_logits[name] for name in CHARTS], dim=0)  # (C,N,10)
    cal_mean = cal_stack.mean(dim=0)  # (N,10)

    # Gauge loss: MSE to consensus
    mse = nn.MSELoss(reduction="mean")
    L_gauge = torch.stack([
        mse(cal_logits[name], cal_mean) for name in CHARTS
    ]).mean()

    # Cycle/reconstruction loss: reconstruct each chart's raw logits from canonical
    # via inverse calibrator and match original logits
    L_cyc = torch.stack([
        mse(calibrators[name].inverse(cal_mean), logits[name]) for name in CHARTS
    ]).mean()

    # Ignition/confidence scalar per chart
    # v = margin = max(logits) - mean(logits)
    v = {}
    s = {}
    ignite = {}
    for name in CHARTS:
        l = logits[name]
        max_logit, _ = l.max(dim=1, keepdim=True)
        v[name] = (max_logit - l.mean(dim=1, keepdim=True)).squeeze(1)  # (N,)
        s[name] = sigmoid(v[name])  # (N,)
        ignite[name] = (s[name] > ignite_thr).float()

    # Coverage target across all charts
    s_all = torch.stack([s[name] for name in CHARTS], dim=0)  # (C,N)
    p_hat = s_all.mean()  # scalar
    L_cov = (p_hat - p_target) ** 2

    # Variance regularizer on v across batch+charts
    v_all = torch.stack([v[name] for name in CHARTS], dim=0)  # (C,N)
    var_v = v_all.var(unbiased=False)
    L_vvar = torch.relu(sigma_target - var_v)

    # Total
    L = L_task + lw.lambda_gauge * L_gauge + lw.lambda_cyc * L_cyc + lw.lambda_coverage * L_cov + lw.lambda_vvar * L_vvar

    # Diagnostics
    with torch.no_grad():
        # per-chart stats
        nll_vals = {k: nlls[k].detach() for k in CHARTS}
        p_ignite_chart = {k: ignite[k].mean().detach() for k in CHARTS}

        # Δv MAD across charts (per sample), averaged
        v_cal = torch.stack([v[k] for k in CHARTS], dim=0)  # (C,N)
        v_mean = v_cal.mean(dim=0, keepdim=True)
        delta_v_mad = (v_cal - v_mean).abs().mean()  # scalar

        # D_alpha: std of s across charts (per sample) averaged
        s_cal = torch.stack([s[k] for k in CHARTS], dim=0)  # (C,N)
        D_alpha = s_cal.std(dim=0, unbiased=False).mean()

        # Conditional (samples where any ignite)
        any_ignite = (torch.stack([ignite[k] for k in CHARTS], dim=0).sum(dim=0) > 0)
        if any_ignite.any():
            s_cond = s_cal[:, any_ignite]
            v_cond = v_cal[:, any_ignite]
            v_mean_c = v_cond.mean(dim=0, keepdim=True)
            delta_v_mad_cond = (v_cond - v_mean_c).abs().mean()
            D_alpha_cond = s_cond.std(dim=0, unbiased=False).mean()
        else:
            delta_v_mad_cond = torch.tensor(float("nan"), device=device)
            D_alpha_cond = torch.tensor(float("nan"), device=device)

        # Pairwise ignition F1 (average over all 6 pairs)
        ign_pairs = []
        chart_list = list(CHARTS)
        for i in range(len(chart_list)):
            for j in range(i + 1, len(chart_list)):
                a = ignite[chart_list[i]].detach().cpu()
                b = ignite[chart_list[j]].detach().cpu()
                ign_pairs.append(f1_binary(a, b))
        ign_f1_avg = float(np.mean(ign_pairs)) if len(ign_pairs) else 0.0

    metrics = {
        "L_task": safe_item(L_task),
        "L_gauge": safe_item(L_gauge),
        "L_cyc": safe_item(L_cyc),
        "L_coverage": safe_item(L_cov),
        "L_vvar": safe_item(L_vvar),
        "p_hat": safe_item(p_hat),
        "var_v": safe_item(var_v),

        "nll_cart2d": safe_item(nll_vals["cart2d"]),
        "nll_polar": safe_item(nll_vals["polar"]),
        "nll_cyl": safe_item(nll_vals["cyl"]),
        "nll_sph": safe_item(nll_vals["sph"]),

        "p_ignite_cart2d": safe_item(p_ignite_chart["cart2d"]),
        "p_ignite_polar": safe_item(p_ignite_chart["polar"]),
        "p_ignite_cyl": safe_item(p_ignite_chart["cyl"]),
        "p_ignite_sph": safe_item(p_ignite_chart["sph"]),

        "ign_f1_avg": ign_f1_avg,
        "delta_v_mad": safe_item(delta_v_mad),
        "D_alpha": safe_item(D_alpha),
        "delta_v_mad_cond": safe_item(delta_v_mad_cond),
        "D_alpha_cond": safe_item(D_alpha_cond),
    }

    return L, metrics


# ----------------------------- Training loop --------------------------

def run_epoch(
    loader: DataLoader,
    judges: Dict[str, MLPJudge],
    calibrators: Dict[str, AffineCalibrator],
    args,
    epoch: int,
    split: str,
    ignite_thr: float,
    optimizer: torch.optim.Optimizer = None,
    scaler: torch.cuda.amp.GradScaler = None,
):
    device = args.device
    training = (optimizer is not None)
    for m in list(judges.values()) + list(calibrators.values()):
        m.train(training)

    lw = LossWeights(
        lambda_gauge=args.lambda_gauge,
        lambda_cyc=args.lambda_cyc,
        lambda_coverage=args.lambda_coverage,
        lambda_vvar=args.lambda_vvar,
    )

    agg = {k: 0.0 for k in [
        "L_task", "L_gauge", "L_cyc", "L_coverage", "L_vvar", "p_hat", "var_v",
        "nll_cart2d", "nll_polar", "nll_cyl", "nll_sph",
        "p_ignite_cart2d", "p_ignite_polar", "p_ignite_cyl", "p_ignite_sph",
        "ign_f1_avg", "delta_v_mad", "D_alpha", "delta_v_mad_cond", "D_alpha_cond"
    ]}
    count = 0

    autocast_enabled = args.amp and (device.type in {"cuda", "mps"})
    amp_dtype = torch.float16 if device.type != "cuda" else torch.float16

    for base3_np, target_np in loader:
        base3 = torch.as_tensor(base3_np, dtype=torch.float32, device=device)
        targets = torch.as_tensor(target_np, dtype=torch.long, device=device)

        with torch.autocast(device_type=device.type, dtype=amp_dtype, enabled=autocast_enabled):
            loss, metrics = compute_losses(
                batch_base3=base3,
                targets=targets,
                judges=judges,
                calibrators=calibrators,
                ignite_thr=ignite_thr,
                lw=lw,
                p_target=args.p_target,
                sigma_target=args.sigma_target,
                device=device,
                jitter_sigma=args.jitter_sigma,
                training=training,
            )

        if training:
            optimizer.zero_grad(set_to_none=True)
            if scaler is not None and device.type == "cuda":
                scaler.scale(loss).step(optimizer)
                scaler.update()
            else:
                loss.backward()
                optimizer.step()

        # aggregate
        bs = base3.shape[0]
        for k, v in metrics.items():
            agg[k] += float(v) * bs
        count += bs

    # normalize
    for k in agg:
        agg[k] = agg[k] / max(1, count)

    # add ignite threshold, split, epoch
    agg["ignite_thr"] = ignite_thr
    agg["split"] = split
    agg["epoch"] = epoch

    return agg


# ------------------------------- Main ---------------------------------

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--pi-file", type=str, required=True)
    p.add_argument("--epochs", type=int, default=55)
    p.add_argument("--batch-size", type=int, default=1024)
    p.add_argument("--L", type=int, default=128)

    # losses / regs
    p.add_argument("--lambda-gauge", type=float, default=1.0)
    p.add_argument("--lambda-cyc", type=float, default=0.01)
    p.add_argument("--lambda-coverage", type=float, default=3.0)
    p.add_argument("--lambda-vvar", type=float, default=0.1)
    p.add_argument("--p-target", type=float, default=0.10)
    p.add_argument("--sigma-target", type=float, default=0.7)

    # ignition schedule
    p.add_argument("--ignite-thr-start", type=float, default=0.45)
    p.add_argument("--ignite-thr-end", type=float, default=0.75)
    p.add_argument("--ignite-thr-epochs", type=int, default=20)

    # model/optim
    p.add_argument("--hidden", type=int, default=64)
    p.add_argument("--lr", type=float, default=2e-3)
    p.add_argument("--weight-decay", type=float, default=1e-4)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--amp", action="store_true", help="Enable AMP (fp16) on CUDA/MPS")
    p.add_argument("--compile", action="store_true", help="torch.compile judges on PyTorch>=2 (MPS supported)")

    # data / apple silicon ergonomics
    p.add_argument("--num-workers", type=int, default=0, help="0 recommended on macOS/MPS")
    p.add_argument("--jitter-sigma", type=float, default=0.01)

    p.add_argument("--csv-out", type=str, default="coords_gate_log.csv")
    args = p.parse_args()

    # device
    args.device = device_pick(prefer_mps=True)
    print(f"[info] device: {args.device}")

    set_seed(args.seed)

    # data
    tr_ds = PiDigits(args.pi_file, L=args.L, split="train", val_frac=0.1)
    va_ds = PiDigits(args.pi_file, L=args.L, split="val", val_frac=0.1)

    pin_mem = (args.device.type == "cuda")
    tr_ld = DataLoader(tr_ds, batch_size=args.batch_size, shuffle=True,
                       num_workers=args.num_workers, pin_memory=pin_mem, drop_last=False)
    va_ld = DataLoader(va_ds, batch_size=args.batch_size, shuffle=False,
                       num_workers=args.num_workers, pin_memory=pin_mem, drop_last=False)

    # models
    judges = {name: MLPJudge(in_dim=3, hidden=args.hidden, out_dim=10).to(args.device) for name in CHARTS}
    calibrators = {name: AffineCalibrator().to(args.device) for name in CHARTS}

    if args.compile and hasattr(torch, "compile"):
        try:
            judges = {k: torch.compile(v) for k, v in judges.items()}
            calibrators = {k: torch.compile(v) for k, v in calibrators.items()}
            print("[info] compiled models with torch.compile")
        except Exception as e:
            print(f"[warn] torch.compile failed ({e}); continuing without.")

    # optimizer (shared)
    params = list()
    for m in list(judges.values()) + list(calibrators.values()):
        params += list(m.parameters())
    opt = torch.optim.AdamW(params, lr=args.lr, weight_decay=args.weight_decay)
    scaler = torch.cuda.amp.GradScaler(enabled=(args.amp and args.device.type == "cuda"))

    # CSV header
    header = ("epoch,split,ignite_thr,"
              "L_task,L_gauge,L_cyc,L_coverage,L_vvar,p_hat,var_v,"
              "ign_f1_avg,delta_v_mad,D_alpha,delta_v_mad_cond,D_alpha_cond,p_ignite_avg,"
              "nll_cart2d,nll_polar,nll_cyl,nll_sph,"
              "p_ignite_cart2d,p_ignite_polar,p_ignite_cyl,p_ignite_sph\n")
    if not os.path.exists(args.csv_out):
        with open(args.csv_out, "w") as f:
            f.write(header)

    # training
    for epoch in range(1, args.epochs + 1):
        thr = ignite_schedule(
            epoch=min(epoch, args.ignite_thr_epochs),
            total_epochs=args.ignite_thr_epochs,
            t0=args.ignite_thr_start,
            t1=args.ignite_thr_end,
            warm=0
        )

        tr = run_epoch(tr_ld, judges, calibrators, args, epoch, "train", thr, optimizer=opt, scaler=scaler)
        va = run_epoch(va_ld, judges, calibrators, args, epoch, "val",   thr, optimizer=None, scaler=None)

        # aggregate / print
        def row(stats: Dict[str, float]) -> str:
            p_ignite_avg = safe_mean([
                stats["p_ignite_cart2d"], stats["p_ignite_polar"],
                stats["p_ignite_cyl"], stats["p_ignite_sph"]
            ])
            cols = [
                stats["epoch"], stats["split"], stats["ignite_thr"],
                stats["L_task"], stats["L_gauge"], stats["L_cyc"], stats["L_coverage"], stats["L_vvar"], stats["p_hat"], stats["var_v"],
                stats["ign_f1_avg"], stats["delta_v_mad"], stats["D_alpha"], stats["delta_v_mad_cond"], stats["D_alpha_cond"], p_ignite_avg,
                stats["nll_cart2d"], stats["nll_polar"], stats["nll_cyl"], stats["nll_sph"],
                stats["p_ignite_cart2d"], stats["p_ignite_polar"], stats["p_ignite_cyl"], stats["p_ignite_sph"],
            ]
            return ",".join(f"{c}" for c in cols) + "\n"

        with open(args.csv_out, "a") as f:
            f.write(row(tr))
            f.write(row(va))

        # brief console summary
        print(f"[{epoch:03d}] thr={thr:.3f}  "
              f"NLL(val): cart={va['nll_cart2d']:.6f} polar={va['nll_polar']:.6f} cyl={va['nll_cyl']:.6f} sph={va['nll_sph']:.6f}  "
              f"p_ignite(val)≈{safe_mean([va['p_ignite_cart2d'],va['p_ignite_polar'],va['p_ignite_cyl'],va['p_ignite_sph']]):.3f}  "
              f"L_gauge(val)={va['L_gauge']:.2e} L_cyc(val)={va['L_cyc']:.2e}")

    print(f"[done] wrote {args.csv_out}")


if __name__ == "__main__":
    main()
