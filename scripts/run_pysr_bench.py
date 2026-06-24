#!/usr/bin/env python3
"""Gate 0 benchmark runner: FASE (nested, leakage-fixed) vs PySR, one protocol.

Reads a dataset manifest, runs FASE_v22.run_benchmark_on_dataset for every
(dataset, seed) pair, and writes:

  <output-dir>/<dataset>__seed<seed>.json   one file per run, written
                                            immediately after the run
                                            completes (crash-safe)
  <output-dir>/summary.json                 paired deltas + wins/losses/ties
  <output-dir>/environment.json             python/numpy/pysr/pmlb versions,
                                            platform, git commit + dirty flag
  <output-dir>/resolved_config.json         the exact configuration this
                                            invocation actually ran with

Usage:
  python scripts/run_pysr_bench.py                       # manifest defaults
  python scripts/run_pysr_bench.py --quick --skip-pysr   # fast FASE-only smoke
  python scripts/run_pysr_bench.py --datasets 579_fri_c0_250_5 --seeds 42
  python scripts/run_pysr_bench.py --list-all-pmlb-regression
"""
import argparse
import json
import os
import platform
import subprocess
import sys
import time

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO_ROOT)

import numpy as np

import FASE_v22 as v22


def git_provenance():
    def run(cmd):
        try:
            return subprocess.check_output(cmd, shell=True, text=True,
                                           cwd=REPO_ROOT,
                                           stderr=subprocess.DEVNULL).strip()
        except Exception:
            return None
    status = run("git status --porcelain")
    return {"commit": run("git rev-parse HEAD"),
            "branch": run("git rev-parse --abbrev-ref HEAD"),
            "dirty": bool(status) if status is not None else None}


def environment():
    env = {"python": sys.version, "platform": platform.platform(),
           "numpy": np.__version__}
    for mod in ("pysr", "pmlb"):
        try:
            env[mod] = __import__(mod).__version__
        except Exception as e:
            env[mod] = f"unavailable: {type(e).__name__}: {e}"
    env["git"] = git_provenance()
    return env


def main():
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--manifest",
                   default=os.path.join(REPO_ROOT, "benchmarks", "pmlb_manifest.json"))
    p.add_argument("--output-dir",
                   default=os.path.join(REPO_ROOT, "outputs", "pysr_bench"))
    p.add_argument("--datasets", nargs="*", help="override manifest dataset list")
    p.add_argument("--seeds", type=int, nargs="*", help="override manifest seeds")
    p.add_argument("--k-folds", type=int, help="override manifest k_folds")
    p.add_argument("--quick", action="store_true",
                   help="small FASE config + tiny PySR budget (smoke only; never publish)")
    p.add_argument("--skip-pysr", action="store_true", help="FASE side only")
    p.add_argument("--ablation", action="store_true",
                   help="also run PySR on [X | FASE fold-model features] (labeled ablation)")
    p.add_argument("--max-datasets", type=int)
    p.add_argument("--list-all-pmlb-regression", action="store_true",
                   help="print every PMLB regression dataset name and exit")
    a = p.parse_args()

    if a.list_all_pmlb_regression:
        from pmlb import regression_dataset_names
        print("\n".join(regression_dataset_names))
        return

    man = {}
    if os.path.exists(a.manifest):
        with open(a.manifest) as f:
            man = json.load(f)
    datasets = a.datasets or man.get("datasets") or []
    if not datasets:
        sys.exit("No datasets: provide --datasets or a manifest with a 'datasets' list.")
    if a.max_datasets:
        datasets = datasets[: a.max_datasets]
    seeds = a.seeds or man.get("seeds") or [42]
    k = a.k_folds or man.get("k_folds") or 5

    pysr_cfg = v22.apply_bench_config(quick=a.quick, pysr_overrides=man.get("pysr"))

    os.makedirs(a.output_dir, exist_ok=True)
    with open(os.path.join(a.output_dir, "environment.json"), "w") as f:
        json.dump(environment(), f, indent=2)
    with open(os.path.join(a.output_dir, "resolved_config.json"), "w") as f:
        json.dump(v22._json_safe({
            "manifest_path": a.manifest, "datasets": datasets, "seeds": seeds,
            "k_folds": k, "quick": a.quick, "skip_pysr": a.skip_pysr,
            "ablation": a.ablation, "pysr": pysr_cfg,
            "fase_config": v22.CONFIG,
        }), f, indent=2, default=str)

    rows = []
    for name in datasets:
        for seed in seeds:
            print(f"\n[bench] === {name} (seed {seed}) ===", flush=True)
            t0 = time.time()
            try:
                res = v22.run_benchmark_on_dataset(
                    name, k_folds=k, seed=seed, pysr_cfg=pysr_cfg,
                    run_pysr=not a.skip_pysr, run_ablation=a.ablation)
            except Exception as e:
                res = {"dataset": name, "seed": seed, "status": "crashed",
                       "error": f"{type(e).__name__}: {e}"}
            res["total_wall_time_s"] = time.time() - t0
            path = os.path.join(a.output_dir, f"{name}__seed{seed}.json")
            with open(path, "w") as f:
                json.dump(v22._json_safe(res), f, indent=2, default=str)
            print(f"[bench] wrote {path} (status={res.get('status')})", flush=True)
            rows.append(res)

    # ---- summary: paired per-(dataset, seed) comparison ----
    summary = {"n_runs": len(rows), "tie_band_abs_delta": 0.01, "runs": []}
    wins = losses = ties = 0
    deltas = []
    for r in rows:
        item = {"dataset": r.get("dataset"), "seed": r.get("seed"),
                "status": r.get("status")}
        f_r2 = (r.get("fase") or {}).get("R2_oof")
        p_r2 = (r.get("pysr_raw") or {}).get("R2_oof")
        item["fase_R2_oof"] = f_r2
        item["pysr_R2_oof"] = p_r2
        if f_r2 is not None and p_r2 is not None:
            d = f_r2 - p_r2
            item["delta_R2_oof"] = d
            deltas.append(d)
            if d > 0.01:
                wins += 1
            elif d < -0.01:
                losses += 1
            else:
                ties += 1
        summary["runs"].append(item)
    if deltas:
        summary.update(fase_wins=int(wins), pysr_wins=int(losses), ties=int(ties),
                       mean_delta_R2=float(np.mean(deltas)),
                       median_delta_R2=float(np.median(deltas)))
    with open(os.path.join(a.output_dir, "summary.json"), "w") as f:
        json.dump(v22._json_safe(summary), f, indent=2, default=str)

    print("\n[bench] ===== Summary =====")
    for item in summary["runs"]:
        fr = item.get("fase_R2_oof")
        pr = item.get("pysr_R2_oof")
        fr_s = f"{fr:.4f}" if isinstance(fr, float) else "  --  "
        pr_s = f"{pr:.4f}" if isinstance(pr, float) else "  --  "
        print(f"  {str(item['dataset']):>28s} seed={item['seed']}  "
              f"FASE={fr_s}  PySR={pr_s}  ({item['status']})")
    if deltas:
        print(f"  FASE wins: {wins} | PySR wins: {losses} | "
              f"ties (|d|<=0.01): {ties} | mean dR2 = {summary['mean_delta_R2']:+.4f}")


if __name__ == "__main__":
    main()
