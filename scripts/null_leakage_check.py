#!/usr/bin/env python3
"""Pure-noise control: the empirical demonstration of the v21 leak.

y ~ N(0,1) is drawn independently of X, so the TRUE out-of-sample R^2 of any
method on this data is <= 0 in expectation. Both protocols run on the same
data and the same outer folds (same kfold_indices seed):

  v21.run_fase_kfold        — feature selection scored on the fold later
                              reported as "OOF" (the leak)
  v22.run_fase_kfold_nested — selection confined to an inner split of the
                              outer-training fold

Any materially positive OOF R^2 under the v21 protocol on this data is
selection leakage, by construction. The v22 number should sit at or below
zero. Runs in minutes at the trimmed default budget; the leak does not need
the full one.

Usage:
  python scripts/null_leakage_check.py
  python scripts/null_leakage_check.py --n 300 --d 8 --k 5 --max-atoms 12
"""
import argparse
import logging
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np

import FASE_v21 as v21
import FASE_v22 as v22


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--n", type=int, default=300)
    p.add_argument("--d", type=int, default=8)
    p.add_argument("--k", type=int, default=5)
    p.add_argument("--seed", type=int, default=7)
    p.add_argument("--max-atoms", type=int, default=12)
    p.add_argument("--verbose", action="store_true")
    a = p.parse_args()

    if not a.verbose:
        logging.getLogger("FASE").setLevel(logging.WARNING)  # silence stage logs

    # Trimmed budget so this finishes in minutes; the structural difference
    # between protocols is what is under test, not the full pipeline.
    v22.apply_bench_config(quick=False)
    v21.CONFIG["MAX_ATOMS"] = a.max_atoms
    v21.CONFIG["MAX_GRAMMAR"] = 0
    v21.CONFIG["USE_RULIAD_STAGE25"] = False
    v21.CONFIG["OGSET"]["bag_boots"] = 2

    rng = np.random.default_rng(0)
    X = rng.normal(size=(a.n, a.d))
    y = rng.normal(size=a.n)  # independent of X: true R^2 <= 0

    print(f"[null] n={a.n} d={a.d} K={a.k} seed={a.seed} "
          f"max_atoms={a.max_atoms}  (y independent of X)")

    leaky = v21.run_fase_kfold(X, y, Sigma=None, K=a.k, seed=a.seed)
    print(f"[null] v21 protocol (selection scored on reported fold): "
          f"OOF R^2 = {leaky['R2_oof']:+.4f}")

    nested = v22.run_fase_kfold_nested(X, y, Sigma=None, K=a.k, seed=a.seed)
    print(f"[null] v22 protocol (nested selection):                  "
          f"OOF R^2 = {nested['R2_oof']:+.4f}")

    print("[null] Interpretation: any materially positive v21 value on pure "
          "noise is selection leakage made visible; v22 should sit at or "
          "below zero.")


if __name__ == "__main__":
    main()
