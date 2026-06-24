#!/usr/bin/env python3
"""Fast regression checks for v23.1 joint refinement and non-degradation."""
import json
import numpy as np
import fase_v23_g1a_coordinate_gate as base
import fase_v23_g1a_constant_refine as ref


def main():
    rng = np.random.default_rng(42)
    n = 160
    a = rng.uniform(-1.5, 8.0, n)
    p = rng.uniform(0.2, 2.0, n)
    X = np.column_stack([a, p])
    y = p * np.sqrt(2.0 + a) / 2.0 + rng.normal(0, 1e-3, n)
    tr = np.arange(120); va = np.arange(120, 160)
    context = [base.X(0), base.X(1)]

    wrong = base.Mul(base.Sqrt(base.Add(base.X(0), base.C(0.5))),
                     base.Sub(base.X(1), base.C(0.5)))
    rcfg = ref.RefineConfig()
    refined, before, after = ref.refine_coordinate_joint(
        wrong, context, X[tr], y[tr], X[va], y[va], rcfg
    )
    constants = ref.enumerate_consts(refined)

    # A coordinate with no constants is unchanged, preserving the original topology.
    mul = base.Mul(base.X(0), base.X(1))
    mul_refined, mb, ma = ref.refine_coordinate_joint(
        mul, [], X[tr], y[tr], X[va], y[va], rcfg
    )

    result = {
        "vieta": {
            "before": str(wrong),
            "after": str(refined),
            "constants": constants,
            "val_R2_before": before,
            "val_R2_after": after,
        },
        "nondegrade_control": {
            "before": str(mul),
            "after": str(mul_refined),
            "unchanged": str(mul) == str(mul_refined),
            "val_before": mb,
            "val_after": ma,
        },
    }
    assert abs(constants[0] - 2.0) < 0.05
    assert abs(constants[1]) < 0.05
    assert after >= 0.99999
    assert str(mul) == str(mul_refined)
    print(json.dumps(result, indent=2))
    print("ALL V23.1 JOINT-REFINEMENT CHECKS PASS")


if __name__ == "__main__":
    main()
