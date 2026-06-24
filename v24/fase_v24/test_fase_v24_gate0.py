#!/usr/bin/env python3
"""Smoke and destroyer-clause checks for FASE v24 Gate 0."""
import json
from fase_v24_modal_recursive_closure import run_lab


def main():
    out = run_lab(quick=True, seed=42, noise=0.0)
    tasks = out["tasks"]

    # In-mode laws must not trigger unnecessary alphabet growth.
    assert tasks["affine_translation"]["observed_status"] == "in_mode"
    assert tasks["linear_scaling"]["observed_status"] == "in_mode"

    # Missing operators should be promoted only when long-horizon and perturbation
    # evidence support them.
    assert tasks["sqrt_missing_operator"]["observed_status"] == "promoted"
    assert tasks["sine_missing_operator"]["observed_status"] == "promoted"
    assert tasks["mobius_missing_operator"]["observed_status"] == "promoted"

    # A nonstationary process with omitted parity must not receive a false generator.
    assert tasks["null_switching"]["observed_status"] == "abstain"
    assert out["summary"]["false_closures"] == 0

    # Perturbation/conjugacy results must be finite.
    for row in tasks.values():
        assert row["conjugacy_audit"]["conjugacy_nrmse"] is not None
        assert row["active_perturbation"]["n_queries"] > 0

    print(json.dumps(out["summary"], indent=2))
    for name, row in tasks.items():
        print(name, "=>", row["observed_status"], row["selected"]["family"], row["selected"]["expression"])
    print("ALL V24 GATE-0 CHECKS PASS")


if __name__ == "__main__":
    main()
