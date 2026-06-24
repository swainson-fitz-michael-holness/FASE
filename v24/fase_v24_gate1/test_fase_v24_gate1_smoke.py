#!/usr/bin/env python3
"""Fast deterministic destroyer-clause test for FASE-v24 Gate 1."""
from dataclasses import asdict

from fase_v24_gate1_operator_genesis import GateConfig, TASK_NAMES, run_condition

EXPECTED_FRAGMENTS = {
    "algebra_distributivity": "Mul(?v0:Expr, Add(?v1:Expr, ?v2:Expr))",
    "lambda_beta_identity": "App(Lam(?v0:Name, Var(?v0:Name)), ?v1:Expr)",
    "chemistry_hydrogenation": "React(Alkene(?v0:Group, ?v1:Group), H2)",
    "biology_transcription": "Express(Gene(?v0:Promoter, Coding(?v1:Sequence)))",
    "structural_swap": "Pair(?v0:Expr, ?v1:Expr)",
}


def main() -> None:
    cfg = asdict(GateConfig(ransac_trials=128))
    rows = []
    for task in TASK_NAMES:
        row = run_condition(task, 42, 12, 0.0, 8, 32, cfg)
        rows.append(row)
        assert row["status_correct"], f"wrong status: {task}: {row['observed_status']}"
        if task in EXPECTED_FRAGMENTS:
            schema = row["selected_schema"]
            assert schema is not None and schema["pass_gate"], f"no promoted schema: {task}"
            assert EXPECTED_FRAGMENTS[task] in schema["pretty"], schema["pretty"]
            assert schema["heldout_exact_accuracy"] == 1.0
            assert schema["perturbation_exact_accuracy"] == 1.0
            assert schema["equivariance_accuracy"] == 1.0
        else:
            assert row["observed_status"] == "abstain"
            assert not row["false_promotion"]

    print("FASE-v24 Gate-1 smoke: ALL DESTROYER CLAUSES PASS")
    for row in rows:
        schema = row.get("selected_schema")
        print(f"  {row['task']:<30} {row['observed_status']:<9} " + (schema['pretty'] if schema else "<none>"))


if __name__ == "__main__":
    main()
