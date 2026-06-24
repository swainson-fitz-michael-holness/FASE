# FASE-v24 Gate 1 — Operator Genesis

## Claim

Repeated concrete anomalous transformations can be folded into a **new executable typed rewrite schema**

\[
\omega : L(\mathbf v) \to R(\mathbf v)
\]

without choosing \(\omega\) from a supplied operator bank, when the schema demonstrates:

1. typed variable abstraction;
2. held-out structural generalization;
3. active-perturbation consistency;
4. equivariance under licensed constant renamings/shifts;
5. bootstrap stability;
6. descriptive compression;
7. abstention on nonstationary, random, or identity controls.

## Representation

The substrate is a typed term tree. The learner uses typed anti-unification to compute a least-general generalization of repeated LHS and RHS transformations. Metavariables inferred on the LHS are linked to matching concrete subterm sequences on the RHS. Every RHS variable must be grounded by the LHS; otherwise the schema is non-executable and cannot be promoted.

Examples generated in the smoke test include:

```text
Mul(?k, Add(?x, ?y)) -> Add(Mul(?k, ?x), Mul(?k, ?y))
App(Lam(?n, Var(?n)), ?arg) -> ?arg
React(Alkene(?r1, ?r2), H2) -> Alkane(?r1, ?r2)
Express(Gene(?p, Coding(?s))) -> RNA(?s)
Pair(?x, ?y) -> Pair(?y, ?x)
```

These schemas are induced from concrete instances; their names and structures are not provided as candidate operators.

## Active perturbation

Bootstrap-induced schemas are treated as competing hypotheses. The system instantiates typed counterfactual LHS terms where the candidates disagree most, queries the synthetic oracle, adds the transformation, and re-induces the schema.

A separate commutative audit checks:

\[
\widehat\omega(gx)=g\widehat\omega(x)
\]

for licensed typed perturbations \(g\), including symbolic renaming and scalar translation.

## Destroyer clauses

Promotion is forbidden if any of the following holds:

- RHS contains an ungrounded variable;
- held-out exact structural accuracy is below threshold;
- counterfactual perturbation accuracy is below threshold;
- equivariance fails;
- bootstrap stability is inadequate;
- compression is insufficient;
- the transformation is identity;
- visible state does not determine a stationary rewrite;
- outputs are unrelated to visible inputs.

## Full matrix

The `full` matrix evaluates:

- 8 tasks;
- 4 seeds: 42, 1337, 2025, 9001;
- train sizes: 4, 8, 16, 32;
- corruption: 0, 0.05, 0.10, 0.20;
- active-query budgets: 0, 8, 32;
- 64 clean held-out transformations per condition.

Total: 1,536 conditions.

The run is checkpointed after every condition and is safe to resume.

## Proposed closure criteria

Gate 1 closes only if the full matrix demonstrates:

- null false-promotion rate \(\le 1\%\);
- positive promotion rate \(\ge 90\%\) for train size \(\ge 8\), corruption \(\le 10\%\), active budget \(\ge 8\);
- exact held-out and perturbation accuracy \(\ge 95\%\) on promoted schemas;
- no task-specific operator templates supplied to the learner;
- all promoted schemas executable and type-grounded;
- active perturbation does not increase false promotion.

Low-support/high-corruption conditions are boundary probes, not mandatory closure regions.

## Current evidence

The local smoke test closes 8/8 conditions with five correct promotions and three correct abstentions. A 192-condition quick run achieved:

- status accuracy: 99.48%;
- positive promotion rate: 99.17%;
- null false-promotion rate: 0%;
- one conservative false abstention at train size 6 with 10% corruption;
- zero false promotions.

This is pre-gate evidence only. The full matrix is authoritative.
