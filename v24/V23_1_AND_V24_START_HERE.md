# FASE v23.1 and v24 — Start Here

## v23.1: preserve v23, add a challenger

`fase_v23_1_joint_refined_gate.py` does not overwrite the original v23 search. It
runs a champion/challenger protocol:

- **baseline:** original v23 discrete coordinate discovery;
- **challenger:** the same topology followed by joint continuous constant refinement;
- both outer-fold predictions are retained and reported;
- refinement is selected only inside the outer-training data.

The fast regression check validates the diagnosed Viète coupling failure and a
non-degradation control:

```bash
python test_fase_v23_1_joint_refined.py
```

A fuller champion/challenger run is:

```bash
python fase_v23_1_joint_refined_gate.py \
  --test vieta \
  --n 160 \
  --json-out runs/v23_1_vieta.json
```

The original v23 files remain the authoritative baseline. The new wrapper is an
experimental challenger, not a destructive replacement.

## v24 Gate 0: modal recursive closure

Enter the `fase_v24` directory and run:

```bash
python test_fase_v24_gate0.py
```

Full report:

```bash
python fase_v24_modal_recursive_closure.py \
  --json-out runs/v24_gate0.json
```

The first Gate-0 lab tests:

- recovery of in-mode recursive generators;
- promotion of withheld square-root, sine, and Möbius operators;
- long-horizon rollout rather than one-step fit alone;
- active perturbations selected by model disagreement;
- symmetry/equivariance audits;
- affine coordinate-change conjugacy;
- abstention on a nonstationary process with hidden parity.

## Perturbation equations

For a licensed symmetry `g`:

\[
\widehat T(gx) \approx g(\widehat T(x)).
\]

For a general invertible coordinate change, re-infer on transformed data and test:

\[
\widehat T_g(gx) \approx g(\widehat T(x)),
\qquad T_g = g\circ T\circ g^{-1}.
\]

The latter is the general statement that applying a constant operation to the data
and applying the same operation to model inference should produce the same transformed
transition, up to estimation error.
