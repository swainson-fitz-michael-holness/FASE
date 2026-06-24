# FASE-v24 PMLB Bridge Probe — 529_pollen

## Status

This is a **conversion-trajectory probe**, not a new generality gate. Its purpose is to breach the native locality of v24's synthetic typed-transition tests and record what survives when the input is an ordinary iid regression table.

The dataset is external to the FASE development suite but is itself synthetic. It contains 3,848 observations, four numeric predictors (`RIDGE`, `NUB`, `CRACK`, `WEIGHT`), and the regression target `DENSITY`/`target`.

## Claim under test

Given an iid table and a predictive model \(\hat f\), can v24 construct a support-aware perturbation graph and promote a reusable operator

\[
\omega_g:\operatorname{State}(x,y)
\longrightarrow
\operatorname{State}(g(x),\psi_g(x,y))
\]

for a repeated constant operation

\[
g_{j,s}(x)=x+\delta_{j,s}e_j,
\]

such that the operator:

1. transfers to untouched empirical neighbor transitions;
2. improves on the identity transition;
3. improves on a pooled, non-modal operator;
4. agrees with the frozen predictive model under a support-safe counterfactual perturbation;
5. is stable and compressive;
6. disappears under target and pair shuffling.

The commutative-square audit is

\[
\hat f(gx)\approx\omega_g(\hat f(x),x).
\]

This is a model-consistency and structural-transfer test. It is **not** evidence that the perturbation is causal or that the rows form a physical time series.

## Construction

### Predictive layer

Every outer fold reports:

- raw-feature Ridge;
- original v23 coordinate branch;
- v23.1 joint-refined branch;
- a non-degrading v23.1 recommendation formed as an inner-CV convex blend with the raw-feature Ridge anchor;
- optional PySR under the same outer fold.

The original v23 and refined branches are never erased. Numerical instability remains visible in the report. The recommended branch may fall back to the raw-feature anchor when a symbolic branch produces non-finite or extreme predictions.

### Empirical perturbation graph

Within each outer-training fold:

1. features are standardized using training statistics only;
2. a k-nearest-neighbor graph is formed;
3. each edge is assigned to the coordinate with the dominant standardized displacement;
4. direction and median step define one of eight candidate modes:
   `RIDGE±`, `NUB±`, `CRACK±`, `WEIGHT±`;
5. one representative edge per source is retained near the mode's constant step.

The same learned step is transferred to the untouched outer-test graph. Geometry checks require the mode to remain axis-dominant and the test step to remain close to the training step.

### Operator language

The numerical rewrite schema is generated from a small typed meta-language:

- identity: \(y'=y\);
- shift: \(y'=y+b\);
- affine output: \(y'=ay+b\);
- context shift: \(y'=y+b+\beta^Tx\);
- context affine: \(y'=ay+b+\beta^Tx\).

Family selection uses grouped inner validation, complexity cost, and optional active perturbation queries. The selected family is then refit on all training transitions.

### Active perturbation

Candidate operators are applied to support-safe counterfactual states. The system queries the frozen v23.1 predictor at states where the candidate operators disagree most and uses this evidence only to discriminate candidate operator families. Untouched empirical outer-test transitions remain the primary validation evidence.

### Destroyer controls

- **Pair shuffle:** preserves the feature graph and target marginal distribution but destroys source-destination correspondence.
- **Target shuffle:** destroys the target relation and fits a fresh null predictor.
- **Identity destroyer:** a promoted operator must improve over carrying the source target unchanged.
- **Pooled destroyer:** a promoted modal operator must improve over one operator fitted across all modes.

## Promotion requirements

A mode is promoted only if all configured checks pass:

- train and test support;
- held-out empirical \(R^2\);
- improvement over identity;
- improvement over pooled transition modeling;
- counterfactual model consistency in both \(R^2\) and NRMSE;
- bootstrap family stability;
- description compression;
- support-safe counterfactual coverage;
- geometric dominance;
- step transfer.

## Horizon classification

The final report returns one of four outcomes:

- `SUPPORTED_EXTERNAL_INDICATION`: at least one mode promotes stably across folds, coverage is adequate, v23.1 remains non-degrading internally, and shuffled controls remain closed.
- `LOCAL_CANDIDATE_ONLY`: fold-local operators exist, but stability or coverage is insufficient.
- `ABSTENTION`: prediction may work, but the table does not license a reusable operator under the tested modes.
- `INVALIDATED_BY_NULL_PROMOTION`: a shuffled control promotes an operator; the bridge evidence is rejected.

## Full matrix

- dataset: `529_pollen`;
- seeds: 42, 1337, 2025, 9001;
- five outer folds per seed;
- full 3,848-row dataset;
- eight real perturbation modes per fold;
- pair-shuffle and target-shuffle controls;
- optional PySR with 1,000 iterations, max size 30, and 180 seconds per fold;
- atomic checkpoint after each fold.

The full run is 20 independent fold jobs. The shell runner launches each fold in a clean Python process to prevent mutable symbolic-search state or numerical-library state from leaking across folds.

## Conversion interpretation

A positive result indicates that the v24 operator-promotion paradigm survives contact with a non-native iid tabular representation. It does not show that the promoted operator is causal, chemically meaningful, or a universal recursive law.

A negative or abstaining result is also informative: it identifies the current horizon at which v23.1 remains predictive but v24 cannot honestly promote the table into a recursive generator.
