# V24-PMLB Bridge Result — 529_pollen

**Horizon classification:** `SUPPORTED_EXTERNAL_INDICATION`

## Predictive layer

| Model | Folds | Mean OOF R² | Median | Minimum |
|---|---:|---:|---:|---:|
| ridge | 3 | 0.7915 | 0.7818 | 0.7758 |
| v23_baseline | 3 | 0.7784 | 0.7793 | 0.7577 |
| v23_1_refined | 3 | 0.7782 | 0.7792 | 0.7577 |
| v23_1_recommended | 3 | 0.7814 | 0.7804 | 0.7602 |

## Operator bridge

- Real promotions: **15 / 24**
- Median transformation coverage: **0.9725**
- Stable modes: **6**

| Mode | Promotion rate | Median empirical R² | Median model-consistency R² | Identity improvement | Pooled improvement |
|---|---:|---:|---:|---:|---:|
| NUB_minus | 0.6667 | 0.4766 | 0.9485 | 0.2282 | 0.0383 |
| NUB_plus | 0.6667 | 0.6104 | 0.9257 | 0.2481 | 0.0606 |
| RIDGE_minus | 1.0000 | 0.6495 | 0.9408 | 0.4276 | 0.2620 |
| RIDGE_plus | 1.0000 | 0.6344 | 0.9615 | 0.4291 | 0.3264 |
| WEIGHT_minus | 1.0000 | 0.5297 | 0.9432 | 0.3246 | 0.1385 |
| WEIGHT_plus | 0.6667 | 0.5297 | 0.9442 | 0.3289 | 0.0777 |

## Destroyer controls

- Pair-shuffle false promotions: **0 / 24**
- Target-shuffle false promotions: **0 / 24**
- Combined false-promotion rate: **0.0000**

## Conversion statement

At least one constant perturbation mode survives empirical, inference, compression, stability, and null-control tests across folds. This indicates that v24-style operator promotion survives a non-native iid tabular setting; it does not establish causality or physical recursion.

This result is a trajectory marker. The dataset is synthetic and iid; promoted operators are support-bounded empirical rewrite schemas, not causal laws.
