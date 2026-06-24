# V24-PMLB Bridge Result — 529_pollen

**Horizon classification:** `SUPPORTED_EXTERNAL_INDICATION`

## Predictive layer

| Model | Folds | Mean OOF R² | Median | Minimum |
|---|---:|---:|---:|---:|
| ridge | 2 | 0.7436 | 0.7436 | 0.7124 |
| v23_baseline | 2 | -184.0499 | -184.0499 | -368.8746 |
| v23_1_refined | 2 | -211.0773 | -211.0773 | -422.9211 |
| v23_1_recommended | 2 | 0.7434 | 0.7434 | 0.7124 |

## Operator bridge

- Real promotions: **3 / 16**
- Median transformation coverage: **0.9667**
- Stable modes: **1**

| Mode | Promotion rate | Median empirical R² | Median model-consistency R² | Identity improvement | Pooled improvement |
|---|---:|---:|---:|---:|---:|
| RIDGE_plus | 1.0000 | 0.4873 | 0.9577 | 0.4491 | 0.3473 |

## Destroyer controls

- Pair-shuffle false promotions: **0 / 16**
- Target-shuffle false promotions: **0 / 16**
- Combined false-promotion rate: **0.0000**

## Conversion statement

At least one constant perturbation mode survives empirical, inference, compression, stability, and null-control tests across folds. This indicates that v24-style operator promotion survives a non-native iid tabular setting; it does not establish causality or physical recursion.

This result is a trajectory marker. The dataset is synthetic and iid; promoted operators are support-bounded empirical rewrite schemas, not causal laws.
