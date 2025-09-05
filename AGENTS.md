# FASE v21 Architecture Documentation (for AI Agents)

## 1. Overview

**FASE (Forward Atomic Search Engine)** is a Python-based system for automated scientific discovery and symbolic regression. Its primary function is to discover a parsimonious, predictive model that explains a target variable `y` from a set of input features `X`, while rigorously respecting the data's error structure (`Sigma`).

The system's philosophy is grounded in **Observer-Gauge Structural Effective Theory (OG-SET)**, which prioritizes **parsimony**, **covariance realism**, and the discovery of **stable, invariant structures**.

## 2. Core Philosophy & Implementation

FASE translates the OG-SET axioms into specific algorithmic components:

* **Parsimony Prior**: The system uses information-theoretic criteria like **MDL (Minimum Description Length)** and **eBIC (Extended Bayesian Information Criterion)** as its primary objective function. New features or operators are only added if they reduce the total description length of the model (i.e., they provide a more compressed explanation of the data). This is implemented in `crit_from_residuals` and `_ebic_bits`.

* **Covariance Realism**: All core fitting and evaluation routines are **GLS-aware** (Generalized Least Squares). They accept a covariance matrix `Sigma` and perform calculations in a whitened space. This ensures that features are judged based on their ability to explain data in directions of high certainty, not just fitting noise. Key functions include `fit_residualization_gls` and `build_ogset`.

* **Structural Realism**: FASE operates by building a basis set of features (an "Atomic Alphabet") in which the relationship to the target becomes simple (linear). This discovers the underlying structure of the data. This is the primary goal of `forward_atomic_search` and the `build_ogset` module.

* **Stability & Invariance**: The system emphasizes the discovery of operators that are **stably selected** across different subsets of the data. The `run_fase_kfold` driver and the internal bagging (`bag_boots`) within `build_ogset` are designed to identify and report on these robust features.

## 3. Architectural Breakdown

FASE v21 uses a multi-stage pipeline within each fold of a K-fold cross-validation process:

1.  **Stage 1: Atomic Alphabet Generation (`build_stage1_candidates`, `forward_atomic_search`)**
    * A massive library of candidate features ("atoms") is generated from the raw inputs `X`. This includes unary transformations (`sin(x)`, `x^2`), random projections, and binary combinations.
    * A greedy forward selection algorithm iteratively adds the single best atom that maximally improves the MDL score. This creates a small, powerful, but potentially suboptimal basis set.

2.  **OG-SET Operator Selection (`build_ogset`)**
    * This is the core discovery module. It takes the *entire* atomic alphabet (thousands of candidates) as a potential "dictionary of physics."
    * It performs its own greedy forward search, guided by eBIC, to find the most evidentially supported subset of operators.
    * It uses internal bagging (stability selection) to report the selection frequency of each chosen operator, prioritizing those that are robustly selected.
    * The operators selected by this module are considered the primary scientific discovery.

3.  **Model Augmentation and Refinement (`Stage 2`, `Stage 2.5`)**
    * The features from the initial greedy search and the stable OG-SET operators are combined.
    * **Stage 2 (`stage2_grammar_search`)**: Optionally adds more complex, non-linear "grammar blocks" (e.g., bilinear interactions) if they further improve the MDL score.
    * **Stage 2.5 (`stage2p5_ruliad_search`)**: Optionally runs a sophisticated, non-greedy evolutionary search ("Mini-Ruliad") to find complex feature combinations that the previous stages may have missed.

4.  **Final Model (`ridge_with_intercept`)**
    * The final model is a simple, regularized linear regression (Ridge) fit on the complete set of discovered and refined features. This model is also fit using GLS if `Sigma` is provided.

## 4. Key Components & Entry Points

* **Main Entry Point**: `run_fase_kfold(X, y, Sigma, ...)` is the primary function. It manages the cross-validation, collects out-of-fold predictions, and computes final stability metrics.
* **Core Discovery Engine**: `build_ogset(...)` is the central function for identifying stable operators from the atomic alphabet.
* **GLS-Aware Residualization**: `fit_residualization_gls(...)` is the key utility that makes the feature construction process `Sigma`-aware.
* **Configuration**: All behavior is controlled by the global `CONFIG` dictionary. The `OGSET` sub-dictionary contains the most important knobs for scientific discovery.

## 5. Inputs and Outputs

* **Inputs**:
    * `X`: `(n_samples, n_features)` numpy array of input features.
    * `y`: `(n_samples,)` numpy array of the target variable.
    * `Sigma`: `(n_samples, n_samples)` or `(n_samples,)` numpy array representing the covariance of the noise in `y`. If `None`, assumes identity (OLS).

* **Outputs**:
    * **Out-of-Fold (OOF) Predictions**: A vector of predictions for the entire dataset, where each sample was predicted by a model that was not trained on it.
    * **OOF Metrics**: R² and MSE calculated from the OOF predictions.
    * **Operator Stability Report**: A list of OG-SET operators and their selection frequency across the K folds. **This is the primary output for scientific interpretation.**
    * **Final Consensus Model Rules**: The `CONFIG["OGSET"]` dictionary contains rules (`final_min_freq`, `final_min_bits`) that can be used to select a final consensus model from the stability report.

## 6. How to Use (Example)

```python
import numpy as np
# from FASE_v21_kfold import run_fase_kfold, make_synthetic

# 1. Load or generate your data
# This synthetic data has a correlated error structure (a non-diagonal Sigma)
X, y, Sigma = make_synthetic(n=500, d=10, gls_noise=True)

# 2. Run the K-fold analysis
# The system will automatically use the Sigma matrix for all internal calculations
results = run_fase_kfold(X, y, Sigma=Sigma, K=5, seed=42)

# 3. Interpret the results
print(f"\nFinal OOF R^2: {results['R2_oof']:.4f}")
print("Most stable operators discovered:")
stable_ops = {k: v for k, v in results['og_stability'].items() if v >= 0.6} # 60% stability threshold
for op, freq in sorted(stable_ops.items(), key=lambda t: -t[1]):
    print(f"  - {op} (found in {freq:.0%} of folds)")
```