# Formal, Auditable Symbolic Engine (FASE)

## Overview

FASE (Finite‑throughput Analysis with Σ‑aware enhancement) is an interpretable
symbolic regression pipeline designed to produce human‑readable equations from
high‑dimensional, noisy data.  It implements a “white‑box” approach to AI:
instead of fitting a black‑box neural network, FASE constructs a transparent
linear model on top of a rich, hand‑selected basis of nonlinear features.  The
architecture draws inspiration from capacity‑limited physical laws in
nanoscale and high‑energy systems and aims to provide robust, auditable
models for safety‑critical domains like brain–computer interfaces (BCIs),
control systems and scientific discovery.  FASE transforms messy inputs into
clear equations that can be inspected, validated and trusted.

## Architecture and Key Features

FASE builds a linear model in three main stages.  Each stage is Σ‑aware,
meaning that it takes into account a general noise covariance matrix Σ by
using Generalized Least Squares (GLS) and residualization.

1. **Stage 1 – Atomic Alphabet**: A candidate “alphabet” of feature
   transformations is constructed from the input variables.  Operations include
   polynomials (e.g., \(x\), \(x^2\), \(x^3\)), nonlinear activation
   functions (ReLU, GELU, tanh, ELU) and handcrafted hybrid symbolic‑machine
   learning (HSML) projections.  The code evaluates each candidate using
   MDL/BIC criteria in the Σ‑geometry and retains those with sufficient gain.

2. **Stage 2 – Grammar & RDMP**: Grammar blocks (bilinear, trilinear and
   invariance structures) are added on top of the atoms.  Before fitting,
   new blocks are residualized against the existing basis using RDMP
   (Σ‑weighted residualization), ensuring that newly added operators capture
   unexplained variance orthogonally to existing features.  Scoring uses a
   weighted MDL/BIC criterion and accepts blocks that yield MDL improvement.

3. **Stage 2.5 – Mini‑Ruliad Refiner**: An evolutionary “mini‑Ruliad” search
   explores combinations of features beyond the grammar.  It uses a small
   energy‑guided search to propose new feature candidates and accepts them
   only when they reduce validation MDL.  This stage brings additional
   expressivity while guaranteeing monotonic MDL descent.

4. **OG‑SET and GLS linear fit**: After feature construction, a GLS fit
   is performed.  A greedy EBIC‑bits step selects a minimal set of operators
   (“OG‑SET”) that yields the best trade‑off between goodness‑of‑fit and
   complexity.  Operators are exported as W‑orthonormal “super‑features” to
   ensure numerical stability.  Stability‑selection bagging and k‑fold
   cross‑validation are used to retain only operators that appear with high
   frequency and consistent sign across folds.

5. **Comparison with PySR**: FASE includes optional integration with
   PySR for baseline comparison.  PySR is an open‑source symbolic regression
   library that searches for interpretable expressions using evolutionary
   algorithms【370106646822031†L45-L53】.  This integration allows users to
   benchmark FASE’s transparent, Σ‑aware approach against a black‑box
   symbolic regression engine.

### Mathematical Foundations

FASE’s scoring and selection procedures are grounded in the following
axioms and definitions:

- **Data and geometry (A1)**: Observations \((X,y)\) are assumed to have
  zero‑mean noise with covariance \(\Sigma\).  GLS weights
  \(W = \Sigma^{-1}\) induce an inner product \(\langle u,v\rangle_W =
  u^\top W v\) and norm \(\|u\|_W^2 = u^\top W u\).

- **Model class (A2)**: The final predictor is linear in the constructed
  feature map \(\Phi(X)\).  Feature construction proceeds through atomic
  transforms, grammar blocks and mini‑Ruliad refiners【93703672366852†L0-L15】.

- **Whitening operator (A3)**: A whitening matrix \(L\) satisfies
  \(L^\top L = W\) so that whitened residuals \(r_w = L(y - \hat{y})\) have
  Euclidean norm equal to the GLS norm.  This underlies RDMP and the GLS
  fit.

- **MDL/BIC scoring (A4)**: Candidate operators and blocks are scored using
  Minimum Description Length/Bayesian Information Criterion in Σ‑geometry;
  evidence gains are converted to bits.

The full theoretical development, including lemmas and theorems showing
that RDMP residualization yields projection‑invariant scoring and that
greedy EBIC‑bits descent terminates, is provided in the accompanying paper.

## FASE v20 vs FASE_v21

FASE_v21 is a significant upgrade over the earlier `FASE.py` (v20).  At a
high level, both versions implement the atomic/grammar/Ruliad stages
described above, but v21 introduces robust cross‑validation, stability
selection and Σ‑weighted residualization that address weaknesses in v20.
The key differences are:

| Aspect | FASE (v20) | FASE_v21 |
|-------|-------------|-----------|
| **Cross‑validation** | Basic k‑fold support with optional stability bagging disabled by default. | Built‑in k‑fold CV driver with out‑of‑fold (OOF) R²/GLS‑R² metrics, operator stability statistics and consensus operator selection【93703672366852†L0-L15】. |
| **Σ‑weighted residualization** | Whiten helper (`chol_whiten`) used in some places; residualization may rely on ordinary least squares. | RDMP (`fit_residualization_gls`) performs Σ‑weighted residualization everywhere; bagging passes fold‑specific Σ so that selection frequencies are GLS‑consistent【93703672366852†L61-L80】. |
| **OG‑SET & bagging** | Optional OG‑SET driver requires external CSV inputs; no built‑in bagging. | OG‑SET bagging integrated: `bag_boots` bootstraps (default 8) and `bag_frac` sampling fraction (0.8) are used to assess operator stability; operators must appear in ≥60 % of bags and have consistent sign to be included【93703672366852†L61-L80】. |
| **W‑orthonormal export** | Uses a simple Cholesky‑based whitening; orthonormal export is optional. | Robust W‑orthonormal export using QR + triangular solve (SVD fallback) ensures that exported super‑features are orthonormal under the Σ‑inner product【93703672366852†L61-L80】. |
| **Evidence accounting** | MDL/AIC/BIC scoring only; no clear notion of evidence in bits. | Evidence gains are converted to bits via ΔBIC→bits; thresholds (e.g., `final_min_bits=6.0`) enforce a minimum amount of evidence before an operator is retained【93703672366852†L61-L80】. |
| **Operator invariances** | Supports bilinear, group, dihedral invariances; costs differ. | Extends invariance blocks and adjusts MDL costs (dihedral/group invariants cost 8–10 bits)【93703672366852†L54-L58】. |
| **Baseline comparison** | Optionally runs PySR with default settings. | Integrates PySR baseline with deterministic search to allow apples‑to‑apples comparison【93703672366852†L61-L89】. |

In short, FASE_v21 emphasises rigorous Σ‑aware model selection, stability
bagging and clear evidence thresholds.  These improvements yield more
reproducible and generalizable models compared with the earlier v20.

## Installation and Usage

1. **Prerequisites**
   - Python ≥ 3.9 with `numpy`, `dataclasses`, `pmlb` (for PMLB datasets),
     and optionally `pysr` for the baseline.
   - Clone the repository or download the source.  Create a virtual
     environment and install dependencies:

   ```bash
   python3 -m venv venv
   source venv/bin/activate
   pip install -r requirements.txt
   pip install pmlb pysr    # optional but recommended
   ```

2. **Running FASE_v21**
   
   The script `FASE_v21.py` runs a k‑fold demonstration on bundled PMLB
   datasets.  From the project root:

   ```bash
   python FASE_v21.py
   ```

   The configuration at the top of the file specifies seeds, datasets and
   maximum number of atoms/grammar blocks【93703672366852†L27-L44】.  To test
   your own data, either:
   - replace the dataset list in the `CONFIG` dict with your own `X` and
     `y` arrays; or
   - prepare an OG‑SET basis (`Yb_csv`, `y_csv`, `Sigma_json`) and set
     `OGSET["enable"]` to `True`.

   During execution the script reports the operators accepted at each
   stage, MDL/val‑MSE improvements and final R²/GLS‑R².  If
   `COMPARE_WITH_PYSR` is `True` then a PySR baseline model is fit for
   comparison.

3. **Running FASE (v20)**
   
   The older `FASE.py` uses a single dataset list and can be run with:

   ```bash
   python FASE.py
   ```

   It produces similar logs but lacks k‑fold CV, stability bagging and
   evidence bits.  For reproducible results we recommend using FASE_v21.

4. **Interpreting the output**
   
   - **OG ops**: The “OG ops” listed under each fold are the operators
     selected by the OG‑SET.
   - **Blocks**: Indicate whether additional grammar blocks (e.g., `ruliad`,
     `perm_invar`, `dihedral_invar`) were injected.
   - **Final R² and GLS‑R²**: Report the fit on the validation fold; OOF
     metrics are summarised at the end.
   - **Operator stability**: Shows the frequency and sign stability of each
     operator across bags and folds.  Operators must meet minimum
     frequency (`final_min_freq`) and sign stability (`final_min_sign_stab`)
     thresholds to be included.

## Empirical Comparison with PySR

The following table compares FASE_v21’s performance to the PySR baseline on
representative PMLB regression datasets.  FASE_v21 consistently achieves
higher R² and lower MSE than PySR, illustrating the benefits of Σ‑aware
residualization, bagging and evidence‑based model selection.

| Dataset | FASE_v21 OOF R² | FASE_v21 OOF MSE | PySR baseline R² | Remarks |
|---------|---------------|-----------------|-----------------|---------|
| **579_fri_c0_250_5** | 0.9360 | 0.0638 | 0.7585 | FASE_v21’s mini‑Ruliad refiner and OG‑SET bagging produce a compact model with eight operators; PySR’s expression has noticeably lower predictive power. |
| **581_fri_c3_500_25** | 0.9337 | 0.0661 | 0.8620 | Σ‑weighted residualization and stability selection reduce overfitting; FASE_v21’s consensus model uses seven OG operators with consistent signs. |
| **582_fri_c1_500_25** | 0.9404 | 0.0595 | 0.7948 | FASE_v21 outperforms PySR by a large margin; the baseline struggles with noisy high‑dimensional data. |

The baseline R² values above were obtained using PySR’s deterministic
configuration (`niterations=250`, `maxsize=18`, `maxdepth=8`)【93703672366852†L61-L89】.

## Roadmap and Applications

FASE aims to become a general‑purpose layer connecting symbolic regression
to safety‑critical systems.  Future releases will include:

- Support for custom covariance structures and heteroscedastic noise models.
- Integration with real‑time BCI data streams.
- User‑defined invariance groups and grammar extensions.
- Further improvements to the Ruliad search and parameter optimization.

FASE is particularly well‑suited to applications where interpretability and
reliability are paramount—scientific law discovery, control of dynamical
systems, anomaly detection and transparent AI.

For more details on the theoretical foundations and proofs of correctness,
see the accompanying paper (`FASE_Paper.md`).
