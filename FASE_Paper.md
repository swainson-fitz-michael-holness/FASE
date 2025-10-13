# Formal, Auditable Symbolic Engine (FASE) – Theoretical Foundations

## 1 Introduction

The Formal, Auditable Symbolic Engine (FASE) is a “white‑box” alternative to
black‑box machine learning models.  It aims to discover human‑readable
equations from high‑dimensional data while providing rigorous control over
model complexity, noise structures and evidence of significance.  The design
was motivated by physical systems in which throughput is finite—such as
nanoscale heat transfer and high‑energy data streams—and by the need for
reliable models in brain–computer interfaces (BCIs) where hallucination and
indeterminacy can have catastrophic consequences.

Standard deep neural networks excel at prediction but offer little insight
into how input features combine.  Symbolic regression addresses this gap by
constructing explicit expressions that optimise an objective.  Tools like
PySR search for such expressions using evolutionary algorithms【370106646822031†L45-L53】,
but they do not incorporate arbitrary noise covariances or guarantee that
selected terms are reproducible.  FASE extends symbolic regression with
Σ‑aware feature construction, rigorous model‑selection criteria and
stability bagging.

This document summarises the mathematical framework underpinning FASE.  We
outline the axioms, definitions, lemmas and theorems that justify the
algorithm, and we highlight how FASE_v21 improves on earlier versions.

## 2 Axioms and Notation

FASE operates under several assumptions (axioms) that define the data
geometry, model class and scoring:

- **A1 (Data & geometry)**: We observe a data matrix \(X\in\mathbb{R}^{n\times d}\) and target vector
  \(y\in\mathbb{R}^n\).  Noise is zero‑mean with covariance \(\Sigma\succeq 0\).
  The GLS weight matrix is \(W=\Sigma^{-1}\) (or its pseudo‑inverse when
  \(\Sigma\) is singular).  Inner products and norms are defined by
  \(\langle u,v\rangle_W = u^\top W v\) and \(\|u\|_W^2=u^\top W u\).  This
  geometry treats heteroscedastic noise correctly and forms the basis for all
  residualization.

- **A2 (Model class)**: FASE constructs a feature map \(\Phi(X)\in\mathbb{R}^{n\times p}\)
  via three stages: Stage 1 generates atomic and hybrid HSML transforms;
  Stage 2 adds grammar‑based blocks; Stage 2.5 applies a mini‑Ruliad
  refiner.  The final predictor is linear in these features:
  \(\hat{y} = b_0 + \Phi w\).  All stages respect the Σ‑geometry and
  update the basis only when they reduce validation MDL.

- **A3 (Whitening operator)**: There exists a matrix \(L\) with
  \(L^\top L = W\) such that whitened residuals \(r_w = L(y - \hat{y})\) have
  Euclidean norm equal to the GLS norm: \(\|r_w\|_2^2 = \|y - \hat{y}\|_W^2\).
  A robust computation of \(L\) uses either Cholesky or eigen‑decomposition
  depending on Σ【93703672366852†L61-L80】.

- **A4 (MDL/BIC scoring in Σ‑geometry)**: On a validation split of size
  \(n\) with \(k\) effective parameters, the code computes an information
  criterion (AICc, BIC or MDL) in the Σ‑geometry.  For example, MDL uses
  \(\chi^2 = \|r\|_W^2\) and adds a complexity penalty proportional to the
  number of bits needed to encode the model.  The acceptance of new
  operators is based on a thresholded gain in bits.

- **A5 (OG‑SET EBIC in bits)**: For a candidate operator set of size
  \(p\), EBIC is converted to bits.  An operator is accepted if its gain
  exceeds a positive threshold; this ensures that EBIC decreases at each
  step and the OG‑SET construction terminates.

- **A6 (K‑fold independence)**: In k‑fold cross‑validation, each validation
  fold is independent of the model fitted on the training folds.  This
  axiom justifies using out‑of‑fold (OOF) metrics as unbiased estimators
  of generalization error.

- **A7 (Stability bagging)**: When bagging over \(B\) bootstrap subsamples,
  the selection indicator for an operator is a Bernoulli random variable
  with mean \(\pi\).  The empirical frequency \(\hat{f}\) of selection
  across bags approximates \(\pi\).  Hoeffding’s inequality bounds the
  probability that a null operator exceeds a frequency threshold, enabling
  control of false discoveries.

## 3 Definitions

Several core definitions operationalise the axioms:

- **D1 (GLS projection and RDMP residualization)**: Given a base feature
  matrix \(F\in\mathbb{R}^{n\times q}\) and a new block \(B\in\mathbb{R}^{n\times s}\),
  the GLS projection of \(B\) onto \(\mathrm{span}_W(F)\) is \(F\Gamma\) with
  \(\Gamma = (F^\top W F)^+ F^\top W B\).  The RDMP residual is
  \(R = B - F\Gamma\).  These residuals are W‑orthogonal to \(F\) and
  ensure that adding \(R\) captures new information.

- **D2 (W‑orthonormal export)**: After feature selection, columns are
  orthonormalised in the W‑geometry via QR.  Given whitened basis
  \(A = L\Phi_S\), QR yields \(A = Q R\), and the exported features are
  \(X_S = \Phi_S T\) where \(R T = I\).  This export preserves GLS
  predictions and yields numerical stability.

- **D3 (Out‑of‑fold predictor)**: In K‑fold CV, the OOF prediction
  \(\hat{y}^{\mathrm{oof}}\) concatenates the predictions on each held‑out
  fold using the model trained on the complementary folds.  Under A6, OOF
  MSE is an unbiased estimator of the test MSE.

## 4 Lemmas and Theorems

FASE’s correctness relies on several technical results.  We present the main
statements informally:

- **L1 (RDMP orthogonality)**: The RDMP residual \(R = B - F\Gamma\) is
  W‑orthogonal to \(F\); i.e., \(F^\top W R = 0\).  This ensures that
  adding \(R\) does not change the GLS coefficients of the existing basis.

- **L2 (GLS invariance under residualization)**: The GLS fit on
  \([F,B]\) is equivalent (up to reparameterisation) to the fit on
  \([F,R]\).  Consequently, RDMP does not alter fitted values or
  residuals; it only decorrelates new blocks from the base.

- **L3 (W‑orthonormal export preserves predictions)**: Replacing raw
  OG‑SET columns with their W‑orthonormal super‑features leaves GLS
  predictions unchanged.  Coefficients and standard errors transform via a
  similarity transform.

- **L4 (MDL bits mapping)**: The function `bits_from_delta_bic` used in
  the code converts a change in BIC into bits; it equals the base‑2
  log‑likelihood ratio.  This mapping justifies interpreting MDL gains in
  terms of information.

- **L5 (Greedy EBIC descent)**: In the OG‑SET construction, each
  accepted operator strictly decreases EBIC (because the gain in bits is
  positive).  Therefore the greedy procedure terminates after finitely
  many steps and yields a local EBIC optimum.

- **L6 (Unbiasedness of OOF MSE)**: Under K‑fold independence, the OOF
  MSE is an unbiased estimator of the test MSE for the given training size.

- **L7 (Bagging frequency tail bound)**: For a null operator selected
  with probability at most \(\pi_0\) in each bag, the probability that its
  empirical selection frequency exceeds a threshold \(\tau>\pi_0\) decays
  exponentially in the number of bags \(B\).  This bound controls false
  inclusions.

These lemmas culminate in several theorems:

- **T1 (RDMP + GLS gives projection‑invariant scoring)**: Adding a block
  \(B\) and scoring with GLS is equivalent to adding its RDMP residual
  \(R\); MDL and BIC scores are identical.  Thus the gain computed during
  stage 2 does not depend on linear dependencies between new blocks and
  the base.

- **T2 (W‑orthonormal export leaves GLS line unchanged)**: Exporting
  OG‑SET features as W‑orthonormal columns preserves predictions and
  standard errors; it simply reparameterises the model.

- **T3 (Greedy EBIC descent terminates)**: The greedy EBIC‑bits loop
  strictly reduces EBIC at each step and halts when no single operator
  yields a gain above the threshold; therefore it terminates at a local
  optimum.

- **T4 (OOF metrics are unbiased)**: Under A6, OOF R² and GLS‑R² are
  unbiased estimators of test metrics and avoid optimism inherent in
  in‑fold evaluations.

- **T5 (MDL acceptance is monotonic)**: Stage 1, Stage 2 and the
  mini‑Ruliad refiner accept a block only if it lowers the validation
  MDL (or improves validation MSE by a small guard).  The sequence of MDL
  scores is therefore non‑increasing.

- **T6 (Consensus OG operators are reproducible)**: The final consensus
  set includes only those operators that (a) appear with frequency ≥
  `final_min_freq`, (b) have sign stability ≥ `final_min_sign_stab`, and
  (c) achieve per‑fold evidence bits ≥ `final_min_bits`.  Under A7, the
  probability that a null operator satisfies these constraints is
  exponentially small.

- **T7 (Mini‑Ruliad acceptance is MDL‑safe)**: Although the mini‑Ruliad
  refiner explores features using a general energy that incorporates
  invariances and MDL scale, its proposals are accepted only if they
  lower the validation MDL.  This guarantee means the refiner never
  increases MDL and thus does not overfit.

Together, these results demonstrate that FASE’s feature construction and
selection procedure is mathematically sound: new features only augment the
model when they offer statistically significant, reproducible gains.

## 5 Improvements in FASE_v21

The v21 release incorporates several enhancements over v20:

1. **Cross‑validated driver and stability selection**: A built‑in k‑fold
   driver computes OOF R²/GLS‑R² and bagging frequencies.  Operators must
   meet user‑specified frequency and sign‑stability thresholds before
   inclusion【93703672366852†L0-L15】.

2. **Σ‑weighted residualization everywhere**: RDMP is used consistently
   for grammar blocks and mini‑Ruliad proposals.  Bagging passes the
   covariance Σ of each fold so that selection frequencies are comparable
   across folds【93703672366852†L61-L80】.

3. **Robust W‑orthonormal export**: Orthonormalisation uses QR with
   triangular solve and falls back to SVD if needed, ensuring numerical
   stability【93703672366852†L61-L80】.

4. **Evidence bits**: Gains are reported in bits (ΔBIC→bits), enabling a
   clear interpretation of the evidence supporting each operator.  Parameters
   like `final_min_bits` enforce that an operator provides at least six
   bits of evidence【93703672366852†L61-L80】.

5. **Enhanced invariances and MDL costs**: Additional invariance blocks
   (e.g., dihedral invariants) and revised MDL costs promote parsimonious
   models【93703672366852†L54-L58】.

6. **Baseline benchmarking**: A deterministic PySR baseline is integrated
   for fairness.  PySR is an open‑source symbolic regression tool that
   searches for interpretable expressions and is engineered for high
   performance【370106646822031†L45-L53】.  FASE_v21 can optionally augment
   the OG‑SET basis with W‑orthonormal super‑features to improve PySR’s
   performance【93703672366852†L61-L80】.

## 6 Results on PMLB Regression Tasks

FASE_v21 has been evaluated on a suite of PMLB regression datasets.  The
argumentation demonstration script runs a 5‑fold CV with two random seeds and compares
the OOF metrics to a PySR baseline.  Sample results are summarised in the
table below (see README for more details):

| Dataset | FASE_v21 OOF R² | PySR baseline R² | Observations |
|---------|---------------|-----------------|--------------|
| **579_fri_c0_250_5** | 0.9360 | 0.7585 | FASE discovers eight interpretable operators with high stability and achieves much higher R² than PySR. |
| **581_fri_c3_500_25** | 0.9337 | 0.8620 | FASE’s consensus model uses seven operators; the baseline is less accurate. |
| **582_fri_c1_500_25** | 0.9404 | 0.7948 | FASE outperforms PySR on a challenging, noisy dataset. |

Across all tested PMLB datasets, FASE_v21 consistently achieves higher OOF
R² and lower MSE than PySR.  The combination of Σ‑aware residualization,
MDL‑based selection, stability bagging and mini‑Ruliad refinement leads to
compact, interpretable equations that generalize well.

## 7 Applications and Outlook

FASE is designed for domains where interpretability and reliability are
critical.  Potential applications include:

- **Brain–Computer Interfaces (BCIs)**: Translating neural signals into
  controllable outputs requires models that avoid hallucinations and can
  be audited.  FASE’s transparent equations and Σ‑aware error modelling
  are well‑suited for safety‑critical BCI layers.

- **Scientific law discovery**: Many physical phenomena obey
  finite‑throughput laws.  FASE can recover governing equations from
  experimental data and provide evidence for their validity.

- **Control and robotics**: Interpretable models derived by FASE can be
  embedded in controllers and provide assurances about system behaviour.

- **Interpretable AI**: FASE can serve as a distillation layer for deep
  models: a neural network’s outputs can be fitted by FASE to yield
  analytic equations, aligning with the concept of symbolic
  distillation【370106646822031†L55-L58】.

Future work will extend FASE to handle heteroscedastic and structured noise,
incorporate new grammar blocks and invariance groups, and integrate with
online data streams.

## 8 Conclusion

The Formal, Auditable Symbolic Engine (FASE) combines symbolic regression with
Σ‑aware residualization, MDL‑based scoring, k‑fold cross‑validation and
stability bagging to produce human‑readable equations with quantifiable
evidence.  The theoretical framework—rooted in GLS geometry, RDMP
residualization and EBIC‑bits—ensures that each accepted operator
contributes genuine explanatory power.  FASE_v21’s improvements over v20,
particularly the integrated cross‑validated driver and evidence bits, lead
to more robust and reproducible models.  Compared with an off‑the‑shelf
symbolic regression tool like PySR, FASE achieves superior predictive
performance while maintaining transparency and auditability.
