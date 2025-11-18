# PMLB Benchmark Suite for Comparing FASE vs PySR

This list selects PMLB datasets (drawn from the provided inventory) that
jointly stress symbolic-regression quality, stability under noise, detection of
interactions, and scalability.  Organize runs into the following modules so
results are defensible and repeatable.

## 1) Symbolic regression with known functional structure
Use these to show FASE’s Σ-aware MDL scoring recovers compact formulas and is
stable across folds.

- Friedman synthetic families: `560_bodyfat`, `564_fried`, `579_fri_c0_250_5`,
  `581_fri_c3_500_25`, `582_fri_c1_500_25`, `583_fri_c1_1000_50`,
  `584_fri_c4_500_25`, `586_fri_c3_1000_25`, `588_fri_c4_1000_100`,
  `589_fri_c2_1000_25`, `590_fri_c0_1000_50`, `591_fri_c1_100_10`,
  `592_fri_c4_1000_25`, `593_fri_c1_1000_10`, `594_fri_c2_100_5`,
  `595_fri_c0_1000_10`, `596_fri_c2_250_5`, `597_fri_c2_500_5`,
  `598_fri_c0_1000_25`, `599_fri_c2_1000_5`, `601_fri_c1_250_5`,
  `602_fri_c3_250_10`, `603_fri_c0_250_50`, `604_fri_c4_500_10`,
  `605_fri_c2_250_25`, `606_fri_c2_1000_10`, `607_fri_c4_1000_50`,
  `608_fri_c3_1000_10`, `609_fri_c0_1000_5`, `611_fri_c3_100_5`,
  `612_fri_c1_1000_5`, `613_fri_c3_250_5`, `615_fri_c4_250_10`,
  `616_fri_c4_500_50`, `617_fri_c3_500_5`, `618_fri_c3_1000_50`,
  `620_fri_c1_1000_25`, `621_fri_c0_100_10`, `622_fri_c2_1000_50`,
  `623_fri_c4_1000_10`, `624_fri_c0_100_5`, `626_fri_c2_500_50`,
  `627_fri_c2_500_10`, `628_fri_c3_1000_5`, `631_fri_c1_500_5`,
  `633_fri_c0_500_25`, `634_fri_c2_100_10`, `635_fri_c0_250_10`,
  `637_fri_c1_500_50`, `641_fri_c1_500_10`, `643_fri_c2_500_25`,
  `644_fri_c4_250_25`, `645_fri_c3_500_50`, `646_fri_c3_500_10`,
  `647_fri_c1_250_10`, `648_fri_c1_250_50`, `649_fri_c0_500_5`,
  `650_fri_c0_500_50`, `651_fri_c0_100_25`, `653_fri_c0_250_25`,
  `654_fri_c0_500_10`, `656_fri_c1_100_5`, `657_fri_c2_250_10`,
  `658_fri_c3_250_25`, `678_visualizing_environmental`.
- Other regression with interpretable structure or tabular physics: `1191_BNG_pbc`,
  `1193_BNG_lowbwt`, `1196_BNG_pharynx`, `1199_BNG_echoMonths`,
  `1201_BNG_breastTumor`, `1203_BNG_pwLinear`, `195_auto_price`,
  `197_cpu_act`, `201_pol`, `207_autoPrice`, `210_cloud`, `215_2dplanes`,
  `218_house_8L`, `225_puma8NH`, `227_cpu_small`, `228_elusage`,
  `229_pwLinear`, `230_machine_cpu`, `294_satellite_image`, `344_mv`,
  `503_wind`, `505_tecator`, `519_vinnie`, `522_pm10`, `542_pollution`,
  `547_no2`, `573_cpu_act`, `574_house_16H`.

## 2) Interaction-heavy and epistatic targets
FASE’s RDMP residualization should expose higher-order interactions; PySR may
struggle without carefully tuned operators.

- GAMETES interaction suites: `GAMETES_Epistasis_2_Way_1000atts_0.4H_EDM_1_EDM_1_1`,
  `GAMETES_Epistasis_2_Way_20atts_0.1H_EDM_1_1`,
  `GAMETES_Epistasis_2_Way_20atts_0.4H_EDM_1_1`,
  `GAMETES_Epistasis_3_Way_20atts_0.2H_EDM_1_1`,
  `GAMETES_Heterogeneity_20atts_1600_Het_0.4_0.2_50_EDM_2_001`,
  `GAMETES_Heterogeneity_20atts_1600_Het_0.4_0.2_75_EDM_2_001`.
- Parity/XOR-like structure: `parity5`, `parity5+5`, `mux6`, `corral`.
- Interaction-heavy but low-noise games: `threeOf9`, `mofn_3_7_10`,
  `cloud`, `tokyo1`, `ring`.

## 3) Logical/boolean rule learning
Highlights FASE’s ability to recover compact decision rules with Σ-aware
regularization.

- Game/logic boards: `tic_tac_toe`, `connect_4`, `kr_vs_kp`, `krkopt`, `chess`.
- Rule-based synthetic: `monk1`, `monk2`, `monk3`, `krkopt`, `parity5+5`.
- Simple discrete signals: `led7`, `led24`, `nursery`, `car`, `car_evaluation`,
  `balance_scale`, `vowel`.

## 4) Clean vs. noisy signal recovery
Pairs of datasets where the target structure is fixed but noise changes;
FASE’s GLS weighting should yield more stable operators.

- `Hill_Valley_with_noise` vs `Hill_Valley_without_noise`.
- `waveform_21` vs `waveform_40`.
- `satimage` (multi-class) and `shuttle` (class-imbalance + noise).
- `pendigits` vs `optdigits` (digit trajectories vs pixel grids).

## 5) High-dimensional, multivariate classification
Use to demonstrate scalability of OG-SET + RDMP vs PySR’s combinatorial search.

- Vision-like: `mnist`, `letter`, `optdigits`, `pendigits`, `mfeat_pixel`,
  `mfeat_factors`, `mfeat_fourier`, `mfeat_karhunen`, `mfeat_morphological`,
  `mfeat_zernike`, `movement_libras`.
- Text/sequence proxies: `dna`, `splice`, `phoneme`, `agaricus_lepiota`,
  `molecular_biology_promoters`.

## 6) Imbalanced or rare-event detection
Show that Σ-aware fitting plus stability bagging improves recall on minority
classes without overfitting.

- Medical/rare outcomes: `haberman`, `hepatitis`, `appendicitis`,
  `postoperative_patient_data`, `analcatdata_apnea1`, `analcatdata_apnea2`,
  `analcatdata_aids`, `analcatdata_fraud`, `analcatdata_happiness`,
  `analcatdata_lawsuit`, `analcatdata_japansolvent`.
- Solar flare and anomaly signals: `solar_flare_1`, `solar_flare_2`, `flare`.
- Fraud/credit skew: `credit_a`, `credit_g`, `churn`, `coil2000`, `adult`.

## 7) Classic tabular classification baselines
Bread-and-butter benchmarks to establish comparable or superior accuracy with
simpler, auditable models.

- Medical: `breast_cancer`, `breast_cancer_wisconsin`, `breast_w`, `wdbc`,
  `lymphography`, `pima`, `diabetes`, `heart_c`, `heart_h`, `heart_statlog`,
  `saheart`, `thyroid` variants (`ann_thyroid`, `new_thyroid`, `allhyper`,
  `allhypo`, `allrep`, `allbp`), `hypothyroid`.
- Biological/chemistry: `yeast`, `ecoli`, `glass`, `glass2`, `ionosphere`,
  `sonar`, `penguins`, `prnn_crabs`, `prnn_fglass`.
- Behavioral/psychology: `confidence`, `haberman`, `sleep`, `shuttle`, `tic_tac_toe`.
- Vote/government: `house_votes_84`, `523_analcatdata_neavote`,
  `527_analcatdata_election2000`.

## 8) Real-world regression (continuous targets)
Demonstrate smooth-response fitting and interpretability on non-synthetic data.

- Economics/finance: `195_auto_price`, `207_autoPrice`, `201_pol`, `230_machine_cpu`,
  `227_cpu_small`, `197_cpu_act`, `573_cpu_act`, `561_cpu`, `562_cpu_small`.
- Physical/environmental: `503_wind`, `505_tecator`, `542_pollution`, `547_no2`,
  `522_pm10`, `574_house_16H`, `218_house_8L`, `537_houses`.
- Social science/education: `1089_USCrime`, `1096_FacultySalaries`, `706_sleuth_case1202`,
  `665_sleuth_case2002`, `687_sleuth_ex1605`, `659_sleuth_ex1714`.

## 9) Additional robustness probes
Use for secondary sweeps or ablation studies (operator cost, bagging count,
whitening strategy).

- Multiclass small-sample: `iris`, `irish`, `segmentation`, `texture`, `vowel`,
  `movement_libras`.
- Binary clean vs synthetic noise: `clean1`, `clean2`.
- Data with ordinal levels: `car`, `nursery`, `tae`.
- Edge cases: `adult`, `bupa`, `cmc`, `optdigits`, `pendigits`, `spambase`,
  `spect`, `spectf`, `auto`, `cars`, `titanic`.

## 10) Running guidance
1. **Cross-validate** within each module (at least 5×2 CV) to report median and
   IQR of GLS-R²/accuracy; fix seeds for PySR for fairness.
2. **Operator cost parity**: Align allowed operators in PySR with FASE’s atomic
   alphabet (polynomials up to cubic, tanh/ReLU/GELU if permitted) to keep
   search spaces comparable.
3. **Evidence thresholds**: Use FASE’s `final_min_bits` (e.g., ≥6 bits) and
   report acceptance frequencies; for PySR log the complexity/score tradeoff so
   readers can see overfitting vs parsimony differences.
4. **Noise sensitivity**: For modules 2–6, report stability-selection frequency
   of each operator; PySR’s equation pool and Pareto front size provide a
   direct contrast.

Running this suite covers synthetic ground truth, interaction discovery,
boolean rules, noise robustness, high-dimensional scaling and class imbalance.
Consistently better GLS-R²/accuracy with simpler formulas across these modules
will provide strong, reproducible evidence of FASE’s effectiveness against PySR.
