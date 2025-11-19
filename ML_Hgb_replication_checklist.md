# Preoperative Hemoglobin ML Study — Full Replication Checklist

This document is a **complete, implementation-oriented checklist** for replicating all results from the “Preoperative hemoglobin in vascular surgery” machine learning study, assuming you already have:

- The **raw VQI data**, and
- The **merged / unified dataset** containing all procedures and variables.

The checklist is structured to be used as a **GitHub issue template or internal SOP** for a replication repo.

---

## 0. Repository & Project Structure

- [ ] Create a new git repository (e.g. `ml-preop-hgb-thresholds`).
- [ ] Add a top-level `README.md` summarizing:
  - [ ] Study aim: ML-derived age-specific preoperative hemoglobin thresholds to reduce mortality in open vascular surgery.
  - [ ] Data source: VQI registry (2012–2020), infrainguinal/suprainguinal bypass and open AAA repair.
  - [ ] Primary outcome: all-cause mortality at any time during follow-up.
  - [ ] Main model: Gradient Boosting Machine (GBM) selected via H2O AutoML.
- [ ] Create directories:
  - [ ] `data_raw/` (if you store raw VQI extracts)
  - [ ] `data_processed/` (final unified analytic dataset)
  - [ ] `notebooks/` (exploratory and replication notebooks)
  - [ ] `src/` (core pipeline scripts)
  - [ ] `models/` (saved H2O models)
  - [ ] `figures/` (ROC, SHAP, PDPs)
  - [ ] `tables/` (exported CSVs for tables)
  - [ ] `config/` (YAML/JSON configs for features, modeling, etc.)

Context: keeping a clear structure makes it easy to track which scripts generate which artifacts (tables, figures, and model objects) and to assert reproducibility via Makefiles or simple shell commands.

---

## 1. Data & Cohort Construction

### 1.1. Cohort Definition

- [ ] Confirm that the unified dataset includes all **open vascular procedures** between **2012–2020**:
  - [ ] Infrainguinal bypass
  - [ ] Suprainguinal bypass
  - [ ] Open abdominal aortic aneurysm (AAA) repair
- [ ] Include all recorded interventions per patient (procedure-level dataset, not strictly patient-level).

Context: The study pools all open arterial surgeries of these three types into a single cohort. There is no split by indication in the final analysis, but reviewer comments note this as a limitation—replication should preserve this design choice.

### 1.2. Outcome Variable

- [ ] Verify existence of `DEAD` variable:
  - [ ] Binary indicator of **all-cause mortality**.
  - [ ] Defined as “any-time mortality during follow-up as recorded in VQI,” **not** 30-day mortality.
- [ ] Set default to “alive” when mortality not recorded (as per original dataset logic).

Context: The outcome represents overall survival status at last recorded follow-up, aligning with the registry’s tracking window (~21 months). The GBM is trained to predict the probability that `DEAD == 1` given preoperative variables.

### 1.3. Feature Universe & Filtering

- [ ] Start from the **77 common features** across the three source datasets.
- [ ] Exclude:
  - [ ] All intra-operative variables.
  - [ ] Other clearly non-essential fields (free-text, meta fields, etc.).
- [ ] Confirm final set of **44 key features** (30 preoperative, 14 postoperative).

Context: The paper explicitly describes a reduction from 77 to 44 features by removing intraoperative and non-essential variables. The final analytic design uses preoperative features for model input but still describes postoperative variables for descriptive and modeling purposes (e.g., for secondary models in supplements).

---

## 2. Preoperative & Postoperative Feature Set

### 2.1. Preoperative Features (Model Inputs)

- [ ] Ensure the following pre-op features (or their exact equivalents) exist and are correctly coded:

  - [ ] Demographics & baseline:
    - [ ] `AGE` — age in years (continuous).
    - [ ] `SEX` — sex at birth (categorical; typically coded as 0/1).
    - [ ] `RACE` — race (categorical).
    - [ ] `ETHNICITY` — indicator for Hispanic/Latino (categorical).
    - [ ] `LIVINGSTATUS` — pre-admission living situation.
  - [ ] Anthropometrics:
    - [ ] `HTCM` — height in centimeters.
    - [ ] `WTLB` or `WEIGHT_KG` — weight (final version uses kilograms).
  - [ ] Comorbidities:
    - [ ] `DIABETES`
    - [ ] `COPD`
    - [ ] `HTN`
    - [ ] `PRIOR_CHF`
    - [ ] `PRIOR_BYPASS`
    - [ ] `PRIOR_CABG`
    - [ ] `PRIOR_CEACAS`
    - [ ] `PRIOR_PCI`
    - [ ] `DIALYSIS`
  - [ ] Preoperative medical therapy:
    - [ ] `PREOP_ASA` — acetylsalicylic acid.
    - [ ] `PREOP_STATIN`
    - [ ] `PREOP_BETABLOCKER`
    - [ ] `PREOP_P2Y` — P2Y12 antagonist.
    - [ ] `PREOP_ACE` — ACE inhibitor/ARB.
    - [ ] `PREOP_ANTICOAG`
  - [ ] Functional & preadmission status:
    - [ ] `PREOP_SMOKING`
    - [ ] `PREOP_AMBUL` — ambulatory status.
    - [ ] `STRESS` — preop stress test.
    - [ ] `TRANSFER` — transfer from hospital/rehab.
  - [ ] Surgical risk classification:
    - [ ] `ASACLASS` — ASA physical status classification.
  - [ ] Laboratory and key pre-op metrics:
    - [ ] `HEMO` — **preoperative hemoglobin (g/dL)**.
    - [ ] `PREOP_CREAT` — preoperative creatinine (mg/dL).

- [ ] Verify expected missingness patterns (e.g., ~23% of rows with ≥1 missing value overall).

Context: These 30 preoperative variables are the main inputs to the GBM for predicting mortality. The modeling approach deliberately retains missing data, relying on tree-based handling of missingness rather than imputation.

### 2.2. Postoperative Features (For Descriptives / Supplement)

- [ ] Ensure postoperative variables are available (for descriptive/supplementary analyses):
  - [ ] `RTOR` — return to OR.
  - [ ] `POSTOP_DYS` — new dysrhythmia.
  - [ ] `ANTIBIOTICSTART` / `ANTIBIOTICEND` — timing of perioperative antibiotics.
  - [ ] `RESPIRATORY` — post-op respiratory complication.
  - [ ] `POSTOP_MI` — new myocardial infarction.
  - [ ] `POSTOP_STROKE` — post-op stroke.
  - [ ] `DC_STATUS` — discharge status.
  - [ ] `LTF_CALC` — follow-up status (loss to follow-up indicator).
  - [ ] `POSTOP_LOS` — length of stay (days).
  - [ ] `TXFUSION` — transfusion units.
  - [ ] `PROC_SURVIVALDAYS` — post-op survival days.

Context: Although the main preoperative GBM focuses on predicting `DEAD`, the supplementary models include these additional outcomes. For full replication, you should compute model performance metrics for each as in Supplemental Table II/III.

---

## 3. Handling Missing Data & Coding

- [ ] **Do not impute** missing values:
  - [ ] Leave NA / missing entries as-is.
  - [ ] Allow the GBM and other ML models to handle missing values natively.
- [ ] Confirm that categorical variables are encoded appropriately for H2O:
  - [ ] Convert to factors where necessary (e.g., `SEX`, `RACE`, `ASACLASS`, etc.).
- [ ] Confirm units:
  - [ ] `HEMO` in g/dL.
  - [ ] `PREOP_CREAT` in mg/dL.
  - [ ] Height in cm, weight in kg (final table corrected to kg).

Context: The paper explicitly notes that missing entries may be informative and are retained; this is an important design choice that must be preserved in replication to match SHAP values and PDPs.

---

## 4. Train/Test Split & AutoML Configuration

### 4.1. Data Split

- [ ] Randomly split the data into:
  - [ ] **Training set:** 80%
  - [ ] **Testing set:** 20%
- [ ] Ensure the split is **stratified on `DEAD`** if possible to maintain prevalence balance (if not specified, document your choice).

### 4.2. H2O AutoML Settings

- [ ] Launch H2O cluster in Python.
- [ ] Define a classification problem:
  - [ ] Response: `DEAD` (binary).
  - [ ] Predictors: 30 preoperative variables.
- [ ] Configure AutoML to explore at least:
  - [ ] GBM models.
  - [ ] Extremely Random Trees / Random Forest-like models.
  - [ ] GLMs.
  - [ ] Deep Learning (ANNs).
  - [ ] Stacked Ensembles.
- [ ] Use **5-fold cross-validation** in AutoML.
- [ ] Specify AUC as the primary leaderboard metric for classification.
- [ ] Allow AutoML to tune:
  - [ ] Number of trees.
  - [ ] Maximum tree depth.
  - [ ] Minimum rows.
  - [ ] Learning rate.
  - [ ] Sample rate.
  - [ ] Minimum split improvement.
  - [ ] Distribution type (appropriate for binary classification).

Context: The paper describes a model comparison phase using H2O AutoML and then selecting the best GBM by AUC. Your AutoML configuration should be generous enough to reasonably rediscover a 70-tree GBM with depth 7 and AUC ~0.785.

---

## 5. Model Training & Selection

### 5.1. Train All Five Model Families

- [ ] From the AutoML leaderboard or direct training, retain:
  - [ ] **Gradient Boosting Machine (GBM)**.
  - [ ] **Extra Trees / Extremely Random Trees** model.
  - [ ] **Generalized Linear Model (GLM)**.
  - [ ] **Artificial Neural Network (ANN)** (H2O Deep Learning).
  - [ ] **Stacked Ensemble** model.
- [ ] Evaluate each on the **held-out 20% test set** using AUC.

### 5.2. Expected Performance (for sanity check)

From the supplemental table (approximate values):

- [ ] Stacked Ensemble: AUC ≈ 0.7903
- [ ] GBM: AUC ≈ 0.7854
- [ ] ANN: AUC ≈ 0.7680
- [ ] Extra Trees: AUC ≈ 0.7601
- [ ] GLM: AUC ≈ 0.7380

- [ ] Confirm your replication achieves **similar ordering and magnitudes**.

### 5.3. Final Model Choice

- [ ] Select the **GBM** as the primary model for interpretation:
  - [ ] Approximately 70 trees.
  - [ ] Maximum depth ≈ 7.
- [ ] Save the final trained GBM object to `models/gbm_preop_hgb.hex` (or similar).

Context: Although the stacked ensemble has slightly higher AUC, the paper chooses GBM for its combination of performance and explainability (SHAP and PDPs are simpler). Replication should mirror this choice and rationale.

---

## 6. Model Interpretation: SHAP & Feature Importance

### 6.1. SHAP Summary Plot

- [ ] Compute SHAP values for the final GBM model on the test set.
- [ ] Produce a **SHAP summary plot** showing the ranked contribution of features.
- [ ] Confirm the **top features** in order are approximately:
  - [ ] `AGE` (very high importance)
  - [ ] `ASACLASS`
  - [ ] `HEMO` (preop hemoglobin)
  - [ ] `PREOP_CREAT`
  - [ ] `PRIOR_CABG`
  - [ ] `WTLB`/`WEIGHT`
  - [ ] `PRIOR_CHF`
  - [ ] `DIABETES`
  - [ ] `COPD`
  - [ ] `PREOP_ANTICOAG`
  - [ ] `HTCM`
  - [ ] `PREOP_BETABLOCKER`
  - [ ] `PREOP_AMBUL`
  - [ ] `PREOP_STATIN`
  - [ ] `PREOP_SMOKING`
  - [ ] `RACE`
  - [ ] `PREOP_ACE`
  - [ ] `TRANSFER`
  - [ ] `STRESS`
  - [ ] `PRIOR_CEACAS`
- [ ] Confirm that **SEX** has **very low importance** (≈ 0.036 relative to HEMO=1.0).

Context: This SHAP analysis is central to the paper’s claim that age and hemoglobin dominate the mortality prediction, while sex is surprisingly unimportant, suggesting that sex-specific anemia thresholds may be less appropriate in this vascular population.

### 6.2. Global Feature Importance

- [ ] Extract H2O’s built-in feature importance for the GBM.
- [ ] Confirm relative importance ranking approximately matches SHAP-based ordering.
- [ ] Document the exact importance value for `SEX` and for `HEMO` (to replicate the 0.036 vs 1.0 contrast).

---

## 7. Partial Dependence Plots (PDPs)

### 7.1. PDP for Age

- [ ] Compute PDP for `AGE` vs predicted probability of `DEAD == 1`:
  - [ ] Use the GBM model.
  - [ ] Cover the observed age range in the cohort.
- [ ] Plot:
  - [ ] X-axis: age in years.
  - [ ] Y-axis: mean predicted mortality ( “Mean response” ).
  - [ ] Background histogram: distribution of ages in the test set.
- [ ] Confirm qualitative shape:
  - [ ] Flat / steady predicted risk up to ~40 years.
  - [ ] Increasing predicted risk beyond ~40.

### 7.2. PDP for Preoperative Hemoglobin

- [ ] Compute PDP for `HEMO` vs predicted mortality:
  - [ ] over the observed hemoglobin range (e.g., 4–20 g/dL).
- [ ] Plot:
  - [ ] X-axis: hemoglobin (g/dL).
  - [ ] Y-axis: mean predicted mortality.
  - [ ] Background histogram: distribution of hemoglobin values.
- [ ] Confirm qualitative shape:
  - [ ] Monotonic decrease in mortality as hemoglobin rises.
  - [ ] Steepest improvement between ~8–14 g/dL.
  - [ ] Flattening / plateau around ≥16 g/dL.

Context: These two PDPs form the core interpretability figures (Figures 3A and 3B), supporting the argument for age-specific thresholds and demonstrating diminishing returns in hemoglobin correction beyond ~16 g/dL.

---

## 8. Age-Stratified Hemoglobin Threshold Derivation

This is the central “threshold” contribution of the paper and must be replicated carefully.

### 8.1. Age Group Definitions

- [ ] Partition the dataset into age groups:
  - [ ] **Under 40** (`AGE < 40`)
  - [ ] **40–49**
  - [ ] **50–59**
  - [ ] **60–69**
  - [ ] **70–79**
  - [ ] **80+`

### 8.2. Group-Specific PDP for Hemoglobin

For each age group:

- [ ] Restrict data to that group (e.g., only patients 40–49).
- [ ] Compute the PDP of `HEMO` vs predicted mortality **within that age group**:
  - [ ] For each candidate hemoglobin value `h`:
    - [ ] Predict mortality using the GBM (holding hemoglobin at `h` within that group).
    - [ ] Compute group-specific mean predicted mortality at `h`.

### 8.3. Identify Protective Hemoglobin Ranges

For a given mortality cutoff (e.g., 10%):

- [ ] For each age group, identify the **widest contiguous interval** `[h_min, h_max]` such that:
  - [ ] For all `h` in `[h_min, h_max]`, the **mean predicted mortality** < cutoff (e.g., 0.10).
- [ ] Record the following for each age group:
  - [ ] Minimum hemoglobin (`h_min`).
  - [ ] Maximum hemoglobin (`h_max`).
  - [ ] Mean predicted mortality within that interval.

### 8.4. Reproducing Table 2 (10% Mortality Cutoff)

- [ ] For **10% cutoff**, ensure values match approximately:

  - Under 40: **7.9 – 18.6 g/dL**, mean mortality ~6%.
  - 40–49: **10.1 – 18.9 g/dL**, mean mortality ~7%.
  - 50–59: **12.5 – 19.6 g/dL**, mean mortality ~8%.
  - 60–69: **16.6 – 18.3 g/dL**, mean mortality ~9%.
  - 70–79: **No range** with predicted mortality <10%.
  - 80+: **No range** with predicted mortality <10%.

- [ ] Mark “None” for age groups where no hemoglobin range satisfies the cutoff.

### 8.5. Supplemental Table IV (Multiple Cutoffs)

- [ ] Repeat the range-finding process for additional mortality cutoffs (e.g., 5%, 20%, 30%). 
- [ ] Populate a table with rows: age group × cutoff, and columns: min Hgb, max Hgb, mean predicted mortality.

Context: This threshold derivation uses model-based predictions rather than observed outcome frequencies. It is inherently a “model-derived” risk calibration exercise and must be implemented exactly as described to obtain similar ranges, especially the very high required hemoglobin for older age and the absence of protective ranges at age ≥70 for low mortality thresholds.

---

## 9. Tables & Figures Reproduction

### 9.1. Table 1 — Descriptive Statistics

- [ ] Compute min, max, mean, and standard deviation for:
  - [ ] `HEMO` (g/dL)
  - [ ] `HTCM` (cm)
  - [ ] `SEX` (encoded numeric)
  - [ ] `TXFUSION` (units)
  - [ ] `WEIGHT_KG` (kg)
- [ ] Ensure units are correctly labeled (e.g., transfusions in units, weight in kg).
- [ ] Export as `tables/Table1_descriptives.csv` (and optionally LaTeX/Word).

### 9.2. Table 2 — Hemoglobin Thresholds by Age (10% Cutoff)

- [ ] Use the outputs from Section 8.4.
- [ ] For each age group, report:
  - [ ] Minimum protective Hgb (g/dL)
  - [ ] Maximum protective Hgb (g/dL)
  - [ ] Mean predicted mortality within range
- [ ] Represent “no valid range” as `None`/`N/A`.

### 9.3. Supplementary Tables I–IV

- [ ] Reproduce variable descriptions (I–II) from your feature metadata.
- [ ] Reproduce model performance summaries (III) from your H2O AutoML results.
- [ ] Reproduce hemoglobin ranges for multiple mortality cutoffs (IV).

### 9.4. Figures

- [ ] **Figure 1 (ROC Curve for GBM)**:
  - [ ] Plot ROC curve on test set.
  - [ ] Report AUC; confirm ≈0.7854.
- [ ] **Figure 2 (SHAP Summary Plot)**:
  - [ ] Plot ranked SHAP values colored by feature value.
  - [ ] Confirm dominance of `AGE`, `ASACLASS`, and `HEMO`.
  - [ ] Highlight low importance of `SEX`.
- [ ] **Figure 3A (PDP — Age)**:
  - [ ] PDP with histogram for `AGE` vs mortality.
  - [ ] Ensure flat region <40 and rising mortality >40.
- [ ] **Figure 3B (PDP — Hemoglobin)**:
  - [ ] PDP with histogram for `HEMO` vs mortality.
  - [ ] Demonstrate 8–14 g/dL as steep improvement zone and plateau ≥16 g/dL.

---

## 10. Environment & Hardware

### 10.1. Hardware (for documentation)

- [ ] Note your actual compute environment.
- [ ] Optionally document matching to the original:
  - [ ] Dell XPS 8940.
  - [ ] 11th Gen Intel Core i5 (6 cores, 12 threads, up to 4.4 GHz).
  - [ ] 16 GB DDR4 RAM @ 3200 MHz.
  - [ ] NVIDIA GeForce RTX 3060 GPU.

### 10.2. Software Versions

- [ ] Record versions of:
  - [ ] Python.
  - [ ] H2O.
  - [ ] SHAP.
  - [ ] Any additional libraries (pandas, numpy, scikit-learn, etc.).
- [ ] Save this in `config/environment.yml` and/or `session_info.md`.

Context: While exact hardware is not essential, documenting the environment supports full reproducibility and can help explain minor discrepancies in performance metrics.

---

## 11. Interpretation & Narrative Consistency

To fully replicate the **scientific narrative** as well as the quantitative results, ensure your replication notes and manuscript text are consistent with the following key interpretive points drawn from the original submission:

- [ ] **Age is the dominant driver** of predicted mortality among pre-op variables.
- [ ] **Preoperative hemoglobin is strongly protective**, with most benefit between 8–14 g/dL and diminishing returns beyond ~16 g/dL.
- [ ] **Sex has minimal influence** in the model despite guideline-based sex-specific anemia thresholds.
- [ ] **Older patients require higher hemoglobin levels** to achieve the same model-predicted mortality as younger patients.
- [ ] **No hemoglobin range** for ages ≥70 yields predicted mortality <10% in the model, reflecting the strong effect of age and comorbidities.
- [ ] Extremely low model-derived minimum hemoglobin levels at high mortality cutoffs (e.g., 4 g/dL for 20–30% cutoffs) are **mathematical artifacts**, not clinically actionable targets.
- [ ] The results should be framed as **guidance for individualized preoperative optimization**, not as rigid transfusion thresholds.

---

## 12. Optional: Repo Automation & Quality Checks

- [ ] Add a `Makefile` or `justfile` with targets such as:
  - [ ] `make data` — construct processed dataset.
  - [ ] `make models` — train models and save them.
  - [ ] `make figures` — regenerate all figures.
  - [ ] `make tables` — regenerate all tables.
- [ ] Add unit-style checks:
  - [ ] Assert that model AUCs fall within ±0.01–0.02 of reported values.
  - [ ] Assert that Table 2 Hgb ranges differ from published values by at most a small tolerance (e.g., ±0.1 g/dL).
- [ ] Add a `REPRODUCIBILITY.md` that links each figure/table to the exact script/notebook that generates it.

---

This checklist plus contextual notes should be sufficient to:

1. Reconstruct the unified dataset and analytic design.
2. Rebuild all machine learning models.
3. Reproduce SHAP, PDPs, and age-stratified hemoglobin thresholds.
4. Regenerate all tables and figures used in the manuscript and supplements.
