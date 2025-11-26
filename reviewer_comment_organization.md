# Organized Reviewer Comments by Topic

This document organizes all reviewer comments from Reviewers 1, 2, and 3 into thematic categories so overlapping issues are grouped together. It does not include responses—only the structured organization to support revision planning.

---

## **1. Clarity, Writing, and Structural Revisions**

### General clarity / simplification

- **R1.1**: Page 9, line 1–12 need simplification.
- **R3.1**: Table of Contents Summary and Key Findings must be shortened (<50 words); expand description of research type (Page 2, line 14).

### Terminology and correctness

- **R2.1**: Typo “morality” → “mortality” (page 3, line 23).
- **R3.2**: "Primary outcome" should be replaced with "outcome of interest."

---

## **2. Cohort Definition, Study Design, and Appropriateness**

### Cohort composition concerns

- **R2.8**: Combining open aneurysm repairs and open revascularizations limits applicability; suggests splitting analyses.
- **R3.5**: Requests rationale for combining AAA repairs and bypass patients given different mortality rates.

### Outcome definition

- **R3.3**: Clarify when all-cause mortality is assessed (30-day vs any follow-up).

### Confounding and model interpretation

- **R3.4**: Questions how models account for confounding; is low Hgb causal or proxy?

---

## **3. Table 1–Related Issues (Variables, Units, Definitions)**

### Variable definitions and odd values

- **R1.5**: "Sex" min 0, max 90 requires explanation.
- **R2.2**: Same concern: sex coded 0–90.

### Units and consistency

- **R2.3**: No units for "transfusion."
- **R2.7**: Weight unit typo; inconsistency between cm and lbs. Suggest BMI instead.

### Additional descriptive statistics

- **R2.6**: Age should also appear numerically in Table 1.

---

## **4. ASA Class and Surgical Urgency (Emergent vs Elective)**

- **R2.4**: Clarify ASA class influence given its SHAP prominence.
- **R2.5**: How were emergent vs elective cases analyzed? Large clinical differences noted.

---

## **5. Figures (2, 3A, 3B) and Their Interpretation**

### Figure 2—SHAP variables

- **R1.3**: Clarify variables on left-hand axis; only defined in supplement.

### Figure 3—Partial dependence plots

- **R1.4**: Clarify what "mean response" on Y-axis represents.
- **R2.6 (related)**: Suggests numerical age summary (complements Figure 3A histogram).

---

## **6. Hemoglobin Threshold Interpretation and Clinical Implications**

### Relationship between Hb and mortality

- **R1.2**: Minimum Hgb for improved outcome? Is targeting Hgb 16 implied? Concerns over transfusion burden/risks.
- **R1.7**: For each age group, how should min/max Hgb be interpreted? Should clinicians target min, max, or midpoints?

### Age-specific patterns and interpretation

- **R3.6**: Do Table 2 results imply patients >70 always exceed 10% mortality? Is age or Hgb driving risk?
- **R3.7**: Supplemental Table IV shows implausibly low min Hgb (e.g., 4 g/dL). Requests:
  - Actual observed Hgb ranges.
  - Whether extremely low values are evenly distributed across age groups.

---

## **7. Supplemental Tables (Model Metrics and ML Methods)**

### Supplemental Table 2—Performance metrics

- **R1.8**: Clarify "best mean per class error" and "best mean absolute error (std dev)."

### Supplemental Table 3—ML method descriptions

- **R1.9**: Appreciates detail but suggests simplifying for non-technical readers.

---

## **8. Study Limitations / Interpretation of Results**

(Overlaps with design and confounding concerns above)

- **R3.4**: Concern about models controlling for confounding.
- **R3.5**: Suggests reevaluating combined cohorts.


---

## **9. Reviewer Comments That Can Be Directly Addressed**

Below is a preliminary assessment of which comments can be addressed through straightforward manuscript edits or clarifying responses, based on the content of the uploaded documents.

### **A. Easy, Direct Fixes (Typos, clarity improvements, missing definitions, labeling issues)**
These can be fully resolved through small manuscript edits.
- **R1.1**: Simplify Page 9 lines 1–12 — text already revised, easy change.
- **R2.1**: Typo “morality” → “mortality.”
- **R3.1**: Shortening summary/Key Findings — already trimmed.
- **R3.2**: Change “primary outcome” to “outcome of interest.”
- **R1.3**: Clarify Figure 2 variable names — already added.
- **R1.4**: Clarify meaning of “mean response” — straightforward.
- **R1.5 / R2.2**: Explain Table 1 “sex” coding issue — must fix table or clarify coding.
- **R2.3**: Add units for transfusion — easy edit.
- **R2.6**: Add age numerically to Table 1 — easy addition.
- **R2.7**: Fix weight units; use consistent metric units — straightforward.
- **R1.8**: Explain model performance metrics — response already drafted.
- **R1.9**: Simplify ML method descriptions — minor text edits.

### **B. Clarifications about modeling approach (straightforward to answer)**
These comments require explanation but not major re-analysis.
- **R1.2**: Clarify that 16 g/dL is not a recommended target; explain diminishing returns — explanation already drafted.
- **R1.7**: Clarify interpretation of min/max Hgb thresholds and clinical implications.
- **R3.3**: Specify mortality timing (follow-up period vs 30-day) — already available.
- **R2.4**: Clarify ASA class influence — interpret SHAP plot.
- **R2.5**: Explain emergent vs elective handling (not included; ASA class as proxy).
- **R3.6**: Clarify interplay of age vs Hgb in determining mortality risk — doable.
- **R3.7**: Explain implausibly low Hgb (from model, not real-world target; based on sample distribution). Provide observed Hgb ranges.

### **C. Substantive Methodological Comments Requiring New Analysis**
The following reviewer concerns identify meaningful methodological questions that we will now address through additional analyses. This section outlines what we currently do, what reviewers are asking, and what new analyses we will add to fully resolve these concerns.

### **C1. Combining AAA + bypass patients (R2.8, R3.5)**
**Current approach:**
- The model pools all open vascular procedures: AAA repairs, suprainguinal bypass, and infrainguinal bypass.
- Surgical subtype is not included as a feature.
- The manuscript does not currently provide justification for combining these clinically distinct populations.

**Reviewer concerns:**
- Mortality risk differs substantially across these procedure types.
- Combining them may obscure meaningful subtype-specific trends.
- Reviewers explicitly request separate cohorts (AAA vs occlusive disease) and potentially even more granular stratification (suprainguinal vs infrainguinal).

**New analysis we will add:**
- Build fully separate GBM models for:
  1. AAA repairs
  2. Suprainguinal bypass
  3. Infrainguinal bypass
- For each subgroup, compute:
  - SHAP summary plots
  - Partial dependence plots for hemoglobin and age
  - Hemoglobin thresholds analogous to Table 2
- Add new supplemental tables and figures comparing:
  - Model performance across subgroups
  - Hemoglobin threshold ranges per subgroup
  - Whether hemoglobin consistently emerges as a strong predictor across populations

**How this will strengthen the manuscript:**
- Demonstrates transparency and methodological rigor
- Addresses heterogeneity concerns directly
- Allows readers to understand population‑specific implications for preoperative hemoglobin

---

### **C2. Confounding and interpretation of hemoglobin effects (R3.4)**
**Current approach:**
- The model includes many covariates (creatinine, ASA, CHF, COPD, etc.).
- SHAP values show hemoglobin as a major contributor to predicted mortality.
- The manuscript does not address whether hemoglobin is acting as a causal variable or proxy for illness severity.

**Reviewer concerns:**
- Whether the model properly accounts for confounding factors
- Whether low hemoglobin is independently predictive or simply correlated with other comorbidities

**New analysis we will add:**
- Compute correlation matrices between hemoglobin and major comorbidities.
- Generate SHAP interaction plots for:
  - Hemoglobin × age
  - Hemoglobin × ASA class
  - Hemoglobin × renal function (creatinine, dialysis)
- Produce stratified partial dependence plots for patients with and without major comorbid conditions.
- Add explanatory text clarifying predictive (not causal) interpretation while demonstrating robustness of hemoglobin’s importance.

**How this will strengthen the manuscript:**
- Directly addresses reviewer concerns about confounding
- Provides clearer insight into the relationship between hemoglobin and mortality risk
- Demonstrates that hemoglobin remains predictive even after accounting for key comorbidities

---

## **D. Additional Comments Requiring New Stratified or Exploratory Analyses**
These reviewer comments require additional data exploration and subgroup‑specific analysis to resolve fully.

### **D1. Hemoglobin distribution issues and clinically implausible values (R3.7)**
**Current approach:**
- Partial dependence plots show hemoglobin effects across the model’s full range, which includes some values that are clinically very low (e.g., 4 g/dL).
- The manuscript does not present the actual observed hemoglobin distribution by age.

**Reviewer concerns:**
- Implausible minimum hemoglobin levels appearing in Supplemental Table IV
- Need clarity on whether extremely low values are modeling artifacts or present in the dataset
- Whether these extreme values differ across age groups

**New analysis we will add:**
- Compute actual observed hemoglobin distributions (min, max, mean, IQR) for each age decade.
- Plot empirical density/histogram of hemoglobin for each age group.
- Compare observed ranges to model-generated ranges used in threshold calculations.
- Recalculate partial dependence curves restricted **only** to observed ranges within each age group.
- Recompute hemoglobin mortality thresholds using these restricted ranges.

**How this will strengthen the manuscript:**
- Eliminates concerns about model extrapolation
- Ensures all reported thresholds are grounded in actual observed patient values
- Provides clinically interpretable hemoglobin ranges

---

### **D2. Population-level heterogeneity and emergent vs elective procedures (R2.5)**
**Current approach:**
- Emergent vs elective status is not explicitly included in the model.
- ASA class is the closest proxy.

**Reviewer concerns:**
- Emergent procedures have substantially different physiologic conditions and mortality risk.
- The absence of modeling this variable may obscure important differences.

**New analysis we will add:**
- Identify emergent vs elective status from VQI data.
- Build stratified models for:
  - Elective cases
  - Emergent cases
- Compare:
  - Mortality rates
  - SHAP importance of hemoglobin
  - Hemoglobin thresholds within each category
- Add supplemental tables summarizing risk differences.

**How this will strengthen the manuscript:**
- Addresses a core reviewer concern
- Demonstrates robustness of hemoglobin–mortality relationship across urgency levels
- Enhances clinical interpretability

---

This revised structure reflects the assumption that we will perform new analyses for all major methodological critiques in Sections C and D, and positions these additions as strengthening the manuscript substantively.. Comments not directly fixable without re-analysis (but can be acknowledged)**
These comments are feasible to address but would require additional modeling, stratification, or data-level exploration. Below is more detail about what reviewers want and what additional analyses could satisfy them.

### **D1. Splitting populations into more granular subgroups (R2.8, R3.5)**
**Current approach:**
- Model treats all open vascular surgeries as a single population.
- Surgical subtype is not included as a feature.
- Thresholds are derived across the full pooled cohort.

**What reviewers are asking:**
- Separate analyses for:
  - AAA vs occlusive disease
  - Suprainguinal vs infrainguinal bypass
- Their concern: mortality risk differs substantially, so thresholds may not generalize across categories.

**What we could add (if choosing to run new analyses):**
- Re-build the GBM model separately for:
  1. AAA repairs only
  2. Suprainguinal bypass
  3. Infrainguinal bypass
- Recompute age-specific hemoglobin thresholds for each.
- Add a supplemental table comparing thresholds across subtypes.
- Add SHAP plots for subgroups to show whether hemoglobin remains a major predictor.

**Minimum-effort enhancement (if not running full subgroup models):**
- Run logistic regression models stratified by procedure type to confirm trends.
- Add supplemental descriptive statistics comparing distributions.

### **D2. Distribution of extremely low hemoglobin values across age groups (R3.7)**
**Current approach:**
- Partial dependence values include observed and extrapolated values.
- Some groups (e.g., age 60–69) show implausibly low minimum Hb (e.g., 4 g/dL) where the model still predicts <20% mortality.
- The manuscript does not report raw distributions by age.

**What reviewers want:**
- Actual observed range of hemoglobin values by age bracket.
- Clarification that extremely low values may appear because of sparse or missing data patterns.

**What we might add with new analysis:**
- Compute empirical minimum, maximum, mean, and IQR of hemoglobin for each age group.
- Plot histograms or density plots of hemoglobin stratified by decade.
- Compare empirical distributions to the ranges used in the partial dependence plots.
- Add text explaining:
  - Whether extremely low values are real or modeling artifacts.
  - Whether certain age groups disproportionately contain low-Hb observations.

### **Optional deeper update (strongest revision):**
- Restrict partial dependence calculations to the **real observed range** for each age group.
- Recalculate thresholds using these constrained ranges.
- Add analysis demonstrating how extrapolation affects threshold interpretation.

This would directly address reviewer concerns about clinical plausibility.

---

If you'd like, I can next: 
- generate a recommended plan for which additional analyses to prioritize; 
- draft the exact explanatory text to insert into the manuscript; 
- or build a structured "Action Plan" section summarizing all proposed additions.
These require justification but do not mandate re-running models.
- **R2.8**: Rationale for combining AAA + bypass patients. Can defend based on dataset size, shared preoperative physiology, prior VQI analyses — but must write carefully.
- **R3.5**: Same concern — rationale can be provided.
- **R3.4**: Confounding in ML models — can explain limits and dependence on SHAP but cannot “solve” confounding fully.

### **D. Comments not directly fixable without re-analysis (but can be acknowledged)**
You can respond, but addressing them fully would require additional modeling.
- **R2.8**: Splitting into separate populations (occlusive vs aneurysm vs suprainguinal/infrainguinal) — would require rebuilding models.
- **R3.7 (partially)**: Testing distribution of extremely low Hb across age groups — unless already calculated, may require additional data queries.

Most of these can still be answered by clarifying limitations and stating that subgroup-specific models are outside scope.

---

If you'd like, I can now convert this into a full author response template where each comment is paired with:
- **Proposed manuscript change**
- **Draft response to reviewer**
- **Location in manuscript**
