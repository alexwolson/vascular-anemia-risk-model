# Modeling Artifacts Included in the Manuscript

This document enumerates all figures, tables, supplemental materials, and other modeling outputs included or planned for inclusion in the revised manuscript. It is organized by section and includes a brief description of the purpose of each artifact.

---

## **1. Main Manuscript Tables**

### **Table 1. Summary of Preoperative Hemoglobin Levels and Patient Demographics**
- Includes minimum, maximum, mean, and standard deviation for hemoglobin, height, weight, sex coding, and transfusion.
- Will be updated for corrected units and clarified variable definitions.

### **Table 2. Hemoglobin Ranges Below 10% Mortality by Age Group**
- Provides minimum and maximum hemoglobin values associated with <10% predicted mortality.
- Will be updated to reflect stratified analyses (AAA, suprainguinal, infrainguinal, emergent vs elective, age-specific observed ranges).

---

## **2. Main Manuscript Figures**

### **Figure 1. ROC Curve for the Leading GBM Model**
- Displays AUC performance of the primary model.
- May be duplicated or supplemented for subgroup-specific models.

### **Figure 2. SHAP Summary Plot for Leading GBM Model**
- Shows relative importance and directional impact of features.
- Supplemental SHAP plots will be added for subgroup models.

### **Figure 3A. Partial Dependence Plot for Age**
- Shows modeled relationship between age and mortality.
- Subgroup-specific versions will be added.

### **Figure 3B. Partial Dependence Plot for Hemoglobin**
- Shows modeled relationship between hemoglobin and mortality.
- Will be updated with:
  - observed-range-constrained PDPs
  - subgroup-specific PDPs
  - interaction PDPs (Hb × age, Hb × ASA, Hb × comorbidity)

---

## **3. Supplemental Tables**

### **Supplemental Table I. Predictor Variable Definitions and Missingness**
- Lists all input variables, categorical status, and % missing.

### **Supplemental Table II. Response Variable Definitions and Model Performance Metrics**
- Includes per-class error and MAE metrics.
- Will be expanded to include subgroup model metrics.

### **Supplemental Table III. Details of Machine Learning Methods Tested**
- Summaries of Extra Trees, GLM, GBM, ANN, Stacked Ensemble.
- Contains model training times and AUCs.
- May include subgroup versions if needed.

### **Supplemental Table IV. Hemoglobin Ranges by Age Group for Multiple Mortality Cutoffs**
- Shows ranges at 5%, 10%, 20%, 30% thresholds.
- Will be recalculated using:
  - stratified models (AAA, SI bypass, II bypass)
  - emergent vs elective stratification
  - observed-range-only analysis

---

## **4. Additional Modeling Artifacts to Add (as Part of the Revised Analysis)**

### **A. Subgroup Model Outputs**
For each subgroup (AAA, suprainguinal bypass, infrainguinal bypass):
- ROC curve
- SHAP summary plot
- Partial dependence plots (age, hemoglobin)
- Hemoglobin thresholds analogous to Table 2
- New supplemental tables with subgroup-specific summary statistics

### **B. Emergent vs Elective Model Outputs**
For emergent and elective cohorts:
- Model performance metrics (AUC, error)
- SHAP summary plots
- PDPs for hemoglobin and age
- Updated threshold tables

### **C. Confounding & Interaction Analysis**
Artifacts to be added:
- Correlation matrix between hemoglobin and comorbidities
- SHAP interaction plots
  - Hb × age
  - Hb × ASA class
  - Hb × creatinine / dialysis
- Stratified PDPs based on presence/absence of major comorbidities

### **D. Hemoglobin Distribution Analysis**
Artifacts to be added:
- Histograms/density plots of hemoglobin by age decade
- Observed vs modeled hemoglobin range comparison tables
- Observed-range-constrained PDPs

---

## **5. Proposed New Supplement Organization**
To accommodate expanded analysis, supplemental materials will be reorganized into:

### **Supplement A: Core Modeling Outputs**
- Predictor/response tables
- Performance metrics
- Modeling methods

### **Supplement B: Subgroup Analyses**
- AAA-specific artifacts
- Suprainguinal bypass artifacts
- Infrainguinal bypass artifacts

### **Supplement C: Emergent vs Elective Analyses**
- Model outputs and threshold tables

### **Supplement D: Confounding & Interaction Analyses**
- Correlations, SHAP interactions, stratified PDPs

### **Supplement E: Age-Specific Hemoglobin Distribution Analyses**
- Distribution plots and constrained PDPs

---

If you'd like, I can now cross-reference each artifact with the reviewer comment it addresses or create a checklist for tracking completion of all modeling outputs.

