# XAI KPI-Check: Explainable Financial KPI-Matching

**Interpretability (SHAP & LIME) for Text-Pair Classification Benchmarked Against KPI-Check (Hillebrand et al., 2022)**

---

## Overview
This repository implements an explainable financial KPI matching pipeline designed to link unstructured narrative statements from financial filings (SEC EDGAR 10-K) to structured financial statement line items.

- **Primary Classifier:** Logistic Regression on interpretable hand-crafted features (Block A) and dense sentence embeddings (Block B).
- **Secondary Model:** Multi-Layer Perceptron (MLP) for non-linear comparison.
- **Explainability:** SHAP (LinearExplainer) and LIME (LimeTabularExplainer) side-by-side attributions.
- **Benchmark:** Comparison against KPI-Check (Hillebrand et al., IEEE BigData 2022, arXiv:2211.06112; 73.00% micro-F1).

---

## Directory Structure
```
xai-kpi-check/
  SCOPE.md              # Scope definition and boundaries
  requirements.txt      # Project dependencies
  README.md             # Project overview and instructions
  data/
    raw/                # Downloaded 10-K filings and download manifest
    processed/          # Extracted KPI sentences and table line items
    labeled/            # Hand-labeled pairs (match / no_match / ambiguous)
  src/
    extraction/         # SEC EDGAR fetcher & parsing scripts
    features/           # Block A (hand-crafted) & Block B (embeddings) feature extractor
    model/              # Classifier training, ablation, and calibration
    xai/                # SHAP and LIME explanation generators
    eval/               # Evaluation, baseline comparison, and ablation tables
  notebooks/            # Exploratory analysis
  results/
    figures/            # Confusion matrices, calibration, SHAP/LIME plots
    tables/             # Model metrics, feature importances, ablation summaries
  docs/
    review2_draft.md    # Review 2 submission draft and structured notes
```
