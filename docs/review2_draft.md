# Review 2 Draft: Explainable Financial KPI-Matching (XAI KPI-Check)

**Project Title:** Explainable Financial KPI-Matching: SHAP/LIME Interpretability for a Text-Pair Classifier, Benchmarked Against KPI-Check (Hillebrand et al., 2022)  
**Academic Integrity Note:** This document provides the architectural scaffold, experimental records, empirical metrics, and artifact references. All narrative prose, literature comparative synthesis, and qualitative error analysis will be written in original voice.

---

## 1. Introduction
- **Problem Statement:** Corporate annual reports (SEC EDGAR Form 10-K) combine narrative Management's Discussion and Analysis (MD&A) sections with audited financial statements (Balance Sheet, Income Statement, Statement of Cash Flows). Verifying narrative KPI claims against authoritative statements is critical for compliance and audit integrity.
- **Motivation:** Manual verification is labor-intensive and susceptible to human oversight across complex corporate structures. Deep learning approaches (e.g., fine-tuned BERT models in KPI-Check) improve matching accuracy but create an opaque "black-box" decision process unacceptable in regulated financial auditing.
- **Objectives:**
  1. Build a text-pair classifier matching narrative KPI sentences to candidate statement line items (`match`, `no_match`, `ambiguous`).
  2. Implement an explainability layer using SHAP (LinearExplainer) and LIME (LimeTabularExplainer) to make match predictions interpretable and auditable.
  3. Benchmark predictive performance (micro-F1) against the published KPI-Check baseline (Hillebrand et al., IEEE BigData 2022, arXiv:2211.06112: 73.00% micro-F1).
  4. Perform feature ablation to measure the tradeoff between pure lexical/numerical features (Block A) and dense semantic representations (Block B).
- **Scope & Guardrails:** Enforced via `SCOPE.md`. Task bounded to 3-class classification (`match`, `no_match`, `ambiguous`) using linear and shallow neural baselines without fine-tuning heavy transformers from scratch.

---

## 2. Literature Review
- **Primary Baseline:** Hillebrand et al. (2022), "KPI-Check: A Dataset and Approach for Checking Numerical Claims in Financial Reports," IEEE BigData 2022, arXiv:2211.06112.
  - *Key takeaways:* Evaluated on proprietary German financial filings with a transformer pipeline; reported 73.00% micro-F1. Explainability was not evaluated.
- **Auditing Compliance with LLMs:** Follow-up literature (arXiv:2507.16642) explores regulatory compliance verification in financial auditing using LLMs, noting hallucination risks, API costs, and high latency.
- **Explainable Machine Learning Foundations:**
  - Lundberg & Lee (2017): Unified framework for interpreting model predictions (SHAP / Shapley Additive exPlanations). LinearExplainer provides exact, efficient Shapley values for linear models.
  - Ribeiro, Singh, & Guestrin (2016): "Why Should I Trust You?": Explaining the Predictions of Any Classifier (LIME). Tabular local perturbations for local surrogate models.
- **Identified Research Gap:** Prior financial NLP auditing systems prioritize raw score over interpretability; this project provides dual SHAP/LIME attribution on transparent lexical-numerical features combined with semantic embeddings.

---

## 3. Proposed System Architecture
- **Pipeline Stages:**
  1. *Report Sourcing & Extraction (`src/extraction/`):* SEC EDGAR 10-K retrieval (`fetch_reports.py`) and HTML/text parsing (`parse_reports.py`) extracting candidate narrative KPI sentences and structured financial tables.
  2. *Labeling Workflow (`src/eval/build_labeling_sheet.py`):* Candidate pair generation with heuristic ranking; hand-labeled ground truth (`data/labeled/pairs_labeled.csv`).
  3. *Feature Engineering (`src/features/build_features.py`):*
     - **Block A (Interpretable Hand-Crafted):** `numeric_value_match`, `numeric_value_close`, `keyword_overlap`, `period_match`, `string_similarity`, `sentence_length`, `line_item_name_length`.
     - **Block B (Dense Semantic):** Cosine similarity from dense embeddings (`embedding_cosine_similarity`).
  4. *Classification Models (`src/model/train.py`, `src/model/calibration.py`):*
     - Logistic Regression (class-weighted, balanced)
     - Multi-Layer Perceptron (MLP)
     - Ablation variants (Block A only, Block B only)
     - Probability calibration assessment (Reliability diagram & Brier score)
  5. *Explainability Layer (`src/xai/`):*
     - SHAP LinearExplainer (`shap_explain.py`)
     - LIME Tabular Explainer (`lime_explain.py`)
     - Side-by-side attribution agreement and qualitative sanity checks.
  6. *Evaluation & Benchmarking (`src/eval/`):*
     - Comparison to KPI-Check 2022 (`compare_to_baseline.py`)
     - Feature ablation summary (`ablation_table.py`)

---

## 4. Dataset Description and Preprocessing
- **Source Filings:** Real SEC EDGAR 10-K filings across 6 major enterprise leaders (Apple, Microsoft, Tesla, Nvidia, Amazon, Alphabet).
  - *Manifest:* `data/raw/manifest.csv`
- **Extracted Corpora:**
  - Candidate sentences filtered by financial keyword lexicon and numerical presence: `data/processed/candidate_kpi_sentences.csv` (473 unique candidate KPI sentences).
  - Structured statement line items: `data/processed/extracted_line_items.csv` (353 unique financial line items).
- **Labeled Dataset (`data/labeled/pairs_labeled.csv`):**
  - Total: 150 hand-labeled pairs.
  - Class Distribution:
    - `match`: 55 instances (36.7%)
    - `no_match`: 55 instances (36.7%)
    - `ambiguous`: 40 instances (26.7%)
  - *Borderline & Ambiguous Cases:* Real-world accounting challenges including:
    - Geographic & operational sub-segments vs consolidated aggregates (e.g., Google Cloud vs Total Revenues, AWS vs Net service sales).
    - Non-GAAP adjusted metrics vs GAAP audited line items (e.g., Adjusted EBITDA vs Operating Income).
    - Ratio/percentage claims vs nominal dollar statement items (e.g., Gross margin percentage of 46.9% vs Total gross margin of $195,201M).
    - Partial rollups (e.g., regulatory automotive credits rolling into automotive sales).

---

## 5. Algorithms and Models Used
- **Classifiers:**
  - Multiclass Logistic Regression (`sklearn.linear_model.LogisticRegression(class_weight='balanced')`).
  - Multi-Layer Perceptron (`sklearn.neural_network.MLPClassifier(hidden_layer_sizes=(32, 16), max_iter=800)`).
- **Probability Calibration:**
  - `CalibratedClassifierCV` (isotonic regression) with reliability curve inspection.
- **Explainability Formulations:**
  - SHAP: Additive feature attribution method computing exact Shapley values via `LinearExplainer`.
  - LIME: Tabular perturber approximating decision boundary with an interpretable local surrogate.

---

## 6. Preliminary Experimental Results
*(Referenced from `results/tables/` and `results/figures/`)*

### 6.1 Model Performance Summary
From `results/tables/model_comparison.csv`:
| Model | Accuracy | Micro-F1 | Macro-F1 | Precision (Micro) | Recall (Micro) |
|---|---|---|---|---|---|
| **Logistic Regression (Block A+B)** | **93.33%** | **93.33%** | **92.80%** | **93.33%** | **93.33%** |
| Logistic Regression (Block A only) | 90.00% | 90.00% | 89.36% | 90.00% | 90.00% |
| MLP Classifier (Block A+B) | 86.67% | 86.67% | 85.93% | 86.67% | 86.67% |
| Logistic Regression (Block B only) | 66.67% | 66.67% | 64.67% | 66.67% | 66.67% |

- *Figures:*
  - `results/figures/confusion_matrix_logistic_regression_full.png`
  - `results/figures/confusion_matrix_mlp_classifier_full.png`
  - `results/figures/confusion_matrix_logistic_regression_block_a_only.png`
  - `results/figures/confusion_matrix_logistic_regression_block_b_only.png`

### 6.2 Model Calibration
From `results/tables/calibration_metrics.csv` & `results/figures/calibration.png`:
- Original Logistic Regression Brier Score: `0.0007`
- Isotonic Calibrated Brier Score: `0.0000`
- Demonstrates near-ideal alignment between model confidence and empirical accuracy.

### 6.3 XAI Interpretability Findings (SHAP & LIME)
From `results/tables/shap_feature_importance.csv`:
| Rank | Feature | Mean Absolute SHAP | Attribution Category |
|---|---|---|---|
| 1 | `numeric_value_match` | 1.2241 | Block A (Numerical) |
| 2 | `numeric_value_close` | 1.2241 | Block A (Numerical) |
| 3 | `embedding_cosine_similarity` | 0.3054 | Block B (Semantic) |
| 4 | `string_similarity` | 0.1784 | Block A (Lexical) |
| 5 | `keyword_overlap` | 0.1118 | Block A (Lexical) |
| 6 | `sentence_length` | 0.0763 | Block A (Structural) |
| 7 | `line_item_name_length` | 0.0468 | Block A (Structural) |
| 8 | `period_match` | 0.0000 | Block A (Temporal) |

- *Global Summary Plot:* `results/figures/shap_summary.png`
- *Dependence Plots:*
  - `results/figures/shap_dependence_top1_numeric_value_match.png`
  - `results/figures/shap_dependence_top2_numeric_value_close.png`
  - `results/figures/shap_dependence_top3_embedding_cosine_similarity.png`
- *Local Instance Analysis (5 Case Studies):*
  - Example 1 (Correct Match): `results/figures/shap_example_1.png` & `results/figures/lime_example_1.png`
  - Example 2 (Correct Match): `results/figures/shap_example_2.png` & `results/figures/lime_example_2.png`
  - Example 3 (Correct No-Match): `results/figures/shap_example_3.png` & `results/figures/lime_example_3.png`
  - Example 4 (Correct No-Match): `results/figures/shap_example_4.png` & `results/figures/lime_example_4.png`
  - Example 5 (Misclassification / Edge Case): `results/figures/shap_example_5.png` & `results/figures/lime_example_5.png`

### 6.4 SHAP vs. LIME Concordance
From `results/tables/shap_vs_lime_agreement.csv`:
| Example ID | Category | SHAP Top 3 Features | LIME Top 3 Features | Agreement Rate |
|---|---|---|---|---|
| 1 | Correct Match (1) | `numeric_value_close`, `numeric_value_match`, `embedding_cosine_similarity` | `numeric_value_close`, `numeric_value_match`, `embedding_cosine_similarity` | 100.0% |
| 2 | Correct Match (2) | `numeric_value_close`, `numeric_value_match`, `embedding_cosine_similarity` | `numeric_value_match`, `numeric_value_close`, `embedding_cosine_similarity` | 100.0% |
| 3 | Correct No-Match (1) | `numeric_value_close`, `numeric_value_match`, `embedding_cosine_similarity` | `numeric_value_match`, `numeric_value_close`, `string_similarity` | 66.7% |
| 4 | Correct No-Match (2) | `numeric_value_close`, `numeric_value_match`, `embedding_cosine_similarity` | `numeric_value_close`, `numeric_value_match`, `keyword_overlap` | 66.7% |
| 5 | Misclassification (Edge) | `numeric_value_close`, `numeric_value_match`, `embedding_cosine_similarity` | `numeric_value_match`, `numeric_value_close`, `sentence_length` | 66.7% |
- **Mean Top-3 Feature Attribution Concordance:** **80.0%**

---

## 7. Comparison of Results & Baseline Benchmark
From `results/tables/baseline_comparison.csv` and `results/figures/baseline_comparison.png`:
| Dimension | KPI-Check (2022 Paper) | This Project |
|---|---|---|
| **Micro-F1 Performance** | **73.00%** | **93.33%** |
| Primary Methodology | BERT-based NER + relation extraction, text-pair classifier | Interpretable hand-crafted + dense embeddings, Logistic Regression |
| Dataset Scale & Provenance | Proprietary German commercial audit reports (auditing firm) | 150 hand-labeled pairs across 6 public SEC 10-K filings |
| Language & Accounting Standard | German (HGB / IFRS commercial filings) | English (US-GAAP SEC Form 10-K filings) |
| Explainability (XAI Layer) | Not evaluated / black-box deep representations | Dual SHAP (LinearExplainer) + LIME interpretability layer |
| Computational Footprint | Heavy fine-tuned Transformer (GPU intensive) | Lightweight linear classifier + exact Shapley computation (CPU real-time) |

### Methodological Caveat & Academic Honesty Note
The experimental score achieved here (93.33% Micro-F1) is higher than the published 73.00% Micro-F1 of Hillebrand et al. (2022). However, these scores **are not directly comparable as a claim of model superiority**. Hillebrand et al. evaluated large-scale, messy, proprietary German filings with full end-to-end relation extraction from OCR text. In contrast, our setup evaluates a focused, curated English corpus of SEC 10-K disclosures designed specifically to test explainability and feature contribution dynamics. The primary contribution of this work is **not** to supersede their architecture, but rather to introduce auditable, transparent XAI layers that address the interpretability deficit in automated financial auditing.

---

## 8. Ablation Analysis
From `results/tables/ablation_summary.csv` and `results/figures/ablation_chart.png`:
| Configuration | Micro-F1 | Macro-F1 | Precision | Recall | SHAP Computation Profile |
|---|---|---|---|---|---|
| **Block A + B Full Set (Logistic Regression)** | **93.33%** | **92.80%** | **93.33%** | **93.33%** | **1.4 ms** (LinearExplainer - Exact) |
| Block A Only (Hand-Crafted Features, LR) | 90.00% | 89.36% | 90.00% | 90.00% | 1.1 ms (LinearExplainer - Exact) |
| Block A + B Full Set (MLP Classifier) | 86.67% | 85.93% | 86.67% | 86.67% | 850.0 ms (KernelExplainer - Sampling) |
| Block B Only (Dense Embeddings, LR) | 66.67% | 64.67% | 66.67% | 66.67% | 1.3 ms (LinearExplainer - Exact) |

### Key Tradeoff Insights
1. **Dense Embeddings Alone are Insufficient (66.67% F1):** Pure semantic similarity cannot differentiate whether a numerical figure ($416B vs $12B) matches the audited statement line item, leading to frequent false matches between related financial items.
2. **Transparent Hand-Crafted Features Dominate (90.00% F1):** Exact and tolerant numeric value matching provide the primary discriminating signal.
3. **Synergy in Combined Model (93.33% F1):** Combining semantic similarity with strict numerical logic resolves ambiguous cases where wording differs from standard statement headers.
4. **Explainability Cost (Linear vs. Non-linear):** LinearSHAP computes exact Shapley attributions in ~1.4 ms per sample on CPU, whereas non-linear MLP models require KernelSHAP sampling taking ~850 ms per sample without providing any performance advantage (86.67% vs 93.33%).
