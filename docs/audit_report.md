# Forensic Audit Report: Inconsistencies, Data Leakage, and Pipeline Remediation

**Project:** Explainable Financial KPI-Matching  
**Date:** 2026-10-09  
**Authority:** Single Source of Truth (`results/metrics_master.json`)  
**Compliance Mandate:** [PROJECT_CONTEXT.md](file:///c:/Users/welcome/Downloads/KPI-CHECK-/PROJECT_CONTEXT.md) Hard Rules 1–6  

---

## 1. Executive Summary & Audit Mandate

A comprehensive forensic audit of the codebase, raw filings, labeling procedures, feature pipelines, and evaluation routines was conducted to identify and resolve discrepancies and data leakage prior to expanding the project architecture.

### Key Audit Findings
1. **Critical Split Leakage:** Under the legacy random 80/20 train/test split, **123 of the 150 hand-labeled GOLD pairs (82.0%) leaked into the training set**, directly violating **Hard Rule 1** (*"GOLD is test-only: never used for training, tuning or calibration"*) and **Hard Rule 2** (*"All splits are by company/filing, never random"*).
2. **Training Calibration Leakage:** In `src/model/calibration.py`, isotonic probability calibration was fitted via `CalibratedClassifierCV(..., cv="prefit")` on $X_{\text{train}}$—the identical dataset on which the base model had already been trained—distorting probability mapping through overfitted training predictions.
3. **Heuristic Label Leakage:** In external dataset ingestion (`scripts/fetch_and_prepare_external_datasets.py`), labels were generated using numeric tolerance (`rel_diff <= 0.02`), making `numeric_value_match` (`rel_diff <= 0.01`) an almost deterministic rule ($98.85\%$ precision and $98.85\%$ recall). Furthermore, the 150 GOLD pairs lacked hard numerical negative distractors.
4. **Spurious Feature Artifact:** `sentence_length` acted as a dataset-source shortcut. SEC EDGAR sentence lengths are uniform across classes ($\sim 70\text{--}74$ characters), but TAT-QA ingested full multi-sentence paragraphs, making `match` (mean $491.8$ chars) and `ambiguous` (mean $433.9$ chars) drastically longer than `no_match` ($283.0$ chars). This inflated `sentence_length` to the #3 most influential SHAP feature.
5. **Score Impact under Leak-Free Protocol:** When the pipeline was remediated by strictly holding out GOLD as test-only and calibrating via 5-fold CV on external training data, **Logistic Regression proved robust (Micro-F1: $86.67\%$ vs legacy $85.71\%$)**, while **MLP Classifier performance collapsed from $85.03\%$ to $68.67\%$ (a $16.36\%$ drop)** due to severe overfitting to external dataset artifacts. Block B (Dense Embeddings only) collapsed from $61.22\%$ to $42.00\%$.

---

## 2. Empirical Discrepancy Breakdown

Every numerical inconsistency identified across presentation slides, drafts, and CSV tables was traced to its empirical root cause:

| Discrepancy | Reported in Slides / Drafts | Verified Empirical Truth | Root Cause | Fix Applied |
| :--- | :--- | :--- | :--- | :--- |
| **1. Candidate Sentences** | Table sums to **833** (AAPL 54, MSFT 297, TSLA 125, NVDA 138, AMZN 123, GOOGL 96); text footer states **473**. | **Raw sum = 833**<br>**Unique total = 473** | The table listed raw sentence counts extracted per HTML filing prior to global deduplication. Global deduplication across filings removed repeated boilerplate clauses, yielding 473 unique sentences. | Both numbers are documented in `metrics_master.json`. Deduplicated per-filing counts: MSFT 149, NVDA 81, AMZN 74, TSLA 72, GOOGL 61, AAPL 36 (Total: 473). |
| **2. Class Distribution** | Slide 5: **55/55/40 pairs** (36.7% / 36.7% / 26.7%).<br>Slide 12: **27.2% amb / 11.6% match / 61.2% no_match**. | **GOLD 150:** 55 match, 55 no_match, 40 ambiguous.<br>**Legacy Test 147:** 17 match (11.6%), 90 no_match (61.2%), 40 ambiguous (27.2%). | Two completely different data subsets were conflated: the 150 hand-labeled GOLD dataset vs the 147-sample test partition of the 731-sample combined dataset. | Hard Rule 1 enforced: GOLD is test-only ($N = 150$). Legacy 147-sample split retired. |
| **3. Test Set Size & "17/17 Match"** | Test set cited as 150 in narrative, but "17/17 Match" claimed in confusion matrix / recall notes. | **Legacy Test: 147** (Support: match = 17, no_match = 90, ambiguous = 40).<br>**LR match recall: 17/17 (100.0%)**. | "17/17" was the exact empirical recall on the 17 match instances inside the legacy 147-sample test set. It was mistakenly conflated with the 55 match pairs in GOLD. | On the leak-free 150-sample GOLD test set, match support is 55, and LR correctly classifies all 55/55 ($100.0\%$ recall, $98.2\%$ precision). |
| **4. MLP Micro-F1** | Slide 8: **86.67% Micro-F1**.<br>Slide 10: **85.03% Micro-F1**. | **Legacy 30-sample test:** $26/30 = 86.67\%$.<br>**Legacy 147-sample test:** $125/147 = 85.03\%$.<br>**Leak-Free GOLD 150:** $68.67\%$. | Slide 8 retained a legacy number from an earlier 30-sample test split ($26/30 = 86.67\%$), while Slide 10 updated to the 147-sample test set ($125/147 = 85.03\%$). | All documentation synchronized to `metrics_master.json`. Under leak-free protocol, MLP achieves 68.67%. |
| **5. Brier Scores** | Slide 8: **0.0000**.<br>Slide 10: **0.0001** (calibrated) vs **0.0003** (original). | **Legacy uncalibrated:** $0.00025002 \to 0.0003$.<br>**Legacy calibrated:** $0.00012324 \to 0.0001$.<br>**Reduction:** $50.71\%$. | Slide 8 rounded 0.0001 to 0.0000 or inherited an early draft that used fewer decimal places. | Enforced uniform 4-decimal precision ($0.0003 \to 0.0001$, $50.71\%$ reduction) for legacy baseline; documented true leak-free calibration ($0.0070 \to 0.0078$). |
| **6. SHAP-LIME Concordance** | Slide 11: **93.3%**.<br>Slide 14 / Abstract: **80.0%**. | **Feature-slot overlap:** $14/15 = 93.33\%$.<br>**Exact instance match:** $4/5 = 80.00\%$. | 93.3% represents the mean top-3 feature overlap rate across all 15 evaluated slots. 80.0% represents the proportion of instances (4 of 5) with 100% top-3 agreement. | Explicitly distinguish mean feature overlap (93.3%) from exact instance concordance (80.0%). |

---

## 3. Forensic Leakage Investigation

### Leakage Check 1: Train/Test Split Leakage
* **Mechanism:** In `src/model/train.py`, `train_test_split(df, test_size=0.20, random_state=42, stratify=df["label"])` was executed on `features.csv` (731 rows).
* **Finding:** The 731 rows contained 150 SEC EDGAR GOLD pairs. The random split allocated:
  * **123 GOLD pairs to `train_split.csv`** ($82.0\%$ of GOLD).
  * **27 GOLD pairs to `test_split.csv`** ($18.0\%$ of GOLD).
* **Violation:** Violated **Hard Rule 1** (*"GOLD is test-only: never used for training, tuning or calibration"*) and **Hard Rule 2** (*"All splits are by company/filing, never random"*).
* **Verdict:** **FAIL — CRITICAL LEAKAGE.**

### Leakage Check 2: Calibration Fitting Protocol
* **Mechanism:** In `src/model/calibration.py`, probability calibration was implemented as:
  ```python
  calibrated_clf = CalibratedClassifierCV(pipeline, method="isotonic", cv="prefit")
  calibrated_clf.fit(X_train, y_train)
  ```
  where `pipeline` was already fit on `(X_train, y_train)`.
* **Finding:** When `cv="prefit"`, `CalibratedClassifierCV` assumes the estimator was fit on a distinct dataset and that the data passed to `fit()` is an independent validation set. Passing the training data violates this assumption: the training predictions are overly confident, so isotonic regression fits an overfitted confidence distribution.
* **Verdict:** **FAIL — TRAINING CALIBRATION LEAKAGE.**

### Leakage Check 3: Heuristic Label Leakage
* **Mechanism:** In `scripts/fetch_and_prepare_external_datasets.py`, pairs from FinQA and TAT-QA were labeled using the rule:
  ```python
  if rel_diff <= 0.02:
      label = "match"
  elif any(word in sent.lower() for word in line_name.lower().split() if len(word) > 3):
      label = "ambiguous"
  else:
      label = "no_match"
  ```
* **Finding:** The primary feature `numeric_value_match` was computed in `src/features/build_features.py` as `rel_diff <= 0.01`. As a result:
  * Across all 731 samples: `numeric_value_match == 1` yielded **86 matches, 1 ambiguous, 0 no-matches** ($98.85\%$ precision, $98.85\%$ recall).
  * Across the 150 GOLD pairs: `numeric_value_match == 1` yielded **55 matches, 1 ambiguous, 0 no-matches** ($98.21\%$ precision, $100.00\%$ recall).
  * The dataset completely lacked **hard numerical negatives** (pairs where numbers match but line items differ conceptually).
* **Verdict:** **FAIL — HEURISTIC LABEL LEAKAGE.**

### Leakage Check 4: Spurious Feature Artifact (`sentence_length`)
* **Mechanism:** `sentence_length` (character length of narrative text) was extracted as a Block A feature.
* **Finding:** Comparing `sentence_length` distributions across data sources reveals severe bias:
  * **SEC EDGAR (Primary 150 pairs):**
    * `match`: mean $70.4 \pm 17.4$ chars (median 68.0)
    * `no_match`: mean $72.3 \pm 17.5$ chars (median 75.0)
    * `ambiguous`: mean $73.6 \pm 15.4$ chars (median 74.0)
    * *Distribution is uniform across classes.*
  * **TAT-QA (362 pairs):**
    * `match`: mean $491.8 \pm 287.0$ chars (median 481.0)
    * `ambiguous`: mean $433.9 \pm 284.3$ chars (median 381.0)
    * `no_match`: mean $283.0 \pm 318.7$ chars (median 134.0)
    * *Multi-sentence paragraphs inflated match and ambiguous lengths by $>70\%$.*
  * **Impact on Interpretability:** SHAP ranked `sentence_length` as the **#3 most important feature** overall (mean $|\text{SHAP}| = 0.4172$). The model learned a dataset-specific artifact: longer text chunk = match/ambiguous.
* **Verdict:** **FAIL — SPURIOUS DATASET SHORTCUT.**

---

## 4. Pipeline Remediation & Protocol Fixes

To achieve full compliance with `PROJECT_CONTEXT.md`, the following architectural changes were implemented:

1. **Strict Held-Out GOLD Test Split (Hard Rule 1):**
   * Modified `src/model/train.py` with an automated split selector (`--split [leak_free|legacy]`).
   * By default (`leak_free`), `train_split.csv` contains exclusively external pairs ($N = 581$, $0$ GOLD pairs).
   * `test_split.csv` contains exclusively the 150 hand-labeled SEC EDGAR pairs ($N = 150$, GOLD).
   * Zero train/test leakage.
2. **Proper Out-of-Fold Calibration (Fix for Leakage 2):**
   * Modified `src/model/calibration.py` to use `CalibratedClassifierCV(pipeline, method="isotonic", cv=5)`.
   * Isotonic regression is fitted strictly on 5-fold out-of-fold validation predictions generated on the training set, eliminating overconfident training probability leakage.
3. **Master Metrics Registry (Hard Rule 3):**
   * Created `results/metrics_master.json` containing every verified metric, data count, and diagnostic result.

---

## 5. Experimental Results: Legacy (With Leakage) vs Leak-Free Comparison

Below is the verified head-to-head comparison between the old pipeline and the remediated leak-free pipeline:

| Metric / Dimension | Legacy Pipeline (Random 80/20, Leaked GOLD) | Leak-Free Pipeline (Train External 581, Test GOLD 150) | Difference / Impact |
| :--- | :---: | :---: | :---: |
| **Training Set Size ($N_{\text{train}}$)** | 584 (incl. 123 GOLD samples) | 581 (0 GOLD samples) | $-3$ samples (100% clean) |
| **Test Set Size ($N_{\text{test}}$)** | 147 (incl. 27 GOLD samples) | 150 (Pure SEC GOLD) | $+3$ samples (100% clean) |
| **LR Full (Block A+B) Micro-F1** | 85.71% | **86.67%** | $+0.96\%$ |
| **LR Full (Block A+B) Macro-F1** | 87.61% | **85.45%** | $-2.16\%$ |
| **LR Full (Block A+B) Accuracy** | 85.71% | **86.67%** | $+0.96\%$ |
| **MLP Classifier Micro-F1** | 85.03% (claimed 86.67%) | **68.67%** | **$-16.36\%$ (Massive Drop)** |
| **MLP Classifier Macro-F1** | 86.37% | **67.23%** | **$-19.14\%$ (Massive Drop)** |
| **LR Block A Only Micro-F1** | 79.59% | **84.00%** | $+4.41\%$ |
| **LR Block B Only Micro-F1** | 61.22% | **42.00%** | **$-19.22\%$ (Embeddings collapse)** |
| **Match Class Precision / Recall** | 100.0% / 100.0% (17/17) | **98.21% / 100.0% (55/55)** | $-1.79\%$ Prec, $100\%$ Rec |
| **No-Match Class Precision / Recall** | 89.66% / 86.67% (78/90) | **82.14% / 83.64% (46/55)** | Realistic baseline |
| **Ambiguous Class Precision / Recall**| 72.09% / 77.50% (31/40) | **76.32% / 72.50% (29/40)** | $+4.23\%$ Prec, $-5.00\%$ Rec |
| **Uncalibrated Brier Score (Match)** | 0.0003 ($0.00025$) | **0.0070** ($0.00699$) | Realistic uncalibrated loss |
| **Calibrated Brier Score (Match)** | 0.0001 ($0.00012$) | **0.0078** ($0.00776$) | Honest out-of-fold calibration |
| **Calibration Error Reduction** | 50.71% reduction | **$-11.04\%$** | External-to-SEC shift |
| **LinearSHAP Latency** | 1.4 ms / sample | 1.4 ms / sample | Unchanged |
| **KernelSHAP (MLP) Latency** | 850.0 ms / sample | 850.0 ms / sample | Unchanged (>600× slower) |
| **SHAP vs LIME Concordance** | 93.3% feature / 80.0% exact | 93.3% feature / 80.0% exact | Unchanged on verified subset |

### Analysis of Score Movements
1. **Why Logistic Regression Maintained Performance ($86.67\%$):**
   * The linear model relies heavily on `numeric_value_match` (positive weight) and `embedding_cosine_similarity` (positive weight). Because the 150 GOLD pairs share the same fundamental numeric and lexical signals, the linear weights generalized well across domains.
2. **Why MLP Performance Collapsed ($85.03\% \to 68.67\%$):**
   * As required by **Hard Rule 5** (*"If something fails or accuracy drops, say so plainly. Never synthesize or invent data"*), this drop is documented openly.
   * The non-linear MLP learned complex multi-feature interactions that fit the specific text formatting and sentence length patterns of TAT-QA and FinQA. When evaluated on real SEC 10-K sentences where `sentence_length` was uniform, the MLP's non-linear decision surface suffered substantial cross-domain error, achieving only $45.45\%$ recall on the match class.
3. **Why Dense Embeddings Alone (Block B) Collapsed ($61.22\% \to 42.00\%$):**
   * Cosine similarity alone cannot differentiate accounting subtleties. Without lexical and numerical constraints, pure embeddings confused ambiguous line items with true matches, achieving only $9.09\%$ recall on `match`.

---

## 6. Recommendations for Future Development

1. **Hard Negative Mining:** Construct challenging negative pairs in SEC EDGAR where line items share the exact same numeric value (e.g., pairing a revenue sentence with a debt line item that shares the identical dollar amount) to force the model to rely on accounting semantics rather than purely numeric coincidence.
2. **Standardize Text Chunking:** Run all external documents through the identical sentence segmentation tokenizer used for SEC filings to eliminate length discrepancies between datasets.
3. **Domain Adaptation for Neural Models:** When re-introducing non-linear classifiers, implement regularized domain adaptation or freeze lexical-numeric rules to prevent shortcut learning.
