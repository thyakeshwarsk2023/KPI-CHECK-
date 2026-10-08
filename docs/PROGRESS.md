# Project Progress Log

A running log of all tasks, deliverables, and milestones for the **Explainable Financial KPI-Matching** capstone project.

---

## Entry 1: 2026-10-08 — Review 2 PPT Overhaul & Experimental Metric Alignment

### What Was Done
1. Re-structured Review 2 presentation deck (`review2_presentation.pptx`):
   - Removed internal file trees and codebase architecture slides (Slide 9).
   - Added rigorous **Methodology & Experimental Validation Framework** slide including Stratified CV protocol, Double-Blind Annotation ($\kappa = 0.8841$), and 5-Fold CV stability boxplot.
   - Formalized **Literature Review & Identified Research Gaps** in a $2 \times 2$ matrix detailing black-box opacity, numerical scale blindness, and latency bottlenecks.
2. Performed deep audit across all tables, charts, plots, and figures in `results/`:
   - Resolved numerical contradictions between text and charts.
   - Synchronized all slide text and tables to match empirical results from `results/tables/model_comparison.csv`, `results/tables/ablation_summary.csv`, and `results/tables/cross_validation_metrics.csv`.
   - Corrected aspect-ratio distortion across figures in `scripts/generate_review2_ppt.py` (preventing vertical/horizontal stretching for wide plots like `calibration.png` and `shap_example_5.png`).
3. Created foundational repo context and running progress log:
   - Created `PROJECT_CONTEXT.md` defining project rules, data sources, and hard constraints.
   - Created `docs/PROGRESS.md` for continuous task tracking.

### Files Created / Modified
- `PROJECT_CONTEXT.md` (Created): Hard rules, data sources, assets, and project scope.
- `docs/PROGRESS.md` (Created): Running task and milestone log.
- `scripts/generate_review2_ppt.py` (Modified): Replaced code tree slide, fixed image geometry, updated metrics.
- `docs/review2_presentation.pptx` (Regenerated): 16-slide widescreen presentation deck.
- `docs/review2_presentation_deck.md` (Modified): Speaker notes synchronized with empirical figures.

### Key Numbers
- **Test Micro-F1 (LR Full A+B):** 85.71% (Macro-F1: 87.61%, Accuracy: 85.71%)
- **MLP Classifier Micro-F1:** 85.03% (Macro-F1: 86.37%)
- **Feature Ablation (Micro-F1):** Block A Only = 79.59% | Block B Only = 61.22% | Synergy = 85.71%
- **5-Fold Stratified CV Micro-F1:** LR Full = 83.99% ± 3.33% | MLP = 86.59% ± 2.31%
- **Inter-Annotator Agreement:** Cohen's Kappa $\kappa = 0.8841$ (92.50% raw agreement across 40 pairs)
- **Isotonic Calibration:** Brier score reduced from 0.0003 to 0.0001 (50.71% reduction)
- **Inference Latency:** LinearSHAP = 1.4 ms/sample vs KernelSHAP (MLP) = 850.0 ms/sample (>600× speedup)
- **Baseline Context:** KPI-Check 2022 = 73.00% (German reports; reference only)

### Open Issues / Next Steps
- Implement `results/metrics_master.json` as the single authoritative ground truth for all document numbers (Rule 3) [COMPLETED in Entry 2].
- Begin data ingestion expansion scripts (`scripts/download_*.py`) adhering to SEC XBRL and QA dataset rules.
- Prepare company/filing-based splits for extended datasets (Rule 2).
- Final review preparation for 16–21 Oct 2026.

---

## Entry 2: 2026-10-09 — Forensic Audit, Data Leakage Remediation & Metrics Registry

### What Was Done
1. **Recomputed All Metrics from Code & Data:**
   - Traced raw candidate sentence sum (833) vs unique global candidate sentences (473).
   - Reconciled class distributions across GOLD 150 (55 match / 55 no_match / 40 ambiguous) vs legacy test split 147 (17 match / 90 no_match / 40 ambiguous).
   - Clarified "17/17 match" claim as empirical recall on the 17 match instances in the legacy 147 test set.
   - Traced MLP Micro-F1 (86.67% on legacy 30-sample test vs 85.03% on legacy 147-sample test vs 68.67% on leak-free GOLD 150).
   - Verified Brier score decimal precision: uncalibrated 0.00025 (0.0003) to calibrated 0.00012 (0.0001) with 50.71% reduction (and legacy rounded 0.0000).
   - Reconciled SHAP-LIME concordance: 93.3% mean feature overlap (14/15 slots) vs 80.0% exact instance concordance (4/5 examples).
2. **Executed 4 Forensic Leakage Checks:**
   - Check 1 (Split Leakage): Found that 123 of 150 GOLD test pairs (82.0%) leaked into `train_split.csv` via random splitting. **Verdict: CRITICAL LEAKAGE.**
   - Check 2 (Calibration Leakage): Found that `CalibratedClassifierCV` used `cv="prefit"` with $X_{\text{train}}$, fitting isotonic regression on overfitted training predictions. **Verdict: TRAINING LEAKAGE.**
   - Check 3 (Heuristic Label Leakage): External labels were assigned via `rel_diff <= 0.02`, making `numeric_value_match` an almost deterministic proxy (98.85% precision & recall). **Verdict: HEURISTIC LEAKAGE.**
   - Check 4 (Spurious Feature Shortcut): TAT-QA paragraphs (mean ~450–492 chars) inflated `match` and `ambiguous` length vs `no_match` (283 chars), while SEC EDGAR sentences are uniform (~70–74 chars), falsely driving `sentence_length` to #3 in SHAP importance. **Verdict: SPURIOUS SHORTCUT.**
3. **Remediated Pipeline Architecture:**
   - Enforced **Hard Rule 1**: GOLD (150 pairs) is held out strictly as test-only ($N_{\text{test}} = 150$). Training is performed exclusively on external benchmarks ($N_{\text{train}} = 581$, 0 GOLD samples).
   - Fixed Calibration: Replaced `cv="prefit"` with 5-fold cross-validation on the training set (`CalibratedClassifierCV(cv=5)`), eliminating training prediction reuse.
   - Updated `src/model/train.py`, `src/model/calibration.py`, and evaluated downstream scripts.
4. **Created Master Registry & Audit Report:**
   - Created `results/metrics_master.json` containing every verified empirical metric as the single source of truth (**Hard Rule 3**).
   - Authored `docs/audit_report.md` detailing all discrepancies, root causes, fixes, and head-to-head comparisons.

### Files Created / Modified
- `results/metrics_master.json` (Created): Authoritative single source of truth for all metrics.
- `docs/audit_report.md` (Created): Full forensic audit report.
- `scripts/build_metrics_master.py` (Created): Automation script generating `metrics_master.json`.
- `scripts/reproduce_and_fix_pipeline.py` (Created): Verification and head-to-head comparison script.
- `src/model/train.py` (Modified): Added leak-free train/test splitting adhering to Hard Rule 1.
- `src/model/calibration.py` (Modified): Enforced 5-fold CV calibration to eliminate training reuse leakage.
- `results/tables/model_comparison.csv` (Regenerated): Updated with leak-free numbers.
- `results/tables/per_class_metrics.csv` (Regenerated): Updated with 150 GOLD test instances.
- `results/tables/calibration_metrics.csv` (Regenerated): Updated with honest out-of-fold calibration.
- `results/tables/ablation_summary.csv` (Regenerated): Updated with leak-free ablation performance.
- `docs/PROGRESS.md` (Modified): Appended Entry 2.

### Key Numbers (Old vs Leak-Free Verified)
- **LR Full (Block A+B) Micro-F1:** Old = 85.71% | **Leak-Free Verified = 86.67%** (+0.96%)
- **LR Full (Block A+B) Macro-F1:** Old = 87.61% | **Leak-Free Verified = 85.45%** (-2.16%)
- **MLP Classifier Micro-F1:** Old = 85.03% (claimed 86.67%) | **Leak-Free Verified = 68.67%** (-16.36% drop due to external artifact overfitting)
- **LR Block A Only Micro-F1:** Old = 79.59% | **Leak-Free Verified = 84.00%** (+4.41%)
- **LR Block B Only Micro-F1:** Old = 61.22% | **Leak-Free Verified = 42.00%** (-19.22% drop without lexical/numeric rules)
- **Match Recall (LR Full):** Old = 100.0% (17/17) | **Leak-Free Verified = 100.0% (55/55 on pure GOLD)**
- **Test Set Size:** Old = 147 (with 27 leaked GOLD pairs) | **Leak-Free Verified = 150 (Pure GOLD)**

---

## Entry 3: 2026-10-09 — SEC XBRL Line-Item Store Ingestion & Validation

### What Was Done
1. **Automated SEC EDGAR XBRL Data Downloader (`scripts/download_sec.py`):**
   - Implemented compliant SEC REST API fetching (`https://data.sec.gov/api/xbrl/companyfacts/CIK{cik}.json`).
   - Configured `User-Agent: s.k.thyakeshwar skthyakeshwar@gmail.com` with max 4 req/sec rate limiting and exponential backoff retry.
   - Downloaded and cached raw JSON files for 33 companies across 6 sectors (Tech, Healthcare, Banking, Energy, Retail, Industrials). Includes all 6 GOLD test set companies (AAPL, MSFT, TSLA, NVDA, AMZN, GOOGL) and mandatory healthcare/banking inclusions (JNJ, JPM).
   - Logged download metadata, file sizes, SHA256 checksums, and timestamps to `data/raw/manifest.csv`.
2. **Built Structured Line-Item Pipeline (`src/data/build_line_items.py`):**
   - Transformed raw XBRL JSON companyfacts into `data/processed/line_items.parquet`.
   - Extracted 15 normalized schema columns covering fiscal years 2021–2025 across Forms 10-K, 10-Q, 10-K/A, 10-Q/A.
   - Formatted values in base units, standardized unit identifiers (`USD`, `shares`), and classified instant vs duration concepts.
3. **Restatement & Deduplication Rule:**
   - Formalized filing-level accession isolation: facts are preserved under their reported `accession` number so narrative claim matching evaluates figures as reported in that specific filing.
   - Deduplicated intra-filing duplicate facts by selecting the latest filed date and complete record.
4. **Executed Automated Sanity Checks & 20-Row Random Spot-Check:**
   - Total extracted records: **166,880 clean line items** across fiscal years 2021–2025.
   - Analyzed top 30 most frequent US-GAAP tags (led by `NetIncomeLoss`, `CashCashEquivalentsRestrictedCashAndRestrictedCashEquivalents`, `EarningsPerShareDiluted`).
   - **Spot-Check Pass Rate:** 20 / 20 (100.0%) spot-checked random rows matched raw SEC EDGAR JSON files with exact decimal equality.
5. **Authored SEC XBRL Datasheet (`docs/datasheet_sec.md`):**
   - Documented dataset specifications, per-company counts, deduplication methodology, and bank-specific taxonomy caveats (e.g., banks report `NetInterestIncome`, `NoninterestIncome`, `LoansAndLeasesReceivable` instead of commercial `Revenues` or `OperatingIncomeLoss`).

### Files Created / Modified
- `scripts/download_sec.py` (Created): Compliant SEC EDGAR downloader with caching, backoff, and manifest logging.
- `src/data/build_line_items.py` (Created): XBRL line-item extraction pipeline producing `data/processed/line_items.parquet`.
- `data/processed/line_items.parquet` (Created): 166,880 structured line-item store (1.12 MB).
- `docs/datasheet_sec.md` (Created): Full datasheet for SEC XBRL dataset.
- `data/raw/manifest.csv` (Updated): Appended metadata for 33 SEC companyfacts JSON files.
- `docs/PROGRESS.md` (Modified): Appended Entry 3.

### Key Metrics
- **Total Line Items:** 166,880
- **Companies / Sectors:** 33 companies across Tech, Healthcare, Banking, Energy, Retail, Industrials
- **Spot-Check Accuracy:** 20 / 20 (100.0% match)
- **Parquet Store Size:** 1.12 MB
