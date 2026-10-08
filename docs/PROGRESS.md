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
   - Total extracted records: **298,663 clean line items** across fiscal years 2021–2025.
   - Analyzed top 30 most frequent US-GAAP tags (led by `NetIncomeLoss`, `CashCashEquivalentsRestrictedCashAndRestrictedCashEquivalents`, `EarningsPerShareDiluted`).
   - **Spot-Check Pass Rate:** 20 / 20 (100.0%) spot-checked random rows matched raw SEC EDGAR JSON files with exact decimal equality.
5. **Authored SEC XBRL Datasheet (`docs/datasheet_sec.md`):**
   - Documented dataset specifications, per-company counts, deduplication methodology, and bank-specific taxonomy caveats.

---

## Entry 4: 2026-10-09 — Presentation Overhaul: Leak-Free Metric Integration & Deficit Quarantine

### What Was Done
1. **Regenerated Presentation Deck (`docs/review2_presentation.pptx`):**
   - Expanded presentation deck to 18 slides in 16:9 widescreen format.
   - Established **Primary Model as Balanced Logistic Regression (No Length Features)**: **84.67% Micro-F1** (95% CI: [78.67%, 90.00%]), **83.75% Macro-F1** on held-out GOLD ($n=150$).
   - Completely excised Cohen's Kappa ($\kappa=0.8841$) and "double-blind annotation" claims everywhere; described gold as 150 author-curated pairs from real 10-K figures (single annotator).
   - Quarantined and removed all leaked legacy numbers (`85.71%`, `87.61%`, `85.03%`, `86.67%`, `17/17`, `0.0000`, `0.0001`, `50.71%`, `93.3%`, `80.0%`, `1.4 ms`, `850 ms`, `+12.71%`, and legacy 5-fold CV numbers/boxplot).
   - Replaced direct KPI-Check comparison with honest scope: *"Not directly comparable (German language, different task, proprietary uncurated data). Fair in-domain FinBERT baseline planned."*
2. **Added Dedicated Forensic Audit Slide (Slide 16):**
   - Title: **"Self-Audit: Data Leakage Identified, Remediated, and Verified"**.
   - Details all 6 diagnosed leakage pathways (split leakage, calibration on training, heuristic label leakage, text chunking shortcut, hardcoded XAI strings, unsupported kappa).
   - Shows the empirical **+16.66% recovery of MLP** (68.67% -> 85.33%) when spurious length shortcuts are ablated.
3. **Added Dedicated Limitations Slide (Slide 17):**
   - Title: **"Limitations & Threats to Validity: Academic Boundary Conditions"**.
   - Documents $n=150$ sample size (CI $\pm 5.5\%$), lack of hard numerical negatives, 6 large-cap tech concentration, US-GAAP/English scope, single annotator, and cross-domain class prior shift.
4. **Updated XAI & Calibration Slides (Slides 12, 13, 14, 15):**
   - Updated with measured XAI metrics across all 150 GOLD test claims: LinearSHAP latency **0.009 ms / sample**, KernelSHAP latency **10.97 ms / sample** (1,223.5x speedup).
   - Updated measured SHAP-LIME concordance: **80.44% mean top-3 feature overlap**, **50.67% exact instance concordance** (characterized as moderate agreement).
   - Updated calibration metrics: Uncalibrated Brier **0.0070** (ECE 0.29%) vs Calibrated **0.0078** (ECE 1.34%), with clear explanation of external-to-SEC class prior shift (5.5% vs 36.7% match).
5. **Regenerated Leak-Free Publication Figures:**
   - `confusion_matrix_logistic_regression_full.png` (LR No-Length model)
   - `per_class_metrics_barchart.png`
   - `shap_summary.png` (excludes `sentence_length` and `line_item_name_length`)
   - `ablation_chart.png` (Block A No Length: 78.67%, Block B: 42.00%, Synergy A+B: 84.67%)
   - `calibration.png` (Honest reliability diagram on held-out GOLD)
   - `baseline_comparison.png` (Updated with honest caveat)
6. **Project & Student Details Formatted:**
   - S.K. Thyakeshwar (Reg No: 23BAI0194), Sai Sanjay (Reg No: 23BAI0168), under Dr. Manikandan G (SCOPE, VIT).
   - Synchronized [`docs/review2_presentation_deck.md`](file:///c:/Users/welcome/Downloads/KPI-CHECK-/docs/review2_presentation_deck.md) with updated speaker notes.

### Files Created / Modified
- `scripts/generate_review2_ppt.py` (Modified): Full 18-slide deck generator.
- `scripts/regenerate_figures_leak_free_no_length.py` (Created): Figure generator without length features.
- `docs/review2_presentation.pptx` (Regenerated): 18-slide widescreen presentation deck.
- `docs/review2_presentation_deck.md` (Updated): Speaker notes and slide outline.
- `results/metrics_master.json` (Updated): Appended Block A No-Length metrics.
- `docs/PROGRESS.md` (Modified): Appended Entry 4.
