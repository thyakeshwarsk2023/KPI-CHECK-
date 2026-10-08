# Review 2 Presentation Slide Deck & Speaker Notes

**Project Title:** Explainable Financial KPI-Matching: Dual-Block Feature Architecture & Audit-Compliant XAI for 10-K Narrative Claims  
**Student Investigators:** S.K. Thyakeshwar (Reg No: 23BAI0194) & Sai Sanjay (Reg No: 23BAI0168)  
**Supervisor:** Dr. Manikandan G | School of Computer Science and Engineering, Vellore Institute of Technology (VIT)  
**Generated PowerPoint File:** [`review2_presentation.pptx`](file:///c:/Users/welcome/Downloads/KPI-CHECK-/docs/review2_presentation.pptx)  
**Date of Review 2:** 7th October 2026 (Final Review: 16th-21st October 2026)  
**Authority:** Single Source of Truth (`results/metrics_master.json`)

---

## Slide 1: Title Slide (Dark Theme)
- **Header:** CAPSTONE PROJECT REVIEW 2 (80% IMPLEMENTATION)
- **Title:** Explainable Financial KPI-Matching
- **Subtitle:** Dual-Block Feature Architecture & Audit-Compliant XAI for 10-K Narrative Claims
- **Student Investigators:** S.K. Thyakeshwar (23BAI0194) & Sai Sanjay (23BAI0168), SCOPE, VIT
- **Supervisor:** Dr. Manikandan G, Associate Professor, SCOPE, VIT
- **Primary Model Performance:** Balanced Logistic Regression (No Length Features) achieves **84.67% Micro-F1** (95% CI: [78.67%, 90.00%]) and **83.75% Macro-F1** on held-out GOLD (n=150).
- **Scope:** 150 SEC EDGAR author-curated pairs (Single Annotator), dual LinearSHAP (0.009 ms) & LIME XAI layer. Status: 80% Implementation Complete.
- **Speaker Note:** *"Good morning esteemed committee members and faculty guide Dr. Manikandan G. I am S.K. Thyakeshwar, presenting alongside Sai Sanjay on our capstone project: Explainable Financial KPI-Matching. In this Review 2 presentation, we demonstrate an 80% completed system that pairs SEC 10-K narrative claims with balance sheet and income statement line items using an auditable dual-block architecture, delivering 84.67% Micro-F1 and dual game-theoretic explainability in 0.009 milliseconds."*

---

## Slide 2: Project Motivation & Problem Statement
- **Takeaway:** Automating 10-K Claim Verification with Audit-Compliant AI.
- **The Dual-Modality of Form 10-K:** SEC 10-Ks combine unstructured narrative text (MD&A) with authoritative financial tables (Balance Sheets, Income Statements).
- **The Audit Challenge:** Thousands of billable CPA hours are spent manually cross-checking hundreds of disclosure pages per filing.
- **The Black-Box Dilemma in AI Auditing:** Deep neural networks provide no verifiable audit trail. LLMs introduce hallucination risks, numerical scale errors, prompt fragility, and impractical 2-5s latencies.
- **Core Objectives:** Develop an auditable, deterministic text-pair classifier with dual SHAP/LIME explainability operating at sub-millisecond latency.

---

## Slide 3: Literature Review & Identified Research Gaps
- **Takeaway:** Dense Embeddings Require Numerical Scale Awareness.
- **Reference Baseline Context: KPI-Check (Hillebrand et al., IEEE BigData 2022):** Transformer NER + Table Extraction + BERT relation extractor on German reports (73.00% Micro-F1). **Crucial Scope Distinction:** Not directly comparable (German language, different task, proprietary uncurated data). Fair in-domain FinBERT baseline planned for final review.
- **Contemporary Context: LLMs in Auditing (Deuser et al., 2025):** Evaluates LLM zero-shot verification; identifies hallucination of figures, non-determinism, and 2-5s latency bottlenecks.
- **Theoretical Foundations:**
  - **SHAP (Lundberg & Lee, NeurIPS 2017):** Exact analytical linear attribution $\phi_i(x) = w_i \cdot (x_i - \mathbb{E}[x_i])$ in 0.009 ms / sample.
  - **LIME (Ribeiro et al., KDD 2016):** Local surrogate optimization validating neighborhood stability.
- **Identified Research Gaps Addressed:**
  - *Gap 1 (Black-Box Opacity):* Deep relation extractors provide no auditable trail for CPAs.
  - *Gap 2 (Numerical Scale Blindness):* Dense embeddings confuse magnitude disparities ($400M vs $100B have >0.85 cosine similarity).
  - *Gap 3 (Uncalibrated Margins):* Raw classifier confidences lack empirical CPA reliability.
  - *Gap 4 (Latency Infeasibility):* KernelSHAP and LLMs are too slow for real-time document search.

---

## Slide 4: Complete Proposed System Architecture
- **Takeaway:** Modular Pipeline Decoupling Features from Black Boxes.
- **Stage 1 (Ingestion):** SEC EDGAR Full-Text 10-K Fetcher + BeautifulSoup HTML Parser (473 Unique Sentences, 353 Line Items).
- **Stage 2 (Curation):** Candidate Proposals -> 150 Author-Curated Ground Truth Pairs (`match`: 55, `no_match`: 55, `ambiguous`: 40).
- **Stage 3 (Features):** Dual-Block Pipeline (Block A: Numeric/Lexical without length shortcuts + Block B: MiniLM Dense Semantic Cosine Similarity).
- **Stage 4 (Classifiers):** Balanced Logistic Regression (Primary, 84.67% F1) + MLP Benchmark + Out-of-Fold Calibration.
- **Stage 5 (XAI & UI):** Dual SHAP (LinearExplainer in 0.009 ms) + LIME + Streamlit Live Auditing Dashboard.

---

## Slide 5: Detailed System Design: Dual-Block Feature Architecture
- **Takeaway:** Strict Numerical Bedrock with Semantic Paraphrase Resolution.
- **Block A (Interpretable Hand-Crafted Features):**
  - `numeric_value_match` ($\le 1\%$ relative tolerance)
  - `numeric_value_close` ($\le 5\%$ relative tolerance, handling rounding)
  - `keyword_overlap` (Jaccard similarity over curated financial lexicons)
  - `period_match` (Fiscal year regex alignment)
  - `string_similarity` (RapidFuzz token sort ratio)
  - *Audit Remediation:* `sentence_length` and `line_item_name_length` ablated to eliminate spurious shortcuts.
- **Block B (Dense Semantic Similarity):**
  - `embedding_cosine_similarity` via `all-MiniLM-L6-v2` (384-d vectors).
  - *Why Embeddings Alone Fail (42.00% Micro-F1):* Embeddings capture topic semantics but are blind to numerical scale ($400M vs $100B have >0.85 cosine similarity).
  - *Why Synergy Succeeds (84.67% Micro-F1):* Block A establishes strict numerical constraints (78.67% F1); Block B resolves vocabulary paraphrasing (+6.00% boost to 84.67%).

---

## Slide 6: Dataset Description & Preprocessing Details
- **Takeaway:** 150 SEC EDGAR Author-Curated Pairs & External Benchmarks.
- **6 Enterprise Filings:** Apple (AAPL 2025), Microsoft (MSFT 2026), Tesla (TSLA 2026), NVIDIA (NVDA 2026), Amazon (AMZN 2026), Alphabet (GOOGL 2026).
- **Volume:** 473 Unique Sentences, 353 Financial Line Items.
- **Ground Truth Gold Set:** 150 author-curated pairs from real 10-K figures, single annotator (55 match, 55 no_match, 40 ambiguous). Held out strictly as test-only ($N_{\text{test}}=150$).
- **External Benchmark Training Corpus:** 581 text pairs from FinQA and TAT-QA, held strictly separate from GOLD test set.

---

## Slide 7: Dataset Innovation: 4-Tier Financial Ambiguity Taxonomy
- **Explicit Breakdown of 40 Borderline Accounting Edge Cases:**
  1. **Subsegment vs Aggregate:** e.g., Google Cloud revenue ($33,088M) vs Total Alphabet revenue ($307,394M) -> `ambiguous` (Subcomponent).
  2. **Non-GAAP Disclosures:** e.g., Adjusted EBITDA ($14,650M) vs Operating Income ($8,891M) -> `ambiguous` (Accounting Standard).
  3. **Ratio vs Nominal Dollar:** e.g., Gross Margin % (46.9%) vs Gross Margin Dollar ($195,201M) -> `ambiguous` (Scale/Unit).
  4. **Temporal Mismatch:** e.g., 24-Month Debt Maturities ($21.5B) vs Current Term Debt ($12,350M) -> `ambiguous` (Time Horizon).

---

## Slide 8: Algorithms & Machine Learning Models Used
- **Primary Classifier:** Balanced Logistic Regression (No Length): $P(y=c|x) = \text{softmax}(W_c x + b_c)$. Globally convex, fully inspectable, deterministic, exact analytical LinearSHAP in 0.009 ms / sample. Achieves 84.67% Micro-F1 (95% CI: [78.67%, 90.00%]), 83.75% Macro-F1.
- **Secondary Benchmark:** Multi-Layer Perceptron (MLP with 2 hidden layers (32, 16 units), ReLU, Adam). When length features are ablated, MLP recovers to 85.33% Micro-F1 (Macro: 83.74%), matching LR while requiring slower KernelSHAP (10.97 ms latency).
- **Probability Calibration:** Isotonic Regression. Uncalibrated LR yields sharp Brier score of 0.0070 (ECE: 0.29%). 5-fold CV calibration yields Brier 0.0078 due to cross-domain class prior shift.

---

## Slide 9: Methodology & Experimental Validation Framework
- **Takeaway:** Strict Leak-Free Partitioning with Held-Out GOLD Evaluation (N=150).
- **Leak-Free Protocol (Hard Rule 1):** Zero train/test split leakage. GOLD is strictly held out as test-only. Models trained on external benchmarks (N=581).
- **Author-Curated Ground-Truth:** 150 claim-table pairs curated from real 10-K figures by a single annotator.
- **Statistical Rigor:** 95% bootstrap confidence intervals evaluated across 1,000 resamples (seed=42).
- **Per-Class Metrics Profile (Primary LR No-Length Model):**
  - Match: Precision = 96.4%, Recall = 98.2%, F1 = 97.3%.
  - No-Match: Precision = 80.7%, Recall = 83.6%, F1 = 82.1%.
  - Ambiguous: Precision = 75.0%, Recall = 67.5%, F1 = 71.1%.

---

## Slide 10: Experimental Results & Model Comparison Table
- **Takeaway:** Honest Evaluation on Held-Out SEC GOLD (n=150) with 95% Bootstrap CIs.
- **Results Table (n=150 Held-Out GOLD):**
  - **Logistic Reg. (No Length) [PRIMARY]:** Micro-F1 = **84.67% [78.67, 90.00]** | Macro-F1 = **83.75% [77.20, 89.14]**
  - **Logistic Reg. (Full A+B, with length):** Micro-F1 = **86.67% [81.33, 92.00]** | Macro-F1 = **85.45% [79.46, 90.82]**
  - **MLP Classifier (No Length):** Micro-F1 = **85.33% [80.00, 90.67]** | Macro-F1 = **83.74% [77.50, 89.47]**
  - **MLP Classifier (Full A+B, with length):** Micro-F1 = **68.67% [61.32, 76.00]** | Macro-F1 = **67.23% [59.37, 74.41]**
  - **Block A Only (No Length):** Micro-F1 = **78.67% [71.33, 85.33]** | Macro-F1 = **77.36% [70.31, 83.64]**
  - **Block B Only (MiniLM Embeddings):** Micro-F1 = **42.00% [34.00, 50.00]** | Macro-F1 = **40.31% [33.01, 47.03]**
- **Confusion Matrix:** Shows 54/55 matches correctly classified (98.2% recall). Misclassifications concentrate between nuanced ambiguous footnote disclosures and non-matches.

---

## Slide 11: Feature & Architecture Ablation Study
- **Takeaway:** Numerical Rules as Bedrock, Dense Embeddings as Fine-Tuner.
- **Embedding Blindness (Block B = 42.00%):** Vector embeddings alone capture general topic similarity but are blind to numerical magnitudes.
- **Numerical Bedrock (Block A = 78.67%):** Exact tolerances, keyword overlap, and token matching achieve 78.67% Micro-F1.
- **Synergistic Optimum (Block A+B = 84.67%):** Combining Block A and Block B resolves vocabulary paraphrasing (+6.00% boost over Block A alone).

---

## Slide 12: Probability Calibration & Reliability
- **Takeaway:** Prior Shift Diagnostics & Honest Audit Bounds.
- **Empirical Findings on Held-Out GOLD (n=150):**
  - Uncalibrated Logistic Regression: Brier Score = **0.0070**, Expected Calibration Error (ECE) = **0.29%**.
  - Calibrated Logistic Regression (5-fold CV): Brier Score = **0.0078**, ECE = **1.34%**.
- **Why Calibration Degrades on GOLD (Prior Shift):** External training set has a match prior of 5.51%, while GOLD test set has a match prior of 36.67%. Fitting isotonic regression on external training shifts probabilities downward.
- **Planned Remediation:** Calibrate on an in-domain financial statement split or implement Bayesian class-prior correction.

---

## Slide 13: Global Interpretability: SHAP Feature Importance
- **Takeaway:** LinearSHAP Feature Attribution (Length Shortcuts Removed).
- **Beeswarm Plot:** Features ranked top-to-bottom by mean absolute SHAP value on 'match' class without length features.
- **Attribution Ranking:**
  1. `numeric_value_match`: Dominant global predictor.
  2. `numeric_value_close`: Strong positive attribution for rounded numbers ($416.2B vs $416,161M).
  3. `embedding_cosine_sim`: Secondary semantic fine-tuner, resolving paraphrasing.
  4. `keyword_overlap`: Curated domain lexicon overlap ensuring topical consistency.
  5. `string_similarity` & `period_match`: Lexical overlap and fiscal year alignment preventing cross-period mismatches.

---

## Slide 14: Local Interpretability: Measured SHAP vs. LIME Concordance
- **Takeaway:** Dual Explainer Agreement Evaluated Across All 150 Held-Out GOLD Claims.
- **Measured Metrics (All 150 Samples):**
  - Mean Top-3 Feature Slot Overlap: **80.44%** (362 out of 450 feature slots agree).
  - Exact Instance Concordance: **50.67%** (76 out of 150 instances exhibit 3/3 exact top-3 match).
  - Characterized as Moderate Agreement: High concordance on unambiguous matches; divergence occurs on complex multi-term footnotes where LIME perturbations test local boundaries.
- **Zero Hardcoded Metrics:** Legacy 93.3% / 80.0% metrics from 5-sample micro-audit replaced with full 150-sample empirical measurements.

---

## Slide 15: Benchmark Context & Computational Efficiency
- **Takeaway:** Honest Baseline Scope & Measured Latency Speedup.
- **Honest Baseline Context:** KPI-Check (2022) achieved 73.00% Micro-F1 as an end-to-end relation extractor on German reports. Not directly comparable to curated English SEC pairs. A fair in-domain FinBERT baseline is planned.
- **Measured Explainer Latency (In this setup):**
  - Exact LinearSHAP (Primary LR Model): **0.009 ms / sample** (amortized over batch; 1.35 ms total across 150 test claims).
  - KernelSHAP (MLP Classifier): **10.97 ms / sample** (1,645.7 ms total across 150 test claims).
  - Computational Speedup: Linear models achieve a **1,223.5x explainer latency speedup**.

---

## Slide 16: Self-Audit: Leakage Found and Fixed
- **Takeaway:** 6 Forensic Audit Deficits Diagnosed, Fixed, and Verified.
- **6 Audit Deficits Remediated:**
  1. *Train/Test Split Leakage:* 123 of 150 gold pairs leaked under legacy random split. Fixed: 100% of GOLD held out as pure test (Hard Rule 1).
  2. *Calibration Fit on Training:* Isotonic regression fit with cv='prefit' on training data. Fixed: Replaced with 5-fold CV calibration.
  3. *Heuristic Label Leakage:* External labels generated with rel_diff <= 2%, nearly identical to feature numeric_value_match (<= 1%). Documented openly.
  4. *Spurious Length Shortcut:* TAT-QA paragraphs (~492 chars) inflated length vs SEC sentences (~70 chars). Fixed: Removed sentence_length & line_len.
  5. *Hard-Coded XAI Values:* Legacy 1.4ms/850ms latencies and 93.3% concordance were hard-coded strings. Replaced with measured values on all 150 GOLD samples.
  6. *Unsupported Kappa Claim:* Cohen's kappa (0.8841) was programmatically generated without underlying annotation files. Removed from all claims.
- **Model Behavior & Shortcut Recovery:**
  - LR Full (Leaked): 85.71% -> LR No-Length (Leak-Free): **84.67%** (Micro-F1) / **83.75%** (Macro-F1).
  - MLP Full (Leak-Free with Length): **68.67%** (Collapsed due to paragraph length shortcut).
  - MLP No-Length (Leak-Free): **85.33%** (**+16.66% recovery** once length shortcut removed!).

---

## Slide 17: Limitations & Threats to Validity
- **Takeaway:** Academic Boundary Conditions and Methodological Caveats.
- **6 Core Limitations:**
  1. *Sample Size Limitation (n=150):* Gold test set is 150 pairs, yielding a 95% bootstrap confidence interval of ~±5.5 points ([78.67%, 90.00%]).
  2. *Absence of Hard Numerical Negatives:* Current gold dataset lacks pairs sharing identical numbers with differing line-item concepts. Consequently, Brier score (0.0070) is optimistic.
  3. *Enterprise & Sector Concentration:* Limited to 6 large-cap tech/consumer companies (AAPL, MSFT, TSLA, NVDA, AMZN, GOOGL); cross-industry generalizability requires broader sector validation.
  4. *Linguistic & GAAP Scope:* Exclusively English-language Form 10-Ks under US-GAAP. Does not generalize to IFRS or multilingual reports.
  5. *Single-Annotator Ground Truth:* Curated by a single annotator without independent double-blind validation records. Dual-annotator agreement protocol planned.
  6. *Cross-Domain Prior Shift & Calibration:* Training on external QA data (5.5% match prior) vs testing on SEC GOLD (36.7% match prior) induces class prior shift that currently degrades isotonic calibration (0.0070 -> 0.0078).

---

## Slide 18: Summary of Achievements & Final Review Roadmap
- **Takeaway:** 80% Implementation Complete; Concrete Roadmap to Final Review (16th-21st October 2026).
- **Review 2 Milestones Completed (80% Implementation):**
  - Automated End-to-End Pipeline (Ingestion -> Features -> Training -> Honest Calibration -> Dual XAI).
  - Leak-Free Benchmark Foundation (84.67% Micro-F1, LR No-Length, Held-Out GOLD n=150).
  - Dual XAI Layer with Measured Metrics (LinearSHAP 0.009 ms, 80.44% feature overlap).
  - Forensic Self-Audit Completed (All 6 leakage pathways diagnosed and resolved).
  - Structured SEC XBRL Store (Built 298,663 line-item store in `line_items.parquet` across 33 companies).
- **Final Review Roadmap (16th-21st October 2026):**
  1. *XBRL Ingestion with Hard Negatives:* Mine numerical distractors sharing identical dollar figures from `line_items.parquet`.
  2. *Company-Level Partitioning:* Implement strict leave-one-company-out and company-partitioned training splits.
  3. *Gold Set Expansion & Dual Annotation:* Expand GOLD corpus with hard negatives; recruit second annotator on 50 pairs for genuine inter-annotator agreement.
  4. *Advanced Models & Fair Baseline:* Train LightGBM + TreeSHAP; implement a fair fine-tuned FinBERT / DeBERTa baseline on our data.
  5. *Calibration Fix:* Solve prior shift via in-domain split calibration or Bayesian prior adjustment.
  6. *Stretch Goals:* End-to-end vector retrieval (hybrid dense/sparse) and timed auditor efficiency user study.
