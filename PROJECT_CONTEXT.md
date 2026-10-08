PROJECT: Explainable Financial KPI-Matching (capstone, VIT).
TASK: Given a claim sentence from a 10-K/10-Q narrative and a candidate financial line item (value, unit, period, us-gaap tag), classify match | no_match | ambiguous, output a calibrated probability, and explain with SHAP/LIME.
EXISTING ASSETS (repo: c:\Users\welcome\Downloads\KPI-CHECK-): 150 hand-labeled pairs from 6 10-Ks (AAPL, MSFT, TSLA, NVDA, AMZN, GOOGL) = GOLD; Block A hand-crafted features; Block B MiniLM cosine similarity; LR and MLP; isotonic calibration; SHAP + LIME; Streamlit UI.
AMBIGUITY TAXONOMY (my contribution): subsegment vs aggregate, non-GAAP, ratio vs nominal, temporal mismatch.
BENCHMARK CONTEXT: KPI-Check (Hillebrand et al., 2022; German reports; arXiv 2211.06112). Not directly comparable to this project.

HARD RULES
1. GOLD is test-only: never used for training, tuning or calibration.
2. All splits are by company/filing, never random.
3. No metric is typed by hand into any document. All numbers come from results/metrics_master.json.
4. Fixed seeds; every config saved.
5. If anything is uncertain, a download fails, or a dataset behaves unexpectedly: STOP and tell me. Never synthesize, invent or silently substitute data.
6. Never claim to beat KPI-Check's 73% directly.
7. Final review is 16-21 Oct 2026. Priority order: Prompts 1 to 7; Prompt 8 is stretch.

DATA SOURCES
- SEC XBRL: data.sec.gov companyfacts and Financial Statement Data Sets. User-Agent: "[s.k.thyakeshwar] [skthyakeshwar@gmail.com]". Max 5 requests/second.
- FiNER-139 (HuggingFace nlpaueb/finer-139; GitHub nlpaueb/finer).
- FinQA (GitHub czyssrs/FinQA or HuggingFace).
- TAT-QA (GitHub NExTplusplus/TAT-QA or HuggingFace next-tat/TAT-QA). Non-commercial license: state this in every report that uses it.
- KPI-Check corpus: German, proprietary. Reference only (annotation guidelines from the paper). Never use as data.

DATA ACCESS RULES
- Download scripts go in scripts/download_*.py; save to data/raw/; log source URL, date, version and checksum; do not commit raw data.
- If load_dataset fails (e.g. loading-script error), pin an older `datasets` version or read the GitHub files directly. Never skip a dataset silently.
- Before writing any conversion code, inspect each dataset's real fields and print 5 examples.
- If a download is impossible in this environment, stop and tell me the exact file I should place in data/raw/.
