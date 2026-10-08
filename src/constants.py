"""
src/constants.py

Central source of truth for features, financial keywords, target classes,
and common file paths across the XAI KPI-Check pipeline.
"""

from pathlib import Path

# Paths
DATA_DIR = Path("data")
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
LABELED_DATA_DIR = DATA_DIR / "labeled"
SAVED_MODELS_DIR = Path("src/model/saved")
RESULTS_DIR = Path("results")
RESULTS_TABLES_DIR = RESULTS_DIR / "tables"
RESULTS_FIGURES_DIR = RESULTS_DIR / "figures"

MANIFEST_PATH = RAW_DATA_DIR / "manifest.csv"
CANDIDATE_SENTENCES_CSV = PROCESSED_DATA_DIR / "candidate_kpi_sentences.csv"
LINE_ITEMS_CSV = PROCESSED_DATA_DIR / "extracted_line_items.csv"
INPUT_LABELED_CSV = LABELED_DATA_DIR / "pairs_labeled.csv"
OUTPUT_FEATURES_CSV = PROCESSED_DATA_DIR / "features.csv"
OUTPUT_EMBEDDINGS_NPY = PROCESSED_DATA_DIR / "raw_embeddings.npy"
TRAIN_SPLIT_CSV = PROCESSED_DATA_DIR / "train_split.csv"
TEST_SPLIT_CSV = PROCESSED_DATA_DIR / "test_split.csv"

# Target classes
TARGET_CLASSES = ["ambiguous", "match", "no_match"]

# Feature Definitions
BLOCK_A_FEATURES = [
    "numeric_value_match",
    "numeric_value_close",
    "keyword_overlap",
    "period_match",
    "string_similarity",
    "sentence_length",
    "line_item_name_length"
]

BLOCK_B_FEATURES = [
    "embedding_cosine_similarity"
]

ALL_FEATURES = BLOCK_A_FEATURES + BLOCK_B_FEATURES

# Domain Lexicon
FINANCIAL_KEYWORDS = [
    "revenue", "net income", "operating income", "gross profit", "cost of sales",
    "gross margin", "operating margin", "ebitda", "operating expenses",
    "total assets", "cash and cash equivalents", "marketable securities",
    "free cash flow", "capital expenditures", "earnings per share", "diluted",
    "sales", "debt", "interest expense", "tax expense", "retained earnings"
]

EMBEDDING_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
