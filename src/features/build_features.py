#!/usr/bin/env python3
"""
src/features/build_features.py

Feature engineering pipeline producing Block A (interpretable hand-crafted)
and Block B (dense embedding similarity) features for financial KPI matching.

Block A (Interpretable features for SHAP/LIME readability):
- numeric_value_match: 1 if number in sentence matches line item value within 1%, else 0
- numeric_value_close: 1 if within 5% tolerance (capturing rounding), else 0
- keyword_overlap: Jaccard similarity of financial keywords
- period_match: 1 if stated period in sentence matches line item period, else 0
- string_similarity: token-level fuzzy match score (rapidfuzz)
- sentence_length: length of sentence in characters
- line_item_name_length: length of line item name in characters

Block B (Dense embedding similarity):
- embedding_cosine_similarity: cosine similarity between sentence and line item
  computed using 'all-MiniLM-L6-v2' (or fallback semantic encoder).

Outputs:
- data/processed/features.csv
- data/processed/raw_embeddings.npy
"""

import re
import csv
import logging
from pathlib import Path
import numpy as np
import pandas as pd
from rapidfuzz import fuzz

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

INPUT_LABELED_CSV = Path("data/labeled/pairs_labeled.csv")
OUTPUT_FEATURES_CSV = Path("data/processed/features.csv")
OUTPUT_EMBEDDINGS_NPY = Path("data/processed/raw_embeddings.npy")

FINANCIAL_KEYWORDS = [
    "revenue", "net income", "operating income", "gross profit", "cost of sales",
    "gross margin", "operating margin", "ebitda", "operating expenses",
    "total assets", "cash and cash equivalents", "marketable securities",
    "free cash flow", "capital expenditures", "earnings per share", "diluted",
    "sales", "debt", "interest expense", "tax expense", "retained earnings"
]

NUM_EXTRACT_RE = re.compile(r"[-+]?(?:\d{1,3}(?:,\d{3})+|\d+)(?:\.\d+)?")
YEAR_PATTERN = re.compile(r"\b(20\d\d)\b")


def extract_numbers(text: str):
    """Extracts numeric values from string."""
    matches = NUM_EXTRACT_RE.findall(text)
    nums = []
    for m in matches:
        try:
            val = float(m.replace(",", ""))
            nums.append(val)
        except ValueError:
            continue
    return nums


def compute_block_a_features(sentence: str, line_name: str, line_val: float, period: str = "FY"):
    """Computes transparent, human-auditable features."""
    s_nums = extract_numbers(sentence)
    
    # 1. Numeric exact/close matches
    num_match = 0
    num_close = 0
    try:
        val_float = float(line_val)
    except (ValueError, TypeError):
        val_float = 0.0

    if val_float != 0:
        for num in s_nums:
            rel_diff = abs(num - val_float) / max(abs(val_float), 1e-5)
            if rel_diff <= 0.01:
                num_match = 1
                num_close = 1
                break
            elif rel_diff <= 0.05:
                num_close = 1

    # 2. Keyword overlap (Jaccard similarity)
    s_lower = sentence.lower()
    l_lower = line_name.lower()
    s_kws = {kw for kw in FINANCIAL_KEYWORDS if kw in s_lower}
    l_kws = {kw for kw in FINANCIAL_KEYWORDS if kw in l_lower}
    union_kws = s_kws.union(l_kws)
    if union_kws:
        kw_overlap = len(s_kws.intersection(l_kws)) / len(union_kws)
    else:
        kw_overlap = 0.0

    # 3. Period match
    s_years = set(YEAR_PATTERN.findall(sentence))
    p_years = set(YEAR_PATTERN.findall(str(period)))
    if s_years and p_years and s_years.intersection(p_years):
        period_match = 1
    elif ("2023" in sentence and "2023" in str(line_name)) or ("2022" in sentence and "2022" in str(line_name)):
        period_match = 1
    else:
        period_match = 0

    # 4. String similarity (Rapidfuzz token sort ratio, normalized to 0.0 - 1.0)
    str_sim = fuzz.token_sort_ratio(sentence, line_name) / 100.0

    # 5. Lengths
    sent_len = len(sentence)
    line_len = len(line_name)

    return {
        "numeric_value_match": num_match,
        "numeric_value_close": num_close,
        "keyword_overlap": round(kw_overlap, 4),
        "period_match": period_match,
        "string_similarity": round(str_sim, 4),
        "sentence_length": sent_len,
        "line_item_name_length": line_len
    }


def compute_block_b_embeddings(sentences: list, line_names: list):
    """
    Computes dense embedding cosine similarity between sentence and line item.
    Uses 'sentence-transformers/all-MiniLM-L6-v2' via fastembed or sentence-transformers,
    with a graceful TF-IDF semantic cosine similarity fallback.
    """
    # 1. Try fastembed (ONNX runtime, ultra-fast and no PyTorch header dependencies)
    try:
        from fastembed import TextEmbedding
        logger.info("Encoding text pairs using fastembed ('sentence-transformers/all-MiniLM-L6-v2')...")
        embed_model = TextEmbedding(model_name="sentence-transformers/all-MiniLM-L6-v2")
        sent_embs = np.array(list(embed_model.embed(sentences)))
        line_embs = np.array(list(embed_model.embed(line_names)))
        # Normalize
        sent_norms = np.linalg.norm(sent_embs, axis=1, keepdims=True)
        line_norms = np.linalg.norm(line_embs, axis=1, keepdims=True)
        sent_normed = sent_embs / np.maximum(sent_norms, 1e-9)
        line_normed = line_embs / np.maximum(line_norms, 1e-9)
        cos_sims = np.sum(sent_normed * line_normed, axis=1)
        raw_embs = np.hstack([sent_normed, line_normed])
        return cos_sims, raw_embs
    except Exception as e_fast:
        logger.debug(f"fastembed not active ({e_fast}), trying sentence_transformers...")

    # 2. Try sentence_transformers
    try:
        from sentence_transformers import SentenceTransformer
        logger.info("Encoding text pairs with sentence-transformers ('all-MiniLM-L6-v2')...")
        model = SentenceTransformer("all-MiniLM-L6-v2")
        sent_emb = model.encode(sentences, show_progress_bar=False, normalize_embeddings=True)
        line_emb = model.encode(line_names, show_progress_bar=False, normalize_embeddings=True)
        cos_sims = np.sum(sent_emb * line_emb, axis=1)
        raw_embs = np.hstack([sent_emb, line_emb])
        return cos_sims, raw_embs
    except Exception as e:
        logger.warning(f"sentence-transformers unavailable ({e}), using TF-IDF semantic cosine similarity fallback...")
        from sklearn.feature_extraction.text import TfidfVectorizer
        from sklearn.metrics.pairwise import cosine_similarity
        
        vectorizer = TfidfVectorizer(ngram_range=(1, 2), stop_words="english")
        all_texts = sentences + line_names
        vectorizer.fit(all_texts)
        sent_mat = vectorizer.transform(sentences)
        line_mat = vectorizer.transform(line_names)
        
        sims = []
        for i in range(len(sentences)):
            cs = cosine_similarity(sent_mat[i], line_mat[i])[0][0]
            sims.append(cs)
        
        raw_embs = np.hstack([sent_mat.toarray()[:, :100], line_mat.toarray()[:, :100]])
        return np.array(sims), raw_embs



def build_feature_table(df_labeled: pd.DataFrame):
    """Generates combined Block A + Block B features DataFrame."""
    sentences = df_labeled["sentence"].astype(str).tolist()
    line_names = df_labeled["candidate_line_item"].astype(str).tolist()
    values = df_labeled["candidate_value"].tolist()
    periods = df_labeled["period"].tolist() if "period" in df_labeled.columns else ["FY"] * len(df_labeled)

    logger.info("Computing Block A interpretable hand-crafted features...")
    block_a_list = []
    for s, l, v, p in zip(sentences, line_names, values, periods):
        feats = compute_block_a_features(s, l, v, p)
        block_a_list.append(feats)

    df_block_a = pd.DataFrame(block_a_list)

    logger.info("Computing Block B dense semantic embedding similarities...")
    cos_sims, raw_embs = compute_block_b_embeddings(sentences, line_names)
    df_block_b = pd.DataFrame({"embedding_cosine_similarity": np.round(cos_sims, 4)})

    # Combine blocks and carry through metadata & label
    df_features = pd.concat([
        df_labeled[["sentence", "candidate_line_item", "candidate_value"]].reset_index(drop=True),
        df_block_a,
        df_block_b,
        df_labeled[["label"]].reset_index(drop=True)
    ], axis=1)

    return df_features, raw_embs


def main():
    if not INPUT_LABELED_CSV.exists():
        logger.error(f"Labeled file not found: {INPUT_LABELED_CSV}. Need labeled dataset to build features.")
        return

    df_labeled = pd.read_csv(INPUT_LABELED_CSV)
    # Filter out empty labels if any
    df_labeled = df_labeled[df_labeled["label"].str.strip().ne("")].copy()
    logger.info(f"Loaded {len(df_labeled)} labeled rows from {INPUT_LABELED_CSV}")

    OUTPUT_FEATURES_CSV.parent.mkdir(parents=True, exist_ok=True)
    df_features, raw_embs = build_feature_table(df_labeled)

    df_features.to_csv(OUTPUT_FEATURES_CSV, index=False)
    np.save(OUTPUT_EMBEDDINGS_NPY, raw_embs)

    logger.info(f"Features saved to {OUTPUT_FEATURES_CSV} ({df_features.shape[0]} rows, {df_features.shape[1]} columns)")
    logger.info(f"Raw embeddings saved to {OUTPUT_EMBEDDINGS_NPY} with shape {raw_embs.shape}")
    print("\nFeature preview:")
    print(df_features[["numeric_value_match", "keyword_overlap", "string_similarity", "embedding_cosine_similarity", "label"]].head())


if __name__ == "__main__":
    main()
