#!/usr/bin/env python3
"""
src/eval/build_labeling_sheet.py

Generates a candidate labeling spreadsheet (data/labeled/pairs_to_label.csv)
pairing narrative KPI sentences with top candidate financial statement line items.
Uses heuristic matching (keyword overlap, numeric proximity, token similarity)
to propose top-3 candidate line items for each sentence.
"""

import re
import csv
import logging
from pathlib import Path
import pandas as pd
from rapidfuzz import fuzz

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

PROCESSED_DIR = Path("data/processed")
LABELED_DIR = Path("data/labeled")
CANDIDATE_SENTENCES_CSV = PROCESSED_DIR / "candidate_kpi_sentences.csv"
LINE_ITEMS_CSV = PROCESSED_DIR / "extracted_line_items.csv"
OUTPUT_PAIRS_CSV = LABELED_DIR / "pairs_to_label.csv"

FINANCIAL_KEYWORDS = [
    "revenue", "net income", "operating income", "gross profit", "cost of sales",
    "margin", "ebitda", "assets", "liabilities", "cash", "debt", "earnings",
    "diluted", "share", "sales", "operating expenses", "cash flow", "tax"
]

NUM_EXTRACT_RE = re.compile(r"[-+]?(?:\d{1,3}(?:,\d{3})+|\d+)(?:\.\d+)?")


def extract_numbers_from_sentence(text: str):
    """Extracts floating point numbers present in a sentence."""
    raw_nums = NUM_EXTRACT_RE.findall(text)
    numbers = []
    for n in raw_nums:
        clean = n.replace(",", "")
        try:
            val = float(clean)
            numbers.append(val)
        except ValueError:
            continue
    return numbers


def score_pair_heuristic(sentence: str, s_nums: list, line_name: str, line_val: float):
    """
    Computes a heuristic ranking score for matching sentence to line item.
    Combines:
    - Numeric value match (high bonus)
    - Keyword set overlap
    - Rapidfuzz fuzzy string matching
    """
    score = 0.0

    # 1. Numeric check
    for num in s_nums:
        if line_val != 0 and num != 0:
            rel_diff = abs(num - line_val) / max(abs(line_val), 1e-5)
            if rel_diff <= 0.01:
                score += 50.0  # Exact/near-exact number match
                break
            elif rel_diff <= 0.05:
                score += 30.0  # Rounded match
                break

    # 2. Financial keyword overlap
    s_lower = sentence.lower()
    l_lower = line_name.lower()
    s_kws = {kw for kw in FINANCIAL_KEYWORDS if kw in s_lower}
    l_kws = {kw for kw in FINANCIAL_KEYWORDS if kw in l_lower}
    shared_kws = s_kws.intersection(l_kws)
    score += len(shared_kws) * 15.0

    # 3. Fuzzy string similarity
    fuzzy_ratio = fuzz.token_sort_ratio(sentence, line_name)
    score += (fuzzy_ratio / 100.0) * 20.0

    return score


def main():
    LABELED_DIR.mkdir(parents=True, exist_ok=True)

    if not CANDIDATE_SENTENCES_CSV.exists() or not LINE_ITEMS_CSV.exists():
        logger.error("Required processed CSVs not found. Run parse_reports.py first.")
        return

    df_sents = pd.read_csv(CANDIDATE_SENTENCES_CSV)
    df_items = pd.read_csv(LINE_ITEMS_CSV)

    logger.info(f"Loaded {len(df_sents)} candidate sentences and {len(df_items)} extracted line items.")

    pairs = []

    # Propose top-3 candidates for each sentence
    for _, sent_row in df_sents.iterrows():
        sentence = str(sent_row["sentence"])
        source_file = str(sent_row.get("source_file", ""))
        s_nums = extract_numbers_from_sentence(sentence)

        # Candidate pool: prioritize same source file, then full pool if few
        pool = df_items[df_items["source_file"] == source_file]
        if len(pool) < 3:
            pool = df_items

        scored_candidates = []
        for _, item_row in pool.iterrows():
            line_name = str(item_row["line_item_name"])
            line_val = float(item_row["value"])
            score = score_pair_heuristic(sentence, s_nums, line_name, line_val)
            scored_candidates.append((score, line_name, line_val))

        scored_candidates.sort(key=lambda x: x[0], reverse=True)
        top3 = scored_candidates[:3]

        for score, line_name, line_val in top3:
            pairs.append({
                "sentence": sentence,
                "candidate_line_item": line_name,
                "candidate_value": line_val,
                "heuristic_score": round(score, 2),
                "label": "",  # To be filled: match, no_match, ambiguous
                "notes": ""
            })

    df_pairs = pd.DataFrame(pairs)
    # Deduplicate sentence + candidate_line_item
    df_pairs.drop_duplicates(subset=["sentence", "candidate_line_item"], inplace=True)
    df_pairs.to_csv(OUTPUT_PAIRS_CSV, index=False)
    logger.info(f"Generated {len(df_pairs)} proposed pairs in {OUTPUT_PAIRS_CSV}")


if __name__ == "__main__":
    main()
