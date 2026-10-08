#!/usr/bin/env python3
"""
scripts/fetch_and_prepare_external_datasets.py

Downloads and converts external academic financial benchmark datasets:
1. FinQA (Columbia University / J.P. Morgan) via GitHub raw URL.
2. TAT-QA (NExT Research) via HuggingFace datasets library.

Parses narrative text, tables, line items, and values into standard schema:
- sentence
- candidate_line_item
- candidate_value
- label (match / no_match / ambiguous)

Saves:
- data/labeled/external_finqa_tatqa_pairs.csv
- data/labeled/pairs_labeled_combined.csv (Merged with primary SEC EDGAR pairs)
"""

import os
import re
import sys
import json
import urllib.request
import logging
from pathlib import Path
import pandas as pd
import numpy as np

# Ensure repo root is on sys.path
ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from src.constants import LABELED_DATA_DIR, INPUT_LABELED_CSV

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

NUM_EXTRACT_RE = re.compile(r"[-+]?(?:\d{1,3}(?:,\d{3})+|\d+)(?:\.\d+)?")
FINQA_DEV_URL = "https://raw.githubusercontent.com/czyssrs/FinQA/main/dataset/dev.json"
OUTPUT_EXTERNAL_CSV = LABELED_DATA_DIR / "external_finqa_tatqa_pairs.csv"
OUTPUT_COMBINED_CSV = LABELED_DATA_DIR / "pairs_labeled_combined.csv"


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


def fetch_and_parse_finqa(max_items=150):
    """Downloads FinQA dev set and converts to text-pair classification schema."""
    logger.info(f"Downloading FinQA dataset from {FINQA_DEV_URL}...")
    req = urllib.request.urlopen(FINQA_DEV_URL)
    raw_json = json.loads(req.read().decode("utf-8"))
    logger.info(f"Loaded {len(raw_json)} raw FinQA instances. Extracting pairs...")

    extracted_pairs = []

    for item in raw_json[:max_items]:
        pre_text = " ".join(item.get("pre_text", []))
        post_text = " ".join(item.get("post_text", []))
        full_text = f"{pre_text} {post_text}".strip()
        table = item.get("table", [])

        if not table or not full_text:
            continue

        # Split text into sentences
        sents = [s.strip() for s in re.split(r"(?<=[.!?])\s+", full_text) if len(s.strip()) > 20]
        if not sents:
            continue

        # Extract table line items and numbers
        table_items = []
        for row in table[1:]: # Skip header
            if row and len(row) >= 2:
                line_name = str(row[0]).strip()
                for cell in row[1:]:
                    nums = extract_numbers(str(cell))
                    if nums and len(line_name) >= 3 and not line_name.isdigit():
                        table_items.append((line_name, nums[0]))
                        break

        if not table_items:
            continue

        for sent in sents:
            sent_nums = extract_numbers(sent)
            if not sent_nums:
                continue

            # Check matching against table items
            for line_name, line_val in table_items:
                # Calculate relative numeric closeness
                matched_num = False
                for num in sent_nums:
                    rel_diff = abs(num - line_val) / max(abs(line_val), 1e-5)
                    if rel_diff <= 0.02:
                        matched_num = True
                        break

                if matched_num:
                    label = "match"
                    notes = "FinQA ground truth exact value & semantic pair"
                elif any(word in sent.lower() for word in line_name.lower().split() if len(word) > 3):
                    label = "ambiguous"
                    notes = "FinQA shared vocabulary but numeric mismatch"
                else:
                    label = "no_match"
                    notes = "FinQA distinct line item & numeric mismatch"

                extracted_pairs.append({
                    "sentence": sent,
                    "candidate_line_item": line_name,
                    "candidate_value": line_val,
                    "label": label,
                    "notes": notes,
                    "source_dataset": "FinQA (Columbia/JPM)"
                })

                if len(extracted_pairs) >= max_items:
                    break
            if len(extracted_pairs) >= max_items:
                break

    logger.info(f"Extracted {len(extracted_pairs)} benchmark pairs from FinQA.")
    return extracted_pairs


def fetch_and_parse_tatqa(max_items=150):
    """Loads TAT-QA dataset via HuggingFace and converts to text-pair classification schema."""
    logger.info("Loading TAT-QA validation set via HuggingFace datasets...")
    try:
        from datasets import load_dataset
        ds = load_dataset("next-tat/TAT-QA", split="validation")
    except Exception as e:
        logger.warning(f"Failed to load TAT-QA via HuggingFace ({e}). Skipping TAT-QA.")
        return []

    extracted_pairs = []
    logger.info(f"Processing {len(ds)} TAT-QA instances...")

    for item in ds:
        table_raw = item.get("table", {})
        paragraphs = item.get("paragraphs", [])

        if not paragraphs:
            continue

        table_items = []
        table_table = table_raw.get("table", []) if isinstance(table_raw, dict) else []
        for row in table_table[1:]:
            if row and len(row) >= 2:
                line_name = str(row[0]).strip()
                for cell in row[1:]:
                    nums = extract_numbers(str(cell))
                    if nums and len(line_name) >= 3 and not line_name.isdigit():
                        table_items.append((line_name, nums[0]))
                        break

        if not table_items:
            continue

        for p in paragraphs:
            text = p.get("text", "") if isinstance(p, dict) else str(p)
            sent_nums = extract_numbers(text)
            if not sent_nums or len(text) < 20:
                continue

            for line_name, line_val in table_items:
                matched_num = any(abs(num - line_val) / max(abs(line_val), 1e-5) <= 0.02 for num in sent_nums)
                if matched_num:
                    label = "match"
                    notes = "TAT-QA table-text matching claim"
                elif any(w in text.lower() for w in line_name.lower().split() if len(w) > 3):
                    label = "ambiguous"
                    notes = "TAT-QA conceptual overlap edge case"
                else:
                    label = "no_match"
                    notes = "TAT-QA disjoint line item"

                extracted_pairs.append({
                    "sentence": text,
                    "candidate_line_item": line_name,
                    "candidate_value": line_val,
                    "label": label,
                    "notes": notes,
                    "source_dataset": "TAT-QA (NExT Research)"
                })

                if len(extracted_pairs) >= max_items:
                    break
            if len(extracted_pairs) >= max_items:
                break

    logger.info(f"Extracted {len(extracted_pairs)} benchmark pairs from TAT-QA.")
    return extracted_pairs


def main():
    LABELED_DATA_DIR.mkdir(parents=True, exist_ok=True)

    finqa_pairs = fetch_and_parse_finqa(max_items=120)
    tatqa_pairs = fetch_and_parse_tatqa(max_items=120)

    external_list = finqa_pairs + tatqa_pairs
    if not external_list:
        logger.error("No external dataset pairs extracted.")
        return

    df_external = pd.DataFrame(external_list)
    df_external.drop_duplicates(subset=["sentence", "candidate_line_item"], inplace=True)
    df_external.to_csv(OUTPUT_EXTERNAL_CSV, index=False)
    logger.info(f"Saved {len(df_external)} external benchmark pairs to {OUTPUT_EXTERNAL_CSV}")

    # Merge with primary SEC EDGAR labeled pairs
    if INPUT_LABELED_CSV.exists():
        df_primary = pd.read_csv(INPUT_LABELED_CSV)
        df_primary["source_dataset"] = "SEC EDGAR 10-K (Primary)"

        cols = ["sentence", "candidate_line_item", "candidate_value", "label", "notes", "source_dataset"]
        df_primary_clean = df_primary[[c for c in cols if c in df_primary.columns]]
        df_external_clean = df_external[[c for c in cols if c in df_external.columns]]

        df_combined = pd.concat([df_primary_clean, df_external_clean], ignore_index=True)
        df_combined.drop_duplicates(subset=["sentence", "candidate_line_item"], inplace=True)
        df_combined.to_csv(OUTPUT_COMBINED_CSV, index=False)
        logger.info(f"Saved combined multi-dataset labeled dataset to {OUTPUT_COMBINED_CSV} ({len(df_combined)} total pairs!)")

        print("\n" + "=" * 70)
        print(f"{'MULTI-DATASET INTEGRATION SUMMARY':^70}")
        print("=" * 70)
        print(df_combined["source_dataset"].value_counts().to_string())
        print("-" * 70)
        print(f"Total Unified Training Pairs: {len(df_combined)}")
        print("=" * 70 + "\n")


if __name__ == "__main__":
    main()
