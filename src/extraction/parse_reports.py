#!/usr/bin/env python3
"""
src/extraction/parse_reports.py

Parses downloaded SEC 10-K filings (HTML or PDF) to:
1. Extract narrative text and segment into financial KPI candidate sentences.
2. Extract structured financial statement line items (Income Statement, Balance Sheet).
3. Filter candidate sentences using financial keywords and numerical figures.
4. Save extracted tables to data/processed/extracted_line_items.csv and
   candidate sentences to data/processed/candidate_kpi_sentences.csv.
5. Prints a detailed per-report summary.
"""

import os
import re
import csv
import logging
from pathlib import Path
from bs4 import BeautifulSoup
import pandas as pd

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

RAW_DIR = Path("data/raw")
PROCESSED_DIR = Path("data/processed")
CANDIDATE_SENTENCES_CSV = PROCESSED_DIR / "candidate_kpi_sentences.csv"
LINE_ITEMS_CSV = PROCESSED_DIR / "extracted_line_items.csv"

# Financial keywords to isolate relevant KPI narrative claims
FINANCIAL_KEYWORDS = [
    "revenue", "net income", "operating income", "gross profit", "cost of sales",
    "gross margin", "operating margin", "ebitda", "operating expenses",
    "total assets", "cash and cash equivalents", "marketable securities",
    "free cash flow", "capital expenditures", "earnings per share", "diluted share",
    "sales", "debt", "interest expense", "tax expense", "retained earnings"
]

NUMBER_PATTERN = re.compile(r"(\$?\d+(?:,\d{3})*(?:\.\d+)?|\d+(?:\.\d+)?%|\b(?:one|two|three|four|five|ten)\b)", re.IGNORECASE)


def split_sentences(text: str):
    """Splits plain text into clean, individual sentences."""
    # Normalize whitespaces and clean html leftovers
    text = re.sub(r"\s+", " ", text).strip()
    # Sentence splitting regex respecting common abbreviations
    raw_sentences = re.split(r"(?<=[.!?])\s+(?=[A-Z0-9])", text)
    cleaned = []
    for s in raw_sentences:
        s = s.strip()
        if 20 <= len(s) <= 400:  # reasonable narrative sentence length
            cleaned.append(s)
    return cleaned


def parse_html_report(filepath: Path):
    """Extracts narrative sentences and financial statement tables from an HTML 10-K."""
    with open(filepath, "r", encoding="utf-8", errors="ignore") as f:
        html_content = f.read()

    soup = BeautifulSoup(html_content, "lxml" if "lxml" in BeautifulSoup.__dict__ else "html.parser")

    # 1. Extract narrative candidate sentences
    # Exclude script, style, and table elements for narrative
    for tag in soup(["script", "style"]):
        tag.decompose()

    # Get all paragraph text
    paragraphs = soup.find_all(["p", "div", "span"])
    full_text_chunks = []
    for p in paragraphs:
        text = p.get_text(" ", strip=True)
        if len(text) > 30:
            full_text_chunks.append(text)
    
    full_text = " ".join(full_text_chunks)
    all_sentences = split_sentences(full_text)

    candidate_sentences = []
    sentence_idx = 1
    for s in all_sentences:
        s_lower = s.lower()
        has_number = bool(NUMBER_PATTERN.search(s))
        has_keyword = any(kw in s_lower for kw in FINANCIAL_KEYWORDS)
        if has_number and has_keyword:
            candidate_sentences.append({
                "sentence": s,
                "source_file": filepath.name,
                "sentence_id": f"{filepath.stem}_sent_{sentence_idx}"
            })
            sentence_idx += 1

    # 2. Extract structured financial statement tables
    extracted_items = []
    tables = soup.find_all("table")
    for t_idx, table in enumerate(tables):
        rows = table.find_all("tr")
        if len(rows) < 3:
            continue
        
        # Check if table has financial context
        table_text = table.get_text(" ", strip=True).lower()
        if not any(k in table_text for k in ["revenue", "assets", "income", "liabilities", "cash", "margin"]):
            continue

        for row in rows:
            cols = [col.get_text(" ", strip=True) for col in row.find_all(["td", "th"])]
            cols = [c for c in cols if c]
            if len(cols) >= 2:
                line_name = cols[0].strip()
                # Check if first column looks like a financial line item
                if any(k in line_name.lower() for k in FINANCIAL_KEYWORDS) and len(line_name) < 100:
                    for val_col in cols[1:]:
                        clean_val = val_col.replace("$", "").replace(",", "").replace("(", "-").replace(")", "").strip()
                        # Verify if this column is numeric
                        try:
                            val_float = float(clean_val)
                            extracted_items.append({
                                "line_item_name": line_name,
                                "value": val_float,
                                "unit": "Millions (USD)",
                                "period": "FY",
                                "source_file": filepath.name
                            })
                            break
                        except ValueError:
                            continue

    return candidate_sentences, extracted_items


def parse_pdf_report(filepath: Path):
    """Extracts narrative text and tables from a PDF report using pdfplumber."""
    try:
        import pdfplumber
    except ImportError:
        logger.warning(f"pdfplumber not available. Skipping PDF parse for {filepath}")
        return [], []

    candidate_sentences = []
    extracted_items = []
    sentence_idx = 1

    with pdfplumber.open(filepath) as pdf:
        for page_idx, page in enumerate(pdf.pages):
            text = page.extract_text()
            if text:
                sents = split_sentences(text)
                for s in sents:
                    s_lower = s.lower()
                    if bool(NUMBER_PATTERN.search(s)) and any(kw in s_lower for kw in FINANCIAL_KEYWORDS):
                        candidate_sentences.append({
                            "sentence": s,
                            "source_file": filepath.name,
                            "sentence_id": f"{filepath.stem}_p{page_idx+1}_{sentence_idx}"
                        })
                        sentence_idx += 1

            tables = page.extract_tables()
            for table in tables:
                for row in table:
                    cols = [str(c).strip() for c in row if c is not None and str(c).strip()]
                    if len(cols) >= 2:
                        line_name = cols[0]
                        if any(k in line_name.lower() for k in FINANCIAL_KEYWORDS):
                            for v in cols[1:]:
                                clean_val = v.replace("$", "").replace(",", "").replace("(", "-").replace(")", "").strip()
                                try:
                                    val_float = float(clean_val)
                                    extracted_items.append({
                                        "line_item_name": line_name,
                                        "value": val_float,
                                        "unit": "Millions (USD)",
                                        "period": "FY",
                                        "source_file": filepath.name
                                    })
                                    break
                                except ValueError:
                                    continue

    return candidate_sentences, extracted_items


def main():
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    report_files = list(RAW_DIR.glob("*.htm")) + list(RAW_DIR.glob("*.html")) + list(RAW_DIR.glob("*.pdf"))

    if not report_files:
        logger.warning(f"No reports found in {RAW_DIR}. Run fetch_reports.py first.")
        return

    all_candidate_sentences = []
    all_line_items = []

    print("=" * 70)
    print(f"{'REPORT PARSING SUMMARY':^70}")
    print("=" * 70)
    print(f"{'File Name':<30} | {'Candidate Sents':<18} | {'Line Items':<15}")
    print("-" * 70)

    for r_file in report_files:
        if r_file.suffix.lower() in [".htm", ".html"]:
            sents, items = parse_html_report(r_file)
        elif r_file.suffix.lower() == ".pdf":
            sents, items = parse_pdf_report(r_file)
        else:
            continue

        print(f"{r_file.name:<30} | {len(sents):<18} | {len(items):<15}")
        all_candidate_sentences.extend(sents)
        all_line_items.extend(items)

    print("-" * 70)
    print(f"Total candidate sentences extracted: {len(all_candidate_sentences)}")
    print(f"Total financial line items extracted: {len(all_line_items)}")
    print("=" * 70)

    # Save to CSV files
    if all_candidate_sentences:
        df_sents = pd.DataFrame(all_candidate_sentences)
        df_sents.drop_duplicates(subset=["sentence"], inplace=True)
        df_sents.to_csv(CANDIDATE_SENTENCES_CSV, index=False)
        logger.info(f"Saved {len(df_sents)} unique candidate sentences to {CANDIDATE_SENTENCES_CSV}")

    if all_line_items:
        df_items = pd.DataFrame(all_line_items)
        df_items.drop_duplicates(subset=["line_item_name", "value", "source_file"], inplace=True)
        df_items.to_csv(LINE_ITEMS_CSV, index=False)
        logger.info(f"Saved {len(df_items)} unique line items to {LINE_ITEMS_CSV}")


if __name__ == "__main__":
    main()
