#!/usr/bin/env python3
"""
src/extraction/fetch_reports.py

Downloads annual reports (Form 10-K) in HTML/PDF format directly from SEC EDGAR.
Uses SEC EDGAR's submissions and company facts API with compliant User-Agent
and rate limiting (under 10 requests/second per SEC Fair Access policy).
Logs downloaded files to data/raw/manifest.csv.
"""

import os
import sys
import time
import csv
import logging
from datetime import datetime
from pathlib import Path
import requests

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

# SEC EDGAR requires a specific User-Agent format: SampleCompany AdminContact@sampledomain.com
SEC_USER_AGENT = "AcademicResearchKPI financial_nlp_research@university.edu"
HEADERS = {
    "User-Agent": SEC_USER_AGENT,
    "Accept-Encoding": "gzip, deflate",
    "Host": "data.sec.gov"
}

# Tickers to CIK mapping for 6 prominent corporate entities
COMPANY_CIKS = {
    "AAPL": "0000320193",
    "MSFT": "0000789019",
    "TSLA": "0001318605",
    "NVDA": "0001045810",
    "AMZN": "0001018724",
    "GOOGL": "0001652044"
}

RAW_DATA_DIR = Path("data/raw")
MANIFEST_PATH = RAW_DATA_DIR / "manifest.csv"


def init_manifest():
    """Initializes manifest.csv if it does not already exist."""
    RAW_DATA_DIR.mkdir(parents=True, exist_ok=True)
    if not MANIFEST_PATH.exists():
        with open(MANIFEST_PATH, mode="w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(["ticker", "year", "form", "filepath", "url", "download_date", "file_size_bytes"])


def log_download(ticker: str, year: str, form: str, filepath: str, url: str, file_size: int):
    """Appends an entry to manifest.csv."""
    with open(MANIFEST_PATH, mode="a", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([
            ticker,
            year,
            form,
            str(filepath),
            url,
            datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            file_size
        ])


def fetch_filing_metadata(cik: str, ticker: str):
    """Fetches submission history for a CIK from data.sec.gov."""
    url = f"https://data.sec.gov/submissions/CIK{cik}.json"
    logger.info(f"Querying SEC submissions for {ticker} (CIK {cik})...")
    time.sleep(0.3)  # SEC rate limit compliance
    resp = requests.get(url, headers=HEADERS, timeout=15)
    if resp.status_code != 200:
        logger.warning(f"Failed to fetch submissions for {ticker}: HTTP {resp.status_code}")
        return []
    
    data = resp.json()
    recent = data.get("filings", {}).get("recent", {})
    forms = recent.get("form", [])
    accession_numbers = recent.get("accessionNumber", [])
    filing_dates = recent.get("filingDate", [])
    primary_docs = recent.get("primaryDocument", [])

    results = []
    for form, acc_num, fdate, pdoc in zip(forms, accession_numbers, filing_dates, primary_docs):
        if form == "10-K":
            year = fdate.split("-")[0]
            clean_acc = acc_num.replace("-", "")
            doc_url = f"https://www.sec.gov/Archives/edgar/data/{int(cik)}/{clean_acc}/{pdoc}"
            results.append({
                "ticker": ticker,
                "year": year,
                "form": form,
                "accession": acc_num,
                "primary_doc": pdoc,
                "filing_date": fdate,
                "url": doc_url
            })
            if len(results) >= 2:  # Get top 1-2 10-Ks per company
                break
    return results


def download_filing(item: dict) -> Path:
    """Downloads a filing HTML/document and saves it to data/raw/."""
    ticker = item["ticker"]
    year = item["year"]
    url = item["url"]
    ext = Path(item["primary_doc"]).suffix or ".htm"
    target_path = RAW_DATA_DIR / f"{ticker}_{year}{ext}"

    if target_path.exists() and target_path.stat().st_size > 1000:
        logger.info(f"Filing already exists: {target_path} ({target_path.stat().st_size} bytes)")
        return target_path

    logger.info(f"Downloading {ticker} {year} 10-K from {url}...")
    headers = {"User-Agent": SEC_USER_AGENT, "Host": "www.sec.gov"}
    time.sleep(0.3)  # SEC rate limit compliance
    resp = requests.get(url, headers=headers, timeout=30)
    if resp.status_code == 200 and len(resp.content) > 1000:
        with open(target_path, "wb") as f:
            f.write(resp.content)
        file_size = target_path.stat().st_size
        logger.info(f"Saved {target_path} ({file_size} bytes)")
        log_download(ticker, year, "10-K", str(target_path), url, file_size)
        return target_path
    else:
        logger.warning(f"Download returned HTTP {resp.status_code} or empty content.")
        return None


def main():
    init_manifest()
    total_downloaded = 0

    for ticker, cik in COMPANY_CIKS.items():
        try:
            filings = fetch_filing_metadata(cik, ticker)
            if not filings:
                logger.warning(f"No 10-K filings found for {ticker}")
                continue
            for filing in filings[:1]:  # 1 recent 10-K per ticker = 6 filings total
                saved = download_filing(filing)
                if saved:
                    total_downloaded += 1
        except Exception as e:
            logger.error(f"Error retrieving filings for {ticker}: {e}")

    logger.info(f"Report fetching complete. Total filings available: {total_downloaded}")
    logger.info(f"Manifest saved to: {MANIFEST_PATH.resolve()}")


if __name__ == "__main__":
    main()
