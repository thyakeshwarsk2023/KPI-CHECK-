"""
scripts/download_sec.py

Downloads SEC EDGAR XBRL companyfacts JSON files for 33 companies across 6 sectors:
Tech (including 6 Gold companies: AAPL, MSFT, TSLA, NVDA, AMZN, GOOGL), Healthcare (incl JNJ),
Banking (incl JPM), Energy, Retail, and Industrials.

Covering Fiscal Years 2021-2025, Forms 10-K and 10-Q.
Enforces SEC EDGAR API compliance:
- User-Agent: s.k.thyakeshwar skthyakeshwar@gmail.com
- Rate limit: max 4 requests/second (below 5 req/sec requirement)
- Retries with exponential backoff
- Raw caching in data/raw/sec/CIK{cik}.json
- Manifest tracking in data/raw/manifest.csv

Methodology Note: Per-company REST API endpoints (https://data.sec.gov/api/xbrl/companyfacts/CIK{cik}.json)
were selected over the bulk companyfacts.zip archive because fetching 33 targeted JSON files (~30 MB total)
is significantly more disk-efficient, reliable, and faster to inspect than decompressing a 1GB+ bulk archive.
"""

import os
import time
import json
import hashlib
import datetime
import urllib.request
import urllib.error
from pathlib import Path

USER_AGENT = "s.k.thyakeshwar skthyakeshwar@gmail.com"
MAX_REQ_PER_SEC = 4
DELAY_BETWEEN_REQUESTS = 1.0 / MAX_REQ_PER_SEC
MAX_RETRIES = 5

RAW_SEC_DIR = Path("data/raw/sec")
MANIFEST_PATH = Path("data/raw/manifest.csv")

TARGET_COMPANIES = [
    # Gold Companies (Tech / Automotive / E-Commerce)
    {"ticker": "AAPL", "cik": "0000320193", "name": "Apple Inc.", "sector": "Tech", "gold": True},
    {"ticker": "MSFT", "cik": "0000789019", "name": "Microsoft Corp.", "sector": "Tech", "gold": True},
    {"ticker": "TSLA", "cik": "0001318605", "name": "Tesla Inc.", "sector": "Tech", "gold": True},
    {"ticker": "NVDA", "cik": "0001045810", "name": "NVIDIA Corp.", "sector": "Tech", "gold": True},
    {"ticker": "AMZN", "cik": "0001018724", "name": "Amazon.com Inc.", "sector": "Tech", "gold": True},
    {"ticker": "GOOGL", "cik": "0001652044", "name": "Alphabet Inc.", "sector": "Tech", "gold": True},
    {"ticker": "ORCL", "cik": "0001341439", "name": "Oracle Corp.", "sector": "Tech", "gold": False},

    # Healthcare (include JNJ)
    {"ticker": "JNJ", "cik": "0000200406", "name": "Johnson & Johnson", "sector": "Healthcare", "gold": False},
    {"ticker": "PFE", "cik": "0000078003", "name": "Pfizer Inc.", "sector": "Healthcare", "gold": False},
    {"ticker": "UNH", "cik": "0000731766", "name": "UnitedHealth Group Inc.", "sector": "Healthcare", "gold": False},
    {"ticker": "LLY", "cik": "0000059478", "name": "Eli Lilly & Co.", "sector": "Healthcare", "gold": False},
    {"ticker": "ABT", "cik": "0000001800", "name": "Abbott Laboratories", "sector": "Healthcare", "gold": False},

    # Banking (include JPM)
    {"ticker": "JPM", "cik": "0000019617", "name": "JPMorgan Chase & Co.", "sector": "Banking", "gold": False},
    {"ticker": "BAC", "cik": "0000070858", "name": "Bank of America Corp.", "sector": "Banking", "gold": False},
    {"ticker": "WFC", "cik": "0000072971", "name": "Wells Fargo & Co.", "sector": "Banking", "gold": False},
    {"ticker": "C", "cik": "0000831001", "name": "Citigroup Inc.", "sector": "Banking", "gold": False},
    {"ticker": "GS", "cik": "0000886982", "name": "Goldman Sachs Group Inc.", "sector": "Banking", "gold": False},
    {"ticker": "MS", "cik": "0000895421", "name": "Morgan Stanley", "sector": "Banking", "gold": False},

    # Energy
    {"ticker": "XOM", "cik": "0000034088", "name": "Exxon Mobil Corp.", "sector": "Energy", "gold": False},
    {"ticker": "CVX", "cik": "0000093410", "name": "Chevron Corp.", "sector": "Energy", "gold": False},
    {"ticker": "COP", "cik": "0001163165", "name": "ConocoPhillips", "sector": "Energy", "gold": False},
    {"ticker": "SLB", "cik": "0000087347", "name": "Schlumberger Ltd.", "sector": "Energy", "gold": False},
    {"ticker": "EOG", "cik": "0000821189", "name": "EOG Resources Inc.", "sector": "Energy", "gold": False},

    # Retail
    {"ticker": "WMT", "cik": "0000104169", "name": "Walmart Inc.", "sector": "Retail", "gold": False},
    {"ticker": "TGT", "cik": "0000027419", "name": "Target Corp.", "sector": "Retail", "gold": False},
    {"ticker": "COST", "cik": "0000909832", "name": "Costco Wholesale Corp.", "sector": "Retail", "gold": False},
    {"ticker": "HD", "cik": "0000354950", "name": "Home Depot Inc.", "sector": "Retail", "gold": False},
    {"ticker": "LOW", "cik": "0000060667", "name": "Lowe's Companies Inc.", "sector": "Retail", "gold": False},

    # Industrials
    {"ticker": "CAT", "cik": "0000018230", "name": "Caterpillar Inc.", "sector": "Industrials", "gold": False},
    {"ticker": "GE", "cik": "0000040545", "name": "General Electric Co.", "sector": "Industrials", "gold": False},
    {"ticker": "HON", "cik": "0000773840", "name": "Honeywell International Inc.", "sector": "Industrials", "gold": False},
    {"ticker": "LMT", "cik": "0000936468", "name": "Lockheed Martin Corp.", "sector": "Industrials", "gold": False},
    {"ticker": "MMM", "cik": "0000066740", "name": "3M Co.", "sector": "Industrials", "gold": False},
]


def fetch_company_facts(cik: str, force_download: bool = False) -> Path:
    """
    Downloads SEC XBRL companyfacts JSON for a zero-padded CIK string.
    Implements retries with exponential backoff and caching.
    """
    RAW_SEC_DIR.mkdir(parents=True, exist_ok=True)
    out_file = RAW_SEC_DIR / f"CIK{cik}.json"

    if out_file.exists() and not force_download:
        print(f"[CACHE] Using cached raw SEC JSON: {out_file}")
        return out_file

    url = f"https://data.sec.gov/api/xbrl/companyfacts/CIK{cik}.json"
    req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})

    for attempt in range(1, MAX_RETRIES + 1):
        try:
            time.sleep(DELAY_BETWEEN_REQUESTS)
            with urllib.request.urlopen(req) as response:
                content = response.read()
                data = json.loads(content)
                with open(out_file, "wb") as f:
                    f.write(content)
                print(f"[DOWNLOADED] CIK{cik} ({data.get('entityName', 'Unknown')}) -> {out_file} ({len(content):,} bytes)")
                return out_file
        except urllib.error.HTTPError as e:
            if e.code in (429, 500, 502, 503, 504):
                backoff = 2 ** attempt
                print(f"[WARN] HTTP {e.code} for CIK{cik}. Retrying in {backoff}s (attempt {attempt}/{MAX_RETRIES})...")
                time.sleep(backoff)
            else:
                print(f"[ERROR] HTTP {e.code} for CIK{cik}: {e.reason}")
                raise
        except Exception as e:
            backoff = 2 ** attempt
            print(f"[WARN] Exception {e} for CIK{cik}. Retrying in {backoff}s...")
            time.sleep(backoff)

    raise RuntimeError(f"Failed to download CIK{cik} after {MAX_RETRIES} attempts.")


def update_manifest(records):
    """
    Updates data/raw/manifest.csv with metadata for each downloaded SEC facts file.
    """
    header = "ticker,cik,company,sector,gold,filepath,url,download_date,file_size_bytes,sha256\n"
    lines = []
    if MANIFEST_PATH.exists():
        with open(MANIFEST_PATH, "r", encoding="utf-8") as f:
            existing = f.readlines()
        if existing and existing[0].startswith("ticker,cik"):
            header = existing[0]

    for rec in records:
        path = rec["filepath"]
        file_size = path.stat().st_size if path.exists() else 0
        sha256 = ""
        if path.exists():
            sha256 = hashlib.sha256(path.read_bytes()).hexdigest()
        
        line = f"{rec['ticker']},{rec['cik']},\"{rec['name']}\",{rec['sector']},{rec['gold']},{path},https://data.sec.gov/api/xbrl/companyfacts/CIK{rec['cik']}.json,{datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')},{file_size},{sha256}\n"
        lines.append(line)

    with open(MANIFEST_PATH, "w", encoding="utf-8") as f:
        f.write(header)
        f.writelines(lines)
    print(f"[MANIFEST] Updated {MANIFEST_PATH} with {len(records)} SEC XBRL companyfacts entries.")


def main():
    print(f"=== SEC EDGAR Data Downloader ===")
    print(f"User-Agent: {USER_AGENT}")
    print(f"Target Companies: {len(TARGET_COMPANIES)} across 6 sectors")
    print(f"Method: Per-company REST API calls (cached in data/raw/sec/)")

    records = []
    for comp in TARGET_COMPANIES:
        out_path = fetch_company_facts(comp["cik"])
        comp_rec = dict(comp)
        comp_rec["filepath"] = out_path
        records.append(comp_rec)

    update_manifest(records)
    print(f"Successfully cached and verified {len(records)} company facts files.")


if __name__ == "__main__":
    main()
