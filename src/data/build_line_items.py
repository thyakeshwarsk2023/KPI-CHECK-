"""
src/data/build_line_items.py

Transforms raw SEC EDGAR XBRL companyfacts JSON files (cached in data/raw/sec/)
into a clean, structured line-item store saved as data/processed/line_items.parquet.

Columns:
- company: Entity name (str)
- cik: 10-digit zero-padded SEC Central Index Key (str)
- sector: Industry sector (Tech, Healthcare, Banking, Energy, Retail, Industrials) (str)
- accession: SEC filing accession number (str)
- form: Filing form (10-K, 10-Q, 10-K/A, 10-Q/A) (str)
- us_gaap_tag: US-GAAP XBRL tag name (str)
- label: Human-readable US-GAAP taxonomy label (str)
- value: Reported value in base units (float64)
- unit: Base unit of measure (USD, shares, USD/shares, etc.) (str)
- period_start: Start date of period YYYY-MM-DD or empty for instant items (str)
- period_end: End / instant date YYYY-MM-DD (str)
- instant_or_duration: 'instant' or 'duration' (str)
- fiscal_year: Fiscal year (2021-2025) (int64)
- fiscal_period: Fiscal period (FY, Q1, Q2, Q3, Q4) (str)
- filed_date: Filing date YYYY-MM-DD (str)

Restatement Handling & Deduplication Rule:
- Each row represents a line item reported within a specific filing accession.
- To handle restatements, when analyzing a narrative claim from a specific filing,
  the line item value reported in that filing's accession is preserved.
- Intra-filing exact duplicate facts for (accession, us_gaap_tag, unit, period_start, period_end)
  are deduplicated keeping the record with the latest filed_date or non-null frame.
"""

import os
import json
import random
import pandas as pd
import numpy as np
from pathlib import Path

RAW_SEC_DIR = Path("data/raw/sec")
MANIFEST_PATH = Path("data/raw/manifest.csv")
PROCESSED_DIR = Path("data/processed")
OUTPUT_PARQUET = PROCESSED_DIR / "line_items.parquet"

# Sector mapping fallback if manifest is missing entries
DEFAULT_SECTOR_MAP = {
    "0000320193": ("Apple Inc.", "Tech"),
    "0000789019": ("Microsoft Corp.", "Tech"),
    "0001318605": ("Tesla Inc.", "Tech"),
    "0001045810": ("NVIDIA Corp.", "Tech"),
    "0001018724": ("Amazon.com Inc.", "Tech"),
    "0001652044": ("Alphabet Inc.", "Tech"),
    "0001341439": ("Oracle Corp.", "Tech"),
    "0000200406": ("Johnson & Johnson", "Healthcare"),
    "0000078003": ("Pfizer Inc.", "Healthcare"),
    "0000731766": ("UnitedHealth Group Inc.", "Healthcare"),
    "0000059478": ("Eli Lilly & Co.", "Healthcare"),
    "0000001800": ("Abbott Laboratories", "Healthcare"),
    "0000019617": ("JPMorgan Chase & Co.", "Banking"),
    "0000070858": ("Bank of America Corp.", "Banking"),
    "0000072971": ("Wells Fargo & Co.", "Banking"),
    "0000831001": ("Citigroup Inc.", "Banking"),
    "0000886982": ("Goldman Sachs Group Inc.", "Banking"),
    "0000895421": ("Morgan Stanley", "Banking"),
    "0000034088": ("Exxon Mobil Corp.", "Energy"),
    "0000093410": ("Chevron Corp.", "Energy"),
    "0001163165": ("ConocoPhillips", "Energy"),
    "0000087347": ("Schlumberger Ltd.", "Energy"),
    "0000821189": ("EOG Resources Inc.", "Energy"),
    "0000104169": ("Walmart Inc.", "Retail"),
    "0000027419": ("Target Corp.", "Retail"),
    "0000909832": ("Costco Wholesale Corp.", "Retail"),
    "0000354950": ("Home Depot Inc.", "Retail"),
    "0000060667": ("Lowe's Companies Inc.", "Retail"),
    "0000018230": ("Caterpillar Inc.", "Industrials"),
    "0000040545": ("General Electric Co.", "Industrials"),
    "0000773840": ("Honeywell International Inc.", "Industrials"),
    "0000936468": ("Lockheed Martin Corp.", "Industrials"),
    "0000066740": ("3M Co.", "Industrials"),
}


def load_sector_map():
    sector_map = {}
    if MANIFEST_PATH.exists():
        try:
            mdf = pd.read_csv(MANIFEST_PATH)
            for _, row in mdf.iterrows():
                if pd.notna(row.get("cik")):
                    cik_str = f"{int(row['cik']):010d}"
                    sector_map[cik_str] = (str(row.get("company", "")), str(row.get("sector", "")))
        except Exception as e:
            print(f"[WARN] Error reading manifest.csv: {e}")
    
    # Fill in defaults if any CIK is missing
    for cik, (comp, sec) in DEFAULT_SECTOR_MAP.items():
        if cik not in sector_map:
            sector_map[cik] = (comp, sec)
            
    return sector_map


def build_line_items_dataframe() -> pd.DataFrame:
    sector_map = load_sector_map()
    json_files = list(RAW_SEC_DIR.glob("CIK*.json"))
    print(f"[BUILD] Processing {len(json_files)} raw SEC XBRL JSON files...")

    records = []
    valid_forms = {"10-K", "10-Q", "10-K/A", "10-Q/A"}
    valid_years = {2021, 2022, 2023, 2024, 2025}

    for json_file in json_files:
        with open(json_file, "r", encoding="utf-8") as f:
            data = json.load(f)

        raw_cik = data.get("cik")
        if raw_cik is None:
            continue
        cik = f"{int(raw_cik):010d}"
        
        default_name, sector = sector_map.get(cik, (data.get("entityName", "Unknown"), "Unknown"))
        company = data.get("entityName") if data.get("entityName") else default_name

        gaap_facts = data.get("facts", {}).get("us-gaap", {})
        for tag, concept in gaap_facts.items():
            label = concept.get("label") or tag
            units = concept.get("units", {})
            for unit_name, items in units.items():
                for item in items:
                    form = str(item.get("form", "")).upper().strip()
                    fy = item.get("fy")
                    
                    if form in valid_forms and fy in valid_years:
                        val = item.get("val")
                        if val is None or pd.isna(val):
                            continue
                        
                        accn = str(item.get("accn", "")).strip()
                        fp = str(item.get("fp", "")).strip()
                        filed = str(item.get("filed", "")).strip()
                        start = str(item.get("start", "")).strip() if item.get("start") else ""
                        end = str(item.get("end", "")).strip() if item.get("end") else ""
                        
                        inst_dur = "instant" if (not start or start == end) else "duration"
                        
                        records.append({
                            "company": company,
                            "cik": cik,
                            "sector": sector,
                            "accession": accn,
                            "form": form,
                            "us_gaap_tag": tag,
                            "label": label,
                            "value": float(val),
                            "unit": unit_name,
                            "period_start": start,
                            "period_end": end,
                            "instant_or_duration": inst_dur,
                            "fiscal_year": int(fy),
                            "fiscal_period": fp,
                            "filed_date": filed
                        })

    df = pd.DataFrame(records)
    print(f"[EXTRACTED] Raw extracted records before deduplication: {len(df):,}")

    # Deduplication rule:
    # Within a single accession, keep the latest filed date record if identical facts exist.
    dedup_cols = ["company", "cik", "accession", "form", "us_gaap_tag", "unit", "period_start", "period_end", "instant_or_duration", "fiscal_year", "fiscal_period"]
    df.sort_values(by=["filed_date"], ascending=True, inplace=True)
    df.drop_duplicates(subset=dedup_cols, keep="last", inplace=True)
    df.sort_values(by=["sector", "company", "fiscal_year", "fiscal_period", "us_gaap_tag"], inplace=True)
    df.reset_index(drop=True, inplace=True)

    print(f"[DEDUPLICATED] Clean structured line items: {len(df):,}")
    return df


def perform_sanity_checks(df: pd.DataFrame, num_spot_checks: int = 20):
    print("\n" + "=" * 70)
    print("=== SANITY CHECKS & DIAGNOSTICS ===")
    print("=" * 70)

    # 1. Total summary stats
    print(f"\n1. Global Row Count: {len(df):,} line items across {df['company'].nunique()} companies")

    # 2. Row counts per sector and company
    print("\n2. Line Item Row Counts per Sector and Company:")
    counts_comp = df.groupby(["sector", "company"]).size().reset_index(name="row_count")
    for sector, group in counts_comp.groupby("sector"):
        print(f"\n  --- Sector: {sector} ---")
        for _, row in group.iterrows():
            print(f"    - {row['company']}: {row['row_count']:,} rows")

    # 3. Row counts per fiscal year
    print("\n3. Row Counts per Fiscal Year:")
    fy_counts = df.groupby("fiscal_year").size()
    for fy, count in fy_counts.items():
        print(f"    - FY {fy}: {count:,} rows")

    # 4. Top 30 most frequent US-GAAP tags
    print("\n4. Top 30 Most Frequent US-GAAP Tags:")
    top_30 = df["us_gaap_tag"].value_counts().head(30)
    for rank, (tag, count) in enumerate(top_30.items(), 1):
        sample_label = df[df["us_gaap_tag"] == tag]["label"].iloc[0]
        print(f"    {rank:2d}. {tag:<45} ({count:,} rows) - '{sample_label}'")

    # 5. 20-row random spot-check against raw JSON
    print(f"\n5. Spot-Check {num_spot_checks} Random Rows Against Raw SEC JSON Files:")
    print("-" * 110)
    print(f"{'Company':<20} | {'FY/FP':<7} | {'Form':<5} | {'US-GAAP Tag':<30} | {'Parquet Value':<15} | {'JSON Value':<15} | {'Match'}")
    print("-" * 110)

    random.seed(42)
    sample_indices = random.sample(range(len(df)), min(num_spot_checks, len(df)))
    mismatches = 0

    for idx in sample_indices:
        row = df.iloc[idx]
        cik = row["cik"]
        json_file = RAW_SEC_DIR / f"CIK{cik}.json"
        
        with open(json_file, "r", encoding="utf-8") as f:
            raw_data = json.load(f)

        gaap = raw_data.get("facts", {}).get("us-gaap", {})
        tag_data = gaap.get(row["us_gaap_tag"], {})
        units_data = tag_data.get("units", {}).get(row["unit"], [])

        # Match entry in JSON by accession, fy, form, and period
        json_val = None
        for item in units_data:
            if (str(item.get("accn", "")).strip() == row["accession"] and
                item.get("fy") == row["fiscal_year"] and
                str(item.get("form", "")).upper().strip() == row["form"] and
                str(item.get("fp", "")).strip() == row["fiscal_period"]):
                
                # Check period start/end match
                s_item = str(item.get("start", "")).strip() if item.get("start") else ""
                e_item = str(item.get("end", "")).strip() if item.get("end") else ""
                if s_item == row["period_start"] and e_item == row["period_end"]:
                    json_val = float(item.get("val"))
                    break

        matches = (json_val is not None) and (abs(json_val - row["value"]) < 1e-4)
        if not matches:
            mismatches += 1
            status = "FAIL"
        else:
            status = "PASS"

        fy_fp = f"{row['fiscal_year']} {row['fiscal_period']}"
        val_pq_str = f"{row['value']:,.0f}" if abs(row['value']) >= 1000 else f"{row['value']:.2f}"
        val_js_str = f"{json_val:,.0f}" if (json_val is not None and abs(json_val) >= 1000) else (f"{json_val:.2f}" if json_val is not None else "NOT FOUND")

        print(f"{row['company'][:20]:<20} | {fy_fp:<7} | {row['form']:<5} | {row['us_gaap_tag'][:30]:<30} | {val_pq_str:<15} | {val_js_str:<15} | {status}")

    print("-" * 110)
    print(f"Spot Check Results: {num_spot_checks - mismatches}/{num_spot_checks} passed completely. Mismatches: {mismatches}")
    assert mismatches == 0, f"Spot check failed with {mismatches} mismatches!"
    print("=" * 70)


def main():
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    df = build_line_items_dataframe()
    
    # Save to Parquet
    df.to_parquet(OUTPUT_PARQUET, index=False)
    print(f"\n[SAVED] Structured line-item store saved to {OUTPUT_PARQUET}")
    print(f"[PARQUET SIZE] {OUTPUT_PARQUET.stat().st_size / (1024*1024):.2f} MB")

    # Perform sanity checks
    perform_sanity_checks(df, num_spot_checks=20)


if __name__ == "__main__":
    main()
