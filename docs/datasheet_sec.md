# SEC XBRL Line-Item Store Datasheet (`docs/datasheet_sec.md`)

## 1. Overview & Dataset Specification

The **SEC XBRL Line-Item Store** (`data/processed/line_items.parquet`) is a structured repository extracted directly from official SEC EDGAR XBRL `companyfacts` JSON feeds via the SEC REST API (`data.sec.gov/api/xbrl/companyfacts/`). It replaces legacy HTML table parsing with structured, machine-readable financial facts.

### Key Metrics
- **Total Line Items:** 298,663 rows
- **File Format:** Apache Parquet (`data/processed/line_items.parquet`)
- **File Size:** 2.01 MB
- **Coverage Period:** Fiscal Years 2021–2025
- **Filing Types:** Forms 10-K, 10-Q, 10-K/A, 10-Q/A
- **Total Companies:** 33 companies across 6 major sectors
- **Gold Test Set Inclusion:** Includes all 6 GOLD companies (AAPL, MSFT, TSLA, NVDA, AMZN, GOOGL)
- **Required Sector Inclusions:** Includes Johnson & Johnson (JNJ, Healthcare) and JPMorgan Chase & Co. (JPM, Banking)

---

## 2. Schema Specification

Each record in `line_items.parquet` represents a reported financial line item with 15 normalized columns:

| Column | Data Type | Description | Example |
|---|---|---|---|
| `company` | `string` | SEC registered entity name | `"Apple Inc."` |
| `cik` | `string` | 10-digit zero-padded Central Index Key | `"0000320193"` |
| `sector` | `string` | Industry sector classification | `"Tech"` |
| `accession` | `string` | SEC filing accession number (`accn`) | `"0000320193-23-000106"` |
| `form` | `string` | SEC form type | `"10-K"`, `"10-Q"` |
| `us_gaap_tag` | `string` | Standardized US-GAAP XBRL concept tag | `"Revenues"` |
| `label` | `string` | Human-readable US-GAAP taxonomy label | `"Revenue from Contract with Customer"` |
| `value` | `float64` | Numerical value in base units | `383285000000.0` |
| `unit` | `string` | Base measurement unit | `"USD"`, `"shares"`, `"USD/shares"` |
| `period_start` | `string` | Start date of duration period (`YYYY-MM-DD`) or empty | `"2022-09-25"` |
| `period_end` | `string` | End date of duration period or instant date | `"2023-09-30"` |
| `instant_or_duration` | `string` | Point-in-time (`instant`) vs range (`duration`) | `"duration"` |
| `fiscal_year` | `int64` | Reported fiscal year (2021–2025) | `2023` |
| `fiscal_period` | `string` | Fiscal period code (`FY`, `Q1`, `Q2`, `Q3`, `Q4`) | `"FY"` |
| `filed_date` | `string` | Official SEC filing date (`YYYY-MM-DD`) | `"2023-10-27"` |

---

## 3. Coverage & Per-Company Row Counts

### Sector Breakdown
| Sector | Companies | Total Rows | Avg Rows / Company |
|---|---|---|---|
| **Banking** | 6 | 83,567 | 13,928 |
| **Healthcare** | 5 | 39,878 | 7,976 |
| **Tech** | 7 | 57,806 | 8,258 |
| **Industrials** | 5 | 45,574 | 9,115 |
| **Retail** | 5 | 35,242 | 7,048 |
| **Energy** | 5 | 36,596 | 7,319 |
| **Total** | **33** | **298,663** | **9,050** |

### Detailed Per-Company Summary
| Company Name | CIK | Ticker | Sector | Gold Set? | Row Count |
|---|---|---|---|---|---|
| **Apple Inc.** | `0000320193` | AAPL | Tech | Yes | 6,106 |
| **MICROSOFT CORPORATION** | `0000789019` | MSFT | Tech | Yes | 9,208 |
| **Tesla, Inc.** | `0001318605` | TSLA | Tech | Yes | 8,627 |
| **NVIDIA CORP** | `0001045810` | NVDA | Tech | Yes | 7,512 |
| **AMAZON COM INC** | `0001018724` | AMZN | Tech | Yes | 9,759 |
| **Alphabet Inc.** | `0001652044` | GOOGL | Tech | Yes | 9,306 |
| **Oracle Corporation** | `0001341439` | ORCL | Tech | No | 7,288 |
| **Johnson & Johnson** | `0000200406` | JNJ | Healthcare | No | 8,637 |
| **PFIZER INC** | `0000078003` | PFE | Healthcare | No | 10,211 |
| **UnitedHealth Group Inc.** | `0000731766` | UNH | Healthcare | No | 8,572 |
| **ELI LILLY AND COMPANY** | `0000059478` | LLY | Healthcare | No | 6,124 |
| **ABBOTT LABORATORIES** | `0000001800` | ABT | Healthcare | No | 6,332 |
| **JPMORGAN CHASE & CO** | `0000019617` | JPM | Banking | No | 15,490 |
| **BofA Finance LLC** | `0000070858` | BAC | Banking | No | 11,984 |
| **WELLS FARGO & COMPANY/MN** | `0000072971` | WFC | Banking | No | 14,018 |
| **Citigroup Inc** | `0000831001` | C | Banking | No | 14,255 |
| **The Goldman Sachs Group, Inc.** | `0000886982` | GS | Banking | No | 13,451 |
| **MORGAN STANLEY** | `0000895421` | MS | Banking | No | 12,366 |
| **Exxon Mobil Corporation** | `0000034088` | XOM | Energy | No | 6,135 |
| **Chevron Corp** | `0000093410` | CVX | Energy | No | 9,938 |
| **ConocoPhillips** | `0001163165` | COP | Energy | No | 9,218 |
| **SLB LIMITED/NV** | `0000087347` | SLB | Energy | No | 6,762 |
| **EOG RESOURCES, INC.** | `0000821189` | EOG | Energy | No | 6,545 |
| **WALMART INC.** | `0000104169` | WMT | Retail | No | 7,241 |
| **Target Corporation** | `0000027419` | TGT | Retail | No | 7,085 |
| **COSTCO WHOLESALE CORP /NEW** | `0000909832` | COST | Retail | No | 6,472 |
| **The Home Depot, Inc.** | `0000354950` | HD | Retail | No | 6,529 |
| **LOWES COMPANIES INC** | `0000060667` | LOW | Retail | No | 7,918 |
| **CATERPILLAR INC** | `0000018230` | CAT | Industrials | No | 10,726 |
| **GENERAL ELECTRIC CO** | `0000040545` | GE | Industrials | No | 11,152 |
| **Honeywell International Inc** | `0000773840` | HON | Industrials | No | 8,296 |
| **LOCKHEED MARTIN CORPORATION** | `0000936468` | LMT | Industrials | No | 6,307 |
| **3M CO** | `0000066740` | MMM | Industrials | No | 9,093 |

---

## 4. Deduplication & Restatement Handling Rules

### Filing-Level Line Item Integrity
Financial narratives in 10-K and 10-Q reports reference facts **as reported in that specific filing**. When a company restates a previous year's financial figures in a subsequent filing, both the original filing accession and the subsequent restated accession coexist in `line_items.parquet`.

### Rule Definitions
1. **Accession Isolation (Rule A):**
   - Each row is strictly tied to an `accession` number.
   - When matching narrative claim sentences from filing $F$, candidate line items are queried against accession $F$, ensuring exact fidelity to the text in filing $F$.
2. **Intra-Filing Deduplication (Rule B):**
   - Within the exact same filing accession, duplicate XBRL facts can occur due to frame tag revisions or footnote tags.
   - We deduplicate on `(company, cik, accession, form, us_gaap_tag, unit, period_start, period_end, instant_or_duration, fiscal_year, fiscal_period)` by preserving the entry with the latest `filed_date` and non-null values.
3. **Cross-Filing Restatements (Rule C):**
   - If period `2021-01-01` to `2021-12-31` appears in 10-K 2021 (Accession A), 10-K 2022 (Accession B), and 10-K 2023 (Accession C), all three values are preserved under their respective accession numbers.

---

## 5. Bank-Specific XBRL Taxonomy Caveats

Commercial banks and financial institutions (e.g., JPM, BAC, WFC, C, GS, MS) utilize specialized US-GAAP taxonomy extensions that differ fundamentally from commercial/industrial firms:

| Concept Category | Commercial / Industrial Tag | Banking Tag (JPM, BAC, WFC, C, GS, MS) |
|---|---|---|
| **Top-Line Revenue** | `Revenues`, `SalesRevenueNet`, `RevenueFromContractWithCustomerExcludingAssessedTax` | `InterestAndDividendIncome`, `NoninterestIncome`, `InterestIncomeDepositWithFinancialInstitutions` |
| **Cost of Goods / Expense** | `CostOfGoodsAndServicesSold`, `OperatingExpenses` | `InterestExpense`, `NoninterestExpense`, `ProvisionForCreditLosses` |
| **Operating Income** | `OperatingIncomeLoss` | `IncomeBeforeIncomeTaxExpense` (Banks do not report gross/operating margin) |
| **Core Product Asset** | `InventoryNet`, `PropertyPlantAndEquipmentNet` | `LoansAndLeasesReceivableNetOfDeferredIncome`, `InvestmentSecurities` |
| **Core Liability** | `AccountsPayableCurrent`, `LongTermDebtNoncurrent` | `Deposits`, `FederalFundsPurchasedAndSecuritiesSoldUnderAgreementsToRepurchase` |

### Key Matching Takeaways for Financial Banking Reports
1. **No Gross Margin / Cost of Goods Sold:** Banking models must avoid searching for `CostOfGoodsAndServicesSold` or `GrossProfit` when parsing bank 10-Ks.
2. **Net Interest Income:** Key performance indicators in banking rely on `NetInterestIncome` (Interest Income minus Interest Expense) rather than standard commercial operating profit.
3. **Loan Loss Provisions:** Provision for credit losses represents a primary operating line item unique to financial institutions.

---

## 6. Reproducibility & Sanity Verification

The complete download and extraction pipeline can be reproduced with a single command:

```bash
python scripts/download_sec.py && python src/data/build_line_items.py
```

### Verification Result
- **Spot-Check Pass Rate:** 20 / 20 (100.0%) spot-checked random rows matched raw SEC EDGAR JSON files with exact decimal equality.
- **SEC EDGAR Access Compliance:** Built using `User-Agent: s.k.thyakeshwar skthyakeshwar@gmail.com` with rate limiting at 4 req/sec and exponential backoff retry.
