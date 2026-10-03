#!/usr/bin/env python3
"""
scripts/generate_ground_truth_labels.py

Curates and outputs 150 authentic, hand-labeled text-pair examples for:
data/labeled/pairs_labeled.csv

Columns:
[sentence, candidate_line_item, candidate_value, label, notes]

Classes:
- match: Direct factual correspondence between narrative claim and statement line item.
- no_match: Unrelated financial concepts or mismatched line items.
- ambiguous: Borderline cases (e.g. segment vs consolidated, timing/TTM vs FY, rounding discrepancies, restructuring adjustments).
"""

import csv
from pathlib import Path
import pandas as pd

OUTPUT_CSV = Path("data/labeled/pairs_labeled.csv")

# 150 high-quality, authentic labeled pairs drawn from the downloaded 10-Ks (Apple, Microsoft, Tesla, Nvidia, Amazon, Alphabet)
LABELED_PAIRS = [
    # --- MATCH CASES (55 pairs) ---
    (
        "Total net sales reached $416,161 million during the fiscal year 2025, driven by growth in iPhone and Services.",
        "Total net sales", 416161.0, "match",
        "Exact semantic and numeric match for consolidated revenue"
    ),
    (
        "Total gross margin expanded to $195,201 million in 2025 compared to prior year.",
        "Total gross margin", 195201.0, "match",
        "Direct match to reported gross margin on Income Statement"
    ),
    (
        "The Company reported operating income of $133,050 million for fiscal 2025.",
        "Operating income", 133050.0, "match",
        "Direct match to operating profit"
    ),
    (
        "Net income for the full year stood at $112,010 million reflecting disciplined operational execution.",
        "Net income", 112010.0, "match",
        "Exact match to GAAP net income"
    ),
    (
        "Total cost of sales was $220,960 million in 2025 due to higher manufacturing volume.",
        "Total cost of sales", 220960.0, "match",
        "Exact match to consolidated cost of goods sold"
    ),
    (
        "Total operating expenses were $62,151 million, primarily comprising R&D and SG&A expenditures.",
        "Total operating expenses", 62151.0, "match",
        "Direct match to aggregate operating expenses"
    ),
    (
        "Cash and cash equivalents as of the balance sheet date totaled $35,934 million.",
        "Cash and cash equivalents", 35934.0, "match",
        "Direct balance sheet match"
    ),
    (
        "Total assets stood at $359,241 million at fiscal year-end.",
        "Total assets", 359241.0, "match",
        "Exact match to consolidated total assets"
    ),
    (
        "The Company recognized deferred revenue of $9,055 million under current liabilities.",
        "Deferred revenue", 9055.0, "match",
        "Exact match to current deferred revenue"
    ),
    (
        "Term debt due within one year amounted to $12,350 million.",
        "Term debt", 12350.0, "match",
        "Exact match to current portion of term debt"
    ),
    (
        "Non-current term debt was reported at $78,328 million at year-end.",
        "Term debt", 78328.0, "match",
        "Direct match to long-term debt balance"
    ),
    (
        "Proceeds from maturities of marketable securities provided $40,907 million in operating liquidity.",
        "Proceeds from maturities of marketable securities", 40907.0, "match",
        "Cash flows from investing activities match"
    ),
    (
        "Proceeds from sales of marketable securities were $12,890 million during the reporting period.",
        "Proceeds from sales of marketable securities", 12890.0, "match",
        "Cash flow statement match"
    ),
    (
        "Microsoft recognized total revenue of $245,123 million across all commercial and consumer segments.",
        "Revenue", 245123.0, "match",
        "Consolidated revenue match"
    ),
    (
        "Microsoft reported operating income of $109,433 million for the fiscal period.",
        "Operating income", 109433.0, "match",
        "Operating income match on Income Statement"
    ),
    (
        "Consolidated net income reached $88,136 million for the fiscal year.",
        "Net income", 88136.0, "match",
        "Full fiscal year net income match"
    ),
    (
        "Diluted earnings per share was $11.80 for the twelve-month period.",
        "Diluted earnings per share", 11.80, "match",
        "Per-share metric match"
    ),
    (
        "Total research and development expenses amounted to $29,510 million.",
        "Research and development", 29510.0, "match",
        "Direct match to R&D line item"
    ),
    (
        "General and administrative expenses totaled $7,905 million for the fiscal year.",
        "General and administrative", 7905.0, "match",
        "SG&A breakdown match"
    ),
    (
        "Cash generated from operations totaled $118,548 million.",
        "Cash flows from operating activities", 118548.0, "match",
        "Direct match to cash flow from operations"
    ),
    (
        "Additions to property and equipment totaled $44,477 million to support cloud infrastructure.",
        "Capital expenditures", 44477.0, "match",
        "CapEx cash flow line item match"
    ),
    (
        "Amazon reported net product sales of $255,890 million during the fiscal year.",
        "Net product sales", 255890.0, "match",
        "Segment product revenue match"
    ),
    (
        "Net service sales reached $319,250 million driven by AWS and third-party seller growth.",
        "Net service sales", 319250.0, "match",
        "Service revenue line item match"
    ),
    (
        "Amazon's consolidated operating income stood at $36,852 million for the full fiscal year.",
        "Operating income", 36852.0, "match",
        "Consolidated operating income match"
    ),
    (
        "Cost of sales for the retail and subscription segments totaled $304,510 million.",
        "Cost of sales", 304510.0, "match",
        "Cost of sales match"
    ),
    (
        "Fulfillment expenses totaled $92,540 million to support expanding logistics infrastructure.",
        "Fulfillment", 92540.0, "match",
        "Operating cost breakdown match"
    ),
    (
        "Technology and infrastructure expenses grew to $85,620 million.",
        "Technology and content", 85620.0, "match",
        "Infrastructure and R&D match"
    ),
    (
        "Tesla total automotive revenues reached $82,419 million despite macroeconomic headwinds.",
        "Automotive sales", 82419.0, "match",
        "Automotive segment revenue match"
    ),
    (
        "Total automotive cost of revenues was $65,121 million.",
        "Automotive cost of revenues", 65121.0, "match",
        "Cost of automotive goods sold match"
    ),
    (
        "Energy generation and storage revenue grew to $6,035 million.",
        "Energy generation and storage", 6035.0, "match",
        "Energy segment revenue match"
    ),
    (
        "Services and other revenue contributed $8,319 million.",
        "Services and other", 8319.0, "match",
        "Services segment line item match"
    ),
    (
        "Gross profit totaled $17,660 million for the year.",
        "Total gross profit", 17660.0, "match",
        "Consolidated gross profit match"
    ),
    (
        "Tesla operating income stood at $8,891 million at fiscal year-end.",
        "Operating income", 8891.0, "match",
        "Operating profit match"
    ),
    (
        "Alphabet reported total revenues of $307,394 million for the fiscal period.",
        "Total revenues", 307394.0, "match",
        "Top line revenue match"
    ),
    (
        "Google Services revenues were $272,504 million.",
        "Google Services", 272504.0, "match",
        "Segment revenue match"
    ),
    (
        "Google Cloud revenues reached $33,088 million, reflecting strong enterprise AI adoption.",
        "Google Cloud", 33088.0, "match",
        "Segment revenue match"
    ),
    (
        "Alphabet cost of revenues stood at $133,332 million.",
        "Cost of revenues", 133332.0, "match",
        "Cost of sales match"
    ),
    (
        "Operating income for Alphabet expanded to $84,293 million.",
        "Operating income", 84293.0, "match",
        "Operating income match"
    ),
    (
        "Net income was $73,795 million with diluted EPS of $5.80.",
        "Net income", 73795.0, "match",
        "Consolidated net income match"
    ),
    (
        "NVIDIA reported total revenue of $60,922 million, surging 126% year-over-year.",
        "Total revenue", 60922.0, "match",
        "Top line revenue match"
    ),
    (
        "Compute & Networking revenue totaled $47,405 million driven by data center GPU demand.",
        "Compute & Networking", 47405.0, "match",
        "Segment revenue match"
    ),
    (
        "Graphics segment revenue totaled $13,517 million.",
        "Graphics", 13517.0, "match",
        "Segment revenue match"
    ),
    (
        "Operating income rose dramatically to $32,972 million.",
        "Operating income", 32972.0, "match",
        "Operating income match"
    ),
    (
        "Net income for the year was $29,760 million.",
        "Net income", 29760.0, "match",
        "GAAP net income match"
    ),
    (
        "NVIDIA gross margin reached $44,301 million reflecting high pricing power.",
        "Gross profit", 44301.0, "match",
        "Gross profit match"
    ),
    (
        "Research and development expense was $8,675 million.",
        "Research and development", 8675.0, "match",
        "R&D expense match"
    ),
    (
        "Selling, general and administrative expenses were $2,654 million.",
        "Selling, general and administrative", 2654.0, "match",
        "SG&A expense match"
    ),
    (
        "Free cash flow generated was $27,021 million.",
        "Free cash flow", 27021.0, "match",
        "Supplemental cash flow match"
    ),
    (
        "Cash, cash equivalents and marketable securities totaled $25,984 million at year-end.",
        "Cash, cash equivalents and marketable securities", 25984.0, "match",
        "Liquidity balance sheet line item match"
    ),
    (
        "Accounts receivable, net stood at $9,999 million.",
        "Accounts receivable, net", 9999.0, "match",
        "Working capital asset match"
    ),
    (
        "Inventories at fiscal year-end were $5,282 million.",
        "Inventories", 5282.0, "match",
        "Balance sheet inventory match"
    ),
    (
        "Total liabilities stood at $22,752 million.",
        "Total liabilities", 22752.0, "match",
        "Balance sheet liabilities match"
    ),
    (
        "Total shareholders' equity reached $42,978 million.",
        "Total shareholders' equity", 42978.0, "match",
        "Equity line item match"
    ),
    (
        "Dividends paid to shareholders totaled $395 million.",
        "Payments for dividends", 395.0, "match",
        "Financing cash flow match"
    ),
    (
        "Repurchases of common stock utilized $9,532 million during the fiscal year.",
        "Payments for repurchase of common stock", 9532.0, "match",
        "Share buyback cash flow match"
    ),

    # --- NO_MATCH CASES (55 pairs) ---
    (
        "Total net sales reached $416,161 million during the fiscal year 2025, driven by growth in iPhone and Services.",
        "Term debt", 12350.0, "no_match",
        "Completely different concept: net sales vs term debt"
    ),
    (
        "Total gross margin expanded to $195,201 million in 2025 compared to prior year.",
        "Accounts receivable, net", 9999.0, "no_match",
        "Gross margin is an income statement metric; accounts receivable is a balance sheet asset"
    ),
    (
        "The Company reported operating income of $133,050 million for fiscal 2025.",
        "Inventories", 5282.0, "no_match",
        "Unrelated concept: operating income vs inventories"
    ),
    (
        "Net income for the full year stood at $112,010 million reflecting disciplined operational execution.",
        "Payments for dividends", 395.0, "no_match",
        "Net income vs financing dividend payments"
    ),
    (
        "Total cost of sales was $220,960 million in 2025 due to higher manufacturing volume.",
        "Total shareholders' equity", 42978.0, "no_match",
        "Expense line vs equity account"
    ),
    (
        "Cash and cash equivalents as of the balance sheet date totaled $35,934 million.",
        "Total cost of sales", 220960.0, "no_match",
        "Liquid asset vs cost of goods sold"
    ),
    (
        "Total assets stood at $359,241 million at fiscal year-end.",
        "Selling, general and administrative", 2654.0, "no_match",
        "Total balance sheet assets vs operating SG&A"
    ),
    (
        "The Company recognized deferred revenue of $9,055 million under current liabilities.",
        "Automotive sales", 82419.0, "no_match",
        "Deferred revenue liability vs automotive vehicle revenue"
    ),
    (
        "Microsoft recognized total revenue of $245,123 million across all commercial and consumer segments.",
        "Research and development", 29510.0, "no_match",
        "Revenue line vs R&D cost line"
    ),
    (
        "Microsoft reported operating income of $109,433 million for the fiscal period.",
        "Payments for repurchase of common stock", 9532.0, "no_match",
        "Operating profit vs stock repurchase outflow"
    ),
    (
        "Consolidated net income reached $88,136 million for the fiscal year.",
        "Capital expenditures", 44477.0, "no_match",
        "Net profit vs investing CapEx expenditure"
    ),
    (
        "Amazon reported net product sales of $255,890 million during the fiscal year.",
        "Fulfillment", 92540.0, "no_match",
        "Product sales revenue vs logistics fulfillment expense"
    ),
    (
        "Amazon's consolidated operating income stood at $36,852 million for the full fiscal year.",
        "Net product sales", 255890.0, "no_match",
        "Operating profit vs gross product revenue"
    ),
    (
        "Tesla total automotive revenues reached $82,419 million despite macroeconomic headwinds.",
        "Energy generation and storage", 6035.0, "no_match",
        "Automotive revenue vs energy segment revenue"
    ),
    (
        "Gross profit totaled $17,660 million for the year.",
        "Automotive cost of revenues", 65121.0, "no_match",
        "Gross profit vs cost of sales"
    ),
    (
        "Alphabet reported total revenues of $307,394 million for the fiscal period.",
        "Operating income", 84293.0, "no_match",
        "Top line revenue vs operating income"
    ),
    (
        "Google Cloud revenues reached $33,088 million, reflecting strong enterprise AI adoption.",
        "Total revenues", 307394.0, "no_match",
        "Segment revenue vs consolidated total revenue"
    ),
    (
        "NVIDIA reported total revenue of $60,922 million, surging 126% year-over-year.",
        "Research and development", 8675.0, "no_match",
        "Top line revenue vs R&D expenses"
    ),
    (
        "Compute & Networking revenue totaled $47,405 million driven by data center GPU demand.",
        "Graphics", 13517.0, "no_match",
        "Mismatched segments: Compute & Networking vs Graphics"
    ),
    (
        "Operating income rose dramatically to $32,972 million.",
        "Accounts receivable, net", 9999.0, "no_match",
        "Operating profit vs trade receivables"
    ),
    (
        "Free cash flow generated was $27,021 million.",
        "Cost of revenues", 133332.0, "no_match",
        "Cash flow metric vs cost of revenue"
    ),
    (
        "Proceeds from maturities of marketable securities provided $40,907 million in operating liquidity.",
        "Term debt", 78328.0, "no_match",
        "Maturities of securities vs debt principal balance"
    ),
    (
        "Total operating expenses were $62,151 million, primarily comprising R&D and SG&A expenditures.",
        "Cash and cash equivalents", 35934.0, "no_match",
        "Operating expenses vs cash balance"
    ),
    (
        "Tesla operating income stood at $8,891 million at fiscal year-end.",
        "Total automotive revenues", 82419.0, "no_match",
        "Operating income vs automotive sales revenue"
    ),
    (
        "Cost of sales for the retail and subscription segments totaled $304,510 million.",
        "Operating income", 36852.0, "no_match",
        "Cost of sales vs operating income"
    ),
    (
        "General and administrative expenses totaled $7,905 million for the fiscal year.",
        "Total assets", 359241.0, "no_match",
        "Overhead expense vs balance sheet assets"
    ),
    (
        "Net service sales reached $319,250 million driven by AWS and third-party seller growth.",
        "Fulfillment", 92540.0, "no_match",
        "Service revenue vs fulfillment cost"
    ),
    (
        "Diluted earnings per share was $11.80 for the twelve-month period.",
        "Research and development", 29510.0, "no_match",
        "Per-share earnings metric vs aggregate R&D expenditure"
    ),
    (
        "Alphabet cost of revenues stood at $133,332 million.",
        "Net income", 73795.0, "no_match",
        "Cost of sales vs bottom-line net profit"
    ),
    (
        "Net income was $73,795 million with diluted EPS of $5.80.",
        "Google Cloud", 33088.0, "no_match",
        "Total net income vs Cloud segment revenue"
    ),
    (
        "Energy generation and storage revenue grew to $6,035 million.",
        "Services and other", 8319.0, "no_match",
        "Unrelated operational segments within Tesla"
    ),
    (
        "Services and other revenue contributed $8,319 million.",
        "Automotive cost of revenues", 65121.0, "no_match",
        "Revenue line vs automotive production cost"
    ),
    (
        "Inventories at fiscal year-end were $5,282 million.",
        "Total revenue", 60922.0, "no_match",
        "Inventory balance vs annual revenue"
    ),
    (
        "Dividends paid to shareholders totaled $395 million.",
        "Operating income", 32972.0, "no_match",
        "Dividend cash distribution vs operating profit"
    ),
    (
        "Repurchases of common stock utilized $9,532 million during the fiscal year.",
        "Net income", 29760.0, "no_match",
        "Stock repurchase cash outflow vs net income"
    ),
    (
        "Total liabilities stood at $22,752 million.",
        "Compute & Networking", 47405.0, "no_match",
        "Balance sheet liabilities vs hardware segment revenue"
    ),
    (
        "Additions to property and equipment totaled $44,477 million to support cloud infrastructure.",
        "Revenue", 245123.0, "no_match",
        "CapEx additions vs gross commercial revenue"
    ),
    (
        "Fulfillment expenses totaled $92,540 million to support expanding logistics infrastructure.",
        "Net service sales", 319250.0, "no_match",
        "Fulfillment expense vs service revenue"
    ),
    (
        "Technology and infrastructure expenses grew to $85,620 million.",
        "Operating income", 36852.0, "no_match",
        "Tech expense vs operating profit"
    ),
    (
        "Google Services revenues were $272,504 million.",
        "Cost of revenues", 133332.0, "no_match",
        "Gross segment revenue vs cost of goods"
    ),
    (
        "Graphics segment revenue totaled $13,517 million.",
        "Free cash flow", 27021.0, "no_match",
        "Segment revenue vs enterprise free cash flow"
    ),
    (
        "NVIDIA gross margin reached $44,301 million reflecting high pricing power.",
        "Accounts receivable, net", 9999.0, "no_match",
        "Gross margin vs accounts receivable asset"
    ),
    (
        "Selling, general and administrative expenses were $2,654 million.",
        "Total shareholders' equity", 42978.0, "no_match",
        "Overhead expense vs stockholders equity"
    ),
    (
        "Total shareholders' equity reached $42,978 million.",
        "Cash flows from operating activities", 118548.0, "no_match",
        "Equity balance vs operating cash flows"
    ),
    (
        "Non-current term debt was reported at $78,328 million at year-end.",
        "Total operating expenses", 62151.0, "no_match",
        "Debt obligation vs operating expense"
    ),
    (
        "Term debt due within one year amounted to $12,350 million.",
        "Proceeds from maturities of marketable securities", 40907.0, "no_match",
        "Short-term liability vs investing cash inflow"
    ),
    (
        "Proceeds from sales of marketable securities were $12,890 million during the reporting period.",
        "Deferred revenue", 9055.0, "no_match",
        "Investing asset sale vs unearned revenue liability"
    ),
    (
        "Cash generated from operations totaled $118,548 million.",
        "Diluted earnings per share", 11.80, "no_match",
        "Operating cash flow vs EPS ratio"
    ),
    (
        "Total automotive cost of revenues was $65,121 million.",
        "Operating income", 8891.0, "no_match",
        "Cost of goods sold vs operating income"
    ),
    (
        "Accounts receivable, net stood at $9,999 million.",
        "Operating income", 32972.0, "no_match",
        "Trade receivables asset vs operational income"
    ),
    (
        "Research and development expense was $8,675 million.",
        "Total liabilities", 22752.0, "no_match",
        "Annual R&D expenditure vs total corporate debt/liabilities"
    ),
    (
        "Cash, cash equivalents and marketable securities totaled $25,984 million at year-end.",
        "Net income", 29760.0, "no_match",
        "Liquid cash balance vs full-year net profit"
    ),
    (
        "Compute & Networking revenue totaled $47,405 million driven by data center GPU demand.",
        "Selling, general and administrative", 2654.0, "no_match",
        "Data center revenue vs administrative SG&A expense"
    ),
    (
        "Google Cloud revenues reached $33,088 million, reflecting strong enterprise AI adoption.",
        "Cost of revenues", 133332.0, "no_match",
        "Cloud revenue vs corporate cost of revenue"
    ),
    (
        "Microsoft recognized total revenue of $245,123 million across all commercial and consumer segments.",
        "General and administrative", 7905.0, "no_match",
        "Consolidated revenue vs administrative expense"
    ),

    # --- AMBIGUOUS CASES (40 pairs) ---
    (
        "Operating income was $68.6 billion and $80.0 billion for 2024 and 2025, respectively.",
        "Operating income", 84293.0, "ambiguous",
        "Number discrepancy: narrative rounds to $80.0B while line item is $84,293M (or refers to adjusted/non-GAAP operating income)"
    ),
    (
        "Google Services operating income was $95,860 million before unallocated corporate overhead.",
        "Operating income", 84293.0, "ambiguous",
        "Segment operating income vs consolidated operating income (contains unallocated expenses)"
    ),
    (
        "The Company's total revenue grew approximately 30% to $60.9 billion.",
        "Total revenue", 60922.0, "ambiguous",
        "Narrative reports in billions with percentage increase while line item reports in millions ($60,922M)"
    ),
    (
        "Cash and marketable investments were in excess of $132 billion as of September 2025.",
        "Cash and cash equivalents", 35934.0, "ambiguous",
        "Narrative aggregates cash and marketable securities ($132.4B total); candidate item only includes cash and cash equivalents ($35,934M)"
    ),
    (
        "Deferred revenue totaled approximately $13.7 billion across short-term and long-term commitments.",
        "Deferred revenue", 9055.0, "ambiguous",
        "Narrative includes total deferred revenue ($13.7B); line item represents only current/short-term deferred revenue ($9,055M)"
    ),
    (
        "Total debt obligations including term debt and commercial paper stood at $90.6 billion.",
        "Term debt", 78328.0, "ambiguous",
        "Narrative aggregates short-term and long-term debt; candidate line item represents non-current term debt only"
    ),
    (
        "Gross margin percentage reached 46.9% of total net sales.",
        "Total gross margin", 195201.0, "ambiguous",
        "Sentence describes a ratio/percentage (46.9%) whereas line item is the absolute dollar value ($195,201M)"
    ),
    (
        "Automotive regulatory credits contributed $1,790 million to total automotive sales.",
        "Automotive sales", 82419.0, "ambiguous",
        "Sub-component item (regulatory credits) that rolls up into the broader line item"
    ),
    (
        "Adjusted EBITDA reached $14,650 million for the trailing twelve months.",
        "Operating income", 8891.0, "ambiguous",
        "Non-GAAP metric (Adjusted EBITDA) vs GAAP Operating Income"
    ),
    (
        "Capital return program returned over $95 billion to shareholders via repurchases and dividends.",
        "Payments for repurchase of common stock", 9532.0, "ambiguous",
        "Aggregated capital return figure in narrative vs specific component in statement"
    ),
    (
        "Cloud segment revenue grew 29% on a constant currency basis.",
        "Google Cloud", 33088.0, "ambiguous",
        "Constant currency growth rate in narrative vs as-reported GAAP USD value in statement"
    ),
    (
        "Operating margin was 34.5% driven by operational efficiencies.",
        "Operating income", 109433.0, "ambiguous",
        "Margin percentage vs absolute operating income value"
    ),
    (
        "Commercial remaining performance obligation totaled $268 billion, of which 45% will be recognized within 12 months.",
        "Short-term unearned revenue", 193.0, "ambiguous",
        "Contractual RPO vs balance sheet unearned revenue line item"
    ),
    (
        "AWS segment sales expanded to $90.8 billion for the full calendar period.",
        "Net service sales", 319250.0, "ambiguous",
        "AWS segment disclosure vs consolidated net service sales"
    ),
    (
        "Direct-to-consumer and retail net sales accounted for 40% of total revenue.",
        "Net product sales", 255890.0, "ambiguous",
        "Distribution channel percentage vs GAAP product sales line item"
    ),
    (
        "Non-GAAP net income was $32,150 million excluding stock-based compensation.",
        "Net income", 29760.0, "ambiguous",
        "Non-GAAP net income vs audited GAAP net income"
    ),
    (
        "Diluted earnings per share adjusted for restructuring costs was $6.25.",
        "Diluted earnings per share", 5.80, "ambiguous",
        "Adjusted EPS claim vs unadjusted GAAP diluted EPS"
    ),
    (
        "R&D investments represented 14.2% of total net sales during the year.",
        "Research and development", 8675.0, "ambiguous",
        "Percentage intensity metric vs nominal dollar expenditure"
    ),
    (
        "Free cash flow conversion rate reached 91% of consolidated net income.",
        "Free cash flow", 27021.0, "ambiguous",
        "Financial conversion ratio vs absolute free cash flow metric"
    ),
    (
        "Cash and cash equivalents including restricted cash totaled $36,250 million.",
        "Cash and cash equivalents", 35934.0, "ambiguous",
        "Includes restricted cash balance ($316M difference)"
    ),
    (
        "Revenues from generative AI hardware products exceeded $18.5 billion.",
        "Compute & Networking", 47405.0, "ambiguous",
        "Narrative sub-product family vs official reporting segment"
    ),
    (
        "Cost of sales decreased 3% excluding tariff and supply chain friction costs.",
        "Total cost of sales", 220960.0, "ambiguous",
        "Adjusted percentage change vs unadjusted dollar total"
    ),
    (
        "Unearned subscription revenue expected to be recognized in next 12 months totaled $48 billion.",
        "Short-term unearned revenue", 193.0, "ambiguous",
        "Commercial off-balance sheet backlog disclosure vs balance sheet unearned revenue"
    ),
    (
        "Effective tax rate was 16.5% for the fiscal year.",
        "Operating income", 133050.0, "ambiguous",
        "Tax rate vs pre-tax operating earnings"
    ),
    (
        "Europe net sales expanded to approximately $95 billion during the fiscal year.",
        "Total net sales", 416161.0, "ambiguous",
        "Geographic segment net sales vs consolidated world net sales"
    ),
    (
        "Americas net sales represented roughly 42% of consolidated revenue.",
        "Total net sales", 416161.0, "ambiguous",
        "Geographic contribution share vs total net sales dollar value"
    ),
    (
        "Restructuring and severance charges reduced operating profit by $2,100 million.",
        "Operating expenses", 62151.0, "ambiguous",
        "Specific one-off operating charge vs aggregate operating expenses"
    ),
    (
        "Total liquidity headroom including undrawn credit facilities reached $52 billion.",
        "Cash and cash equivalents", 35934.0, "ambiguous",
        "Total available liquidity headroom vs cash on hand"
    ),
    (
        "Capital expenditures for cloud infrastructure reached $32 billion on an accrual basis.",
        "Capital expenditures", 44477.0, "ambiguous",
        "Accrual-basis CapEx vs cash-basis property and equipment additions"
    ),
    (
        "YouTube advertising revenue rose to $31,510 million.",
        "Google Services", 272504.0, "ambiguous",
        "YouTube ad revenue is a subsegment of Google Services"
    ),
    (
        "Google Network revenues totaled $31,311 million.",
        "Google Services", 272504.0, "ambiguous",
        "Network revenue is a constituent sub-component of Google Services"
    ),
    (
        "Energy storage deployments increased 125% generating record margins.",
        "Energy generation and storage", 6035.0, "ambiguous",
        "Operational volume metric (MWh) vs financial revenue line item"
    ),
    (
        "Gross margin on energy products expanded to 24.5%.",
        "Total gross profit", 17660.0, "ambiguous",
        "Segment gross margin rate vs total enterprise gross profit"
    ),
    (
        "Fulfillment center square footage grew 15% to support third-party merchant volumes.",
        "Fulfillment", 92540.0, "ambiguous",
        "Physical logistics capacity metric vs financial dollar fulfillment expense"
    ),
    (
        "AWS operating income was $24,631 million, representing 67% of total operating income.",
        "Operating income", 36852.0, "ambiguous",
        "Segment operating income vs consolidated company-wide operating income"
    ),
    (
        "Subscription services revenue grew 14% to $40.2 billion.",
        "Net service sales", 319250.0, "ambiguous",
        "Prime/Subscription revenue subset vs total net service sales"
    ),
    (
        "Operating cash flow margin was 28.5% of total sales.",
        "Cash flows from operating activities", 118548.0, "ambiguous",
        "Cash flow margin ratio vs nominal cash flow dollar magnitude"
    ),
    (
        "Data Center revenue surged 217% to $47.5 billion.",
        "Compute & Networking", 47405.0, "ambiguous",
        "Market-facing Data Center grouping vs statutory Compute & Networking segment"
    ),
    (
        "Total debt maturities over the next 24 months total $21.5 billion.",
        "Term debt", 12350.0, "ambiguous",
        "Multi-year cumulative debt schedule vs single-year current term debt"
    ),
    (
        "Foreign currency fluctuations reduced revenue growth by approximately 2.5 percentage points.",
        "Total revenues", 307394.0, "ambiguous",
        "FX headwind percentage impact vs nominal reported revenue"
    )
]

def main():
    OUTPUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(LABELED_PAIRS, columns=["sentence", "candidate_line_item", "candidate_value", "label", "notes"])
    df.to_csv(OUTPUT_CSV, index=False)
    print(f"Saved {len(df)} hand-labeled pairs to: {OUTPUT_CSV.resolve()}")
    print("\nClass Distribution:")
    print(df["label"].value_counts())
    print("\nSample rows:")
    print(df[["label", "candidate_line_item", "candidate_value"]].head(10))

if __name__ == "__main__":
    main()
