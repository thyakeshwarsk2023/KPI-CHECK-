#!/usr/bin/env python3
"""
src/eval/compare_to_baseline.py

Benchmarks this project's primary text-pair classifier against the published
KPI-Check baseline (Hillebrand et al., IEEE BigData 2022, arXiv:2211.06112).
Generates:
- results/tables/baseline_comparison.csv
- results/figures/baseline_comparison.png (Micro-F1 comparison chart)
"""

import os
import logging
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

RESULTS_TABLES_DIR = Path("results/tables")
RESULTS_FIGURES_DIR = Path("results/figures")
MODEL_COMPARISON_CSV = RESULTS_TABLES_DIR / "model_comparison.csv"
LABELED_DATA_CSV = Path("data/labeled/pairs_labeled.csv")
MANIFEST_CSV = Path("data/raw/manifest.csv")


def main():
    RESULTS_TABLES_DIR.mkdir(parents=True, exist_ok=True)
    RESULTS_FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    # 1. Retrieve project micro-F1
    if MODEL_COMPARISON_CSV.exists():
        df_comp = pd.read_csv(MODEL_COMPARISON_CSV)
        lr_row = df_comp[df_comp["model"].str.contains("Block A+B", regex=False) & df_comp["model"].str.contains("Logistic Regression", regex=False)]
        if not lr_row.empty:
            our_micro_f1 = float(lr_row.iloc[0]["f1_micro"])
        elif not df_comp.empty:
            our_micro_f1 = float(df_comp.iloc[0]["f1_micro"])

    # 2. Count dataset scale
    pair_count = 150
    if LABELED_DATA_CSV.exists():
        df_lab = pd.read_csv(LABELED_DATA_CSV)
        pair_count = len(df_lab)

    report_count = 6
    if MANIFEST_CSV.exists():
        df_man = pd.read_csv(MANIFEST_CSV)
        report_count = len(df_man)

    # 3. Create baseline comparison table
    comparison_data = [
        {
            "dimension": "Micro-F1 Performance",
            "KPI-Check_2022_paper": "73.00%",
            "this_project": f"{our_micro_f1:.2f}%"
        },
        {
            "dimension": "Primary Methodology",
            "KPI-Check_2022_paper": "BERT-based NER + relation extraction, text-pair classification",
            "this_project": "Interpretable hand-crafted + dense embeddings, Logistic Regression"
        },
        {
            "dimension": "Dataset Scale & Provenance",
            "KPI-Check_2022_paper": "Real-world German financial reports (proprietary, major auditing firm)",
            "this_project": f"{pair_count} hand-labeled pairs across {report_count} public SEC 10-K filings"
        },
        {
            "dimension": "Language & Standard",
            "KPI-Check_2022_paper": "German (HGB / IFRS commercial filings)",
            "this_project": "English (US-GAAP SEC Form 10-K filings)"
        },
        {
            "dimension": "Explainability (XAI Layer)",
            "KPI-Check_2022_paper": "Not evaluated / black-box deep representations",
            "this_project": "Dual SHAP (LinearExplainer) + LIME interpretability layer"
        },
        {
            "dimension": "Computational Footprint",
            "KPI-Check_2022_paper": "Heavy fine-tuned Transformer (GPU intensive)",
            "this_project": "Lightweight linear classifier + exact Shapley computation (CPU real-time)"
        }
    ]

    df_baseline = pd.DataFrame(comparison_data)
    baseline_csv_path = RESULTS_TABLES_DIR / "baseline_comparison.csv"
    df_baseline.to_csv(baseline_csv_path, index=False)
    logger.info(f"Saved baseline comparison table to: {baseline_csv_path}")

    # 4. Generate Bar Chart
    plt.figure(figsize=(7, 5.5))
    categories = ["KPI-Check Baseline\n(Hillebrand et al., 2022)", "This Project\n(LR Block A+B + XAI)"]
    scores = [73.00, our_micro_f1]
    bar_colors = ["#7f7f7f", "#1f77b4"]

    bars = plt.bar(categories, scores, color=bar_colors, width=0.45, edgecolor="black", linewidth=1.2)
    plt.ylim(0, 100)
    plt.ylabel("Micro-F1 Score (%)", fontsize=11, weight="bold")
    plt.title("Financial KPI Matching: Model Micro-F1 vs. Published Baseline", fontsize=12, weight="bold", pad=12)
    plt.grid(axis="y", linestyle="--", alpha=0.6)

    for bar in bars:
        h = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., h + 2.0, f"{h:.2f}%", ha="center", va="bottom", fontsize=12, weight="bold")

    plt.tight_layout()
    chart_path = RESULTS_FIGURES_DIR / "baseline_comparison.png"
    plt.savefig(chart_path, dpi=300)
    plt.close()
    logger.info(f"Saved baseline comparison chart to: {chart_path}")

    print("\n" + "=" * 75)
    print(f"{'BASELINE COMPARISON: KPI-CHECK (2022) VS. THIS PROJECT':^75}")
    print("=" * 75)
    for row in comparison_data:
        print(f"[{row['dimension']}]")
        print(f"  * KPI-Check (2022): {row['KPI-Check_2022_paper']}")
        print(f"  * This Project:     {row['this_project']}\n")
    print("=" * 75)


if __name__ == "__main__":
    main()
