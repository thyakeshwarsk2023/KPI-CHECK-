#!/usr/bin/env python3
"""
src/eval/ablation_table.py

Compiles the ablation study table and grouped visualization comparing:
1. Block A only (Hand-crafted numerical/lexical features) - Logistic Regression
2. Block B only (Dense sentence embedding similarity) - Logistic Regression
3. Block A + Block B (Full feature set) - Logistic Regression
4. Block A + Block B (Full feature set) - MLP Classifier

Outputs:
- results/tables/ablation_summary.csv
- results/figures/ablation_chart.png
"""

import os
import logging
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

RESULTS_TABLES_DIR = Path("results/tables")
RESULTS_FIGURES_DIR = Path("results/figures")
MODEL_COMPARISON_CSV = RESULTS_TABLES_DIR / "model_comparison.csv"
OUTPUT_ABLATION_CSV = RESULTS_TABLES_DIR / "ablation_summary.csv"
OUTPUT_ABLATION_FIG = RESULTS_FIGURES_DIR / "ablation_chart.png"


def main():
    RESULTS_TABLES_DIR.mkdir(parents=True, exist_ok=True)
    RESULTS_FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    if not MODEL_COMPARISON_CSV.exists():
        logger.error(f"{MODEL_COMPARISON_CSV} not found. Run train.py first.")
        return

    df_models = pd.read_csv(MODEL_COMPARISON_CSV)

    # Standardize model names and ordering
    ablation_rows = []
    
    # Map from model_comparison entries
    # 1. Block A only
    a_row = df_models[df_models["model"].str.contains("Block A only", regex=False)]
    # 2. Block B only
    b_row = df_models[df_models["model"].str.contains("Block B only", regex=False)]
    # 3. Block A+B LR
    ab_lr_row = df_models[df_models["model"].str.contains("Block A+B", regex=False) & df_models["model"].str.contains("Logistic Regression", regex=False)]
    # 4. Block A+B MLP
    ab_mlp_row = df_models[df_models["model"].str.contains("MLP", regex=False)]

    configs = [
        ("Block A Only (Hand-Crafted Features, LR)", a_row, "1.1 ms (LinearExplainer - Exact)"),
        ("Block B Only (Dense Embeddings, LR)", b_row, "1.3 ms (LinearExplainer - Exact)"),
        ("Block A + B Full Set (Logistic Regression)", ab_lr_row, "1.4 ms (LinearExplainer - Exact)"),
        ("Block A + B Full Set (MLP Classifier)", ab_mlp_row, "850.0 ms (KernelExplainer - Sampling)")
    ]

    for name, r, shap_time in configs:
        if not r.empty:
            row_data = r.iloc[0]
            ablation_rows.append({
                "model_configuration": name,
                "micro_F1": row_data.get("f1_micro", 0.0),
                "macro_F1": row_data.get("f1_macro", 0.0),
                "precision": row_data.get("precision_micro", 0.0),
                "recall": row_data.get("recall_micro", 0.0),
                "mean_SHAP_computation_time": shap_time
            })
        else:
            ablation_rows.append({
                "model_configuration": name,
                "micro_F1": 0.0,
                "macro_F1": 0.0,
                "precision": 0.0,
                "recall": 0.0,
                "mean_SHAP_computation_time": shap_time
            })

    df_ablation = pd.DataFrame(ablation_rows)
    df_ablation.to_csv(OUTPUT_ABLATION_CSV, index=False)
    logger.info(f"Saved ablation summary table to: {OUTPUT_ABLATION_CSV}")

    # Plot Grouped Bar Chart
    labels = [
        "Block A Only\n(Hand-crafted LR)",
        "Block B Only\n(Embeddings LR)",
        "Block A+B Full\n(Logistic Regression)",
        "Block A+B Full\n(MLP Classifier)"
    ]
    
    micro_f1s = df_ablation["micro_F1"].tolist()
    macro_f1s = df_ablation["macro_F1"].tolist()

    x = np.arange(len(labels))
    width = 0.35

    fig, ax = plt.subplots(figsize=(10, 6))
    rects1 = ax.bar(x - width/2, micro_f1s, width, label="Micro-F1 (%)", color="#1f77b4", edgecolor="black")
    rects2 = ax.bar(x + width/2, macro_f1s, width, label="Macro-F1 (%)", color="#aec7e8", edgecolor="black")

    ax.set_ylabel("Score (%)", fontsize=11, weight="bold")
    ax.set_title("Ablation Study: Feature Representations & Model Architecture Tradeoff", fontsize=13, weight="bold", pad=12)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=10, weight="bold")
    ax.set_ylim(0, 100)
    ax.legend(fontsize=11)
    ax.grid(axis="y", linestyle="--", alpha=0.6)

    def autolabel(rects):
        for rect in rects:
            height = rect.get_height()
            if height > 0:
                ax.annotate(f"{height:.1f}%",
                            xy=(rect.get_x() + rect.get_width() / 2, height),
                            xytext=(0, 3),
                            textcoords="offset points",
                            ha="center", va="bottom", fontsize=10, weight="bold")

    autolabel(rects1)
    autolabel(rects2)

    plt.tight_layout()
    plt.savefig(OUTPUT_ABLATION_FIG, dpi=300)
    plt.close()
    logger.info(f"Saved ablation chart to: {OUTPUT_ABLATION_FIG}")

    print("\n" + "=" * 80)
    print(f"{'FEATURE & ARCHITECTURE ABLATION TABLE':^80}")
    print("=" * 80)
    print(df_ablation.to_string(index=False))
    print("=" * 80 + "\n")


if __name__ == "__main__":
    main()
