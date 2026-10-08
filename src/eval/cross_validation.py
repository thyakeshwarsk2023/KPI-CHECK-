#!/usr/bin/env python3
"""
src/eval/cross_validation.py

Executes 5-Fold Stratified Cross-Validation across all feature blocks and architectures:
1. Logistic Regression (Block A + B)
2. Logistic Regression (Block A Only)
3. Logistic Regression (Block B Only)
4. MLP Classifier (Block A + B)

Computes mean ± standard deviation for:
- Micro-F1
- Macro-F1
- Accuracy
- Precision (Micro & Macro)
- Recall (Micro & Macro)

Outputs:
- results/tables/cross_validation_metrics.csv
- results/figures/cv_performance_distribution.png
"""

import sys
import logging
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score

# Ensure root on sys.path
ROOT_DIR = Path(__file__).resolve().parents[2]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from src.constants import (
    ALL_FEATURES,
    BLOCK_A_FEATURES,
    BLOCK_B_FEATURES,
    OUTPUT_FEATURES_CSV,
    RESULTS_TABLES_DIR,
    RESULTS_FIGURES_DIR
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def main():
    RESULTS_TABLES_DIR.mkdir(parents=True, exist_ok=True)
    RESULTS_FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    if not OUTPUT_FEATURES_CSV.exists():
        logger.error(f"Features file {OUTPUT_FEATURES_CSV} not found. Run build_features.py first.")
        return

    df = pd.read_csv(OUTPUT_FEATURES_CSV)
    y = df["label"].values
    classes = sorted(list(set(y)))
    logger.info(f"Loaded {len(df)} samples across classes: {classes}")

    models_config = [
        {
            "name": "lr_full",
            "display": "Logistic Regression (Block A+B)",
            "features": ALL_FEATURES,
            "pipeline": lambda: Pipeline([
                ("scaler", StandardScaler()),
                ("clf", LogisticRegression(class_weight="balanced", max_iter=1000, random_state=42))
            ])
        },
        {
            "name": "lr_block_a",
            "display": "Logistic Regression (Block A only)",
            "features": BLOCK_A_FEATURES,
            "pipeline": lambda: Pipeline([
                ("scaler", StandardScaler()),
                ("clf", LogisticRegression(class_weight="balanced", max_iter=1000, random_state=42))
            ])
        },
        {
            "name": "lr_block_b",
            "display": "Logistic Regression (Block B only)",
            "features": BLOCK_B_FEATURES,
            "pipeline": lambda: Pipeline([
                ("scaler", StandardScaler()),
                ("clf", LogisticRegression(class_weight="balanced", max_iter=1000, random_state=42))
            ])
        },
        {
            "name": "mlp_full",
            "display": "MLP Classifier (Block A+B)",
            "features": ALL_FEATURES,
            "pipeline": lambda: Pipeline([
                ("scaler", StandardScaler()),
                ("clf", MLPClassifier(hidden_layer_sizes=(32, 16), max_iter=800, random_state=42))
            ])
        }
    ]

    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    cv_summary_rows = []
    all_fold_records = []

    print("\n" + "=" * 80)
    print(f"{'5-FOLD STRATIFIED CROSS-VALIDATION BENCHMARK':^80}")
    print("=" * 80)

    for cfg in models_config:
        name = cfg["name"]
        display = cfg["display"]
        feats = cfg["features"]
        X_sub = df[feats].values

        fold_accs = []
        fold_f1_micros = []
        fold_f1_macros = []
        fold_prec_macros = []
        fold_rec_macros = []

        for fold_idx, (train_idx, test_idx) in enumerate(skf.split(X_sub, y), start=1):
            pipe = cfg["pipeline"]()
            X_tr, y_tr = X_sub[train_idx], y[train_idx]
            X_te, y_te = X_sub[test_idx], y[test_idx]

            pipe.fit(X_tr, y_tr)
            y_pred = pipe.predict(X_te)

            acc = accuracy_score(y_te, y_pred) * 100
            f1_mic = f1_score(y_te, y_pred, average="micro", zero_division=0) * 100
            f1_mac = f1_score(y_te, y_pred, average="macro", zero_division=0) * 100
            prec_mac = precision_score(y_te, y_pred, average="macro", zero_division=0) * 100
            rec_mac = recall_score(y_te, y_pred, average="macro", zero_division=0) * 100

            fold_accs.append(acc)
            fold_f1_micros.append(f1_mic)
            fold_f1_macros.append(f1_mac)
            fold_prec_macros.append(prec_mac)
            fold_rec_macros.append(rec_mac)

            all_fold_records.append({
                "model": display,
                "fold": fold_idx,
                "accuracy": acc,
                "f1_micro": f1_mic,
                "f1_macro": f1_mac
            })

        mean_acc = np.mean(fold_accs)
        std_acc = np.std(fold_accs)
        mean_mic = np.mean(fold_f1_micros)
        std_mic = np.std(fold_f1_micros)
        mean_mac = np.mean(fold_f1_macros)
        std_mac = np.std(fold_f1_macros)
        mean_prec = np.mean(fold_prec_macros)
        std_prec = np.std(fold_prec_macros)
        mean_rec = np.mean(fold_rec_macros)
        std_rec = np.std(fold_rec_macros)

        cv_summary_rows.append({
            "model": display,
            "mean_f1_micro": round(mean_mic, 2),
            "std_f1_micro": round(std_mic, 2),
            "f1_micro_display": f"{mean_mic:.2f} +/- {std_mic:.2f}%",
            "mean_f1_macro": round(mean_mac, 2),
            "std_f1_macro": round(std_mac, 2),
            "f1_macro_display": f"{mean_mac:.2f} +/- {std_mac:.2f}%",
            "mean_accuracy": round(mean_acc, 2),
            "std_accuracy": round(std_acc, 2),
            "mean_precision_macro": round(mean_prec, 2),
            "mean_recall_macro": round(mean_rec, 2)
        })

        print(f"Model: {display}")
        print(f"  Micro-F1: {mean_mic:.2f}% +/- {std_mic:.2f}%")
        print(f"  Macro-F1: {mean_mac:.2f}% +/- {std_mac:.2f}%")
        print(f"  Accuracy: {mean_acc:.2f}% +/- {std_acc:.2f}%\n")

    df_summary = pd.DataFrame(cv_summary_rows)
    cv_table_path = RESULTS_TABLES_DIR / "cross_validation_metrics.csv"
    df_summary.to_csv(cv_table_path, index=False)
    logger.info(f"Saved 5-fold CV metrics to: {cv_table_path}")

    # Plot Boxplot / Fold Distribution
    df_folds = pd.DataFrame(all_fold_records)
    plt.figure(figsize=(10, 6))
    sns.boxplot(
        data=df_folds, x="model", y="f1_micro", palette="Blues_r", width=0.45,
        boxprops=dict(alpha=0.85), showmeans=True,
        meanprops={"marker": "o", "markerfacecolor": "red", "markeredgecolor": "black", "markersize": "7"}
    )
    sns.stripplot(data=df_folds, x="model", y="f1_micro", color="black", size=6, jitter=0.1, alpha=0.7)
    plt.title("5-Fold Stratified Cross-Validation: Micro-F1 Stability Across Architectures", fontsize=12, weight="bold", pad=12)
    plt.ylabel("Micro-F1 Score (%)", fontsize=11, weight="bold")
    plt.xlabel("Model Configuration", fontsize=11, weight="bold")
    plt.xticks(rotation=15, ha="right", fontsize=10, weight="bold")
    plt.ylim(50, 100)
    plt.grid(axis="y", linestyle="--", alpha=0.6)
    plt.tight_layout()

    fig_out = RESULTS_FIGURES_DIR / "cv_performance_distribution.png"
    plt.savefig(fig_out, dpi=300)
    plt.close()
    logger.info(f"Saved CV distribution plot to: {fig_out}")

    print("=" * 80)
    print(df_summary[["model", "f1_micro_display", "f1_macro_display", "mean_accuracy"]].to_string(index=False))
    print("=" * 80 + "\n")


if __name__ == "__main__":
    main()
