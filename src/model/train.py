#!/usr/bin/env python3
"""
src/model/train.py

Trains and evaluates text-pair classifiers on Block A and Block B features:
1. Logistic Regression (class_weight='balanced') on Block A + Block B.
2. MLP Classifier (sklearn MLPClassifier) on Block A + Block B.
3. Ablation Models:
   - Logistic Regression on Block A only (Hand-crafted features).
   - Logistic Regression on Block B only (Embedding similarity).
4. Saves models to src/model/saved/{model_name}.pkl.
5. Saves evaluation metrics to results/tables/model_comparison.csv.
6. Generates confusion matrix heatmaps in results/figures/.
"""

import os
import joblib
import logging
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, confusion_matrix, classification_report
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

FEATURES_CSV = Path("data/processed/features.csv")
SAVED_MODELS_DIR = Path("src/model/saved")
RESULTS_TABLES_DIR = Path("results/tables")
RESULTS_FIGURES_DIR = Path("results/figures")
SPLITS_DIR = Path("data/processed")

BLOCK_A_FEATURES = [
    "numeric_value_match", "numeric_value_close", "keyword_overlap",
    "period_match", "string_similarity", "sentence_length", "line_item_name_length"
]
BLOCK_B_FEATURES = ["embedding_cosine_similarity"]
ALL_FEATURES = BLOCK_A_FEATURES + BLOCK_B_FEATURES


def plot_confusion_matrix(cm, classes, title, output_path):
    """Plots and saves a styled confusion matrix heatmap."""
    plt.figure(figsize=(6, 5))
    sns.heatmap(
        cm, annot=True, fmt="d", cmap="Blues",
        xticklabels=classes, yticklabels=classes,
        cbar=True, annot_kws={"size": 13, "weight": "bold"}
    )
    plt.title(title, fontsize=13, weight="bold", pad=12)
    plt.ylabel("Ground Truth", fontsize=11, weight="bold")
    plt.xlabel("Predicted Label", fontsize=11, weight="bold")
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()
    logger.info(f"Saved confusion matrix: {output_path}")


def evaluate_model(model_name: str, pipeline, X_train, y_train, X_test, y_test, classes):
    """Trains pipeline, scores on test set, and returns metrics dict and predictions."""
    pipeline.fit(X_train, y_train)
    y_pred = pipeline.predict(X_test)

    acc = accuracy_score(y_test, y_pred)
    prec_macro = precision_score(y_test, y_pred, average="macro", zero_division=0)
    rec_macro = recall_score(y_test, y_pred, average="macro", zero_division=0)
    f1_macro = f1_score(y_test, y_pred, average="macro", zero_division=0)
    prec_micro = precision_score(y_test, y_pred, average="micro", zero_division=0)
    rec_micro = recall_score(y_test, y_pred, average="micro", zero_division=0)
    f1_micro = f1_score(y_test, y_pred, average="micro", zero_division=0)

    cm = confusion_matrix(y_test, y_pred, labels=classes)

    metrics = {
        "model": model_name,
        "accuracy": round(acc * 100, 2),
        "f1_micro": round(f1_micro * 100, 2),
        "f1_macro": round(f1_macro * 100, 2),
        "precision_micro": round(prec_micro * 100, 2),
        "recall_micro": round(rec_micro * 100, 2),
        "precision_macro": round(prec_macro * 100, 2),
        "recall_macro": round(rec_macro * 100, 2)
    }

    return metrics, cm, y_pred


def main():
    SAVED_MODELS_DIR.mkdir(parents=True, exist_ok=True)
    RESULTS_TABLES_DIR.mkdir(parents=True, exist_ok=True)
    RESULTS_FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    if not FEATURES_CSV.exists():
        logger.error(f"Features file {FEATURES_CSV} does not exist. Run build_features.py first.")
        return

    df = pd.read_csv(FEATURES_CSV)
    logger.info(f"Loaded {len(df)} samples from {FEATURES_CSV}")

    # Validate label classes
    classes = sorted(df["label"].unique().tolist())
    logger.info(f"Target classes ({len(classes)}): {classes}")

    import argparse
    parser = argparse.ArgumentParser(description="Train text-pair matching classifiers.")
    parser.add_argument("--split", choices=["leak_free", "legacy"], default="leak_free",
                        help="Data split protocol. 'leak_free' (default) holds out 150 GOLD pairs as test-only (Hard Rule 1). 'legacy' reproduces the flawed 80/20 random split.")
    args = parser.parse_args()

    # Split dataset
    comb_labeled = Path("data/labeled/pairs_labeled_combined.csv")
    if args.split == "leak_free" and comb_labeled.exists():
        df_comb = pd.read_csv(comb_labeled)
        df["source_dataset"] = df_comb["source_dataset"]
        train_df = df[df["source_dataset"].isin(["TAT-QA (NExT Research)", "FinQA (Columbia/JPM)"])].copy().reset_index(drop=True)
        test_df = df[df["source_dataset"] == "SEC EDGAR 10-K (Primary)"].copy().reset_index(drop=True)
        logger.info(f"Leak-Free Split Enforced (Hard Rule 1): {len(train_df)} external train samples, {len(test_df)} GOLD test samples (0 overlap).")
    else:
        if args.split == "leak_free":
            logger.warning("Combined dataset not found. Falling back to random stratified split.")
        else:
            logger.warning("WARNING: Running in 'legacy' mode with known data leakage (GOLD samples present in training set).")
        train_df, test_df = train_test_split(
            df, test_size=0.20, random_state=42, stratify=df["label"]
        )

    train_df.to_csv(SPLITS_DIR / "train_split.csv", index=False)
    test_df.to_csv(SPLITS_DIR / "test_split.csv", index=False)
    logger.info(f"Split data: {len(train_df)} train, {len(test_df)} test. Saved splits to data/processed/")

    y_train = train_df["label"]
    y_test = test_df["label"]

    models_config = [
        {
            "name": "logistic_regression_full",
            "display": "Logistic Regression (Block A+B)",
            "features": ALL_FEATURES,
            "pipeline": Pipeline([
                ("scaler", StandardScaler()),
                ("clf", LogisticRegression(class_weight="balanced", max_iter=1000, random_state=42))
            ])
        },
        {
            "name": "mlp_classifier_full",
            "display": "MLP Classifier (Block A+B)",
            "features": ALL_FEATURES,
            "pipeline": Pipeline([
                ("scaler", StandardScaler()),
                ("clf", MLPClassifier(hidden_layer_sizes=(32, 16), max_iter=800, random_state=42))
            ])
        },
        {
            "name": "logistic_regression_block_a_only",
            "display": "Logistic Regression (Block A only)",
            "features": BLOCK_A_FEATURES,
            "pipeline": Pipeline([
                ("scaler", StandardScaler()),
                ("clf", LogisticRegression(class_weight="balanced", max_iter=1000, random_state=42))
            ])
        },
        {
            "name": "logistic_regression_block_b_only",
            "display": "Logistic Regression (Block B only)",
            "features": BLOCK_B_FEATURES,
            "pipeline": Pipeline([
                ("scaler", StandardScaler()),
                ("clf", LogisticRegression(class_weight="balanced", max_iter=1000, random_state=42))
            ])
        }
    ]

    all_metrics = []

    print("\n" + "=" * 80)
    print(f"{'MODEL TRAINING & EVALUATION BENCHMARK':^80}")
    print("=" * 80)

    for cfg in models_config:
        name = cfg["name"]
        display = cfg["display"]
        feats = cfg["features"]
        pipe = cfg["pipeline"]

        X_tr = train_df[feats]
        X_te = test_df[feats]

        metrics, cm, _ = evaluate_model(display, pipe, X_tr, y_train, X_te, y_test, classes)
        all_metrics.append(metrics)

        # Save model pipeline
        model_save_path = SAVED_MODELS_DIR / f"{name}.pkl"
        joblib.dump(pipe, model_save_path)
        logger.info(f"Model saved to {model_save_path}")

        # Save confusion matrix figure
        cm_fig_path = RESULTS_FIGURES_DIR / f"confusion_matrix_{name}.png"
        plot_confusion_matrix(cm, classes, f"Confusion Matrix: {display}", cm_fig_path)

        print(f"\nModel: {display}")
        print(f"Accuracy: {metrics['accuracy']}% | Micro-F1: {metrics['f1_micro']}% | Macro-F1: {metrics['f1_macro']}%")

    df_metrics = pd.DataFrame(all_metrics)
    comparison_csv_path = RESULTS_TABLES_DIR / "model_comparison.csv"
    df_metrics.to_csv(comparison_csv_path, index=False)
    logger.info(f"Model comparison table saved to: {comparison_csv_path}")

    print("\n" + "=" * 80)
    print(df_metrics.to_string(index=False))
    print("=" * 80 + "\n")


if __name__ == "__main__":
    main()
