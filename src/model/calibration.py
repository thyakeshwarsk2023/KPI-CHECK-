#!/usr/bin/env python3
"""
src/model/calibration.py

Evaluates and refines probability calibration of the primary text-pair classifier:
1. Computes Brier score loss and expected calibration error.
2. Plots a reliability diagram (predicted probability vs. empirical frequency).
3. Applies CalibratedClassifierCV (isotonic calibration) to produce before/after comparisons.
4. Saves figure to results/figures/calibration.png and summary to results/tables/calibration_metrics.csv.
"""

import os
import joblib
import logging
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.calibration import calibration_curve, CalibratedClassifierCV
from sklearn.metrics import brier_score_loss
from sklearn.preprocessing import label_binarize

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

SAVED_MODELS_DIR = Path("src/model/saved")
SPLITS_DIR = Path("data/processed")
RESULTS_FIGURES_DIR = Path("results/figures")
RESULTS_TABLES_DIR = Path("results/tables")

BLOCK_A_FEATURES = [
    "numeric_value_match", "numeric_value_close", "keyword_overlap",
    "period_match", "string_similarity", "sentence_length", "line_item_name_length"
]
BLOCK_B_FEATURES = ["embedding_cosine_similarity"]
ALL_FEATURES = BLOCK_A_FEATURES + BLOCK_B_FEATURES


def main():
    RESULTS_FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    RESULTS_TABLES_DIR.mkdir(parents=True, exist_ok=True)

    model_path = SAVED_MODELS_DIR / "logistic_regression_full.pkl"
    train_path = SPLITS_DIR / "train_split.csv"
    test_path = SPLITS_DIR / "test_split.csv"

    if not model_path.exists() or not test_path.exists():
        logger.error("Required model or test split not found. Run train.py first.")
        return

    pipeline = joblib.load(model_path)
    train_df = pd.read_csv(train_path)
    test_df = pd.read_csv(test_path)

    X_train = train_df[ALL_FEATURES]
    y_train = train_df["label"]
    X_test = test_df[ALL_FEATURES]
    y_test = test_df["label"]

    classes = pipeline.classes_
    target_class = "match" if "match" in classes else classes[0]
    target_idx = list(classes).index(target_class)

    # 1. Uncalibrated predictions
    probs_uncal = pipeline.predict_proba(X_test)[:, target_idx]
    y_test_binary = (y_test == target_class).astype(int)
    brier_uncal = brier_score_loss(y_test_binary, probs_uncal)

    # 2. Fit Isotonic Calibrated Classifier (5-Fold CV on training set to prevent training leakage)
    calibrated_clf = CalibratedClassifierCV(pipeline, method="isotonic", cv=5)
    calibrated_clf.fit(X_train, y_train)
    probs_cal = calibrated_clf.predict_proba(X_test)[:, target_idx]
    brier_cal = brier_score_loss(y_test_binary, probs_cal)

    # 3. Calibration Curves
    prob_true_uncal, prob_pred_uncal = calibration_curve(y_test_binary, probs_uncal, n_bins=5, strategy="uniform")
    prob_true_cal, prob_pred_cal = calibration_curve(y_test_binary, probs_cal, n_bins=5, strategy="uniform")

    # 4. Plotting Side-by-Side Reliability Diagram
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))

    # Reliability diagram
    ax1.plot([0, 1], [0, 1], "k--", label="Perfect Calibration (Ideal)")
    ax1.plot(prob_pred_uncal, prob_true_uncal, "s-", color="#d9534f", label=f"Original LR (Brier: {brier_uncal:.4f})")
    ax1.plot(prob_pred_cal, prob_true_cal, "o-", color="#2e6da4", label=f"Isotonic Calibrated (Brier: {brier_cal:.4f})")
    ax1.set_xlabel(f"Mean Predicted Probability ('{target_class}')", fontsize=11, weight="bold")
    ax1.set_ylabel(f"Empirical Fraction of Positives", fontsize=11, weight="bold")
    ax1.set_title("Reliability Diagram (Calibration Curve)", fontsize=12, weight="bold")
    ax1.legend(loc="best")
    ax1.grid(True, linestyle="--", alpha=0.6)

    # Predicted probability distributions
    ax2.hist(probs_uncal, range=(0, 1), bins=10, label="Original Probabilities", histtype="step", lw=2, color="#d9534f")
    ax2.hist(probs_cal, range=(0, 1), bins=10, label="Calibrated Probabilities", histtype="step", lw=2, color="#2e6da4")
    ax2.set_xlabel(f"Predicted Probability ('{target_class}')", fontsize=11, weight="bold")
    ax2.set_ylabel("Count of Test Instances", fontsize=11, weight="bold")
    ax2.set_title("Probability Distribution Histogram", fontsize=12, weight="bold")
    ax2.legend(loc="best")
    ax2.grid(True, linestyle="--", alpha=0.6)

    plt.tight_layout()
    fig_out = RESULTS_FIGURES_DIR / "calibration.png"
    plt.savefig(fig_out, dpi=300)
    plt.close()
    logger.info(f"Calibration plot saved to: {fig_out}")

    # Save summary table
    df_metrics = pd.DataFrame([
        {"stage": "Original Logistic Regression", "brier_score": round(brier_uncal, 4), "improvement": "Baseline"},
        {"stage": "Isotonic Calibrated", "brier_score": round(brier_cal, 4), "improvement": f"{((brier_uncal - brier_cal)/max(brier_uncal,1e-6))*100:.2f}% reduction"}
    ])
    table_out = RESULTS_TABLES_DIR / "calibration_metrics.csv"
    df_metrics.to_csv(table_out, index=False)
    logger.info(f"Calibration metrics saved to: {table_out}")

    print("\n" + "=" * 60)
    print(f"{'PROBABILITY CALIBRATION RESULTS':^60}")
    print("=" * 60)
    print(f"Target Class: {target_class}")
    print(f"Original Brier Score:   {brier_uncal:.4f}")
    print(f"Calibrated Brier Score: {brier_cal:.4f}")
    print(f"Brier Improvement:      {((brier_uncal - brier_cal)/max(brier_uncal,1e-6))*100:.2f}%")
    print("=" * 60 + "\n")


if __name__ == "__main__":
    main()
