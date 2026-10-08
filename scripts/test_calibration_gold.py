#!/usr/bin/env python3
"""
scripts/test_calibration_gold.py

Computes Brier score and Expected Calibration Error (ECE) for uncalibrated vs calibrated
Logistic Regression on the pure 150 GOLD test instances (Hard Rule 1 compliant: no tuning on GOLD).
"""

import sys
import logging
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.metrics import brier_score_loss
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.calibration import CalibratedClassifierCV
from sklearn.pipeline import Pipeline

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

ALL_FEATURES = [
    "numeric_value_match", "numeric_value_close", "keyword_overlap",
    "period_match", "string_similarity", "sentence_length",
    "line_item_name_length", "embedding_cosine_similarity"
]

def compute_ece(y_true, probs, n_bins=10):
    """Computes Expected Calibration Error (ECE)."""
    bin_boundaries = np.linspace(0, 1, n_bins + 1)
    ece = 0.0
    total_samples = len(y_true)

    for i in range(n_bins):
        bin_lower = bin_boundaries[i]
        bin_upper = bin_boundaries[i + 1]

        # Samples in bin i
        if i == n_bins - 1:
            in_bin = (probs >= bin_lower) & (probs <= bin_upper)
        else:
            in_bin = (probs >= bin_lower) & (probs < bin_upper)

        bin_size = np.sum(in_bin)
        if bin_size > 0:
            avg_acc = np.mean(y_true[in_bin])
            avg_conf = np.mean(probs[in_bin])
            ece += (bin_size / total_samples) * np.abs(avg_acc - avg_conf)

    return ece

def main():
    df_all_feats = pd.read_csv("data/processed/features.csv")
    df_comb_labeled = pd.read_csv("data/labeled/pairs_labeled_combined.csv")
    df_all_feats["source_dataset"] = df_comb_labeled["source_dataset"]

    train_df = df_all_feats[df_all_feats["source_dataset"].isin(["TAT-QA (NExT Research)", "FinQA (Columbia/JPM)"])].copy().reset_index(drop=True)
    test_df = df_all_feats[df_all_feats["source_dataset"] == "SEC EDGAR 10-K (Primary)"].copy().reset_index(drop=True)

    X_train = train_df[ALL_FEATURES]
    y_train = train_df["label"]
    X_test = test_df[ALL_FEATURES]
    y_test = test_df["label"]

    lr_pipe = Pipeline([
        ("scaler", StandardScaler()),
        ("clf", LogisticRegression(class_weight="balanced", max_iter=1000, random_state=42))
    ])
    lr_pipe.fit(X_train, y_train)

    classes = list(lr_pipe.classes_)
    target_idx = classes.index("match")
    y_test_binary = (y_test == "match").astype(int)

    # 1. Uncalibrated predictions on GOLD test set
    probs_uncal = lr_pipe.predict_proba(X_test)[:, target_idx]
    brier_uncal = brier_score_loss(y_test_binary, probs_uncal)
    ece_uncal = compute_ece(y_test_binary.values, probs_uncal)

    # 2. 5-Fold CV Isotonic Calibrated Classifier fit on training set
    cal_clf = CalibratedClassifierCV(lr_pipe, method="isotonic", cv=5)
    cal_clf.fit(X_train, y_train)

    probs_cal = cal_clf.predict_proba(X_test)[:, target_idx]
    brier_cal = brier_score_loss(y_test_binary, probs_cal)
    ece_cal = compute_ece(y_test_binary.values, probs_cal)

    res = {
        "uncalibrated": {
            "brier_score": round(brier_uncal, 6),
            "ece": round(ece_uncal, 6)
        },
        "calibrated": {
            "brier_score": round(brier_cal, 6),
            "ece": round(ece_cal, 6)
        },
        "deltas": {
            "brier_change": round(brier_cal - brier_uncal, 6),
            "ece_change": round(ece_cal - ece_uncal, 6),
            "brier_pct_change": round(((brier_cal - brier_uncal) / brier_uncal) * 100, 2),
            "ece_pct_change": round(((ece_cal - ece_uncal) / ece_uncal) * 100, 2)
        }
    }

    print("=" * 80)
    print(f"{'PROBABILITY CALIBRATION ANALYSIS ON GOLD TEST SET (N=150)':^80}")
    print("=" * 80)
    print(f"Uncalibrated Brier Score: {res['uncalibrated']['brier_score']:.6f} | ECE: {res['uncalibrated']['ece']:.6f}")
    print(f"Calibrated Brier Score:   {res['calibrated']['brier_score']:.6f} | ECE: {res['calibrated']['ece']:.6f}")
    print(f"Brier Score Delta:        {res['deltas']['brier_change']:+.6f} ({res['deltas']['brier_pct_change']}%)")
    print(f"ECE Delta:                {res['deltas']['ece_change']:+.6f} ({res['deltas']['ece_pct_change']}%)")
    print("=" * 80)

if __name__ == "__main__":
    main()
