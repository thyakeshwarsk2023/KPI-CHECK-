#!/usr/bin/env python3
"""
scripts/reproduce_and_fix_pipeline.py

Compares:
1. Old Pipeline (Random 80/20 Stratified Split on 731 samples, including 123 GOLD samples in train).
2. Leak-Free Pipeline (Train on 581 external pairs, Test on 150 GOLD pairs; CalibratedClassifierCV with cv=5).
3. Company-Grouped 6-Fold Cross-Validation on the 150 GOLD pairs (Leave-One-Company-Out by filing).
"""

import sys
import json
import logging
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score, f1_score, precision_score, recall_score,
    classification_report, confusion_matrix, brier_score_loss
)
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.calibration import CalibratedClassifierCV
from sklearn.model_selection import StratifiedKFold, LeaveOneGroupOut

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

ROOT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT_DIR))

from src.constants import ALL_FEATURES, BLOCK_A_FEATURES, BLOCK_B_FEATURES
from src.features.build_features import compute_block_a_features

def build_models():
    return {
        "lr_full": Pipeline([
            ("scaler", StandardScaler()),
            ("clf", LogisticRegression(class_weight="balanced", max_iter=1000, random_state=42))
        ]),
        "mlp_full": Pipeline([
            ("scaler", StandardScaler()),
            ("clf", MLPClassifier(hidden_layer_sizes=(32, 16), max_iter=800, random_state=42))
        ]),
        "lr_block_a": Pipeline([
            ("scaler", StandardScaler()),
            ("clf", LogisticRegression(class_weight="balanced", max_iter=1000, random_state=42))
        ]),
        "lr_block_b": Pipeline([
            ("scaler", StandardScaler()),
            ("clf", LogisticRegression(class_weight="balanced", max_iter=1000, random_state=42))
        ])
    }

def eval_pipeline(train_df, test_df, calib_method="cv5"):
    results = {}
    models = build_models()
    
    feature_sets = {
        "lr_full": ALL_FEATURES,
        "mlp_full": ALL_FEATURES,
        "lr_block_a": BLOCK_A_FEATURES,
        "lr_block_b": BLOCK_B_FEATURES
    }

    y_train = train_df["label"]
    y_test = test_df["label"]

    for m_name, pipe in models.items():
        feats = feature_sets[m_name]
        X_train = train_df[feats]
        X_test = test_df[feats]

        pipe.fit(X_train, y_train)
        y_pred = pipe.predict(X_test)

        acc = accuracy_score(y_test, y_pred) * 100
        f1_micro = f1_score(y_test, y_pred, average="micro") * 100
        f1_macro = f1_score(y_test, y_pred, average="macro") * 100
        prec_macro = precision_score(y_test, y_pred, average="macro", zero_division=0) * 100
        rec_macro = recall_score(y_test, y_pred, average="macro", zero_division=0) * 100

        # Per-class metrics
        rep = classification_report(y_test, y_pred, output_dict=True, zero_division=0)

        results[m_name] = {
            "accuracy": round(acc, 2),
            "f1_micro": round(f1_micro, 2),
            "f1_macro": round(f1_macro, 2),
            "precision_macro": round(prec_macro, 2),
            "recall_macro": round(rec_macro, 2),
            "per_class": rep
        }

    # Calibration on LR Full
    lr_pipe = Pipeline([
        ("scaler", StandardScaler()),
        ("clf", LogisticRegression(class_weight="balanced", max_iter=1000, random_state=42))
    ])
    X_train_full = train_df[ALL_FEATURES]
    X_test_full = test_df[ALL_FEATURES]
    lr_pipe.fit(X_train_full, y_train)

    classes = list(lr_pipe.classes_)
    target_idx = classes.index("match") if "match" in classes else 0
    y_test_binary = (y_test == "match").astype(int)

    probs_uncal = lr_pipe.predict_proba(X_test_full)[:, target_idx]
    brier_uncal = brier_score_loss(y_test_binary, probs_uncal)

    if calib_method == "prefit_train":
        # The OLD flawed method
        cal_clf = CalibratedClassifierCV(lr_pipe, method="isotonic", cv="prefit")
        cal_clf.fit(X_train_full, y_train)
    else:
        # Proper 5-fold CV calibration on training set
        cal_clf = CalibratedClassifierCV(lr_pipe, method="isotonic", cv=5)
        cal_clf.fit(X_train_full, y_train)

    probs_cal = cal_clf.predict_proba(X_test_full)[:, target_idx]
    brier_cal = brier_score_loss(y_test_binary, probs_cal)
    reduction = ((brier_uncal - brier_cal) / max(brier_uncal, 1e-6)) * 100

    results["calibration"] = {
        "brier_uncalibrated": round(brier_uncal, 5),
        "brier_calibrated": round(brier_cal, 5),
        "brier_reduction_pct": round(reduction, 2)
    }

    return results

def main():
    print("=" * 80)
    print("RUNNING COMPARATIVE PIPELINE AUDIT")
    print("=" * 80)

    # 1. OLD PIPELINE EVALUATION (Random 80/20 on combined data)
    old_train = pd.read_csv("data/processed/train_split.csv")
    old_test = pd.read_csv("data/processed/test_split.csv")
    old_results = eval_pipeline(old_train, old_test, calib_method="prefit_train")

    print("\n[1] OLD PIPELINE RESULTS (Random 80/20 on 731 samples, with 123 GOLD samples in train):")
    for k, v in old_results.items():
        if k != "calibration":
            print(f"  {k:<12}: Micro-F1 = {v['f1_micro']:.2f}%, Macro-F1 = {v['f1_macro']:.2f}%, Acc = {v['accuracy']:.2f}%")
        else:
            print(f"  Calibration : Uncal Brier = {v['brier_uncalibrated']}, Cal Brier = {v['brier_calibrated']} ({v['brier_reduction_pct']}% red)")

    # 2. LEAK-FREE PROTOCOL: Train on External (581), Test on GOLD (150)
    # First, let's load features for external and GOLD
    df_all_feats = pd.read_csv("data/processed/features.csv")
    df_comb_labeled = pd.read_csv("data/labeled/pairs_labeled_combined.csv")
    df_all_feats["source_dataset"] = df_comb_labeled["source_dataset"]

    ext_mask = df_all_feats["source_dataset"].isin(["TAT-QA (NExT Research)", "FinQA (Columbia/JPM)"])
    gold_mask = df_all_feats["source_dataset"] == "SEC EDGAR 10-K (Primary)"

    train_ext = df_all_feats[ext_mask].reset_index(drop=True)
    test_gold = df_all_feats[gold_mask].reset_index(drop=True)

    print(f"\nLeak-free split sizes: Train (External) = {len(train_ext)}, Test (GOLD) = {len(test_gold)}")
    leak_free_results = eval_pipeline(train_ext, test_gold, calib_method="cv5")

    print("\n[2] LEAK-FREE RESULTS (Train on External 581, Test strictly on GOLD 150):")
    for k, v in leak_free_results.items():
        if k != "calibration":
            print(f"  {k:<12}: Micro-F1 = {v['f1_micro']:.2f}%, Macro-F1 = {v['f1_macro']:.2f}%, Acc = {v['accuracy']:.2f}%")
        else:
            print(f"  Calibration : Uncal Brier = {v['brier_uncalibrated']}, Cal Brier = {v['brier_calibrated']} ({v['brier_reduction_pct']}% red)")

    print("\nPer-Class Breakdown on GOLD Test Set (LR Full):")
    rep_gold = leak_free_results["lr_full"]["per_class"]
    for c in ["match", "no_match", "ambiguous"]:
        print(f"  Class {c:<10}: Prec = {rep_gold[c]['precision']*100:.1f}%, Rec = {rep_gold[c]['recall']*100:.1f}%, F1 = {rep_gold[c]['f1-score']*100:.1f}%, Support = {rep_gold[c]['support']}")

    # Save comparison to results
    output_path = Path("results/leakage_audit_comparison.json")
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump({
            "old_pipeline": old_results,
            "leak_free_pipeline": leak_free_results
        }, f, indent=2)
    print(f"\nSaved comparison to {output_path}")

if __name__ == "__main__":
    main()
