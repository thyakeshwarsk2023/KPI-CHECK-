#!/usr/bin/env python3
"""
scripts/test_length_shortcut_ablation.py

Tests the length-shortcut hypothesis by retraining Logistic Regression and MLP
on the leak-free split with and without length features (sentence_length, line_item_name_length).
"""

import sys
import logging
from pathlib import Path
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

ALL_FEATURES = [
    "numeric_value_match", "numeric_value_close", "keyword_overlap",
    "period_match", "string_similarity", "sentence_length",
    "line_item_name_length", "embedding_cosine_similarity"
]

NO_LENGTH_FEATURES = [
    "numeric_value_match", "numeric_value_close", "keyword_overlap",
    "period_match", "string_similarity", "embedding_cosine_similarity"
]

def main():
    df_all_feats = pd.read_csv("data/processed/features.csv")
    df_comb_labeled = pd.read_csv("data/labeled/pairs_labeled_combined.csv")
    df_all_feats["source_dataset"] = df_comb_labeled["source_dataset"]

    train_df = df_all_feats[df_all_feats["source_dataset"].isin(["TAT-QA (NExT Research)", "FinQA (Columbia/JPM)"])].copy().reset_index(drop=True)
    test_df = df_all_feats[df_all_feats["source_dataset"] == "SEC EDGAR 10-K (Primary)"].copy().reset_index(drop=True)

    y_train = train_df["label"]
    y_test = test_df["label"]

    models = {
        "lr_all_features": (ALL_FEATURES, Pipeline([("scaler", StandardScaler()), ("clf", LogisticRegression(class_weight="balanced", max_iter=1000, random_state=42))])),
        "lr_no_length": (NO_LENGTH_FEATURES, Pipeline([("scaler", StandardScaler()), ("clf", LogisticRegression(class_weight="balanced", max_iter=1000, random_state=42))])),
        "mlp_all_features": (ALL_FEATURES, Pipeline([("scaler", StandardScaler()), ("clf", MLPClassifier(hidden_layer_sizes=(32, 16), max_iter=800, random_state=42))])),
        "mlp_no_length": (NO_LENGTH_FEATURES, Pipeline([("scaler", StandardScaler()), ("clf", MLPClassifier(hidden_layer_sizes=(32, 16), max_iter=800, random_state=42))]))
    }

    results = {}
    for m_name, (feats, pipe) in models.items():
        pipe.fit(train_df[feats], y_train)
        y_pred = pipe.predict(test_df[feats])

        acc = accuracy_score(y_test, y_pred) * 100
        f1_mic = f1_score(y_test, y_pred, average="micro") * 100
        f1_mac = f1_score(y_test, y_pred, average="macro") * 100

        results[m_name] = {
            "accuracy": round(acc, 2),
            "f1_micro": round(f1_mic, 2),
            "f1_macro": round(f1_mac, 2)
        }

    print("=" * 75)
    print(f"{'LENGTH-SHORTCUT ABLATION TEST RESULTS (GOLD TEST SET N=150)':^75}")
    print("=" * 75)
    print(f"{'Model Configuration':<35} | {'Micro-F1':<12} | {'Macro-F1':<12} | {'Accuracy':<12}")
    print("-" * 75)
    print(f"{'Logistic Regression (All Features)':<35} | {results['lr_all_features']['f1_micro']:<12.2f}% | {results['lr_all_features']['f1_macro']:<12.2f}% | {results['lr_all_features']['accuracy']:<12.2f}%")
    print(f"{'Logistic Regression (No Length)':<35} | {results['lr_no_length']['f1_micro']:<12.2f}% | {results['lr_no_length']['f1_macro']:<12.2f}% | {results['lr_no_length']['accuracy']:<12.2f}%")
    print(f"{'MLP Classifier (All Features)':<35} | {results['mlp_all_features']['f1_micro']:<12.2f}% | {results['mlp_all_features']['f1_macro']:<12.2f}% | {results['mlp_all_features']['accuracy']:<12.2f}%")
    print(f"{'MLP Classifier (No Length)':<35} | {results['mlp_no_length']['f1_micro']:<12.2f}% | {results['mlp_no_length']['f1_macro']:<12.2f}% | {results['mlp_no_length']['accuracy']:<12.2f}%")
    print("=" * 75)

if __name__ == "__main__":
    main()
