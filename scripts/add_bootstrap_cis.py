#!/usr/bin/env python3
"""
scripts/add_bootstrap_cis.py

Computes 1,000-iteration bootstrap 95% Confidence Intervals (percentile method, seed=42)
for all Micro-F1 and Macro-F1 scores across legacy, leak-free, and ablation models,
and injects them directly into results/metrics_master.json.
"""

import sys
import json
import logging
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.metrics import f1_score
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

ALL_FEATURES = [
    "numeric_value_match", "numeric_value_close", "keyword_overlap",
    "period_match", "string_similarity", "sentence_length",
    "line_item_name_length", "embedding_cosine_similarity"
]

BLOCK_A_FEATURES = [
    "numeric_value_match", "numeric_value_close", "keyword_overlap",
    "period_match", "string_similarity", "sentence_length",
    "line_item_name_length"
]

BLOCK_B_FEATURES = ["embedding_cosine_similarity"]

NO_LENGTH_FEATURES = [
    "numeric_value_match", "numeric_value_close", "keyword_overlap",
    "period_match", "string_similarity", "embedding_cosine_similarity"
]

def bootstrap_f1_cis(y_true, y_pred, n_bootstraps=1000, seed=42):
    rng = np.random.RandomState(seed)
    n_samples = len(y_true)
    
    micro_scores = []
    macro_scores = []

    y_true_arr = np.array(y_true)
    y_pred_arr = np.array(y_pred)

    for _ in range(n_bootstraps):
        idx = rng.randint(0, n_samples, size=n_samples)
        y_t_b = y_true_arr[idx]
        y_p_b = y_pred_arr[idx]

        mic = f1_score(y_t_b, y_p_b, average="micro") * 100
        mac = f1_score(y_t_b, y_p_b, average="macro") * 100

        micro_scores.append(mic)
        macro_scores.append(mac)

    mic_lower = float(np.percentile(micro_scores, 2.5))
    mic_upper = float(np.percentile(micro_scores, 97.5))
    mac_lower = float(np.percentile(macro_scores, 2.5))
    mac_upper = float(np.percentile(macro_scores, 97.5))

    return {
        "f1_micro_ci_95": [round(mic_lower, 2), round(mic_upper, 2)],
        "f1_macro_ci_95": [round(mac_lower, 2), round(mac_upper, 2)],
        "f1_micro_ci_str": f"[{mic_lower:.2f}%, {mic_upper:.2f}%]",
        "f1_macro_ci_str": f"[{mac_lower:.2f}%, {mac_upper:.2f}%]"
    }

def main():
    df_all_feats = pd.read_csv("data/processed/features.csv")
    df_comb_labeled = pd.read_csv("data/labeled/pairs_labeled_combined.csv")
    df_all_feats["source_dataset"] = df_comb_labeled["source_dataset"]

    # Splits
    old_train = pd.read_csv("data/processed/train_split.csv")
    old_test = pd.read_csv("data/processed/test_split.csv")

    train_ext = df_all_feats[df_all_feats["source_dataset"].isin(["TAT-QA (NExT Research)", "FinQA (Columbia/JPM)"])].copy().reset_index(drop=True)
    test_gold = df_all_feats[df_all_feats["source_dataset"] == "SEC EDGAR 10-K (Primary)"].copy().reset_index(drop=True)

    master_path = Path("results/metrics_master.json")
    with open(master_path, "r", encoding="utf-8") as f:
        master = json.load(f)

    # 1. Legacy Pipeline Models CIs
    logger.info("Computing bootstrap CIs for legacy pipeline models...")
    models_config = {
        "logistic_regression_full": ALL_FEATURES,
        "mlp_classifier_full": ALL_FEATURES,
        "logistic_regression_block_a_only": BLOCK_A_FEATURES,
        "logistic_regression_block_b_only": BLOCK_B_FEATURES
    }

    for m_name, feats in models_config.items():
        if "mlp" in m_name:
            clf = MLPClassifier(hidden_layer_sizes=(32, 16), max_iter=800, random_state=42)
        else:
            clf = LogisticRegression(class_weight="balanced", max_iter=1000, random_state=42)
        pipe = Pipeline([("scaler", StandardScaler()), ("clf", clf)])
        pipe.fit(old_train[feats], old_train["label"])
        y_pred = pipe.predict(old_test[feats])

        cis = bootstrap_f1_cis(old_test["label"], y_pred)
        master["legacy_pipeline_metrics_with_leakage"][m_name].update(cis)

    # 2. Leak-Free Pipeline Models CIs
    logger.info("Computing bootstrap CIs for leak-free pipeline models...")
    for m_name, feats in models_config.items():
        if "mlp" in m_name:
            clf = MLPClassifier(hidden_layer_sizes=(32, 16), max_iter=800, random_state=42)
        else:
            clf = LogisticRegression(class_weight="balanced", max_iter=1000, random_state=42)
        pipe = Pipeline([("scaler", StandardScaler()), ("clf", clf)])
        pipe.fit(train_ext[feats], train_ext["label"])
        y_pred = pipe.predict(test_gold[feats])

        cis = bootstrap_f1_cis(test_gold["label"], y_pred)
        master["leak_free_pipeline_metrics"][m_name].update(cis)

    # 3. Length-Ablated Models CIs
    logger.info("Computing bootstrap CIs for length-ablated models...")
    length_ablation_models = {
        "logistic_regression_no_length": (NO_LENGTH_FEATURES, LogisticRegression(class_weight="balanced", max_iter=1000, random_state=42)),
        "mlp_classifier_no_length": (NO_LENGTH_FEATURES, MLPClassifier(hidden_layer_sizes=(32, 16), max_iter=800, random_state=42))
    }

    master["length_ablation_metrics"] = {}
    for m_name, (feats, clf) in length_ablation_models.items():
        pipe = Pipeline([("scaler", StandardScaler()), ("clf", clf)])
        pipe.fit(train_ext[feats], train_ext["label"])
        y_pred = pipe.predict(test_gold[feats])

        f1_mic = f1_score(test_gold["label"], y_pred, average="micro") * 100
        f1_mac = f1_score(test_gold["label"], y_pred, average="macro") * 100
        cis = bootstrap_f1_cis(test_gold["label"], y_pred)

        master["length_ablation_metrics"][m_name] = {
            "f1_micro": round(f1_mic, 2),
            "f1_macro": round(f1_mac, 2),
            "f1_micro_ci_95": cis["f1_micro_ci_95"],
            "f1_macro_ci_95": cis["f1_macro_ci_95"],
            "f1_micro_ci_str": cis["f1_micro_ci_str"],
            "f1_macro_ci_str": cis["f1_macro_ci_str"]
        }

    with open(master_path, "w", encoding="utf-8") as f:
        json.dump(master, f, indent=2)

    logger.info(f"Successfully injected 95% CIs into {master_path}")
    print("BOOTSTRAP 95% CIs COMPUTATION COMPLETE!")

if __name__ == "__main__":
    main()
