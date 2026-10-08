#!/usr/bin/env python3
"""
scripts/recompute_xai_gold.py

Recomputes LinearSHAP latency, KernelSHAP latency, and SHAP-LIME concordance
for the leak-free models on ALL 150 GOLD test instances.
"""

import sys
import time
import logging
from pathlib import Path
import numpy as np
import pandas as pd
import shap
from lime.lime_tabular import LimeTabularExplainer
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

    # Fit scaler and models
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    lr_clf = LogisticRegression(class_weight="balanced", max_iter=1000, random_state=42)
    lr_clf.fit(X_train_scaled, y_train)

    mlp_clf = MLPClassifier(hidden_layer_sizes=(32, 16), max_iter=800, random_state=42)
    mlp_clf.fit(X_train_scaled, y_train)

    # 1. LinearSHAP Latency on 150 GOLD samples
    logger.info("Computing LinearSHAP attributions across all 150 GOLD test samples...")
    t0 = time.perf_counter()
    linear_explainer = shap.LinearExplainer(lr_clf, X_train_scaled)
    shap_values_lr = linear_explainer.shap_values(X_test_scaled)
    t1 = time.perf_counter()
    linear_shap_total_ms = (t1 - t0) * 1000
    linear_shap_per_sample_ms = linear_shap_total_ms / len(X_test)

    # 2. KernelSHAP Latency on 150 GOLD samples (using a representative background sample of 50 for KernelExplainer)
    logger.info("Computing KernelSHAP attributions for MLP across all 150 GOLD test samples...")
    bg_data = shap.sample(X_train_scaled, 50, random_state=42)
    kernel_explainer = shap.KernelExplainer(mlp_clf.predict_proba, bg_data)
    
    t0 = time.perf_counter()
    # To compute KernelSHAP across all 150 GOLD test samples safely:
    shap_values_mlp = kernel_explainer.shap_values(X_test_scaled, nsamples=100)
    t1 = time.perf_counter()
    kernel_shap_total_ms = (t1 - t0) * 1000
    kernel_shap_per_sample_ms = kernel_shap_total_ms / len(X_test)

    # 3. LIME Attributions on 150 GOLD samples
    logger.info("Computing LIME attributions across all 150 GOLD test samples...")
    lime_explainer = LimeTabularExplainer(
        training_data=X_train_scaled,
        feature_names=ALL_FEATURES,
        class_names=lr_clf.classes_,
        mode="classification",
        random_state=42
    )

    # Calculate SHAP vs LIME concordance across ALL 150 GOLD test samples
    shared_slots = 0
    total_slots = len(X_test) * 3
    exact_matches = 0

    classes = list(lr_clf.classes_)

    for i in range(len(X_test)):
        # Model predicted class index for instance i
        pred_label = lr_clf.predict(X_test_scaled[i:i+1])[0]
        c_idx = classes.index(pred_label)

        # Get SHAP top-3 feature indices for predicted class
        if isinstance(shap_values_lr, list):
            sv = shap_values_lr[c_idx][i]
        elif len(shap_values_lr.shape) == 3:
            sv = shap_values_lr[i, :, c_idx]
        else:
            sv = shap_values_lr[i]

        sv_1d = np.array(sv).flatten()
        shap_top3 = set(np.argsort(np.abs(sv_1d))[-3:])

        # Get LIME top-3 feature indices for predicted class
        exp = lime_explainer.explain_instance(
            X_test_scaled[i],
            lr_clf.predict_proba,
            num_features=3,
            labels=(c_idx,)
        )
        lime_feat_names = [pair[0] for pair in exp.as_list(label=c_idx)]
        # Map feature string (which might contain conditions or name) to ALL_FEATURES index
        lime_top3 = set()
        for fname in lime_feat_names:
            for feat_idx, real_name in enumerate(ALL_FEATURES):
                if real_name in fname:
                    lime_top3.add(feat_idx)
                    break
        
        overlap = len(shap_top3.intersection(lime_top3))
        shared_slots += overlap
        if overlap == 3:
            exact_matches += 1

    mean_feature_agreement_pct = (shared_slots / total_slots) * 100
    exact_instance_concordance_pct = (exact_matches / len(X_test)) * 100

    print("=" * 85)
    print(f"{'EMPIRICAL XAI LATENCY & CONCORDANCE RESULTS (GOLD TEST SET N=150)':^85}")
    print("=" * 85)
    print(f"LinearSHAP Total Time (150 samples): {linear_shap_total_ms:.2f} ms")
    print(f"LinearSHAP Per-Sample Latency:     {linear_shap_per_sample_ms:.2f} ms / sample")
    print(f"KernelSHAP Total Time (150 samples): {kernel_shap_total_ms:.2f} ms")
    print(f"KernelSHAP Per-Sample Latency:     {kernel_shap_per_sample_ms:.2f} ms / sample")
    print(f"Latency Speedup Factor (Kernel/Linear): {kernel_shap_per_sample_ms / max(linear_shap_per_sample_ms, 1e-5):.1f}x")
    print("-" * 85)
    print(f"SHAP-LIME Mean Top-3 Feature Overlap Rate (150 samples): {mean_feature_agreement_pct:.2f}% ({shared_slots}/{total_slots} slots)")
    print(f"SHAP-LIME Exact Top-3 Instance Concordance (150 samples): {exact_instance_concordance_pct:.2f}% ({exact_matches}/{len(X_test)} instances)")
    print("=" * 85)

if __name__ == "__main__":
    main()
