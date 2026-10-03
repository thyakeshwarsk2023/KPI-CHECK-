#!/usr/bin/env python3
"""
src/xai/shap_explain.py

SHAP interpretability analysis for the primary Logistic Regression text-pair classifier:
1. Loads trained model (StandardScaler + LogisticRegression).
2. Computes exact Shapley values using shap.LinearExplainer (with fallback to shap.Explainer).
3. Generates:
   - Global SHAP summary plot (results/figures/shap_summary.png)
   - Dependence plots for top 3 most influential features
   - Local explanation waterfall/bar plots for 5 representative cases:
     (2 correct matches, 2 correct no-matches, 1 misclassification/borderline)
   - Features ranked by mean absolute SHAP value (results/tables/shap_feature_importance.csv)
4. Exports chosen example indices to data/processed/selected_5_examples.json for LIME alignment.
"""

import os
import json
import joblib
import logging
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

SAVED_MODELS_DIR = Path("src/model/saved")
SPLITS_DIR = Path("data/processed")
RESULTS_FIGURES_DIR = Path("results/figures")
RESULTS_TABLES_DIR = Path("results/tables")
EXAMPLES_JSON = Path("data/processed/selected_5_examples.json")

BLOCK_A_FEATURES = [
    "numeric_value_match", "numeric_value_close", "keyword_overlap",
    "period_match", "string_similarity", "sentence_length", "line_item_name_length"
]
BLOCK_B_FEATURES = ["embedding_cosine_similarity"]
ALL_FEATURES = BLOCK_A_FEATURES + BLOCK_B_FEATURES


def select_5_evaluation_examples(y_true, y_pred):
    """
    Selects 5 representative test set instances:
    - 2 Correct Matches
    - 2 Correct No-Matches
    - 1 Misclassification or ambiguous edge case
    """
    correct_matches = []
    correct_nomatches = []
    misclassified = []

    for idx, (yt, yp) in enumerate(zip(y_true, y_pred)):
        if yt == "match" and yp == "match":
            correct_matches.append(idx)
        elif yt == "no_match" and yp == "no_match":
            correct_nomatches.append(idx)
        elif yt != yp:
            misclassified.append(idx)

    # Fallback selections if categories are sparse
    if len(correct_matches) < 2:
        correct_matches = [i for i, y in enumerate(y_true) if y == "match"][:2]
    if len(correct_nomatches) < 2:
        correct_nomatches = [i for i, y in enumerate(y_true) if y == "no_match"][:2]
    if not misclassified:
        # Pick the lowest prediction confidence or ambiguous case
        misclassified = [i for i, y in enumerate(y_true) if y == "ambiguous"] or [0]

    selected = [
        {"index": int(correct_matches[0]), "category": "Correct Match (1)"},
        {"index": int(correct_matches[1] if len(correct_matches) > 1 else correct_matches[0]), "category": "Correct Match (2)"},
        {"index": int(correct_nomatches[0]), "category": "Correct No-Match (1)"},
        {"index": int(correct_nomatches[1] if len(correct_nomatches) > 1 else correct_nomatches[0]), "category": "Correct No-Match (2)"},
        {"index": int(misclassified[0]), "category": "Misclassification / Edge Case"}
    ]
    return selected


def main():
    import shap

    RESULTS_FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    RESULTS_TABLES_DIR.mkdir(parents=True, exist_ok=True)

    model_path = SAVED_MODELS_DIR / "logistic_regression_full.pkl"
    train_path = SPLITS_DIR / "train_split.csv"
    test_path = SPLITS_DIR / "test_split.csv"

    if not model_path.exists() or not test_path.exists():
        logger.error("Model or test split not found. Run train.py first.")
        return

    pipeline = joblib.load(model_path)
    scaler = pipeline.named_steps["scaler"]
    clf = pipeline.named_steps["clf"]

    train_df = pd.read_csv(train_path)
    test_df = pd.read_csv(test_path)

    X_train_raw = train_df[ALL_FEATURES]
    X_test_raw = test_df[ALL_FEATURES]
    y_test = test_df["label"].tolist()

    X_train_scaled = scaler.transform(X_train_raw)
    X_test_scaled = scaler.transform(X_test_raw)

    y_pred = pipeline.predict(X_test_raw).tolist()
    classes = list(clf.classes_)
    target_class = "match" if "match" in classes else classes[0]
    target_class_idx = classes.index(target_class)

    logger.info(f"Computing exact Shapley values via LinearExplainer for target class '{target_class}'...")
    try:
        explainer = shap.LinearExplainer(clf, X_train_scaled, feature_names=ALL_FEATURES)
        shap_values_raw = explainer.shap_values(X_test_scaled)
    except Exception as e:
        logger.warning(f"LinearExplainer direct call failed ({e}), using shap.Explainer...")
        explainer = shap.Explainer(clf, X_train_scaled, feature_names=ALL_FEATURES)
        shap_values_raw = explainer(X_test_scaled).values

    # Handle shape of shap_values
    if isinstance(shap_values_raw, list):
        # Multiclass list of arrays: select class index
        shap_vals_target = shap_values_raw[target_class_idx]
    elif len(shap_values_raw.shape) == 3:
        # Shape: (samples, features, classes)
        shap_vals_target = shap_values_raw[:, :, target_class_idx]
    else:
        shap_vals_target = shap_values_raw

    # 1. Global Feature Importance Table
    mean_abs_shap = np.mean(np.abs(shap_vals_target), axis=0)
    df_importance = pd.DataFrame({
        "feature": ALL_FEATURES,
        "mean_abs_shap": np.round(mean_abs_shap, 4)
    }).sort_values(by="mean_abs_shap", ascending=False)

    importance_csv = RESULTS_TABLES_DIR / "shap_feature_importance.csv"
    df_importance.to_csv(importance_csv, index=False)
    logger.info(f"Saved SHAP feature importance: {importance_csv}")

    # 2. Global SHAP Summary Plot
    plt.figure(figsize=(10, 6))
    shap.summary_plot(
        shap_vals_target,
        X_test_raw,
        feature_names=ALL_FEATURES,
        show=False,
        plot_type="dot"
    )
    plt.title(f"Global SHAP Summary Plot (Impact on '{target_class}' Classification)", fontsize=13, weight="bold", pad=15)
    plt.tight_layout()
    summary_plot_path = RESULTS_FIGURES_DIR / "shap_summary.png"
    plt.savefig(summary_plot_path, dpi=300, bbox_inches="tight")
    plt.close()
    logger.info(f"Saved global SHAP summary plot: {summary_plot_path}")

    # 3. Dependence Plots for Top 3 Features
    top_3_features = df_importance["feature"].iloc[:3].tolist()
    for rank, feat in enumerate(top_3_features, start=1):
        feat_idx = ALL_FEATURES.index(feat)
        plt.figure(figsize=(7, 5))
        plt.scatter(
            X_test_raw[feat],
            shap_vals_target[:, feat_idx],
            c=X_test_raw["embedding_cosine_similarity"] if feat != "embedding_cosine_similarity" else X_test_raw["keyword_overlap"],
            cmap="viridis", alpha=0.8, edgecolors="none"
        )
        cbar = plt.colorbar()
        cbar.set_label("Interaction Context", fontsize=10)
        plt.axhline(0, color="gray", linestyle="--", alpha=0.7)
        plt.xlabel(feat, fontsize=11, weight="bold")
        plt.ylabel(f"SHAP value for '{target_class}'", fontsize=11, weight="bold")
        plt.title(f"SHAP Dependence Plot: {feat} (Rank {rank})", fontsize=12, weight="bold")
        plt.tight_layout()
        dep_path = RESULTS_FIGURES_DIR / f"shap_dependence_top{rank}_{feat}.png"
        plt.savefig(dep_path, dpi=300)
        plt.close()
        logger.info(f"Saved dependence plot: {dep_path}")

    # 4. Select 5 Representative Instances for Local Explanations
    selected_examples = select_5_evaluation_examples(y_test, y_pred)
    with open(EXAMPLES_JSON, "w", encoding="utf-8") as f:
        json.dump(selected_examples, f, indent=2)
    logger.info(f"Exported selected 5 examples to: {EXAMPLES_JSON}")

    # Generate individual waterfall/bar plots for each example
    for i, ex in enumerate(selected_examples, start=1):
        idx = ex["index"]
        cat = ex["category"]
        inst_row = test_df.iloc[idx]
        inst_shap = shap_vals_target[idx]
        
        sent_txt = str(inst_row["sentence"])
        if len(sent_txt) > 85:
            sent_txt = sent_txt[:82] + "..."
        line_txt = str(inst_row["candidate_line_item"])
        true_lbl = inst_row["label"]
        pred_lbl = y_pred[idx]

        # Horizontal bar attribution plot
        plt.figure(figsize=(9, 4.5))
        sorted_indices = np.argsort(inst_shap)
        y_pos = np.arange(len(ALL_FEATURES))
        colors = ["#2ca02c" if val >= 0 else "#d62728" for val in inst_shap[sorted_indices]]
        
        plt.barh(y_pos, inst_shap[sorted_indices], color=colors, align="center")
        plt.yticks(y_pos, [ALL_FEATURES[j] for j in sorted_indices], fontsize=10)
        plt.axvline(0, color="black", linestyle="-", linewidth=0.8)
        plt.xlabel(f"SHAP Attribution (Towards '{target_class}')", fontsize=10, weight="bold")
        
        title_str = (
            f"SHAP Local Explanation #{i} [{cat}]\n"
            f"Sentence: \"{sent_txt}\"\n"
            f"Line Item: \"{line_txt}\" | True: '{true_lbl}' | Pred: '{pred_lbl}'"
        )
        plt.title(title_str, fontsize=10, weight="bold", pad=10)
        plt.tight_layout()
        
        local_plot_path = RESULTS_FIGURES_DIR / f"shap_example_{i}.png"
        plt.savefig(local_plot_path, dpi=300)
        plt.close()
        logger.info(f"Saved local SHAP explanation #{i}: {local_plot_path}")

    print("\n" + "=" * 60)
    print(f"{'SHAP EXPLANATION PIPELINE COMPLETE':^60}")
    print("=" * 60)
    print("Top Features by Global Importance:")
    print(df_importance.to_string(index=False))
    print("=" * 60 + "\n")


if __name__ == "__main__":
    main()
