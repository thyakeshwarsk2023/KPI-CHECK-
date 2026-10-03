#!/usr/bin/env python3
"""
src/xai/lime_explain.py

LIME interpretability analysis for the primary Logistic Regression text-pair classifier:
1. Loads trained model pipeline (scaler + LogisticRegression).
2. Uses lime.lime_tabular.LimeTabularExplainer on the feature space.
3. Explains the exact SAME 5 test instances selected in shap_explain.py.
4. Generates explanation figures: results/figures/lime_example_{i}.png.
5. Computes top-3 feature attribution agreement between SHAP and LIME:
   results/tables/shap_vs_lime_agreement.csv.
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


def clean_lime_feature_name(raw_name: str):
    """Maps LIME rule strings (e.g. 'keyword_overlap > 0.35') back to canonical feature names."""
    for feat in ALL_FEATURES:
        if feat in raw_name:
            return feat
    return raw_name


def main():
    import lime
    import lime.lime_tabular
    import shap

    RESULTS_FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    RESULTS_TABLES_DIR.mkdir(parents=True, exist_ok=True)

    model_path = SAVED_MODELS_DIR / "logistic_regression_full.pkl"
    train_path = SPLITS_DIR / "train_split.csv"
    test_path = SPLITS_DIR / "test_split.csv"

    if not model_path.exists() or not test_path.exists() or not EXAMPLES_JSON.exists():
        logger.error("Required model, test split, or examples JSON not found. Run train.py and shap_explain.py first.")
        return

    pipeline = joblib.load(model_path)
    train_df = pd.read_csv(train_path)
    test_df = pd.read_csv(test_path)

    with open(EXAMPLES_JSON, "r", encoding="utf-8") as f:
        selected_examples = json.load(f)

    X_train_raw = train_df[ALL_FEATURES]
    X_test_raw = test_df[ALL_FEATURES]

    classes = list(pipeline.classes_)
    target_class = "match" if "match" in classes else classes[0]
    target_idx = classes.index(target_class)

    # Recompute SHAP values on test set to compare per-instance attributions directly
    scaler = pipeline.named_steps["scaler"]
    clf = pipeline.named_steps["clf"]
    X_train_scaled = scaler.transform(X_train_raw)
    X_test_scaled = scaler.transform(X_test_raw)

    try:
        shap_explainer = shap.LinearExplainer(clf, X_train_scaled, feature_names=ALL_FEATURES)
        shap_vals_raw = shap_explainer.shap_values(X_test_scaled)
    except Exception:
        shap_explainer = shap.Explainer(clf, X_train_scaled, feature_names=ALL_FEATURES)
        shap_vals_raw = shap_explainer(X_test_scaled).values

    if isinstance(shap_vals_raw, list):
        shap_vals_target = shap_vals_raw[target_idx]
    elif len(shap_vals_raw.shape) == 3:
        shap_vals_target = shap_vals_raw[:, :, target_idx]
    else:
        shap_vals_target = shap_vals_raw

    # Initialize LIME Tabular Explainer
    lime_explainer = lime.lime_tabular.LimeTabularExplainer(
        training_data=X_train_raw.values,
        feature_names=ALL_FEATURES,
        class_names=classes,
        mode="classification",
        random_state=42
    )

    agreement_records = []

    print("\n" + "=" * 70)
    print(f"{'LIME LOCAL EXPLANATIONS & SHAP CONCORDANCE':^70}")
    print("=" * 70)

    for i, ex in enumerate(selected_examples, start=1):
        idx = ex["index"]
        cat = ex["category"]
        inst_row = test_df.iloc[idx]
        inst_features = X_test_raw.iloc[idx].values
        
        # Predict with pipeline
        pred_probs = pipeline.predict_proba([inst_features])[0]
        pred_label = classes[np.argmax(pred_probs)]
        true_label = inst_row["label"]

        # Run LIME explanation
        exp = lime_explainer.explain_instance(
            data_row=inst_features,
            predict_fn=pipeline.predict_proba,
            num_features=len(ALL_FEATURES),
            labels=[target_idx]
        )

        lime_list = exp.as_list(label=target_idx)
        # Extract features and weights
        lime_features = [clean_lime_feature_name(item[0]) for item in lime_list]
        lime_weights = [item[1] for item in lime_list]

        # Plot LIME local explanation
        plt.figure(figsize=(9, 4.5))
        y_pos = np.arange(len(lime_features))
        colors = ["#2ca02c" if w >= 0 else "#d62728" for w in lime_weights]
        
        plt.barh(y_pos, lime_weights, color=colors, align="center")
        plt.yticks(y_pos, lime_features, fontsize=10)
        plt.axvline(0, color="black", linestyle="-", linewidth=0.8)
        plt.xlabel(f"LIME Weight (Towards '{target_class}')", fontsize=10, weight="bold")
        
        sent_txt = str(inst_row["sentence"])
        if len(sent_txt) > 85:
            sent_txt = sent_txt[:82] + "..."
        line_txt = str(inst_row["candidate_line_item"])

        title_str = (
            f"LIME Local Explanation #{i} [{cat}]\n"
            f"Sentence: \"{sent_txt}\"\n"
            f"Line Item: \"{line_txt}\" | True: '{true_label}' | Pred: '{pred_label}'"
        )
        plt.title(title_str, fontsize=10, weight="bold", pad=10)
        plt.tight_layout()
        
        lime_plot_path = RESULTS_FIGURES_DIR / f"lime_example_{i}.png"
        plt.savefig(lime_plot_path, dpi=300)
        plt.close()
        logger.info(f"Saved LIME explanation #{i}: {lime_plot_path}")

        # Compute Agreement between SHAP and LIME
        # Top 3 features by absolute attribution
        inst_shap = shap_vals_target[idx]
        top3_shap_indices = np.argsort(np.abs(inst_shap))[::-1][:3]
        top3_shap = [ALL_FEATURES[k] for k in top3_shap_indices]

        # Top 3 features by absolute LIME weight
        sorted_lime_by_abs = sorted(lime_list, key=lambda x: abs(x[1]), reverse=True)[:3]
        top3_lime = [clean_lime_feature_name(item[0]) for item in sorted_lime_by_abs]

        overlap = set(top3_shap).intersection(set(top3_lime))
        agreement_ratio = len(overlap) / 3.0

        agreement_records.append({
            "example_id": i,
            "category": cat,
            "true_label": true_label,
            "pred_label": pred_label,
            "shap_top_3": ", ".join(top3_shap),
            "lime_top_3": ", ".join(top3_lime),
            "shared_features_count": len(overlap),
            "shared_features": ", ".join(overlap),
            "agreement_rate": round(agreement_ratio * 100, 1)
        })

    df_agree = pd.DataFrame(agreement_records)
    agree_csv_path = RESULTS_TABLES_DIR / "shap_vs_lime_agreement.csv"
    df_agree.to_csv(agree_csv_path, index=False)
    logger.info(f"Saved SHAP vs LIME agreement table: {agree_csv_path}")

    avg_agreement = df_agree["agreement_rate"].mean()
    print("\n" + "=" * 70)
    print(df_agree[["example_id", "category", "shared_features_count", "agreement_rate"]].to_string(index=False))
    print("-" * 70)
    print(f"Mean Top-3 Feature Attribution Concordance: {avg_agreement:.1f}%")
    print("=" * 70 + "\n")


if __name__ == "__main__":
    main()
