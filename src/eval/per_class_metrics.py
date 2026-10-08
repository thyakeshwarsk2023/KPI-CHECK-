#!/usr/bin/env python3
"""
src/eval/per_class_metrics.py

Computes granular per-class performance metrics (Precision, Recall, F1-Score, Support)
and error distribution across target classes ('match', 'no_match', 'ambiguous'):
- Highlights performance on the challenging 'ambiguous' category.
- Evaluates human-in-the-loop audit triage (Auto-Cleared vs Auditor-Referred rate).
- Outputs:
  - results/tables/per_class_metrics.csv
  - results/tables/audit_triage_summary.csv
  - results/figures/per_class_metrics_barchart.png
"""

import sys
import logging
from pathlib import Path
import numpy as np
import pandas as pd
import joblib
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report

ROOT_DIR = Path(__file__).resolve().parents[2]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from src.constants import (
    ALL_FEATURES,
    SAVED_MODELS_DIR,
    TEST_SPLIT_CSV,
    RESULTS_TABLES_DIR,
    RESULTS_FIGURES_DIR,
    TARGET_CLASSES
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def main():
    RESULTS_TABLES_DIR.mkdir(parents=True, exist_ok=True)
    RESULTS_FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    model_path = SAVED_MODELS_DIR / "logistic_regression_full.pkl"
    if not model_path.exists() or not TEST_SPLIT_CSV.exists():
        logger.error("Model or test split not found. Run train.py first.")
        return

    pipeline = joblib.load(model_path)
    test_df = pd.read_csv(TEST_SPLIT_CSV)

    X_test = test_df[ALL_FEATURES]
    y_test = test_df["label"]

    y_pred = pipeline.predict(X_test)
    y_proba = pipeline.predict_proba(X_test)
    max_proba = np.max(y_proba, axis=1)

    # 1. Classification report dictionary
    report = classification_report(y_test, y_pred, output_dict=True, zero_division=0)

    per_class_rows = []
    for cls in TARGET_CLASSES:
        if cls in report:
            per_class_rows.append({
                "class": cls,
                "precision": round(report[cls]["precision"] * 100, 2),
                "recall": round(report[cls]["recall"] * 100, 2),
                "f1_score": round(report[cls]["f1-score"] * 100, 2),
                "support": int(report[cls]["support"])
            })

    df_per_class = pd.DataFrame(per_class_rows)
    out_csv = RESULTS_TABLES_DIR / "per_class_metrics.csv"
    df_per_class.to_csv(out_csv, index=False)
    logger.info(f"Saved per-class metrics to: {out_csv}")

    # 2. Audit Triage Analysis (Human-in-the-Loop)
    # Instances with high confidence (>= 0.70) and predicted match are Auto-Cleared
    # All others (ambiguous, low confidence, or discrepancy) are flagged for human review
    auto_cleared = (y_pred == "match") & (max_proba >= 0.70)
    flagged_ambiguous = (y_pred == "ambiguous")
    flagged_discrepancy = (y_pred == "no_match")
    low_confidence = (max_proba < 0.70) & (~flagged_ambiguous)

    total_test = len(test_df)
    triage_data = [
        {
            "triage_tier": "Auto-Verified Match (High Confidence >=70%)",
            "count": int(np.sum(auto_cleared)),
            "percentage": f"{(np.sum(auto_cleared)/total_test)*100:.1f}%",
            "action": "Direct insertion to audit workpaper"
        },
        {
            "triage_tier": "Flagged Ambiguous (Borderline / Accounting Nuance)",
            "count": int(np.sum(flagged_ambiguous)),
            "percentage": f"{(np.sum(flagged_ambiguous)/total_test)*100:.1f}%",
            "action": "Escalate to CPA for segment / non-GAAP inspection"
        },
        {
            "triage_tier": "Discrepancy (Confirmed No-Match)",
            "count": int(np.sum(flagged_discrepancy)),
            "percentage": f"{(np.sum(flagged_discrepancy)/total_test)*100:.1f}%",
            "action": "Flag numerical / semantic inconsistency in filing"
        },
        {
            "triage_tier": "Low Confidence (<70% Margin)",
            "count": int(np.sum(low_confidence)),
            "percentage": f"{(np.sum(low_confidence)/total_test)*100:.1f}%",
            "action": "Route for secondary verification"
        }
    ]
    df_triage = pd.DataFrame(triage_data)
    triage_csv = RESULTS_TABLES_DIR / "audit_triage_summary.csv"
    df_triage.to_csv(triage_csv, index=False)
    logger.info(f"Saved audit triage summary to: {triage_csv}")

    # 3. Bar Chart of Per-Class Metrics
    classes = df_per_class["class"].tolist()
    x = np.arange(len(classes))
    width = 0.25

    plt.figure(figsize=(8, 5))
    plt.bar(x - width, df_per_class["precision"], width, label="Precision", color="#3470a3")
    plt.bar(x, df_per_class["recall"], width, label="Recall", color="#5cb85c")
    plt.bar(x + width, df_per_class["f1_score"], width, label="F1-Score", color="#f0ad4e")

    plt.xticks(x, [c.replace("_", " ").title() for c in classes], fontsize=11, weight="bold")
    plt.ylabel("Score (%)", fontsize=11, weight="bold")
    plt.title("Per-Class Classification Metrics (Primary Logistic Regression)", fontsize=12, weight="bold", pad=12)
    plt.ylim(0, 110)
    plt.legend(loc="lower right", fontsize=10)
    plt.grid(axis="y", linestyle="--", alpha=0.6)
    plt.tight_layout()

    fig_out = RESULTS_FIGURES_DIR / "per_class_metrics_barchart.png"
    plt.savefig(fig_out, dpi=300)
    plt.close()
    logger.info(f"Saved per-class bar chart to: {fig_out}")

    print("\n" + "=" * 70)
    print(f"{'PER-CLASS EVALUATION METRICS (TEST SET)':^70}")
    print("=" * 70)
    print(df_per_class.to_string(index=False))
    print("-" * 70)
    print("AUDIT WORKFLOW TRIAGE BREAKDOWN:")
    print(df_triage[["triage_tier", "count", "percentage", "action"]].to_string(index=False))
    print("=" * 70 + "\n")


if __name__ == "__main__":
    main()
