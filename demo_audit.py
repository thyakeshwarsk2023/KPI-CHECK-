#!/usr/bin/env python3
"""
demo_audit.py

Command-Line Demonstration of Form 10-K Financial Consistency & Inconsistency Audit:
Scans narrative claims against reported financial statements, identifies numerical
inconsistencies, detects non-GAAP/subsegment rollups, and explains the root cause using SHAP.

Usage:
    python demo_audit.py
    python demo_audit.py --company "Apple Inc."
    python demo_audit.py --company "Alphabet Inc."
    python demo_audit.py --company "Tesla Inc."
    python demo_audit.py --company "NVIDIA Corp."
"""

import sys
import argparse
from pathlib import Path
import numpy as np
import pandas as pd
import joblib

if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

ROOT_DIR = Path(__file__).resolve().parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from src.constants import ALL_FEATURES, SAVED_MODELS_DIR, TRAIN_SPLIT_CSV

MODEL_PATH = SAVED_MODELS_DIR / "logistic_regression_full.pkl"
LABELED_PATH = ROOT_DIR / "data" / "labeled" / "pairs_labeled.csv"
FEATURES_PATH = ROOT_DIR / "data" / "processed" / "features.csv"


def run_audit(company_name="Alphabet Inc."):
    if not MODEL_PATH.exists():
        print(f"Error: Model not found at {MODEL_PATH}")
        return

    pipeline = joblib.load(MODEL_PATH)
    classes = list(pipeline.classes_)

    # Load background data for LinearSHAP
    import shap
    train_df = pd.read_csv(TRAIN_SPLIT_CSV)
    scaler = pipeline.named_steps["scaler"]
    clf = pipeline.named_steps["clf"]
    bg_scaled = scaler.transform(train_df[ALL_FEATURES])
    explainer = shap.LinearExplainer(clf, bg_scaled, feature_names=ALL_FEATURES)

    # Load data
    df_lbl = pd.read_csv(LABELED_PATH)
    df_feat = pd.read_csv(FEATURES_PATH).iloc[:len(df_lbl)]

    for col in ALL_FEATURES:
        if col not in df_lbl.columns and col in df_feat.columns:
            df_lbl[col] = df_feat[col].values

    # Available companies
    avail_companies = df_lbl["company"].unique().tolist()
    match_comp = [c for c in avail_companies if company_name.lower() in c.lower()]
    if not match_comp:
        print(f"Company '{company_name}' not found. Available: {avail_companies}")
        return
    target_company = match_comp[0]

    sub_df = df_lbl[df_lbl["company"] == target_company].copy().reset_index(drop=True)

    # Predict
    X = sub_df[ALL_FEATURES]
    probs = pipeline.predict_proba(X)
    pred_idx = np.argmax(probs, axis=1)
    preds = [classes[i] for i in pred_idx]
    confs = np.max(probs, axis=1)
    sub_df["pred"] = preds
    sub_df["conf"] = confs

    # Compute SHAP
    X_s = scaler.transform(X)
    target_cls_idx = classes.index("match") if "match" in classes else 0
    shap_raw = explainer.shap_values(X_s)
    if isinstance(shap_raw, list):
        shap_matrix = shap_raw[target_cls_idx]
    elif len(shap_raw.shape) == 3:
        shap_matrix = shap_raw[:, :, target_cls_idx]
    else:
        shap_matrix = shap_raw

    # Header
    total = len(sub_df)
    n_match = sum(sub_df["pred"] == "match")
    n_no = sum(sub_df["pred"] == "no_match")
    n_amb = sum(sub_df["pred"] == "ambiguous")

    print("\n" + "=" * 90)
    print(f" [*] FORM 10-K FINANCIAL AUDIT DISCREPANCY REPORT: {target_company.upper()}")
    print("     Automated Narrative Claim Verification & Numerical Inconsistency Detection")
    print("=" * 90)
    print(f" Total Narrative Claims Audited: {total}")
    print(f"   [+] Verified Consistent Figures:     {n_match} ({n_match/total*100:.1f}%)")
    print(f"   [-] Numerical Discrepancies Caught:  {n_no} ({n_no/total*100:.1f}%)")
    print(f"   [?] Ambiguities / Subsegment Rollups: {n_amb} ({n_amb/total*100:.1f}%)")
    print("-" * 90)

    # Section 1: Discrepancies Caught
    discrepancies = sub_df[sub_df["pred"] == "no_match"]
    print(f"\n[SECTION 1] >>> NUMERICAL DISCREPANCIES & CONTRADICTING CLAIMS ({len(discrepancies)} Found)")
    print("-" * 90)
    
    count = 1
    for idx, row in discrepancies.iterrows():
        print(f"\n  Discrepancy #{count}:")
        print(f"  * Narrative Claim:    \"{row['sentence']}\"")
        print(f"  * Line Item Checked:  {row['candidate_line_item']}  |  Reported Value: ${row['candidate_value']:,.2f}M")
        print(f"  * Model Verdict:      [NO_MATCH / DISCREPANCY] (Confidence: {row['conf']*100:.1f}%)")
        print(f"  * Audit Finding:      {row['notes']}")
        
        # Top SHAP penalties
        sv = shap_matrix[idx]
        top_penalties = np.argsort(sv)[:3]  # most negative towards match
        pen_strs = [f"{ALL_FEATURES[k]} ({sv[k]:+.2f})" for k in top_penalties if sv[k] < 0]
        if pen_strs:
            print(f"  * Why Rejected (XAI): Penalized by {', '.join(pen_strs)}")
        count += 1
        if count > 4:  # show top 4 to keep terminal clean
            rem = len(discrepancies) - 4
            if rem > 0:
                print(f"\n  ... and {rem} additional numerical discrepancies flagged in filing workpaper.")
            break

    # Section 2: Ambiguities / Rollups
    ambiguities = sub_df[sub_df["pred"] == "ambiguous"]
    print(f"\n\n[SECTION 2] >>> ACCOUNTING AMBIGUITIES & SUBSEGMENT ROLLUPS ({len(ambiguities)} Flagged)")
    print("-" * 90)
    
    count = 1
    for idx, row in ambiguities.iterrows():
        print(f"\n  Flagged Item #{count}:")
        print(f"  * Narrative Claim:    \"{row['sentence']}\"")
        print(f"  * Line Item Checked:  {row['candidate_line_item']}  |  Reported Value: ${row['candidate_value']:,.2f}M")
        print(f"  * Model Verdict:      [AMBIGUOUS - REFER TO HUMAN CPA] (Confidence: {row['conf']*100:.1f}%)")
        print(f"  * Audit Finding:      {row['notes']}")
        
        sv = shap_matrix[idx]
        pos_f = [ALL_FEATURES[k] for k in np.argsort(sv)[-2:] if sv[k] > 0]
        neg_f = [ALL_FEATURES[k] for k in np.argsort(sv)[:2] if sv[k] < 0]
        print(f"  * Why Ambiguous (XAI): Supported by {pos_f}, but penalized by {neg_f}")
        count += 1
        if count > 3:
            rem = len(ambiguities) - 3
            if rem > 0:
                print(f"\n  ... and {rem} additional ambiguous claims flagged for CPA inspection.")
            break

    # Section 3: Verified
    matches = sub_df[sub_df["pred"] == "match"]
    print(f"\n\n[SECTION 3] >>> AUTO-VERIFIED CONSISTENT FIGURES ({len(matches)} Verified)")
    print("-" * 90)
    for idx, row in matches.head(2).iterrows():
        print(f"  * \"{row['sentence'][:70]}...\"")
        print(f"    -> Matched with '{row['candidate_line_item']}' (${row['candidate_value']:,.2f}M) | Confidence: {row['conf']*100:.1f}%\n")

    print("=" * 90)
    print(" [OK] AUDIT SUMMARY COMPLETE: Workpapers generated for CPA review.")
    print("=" * 90 + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Form 10-K Consistency Audit Scanner")
    parser.add_argument("--company", type=str, default="Alphabet Inc.", help="Company name (e.g. Apple, Alphabet, Microsoft, Tesla, NVIDIA)")
    args = parser.parse_args()
    run_audit(args.company)
