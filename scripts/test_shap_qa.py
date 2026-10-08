import sys
from pathlib import Path
ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

import pandas as pd
import numpy as np
import joblib
import shap
from src.app import load_model_and_explainer, load_filings_dataset, FEATURE_DESCRIPTIONS
from src.constants import ALL_FEATURES

p, explainer, _, _ = load_model_and_explainer()
df = load_filings_dataset()
classes = list(p.classes_)
scaler = p.named_steps['scaler']

test_cases = [
    ("MATCH Example", df[df['label'] == 'match'].iloc[0]),
    ("NO_MATCH Example", df[df['label'] == 'no_match'].iloc[0]),
    ("AMBIGUOUS Example", df[df['label'] == 'ambiguous'].iloc[0])
]

for title, row in test_cases:
    X_df = row[ALL_FEATURES].to_frame().T
    X_s = scaler.transform(X_df)
    probs = p.predict_proba(X_df)[0]
    pred_idx = int(np.argmax(probs))
    pred_class = classes[pred_idx]
    pred_prob = probs[pred_idx]
    
    # SHAP extracted for predicted class
    sv = explainer.shap_values(X_s)[0, :, pred_idx]
    
    tuples = list(zip(ALL_FEATURES, X_df.iloc[0].values, sv))
    supp = sorted([(f, v) for f, _, v in tuples if v > 0], key=lambda x: x[1], reverse=True)
    opp = sorted([(f, v) for f, _, v in tuples if v < 0], key=lambda x: x[1])
    
    print("=" * 80)
    print(f"CASE: {title}")
    print(f"Claim: \"{row['sentence'][:65]}...\"")
    print(f"Line Item: {row['candidate_line_item']} | Stmt Value: ${row['candidate_value']:,.2f}M")
    print(f"Predicted Class: {pred_class.upper()} (Class idx: {pred_idx}) | Predicted Probability: {pred_prob*100:.2f}%")
    print(f"SHAP Extraction Index: {pred_idx} (Exact Match: {classes[pred_idx] == pred_class})")
    print("-" * 80)
    print(f"TOP FACTORS SUPPORTING DECISION ({pred_class.upper()}):")
    for f, v in supp[:3]:
        print(f"  + {f:28}: +{v:.3f} (supports {pred_class.upper()})")
    print(f"TOP FACTORS OPPOSING DECISION ({pred_class.upper()}):")
    for f, v in opp[:3]:
        print(f"  - {f:28}: {v:.3f} (opposes {pred_class.upper()})")
    print("\nTOP 4 CONTRIBUTIONS TABLE:")
    sorted_table = sorted(tuples, key=lambda x: abs(x[2]), reverse=True)[:4]
    for f, val, c in sorted_table:
        eff = f"Supports {pred_class.upper()}" if c > 0 else f"Opposes {pred_class.upper()}"
        print(f"  {f:28} | Val: {val:<8} | SHAP: {c:+.3f} | {eff}")
    print()
