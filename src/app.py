#!/usr/bin/env python3
"""
src/app.py

Interactive Demonstration Interface for Explainable Financial KPI Matching:
1. User can choose curated test presets or provide custom narrative and line item inputs.
2. Computes Block A hand-crafted and real Block B MiniLM embedding features dynamically.
3. Runs inference through the trained Logistic Regression classifier.
4. Displays prediction label ('match', 'no_match', 'ambiguous') and calibrated probabilities.
5. Surfaces Auditor Workflow Recommendation Tiers (Auto-Verified, Flag for Review, Discrepancy).
6. Renders a live SHAP attribution plot explaining the decision in real time.
"""

import sys
from pathlib import Path
import re
import numpy as np
import pandas as pd
import joblib
import matplotlib.pyplot as plt
from rapidfuzz import fuzz
import streamlit as st

# Ensure repository root is in python path
ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from src.constants import (
    ALL_FEATURES,
    BLOCK_A_FEATURES,
    BLOCK_B_FEATURES,
    FINANCIAL_KEYWORDS,
    SAVED_MODELS_DIR,
    TRAIN_SPLIT_CSV
)
from src.features.embed_helper import get_embedder, compute_pair_similarity

st.set_page_config(
    page_title="XAI KPI-Check Demo",
    page_icon="📊",
    layout="wide"
)

SAVED_MODEL_PATH = SAVED_MODELS_DIR / "logistic_regression_full.pkl"
NUM_EXTRACT_RE = re.compile(r"[-+]?(?:\d{1,3}(?:,\d{3})+|\d+)(?:\.\d+)?")
YEAR_PATTERN = re.compile(r"\b(20\d\d)\b")

PRESETS = {
    "Custom Input": {
        "sentence": "Total net sales reached $416,161 million during the fiscal year 2025, driven by growth in iPhone and Services.",
        "line_item": "Total net sales",
        "value": 416161.0,
        "note": "Custom input"
    },
    "Case 1: Total Net Sales (Correct Match)": {
        "sentence": "Cost of sales for the retail and subscription segments totaled $304,510 million.",
        "line_item": "Cost of sales",
        "value": 304510.0,
        "note": "Exact numerical and keyword agreement ($304,510M)."
    },
    "Case 2: Share Repurchases (Correct Match)": {
        "sentence": "Repurchases of common stock utilized $9,532 million during the fiscal year.",
        "line_item": "Payments for repurchase of common stock",
        "value": 9532.0,
        "note": "Direct financing cash outflow match ($9,532M)."
    },
    "Case 3: Tech Expenses vs Operating Income (Correct No-Match)": {
        "sentence": "Technology and infrastructure expenses grew to $85,620 million.",
        "line_item": "Operating income",
        "value": 36852.0,
        "note": "Unrelated line items; numerical figures differ ($85,620M vs $36,852M)."
    },
    "Case 4: Cash Balances vs Net Income (Correct No-Match)": {
        "sentence": "Cash, cash equivalents and marketable securities totaled $25,984 million at year-end.",
        "line_item": "Net income",
        "value": 29760.0,
        "note": "Balance sheet asset vs Income statement profit."
    },
    "Case 5: Google Cloud vs Total Revenue (Segment Rollup - Ambiguous)": {
        "sentence": "Google Cloud revenues reached $33,088 million, reflecting strong enterprise AI adoption.",
        "line_item": "Total revenues",
        "value": 307394.0,
        "note": "Shared keyword 'revenue' pulls model toward ambiguous despite value mismatch ($33,088M vs $307,394M)."
    }
}


@st.cache_resource
def load_model_and_explainer():
    """Loads trained pipeline, cached background data for SHAP, and real MiniLM embedder."""
    if not SAVED_MODEL_PATH.exists():
        return None, None, None, None
    pipeline = joblib.load(SAVED_MODEL_PATH)
    train_df = pd.read_csv(TRAIN_SPLIT_CSV) if TRAIN_SPLIT_CSV.exists() else None
    
    explainer = None
    try:
        import shap
        scaler = pipeline.named_steps["scaler"]
        clf = pipeline.named_steps["clf"]
        if train_df is not None:
            bg_scaled = scaler.transform(train_df[ALL_FEATURES])
            explainer = shap.LinearExplainer(clf, bg_scaled, feature_names=ALL_FEATURES)
    except Exception as e:
        st.warning(f"SHAP explainer init note: {e}")

    embedder = get_embedder()
    return pipeline, explainer, train_df, embedder


def extract_features_single(sentence: str, line_name: str, line_val: float, embedder=None):
    """
    Computes exact feature vector for a single pair using the real
    MiniLM-L6 dense embedding model and Block A hand-crafted heuristics.
    """
    raw_nums = NUM_EXTRACT_RE.findall(sentence)
    nums = []
    for n in raw_nums:
        clean = n.replace(",", "")
        try:
            nums.append(float(clean))
        except ValueError:
            continue

    num_match = 0
    num_close = 0
    try:
        v_float = float(line_val)
    except (ValueError, TypeError):
        v_float = 0.0

    if v_float != 0:
        for num in nums:
            diff = abs(num - v_float) / max(abs(v_float), 1e-5)
            if diff <= 0.01:
                num_match = 1
                num_close = 1
                break
            elif diff <= 0.05:
                num_close = 1

    s_lower = sentence.lower()
    l_lower = line_name.lower()
    s_kws = {kw for kw in FINANCIAL_KEYWORDS if kw in s_lower}
    l_kws = {kw for kw in FINANCIAL_KEYWORDS if kw in l_lower}
    union = s_kws.union(l_kws)
    kw_overlap = len(s_kws.intersection(l_kws)) / len(union) if union else 0.0

    s_years = set(YEAR_PATTERN.findall(sentence))
    l_years = set(YEAR_PATTERN.findall(line_name))
    period_match = 1 if (s_years and l_years and s_years.intersection(l_years)) else 0

    str_sim = fuzz.token_sort_ratio(sentence, line_name) / 100.0
    sent_len = len(sentence)
    line_len = len(line_name)

    # Real MiniLM dense embedding cosine similarity (replaces previous proxy)
    emb_sim = round(compute_pair_similarity(sentence, line_name, embedder=embedder), 4)

    return pd.DataFrame([{
        "numeric_value_match": num_match,
        "numeric_value_close": num_close,
        "keyword_overlap": round(kw_overlap, 4),
        "period_match": period_match,
        "string_similarity": round(str_sim, 4),
        "sentence_length": sent_len,
        "line_item_name_length": line_len,
        "embedding_cosine_similarity": emb_sim
    }])


def main():
    st.title("📊 XAI KPI-Check: Explainable Financial KPI Matching")
    st.markdown("""
    **Interpretability (SHAP & LIME) for Financial Statement Verification**  
    Benchmarked against KPI-Check *(Hillebrand et al., IEEE BigData 2022)* | Feature-Aligned Inference Engine.
    """)

    pipeline, explainer, _, embedder = load_model_and_explainer()

    if pipeline is None:
        st.error("Trained model not found at `src/model/saved/logistic_regression_full.pkl`. Please run `python run_pipeline.py` first.")
        return

    backend_name = embedder[0] if embedder else "local"
    st.sidebar.markdown(f"**Embedding Model:** `all-MiniLM-L6-v2` ({backend_name})")
    st.sidebar.header("Select Evaluation Preset")
    preset_choice = st.sidebar.selectbox("Test Scenarios", list(PRESETS.keys()))
    preset_data = PRESETS[preset_choice]

    col1, col2 = st.columns([1, 1])

    with col1:
        st.subheader("1. Input Financial Pair")
        if preset_choice != "Custom Input":
            st.info(f"**Context:** {preset_data['note']}")

        sentence_input = st.text_area("Narrative KPI Sentence", value=preset_data["sentence"], height=120)

        line_col1, line_col2 = st.columns([2, 1])
        with line_col1:
            line_item_input = st.text_input("Candidate Line Item Name", value=preset_data["line_item"])
        with line_col2:
            candidate_val_input = st.number_input("Statement Value ($M)", value=float(preset_data["value"]), format="%.2f")

        predict_btn = st.button("🔍 Classify & Explain", type="primary", use_container_width=True)

    with col2:
        st.subheader("2. Prediction & Model Attribution")
        if predict_btn or sentence_input:
            df_feats = extract_features_single(sentence_input, line_item_input, candidate_val_input, embedder=embedder)
            classes = list(pipeline.classes_)
            probs = pipeline.predict_proba(df_feats[ALL_FEATURES])[0]
            pred_class = classes[np.argmax(probs)]
            pred_prob = np.max(probs)

            # Visual badge
            color_map = {"match": "#28a745", "no_match": "#dc3545", "ambiguous": "#ffc107"}
            badge_color = color_map.get(pred_class, "#6c757d")
            st.markdown(
                f"""
                <div style="background-color: {badge_color}22; border-left: 6px solid {badge_color}; padding: 12px; border-radius: 4px; margin-bottom: 15px;">
                    <h3 style="margin: 0; color: {badge_color};">Prediction: <strong>{pred_class.upper()}</strong> ({pred_prob*100:.1f}% confidence)</h3>
                </div>
                """,
                unsafe_allow_html=True
            )

            # Human-in-the-loop Auditor Decision Tier
            if pred_class == "match" and pred_prob >= 0.70:
                st.success("🟢 **Auditor Decision: AUTO-VERIFIED MATCH** — Numeric and semantic agreement validated for automated audit workpaper.")
            elif pred_class == "ambiguous" or pred_prob < 0.70:
                st.warning("🟡 **Auditor Decision: REFER TO HUMAN AUDITOR** — Borderline confidence / segment rollup flagged for CPA inspection.")
            else:
                st.error("🔴 **Auditor Decision: VERIFICATION DISCREPANCY** — Line item does not support narrative claim.")

            # Probabilities distribution
            p_df = pd.DataFrame({"Class": classes, "Probability": [f"{p*100:.1f}%" for p in probs]})
            st.dataframe(p_df.T, use_container_width=True)

            # SHAP attribution
            if explainer is not None:
                scaler = pipeline.named_steps["scaler"]
                X_scaled = scaler.transform(df_feats[ALL_FEATURES])
                target_idx = classes.index("match") if "match" in classes else 0
                
                shap_raw = explainer.shap_values(X_scaled)
                if isinstance(shap_raw, list):
                    shap_target = shap_raw[target_idx][0]
                elif len(shap_raw.shape) == 3:
                    shap_target = shap_raw[0, :, target_idx]
                else:
                    shap_target = shap_raw[0]

                st.write("**Feature Attributions towards 'Match' (Exact LinearSHAP):**")
                fig, ax = plt.subplots(figsize=(8, 4))
                sorted_idx = np.argsort(shap_target)
                y_pos = np.arange(len(ALL_FEATURES))
                colors = ["#28a745" if v >= 0 else "#dc3545" for v in shap_target[sorted_idx]]
                
                ax.barh(y_pos, shap_target[sorted_idx], color=colors)
                ax.set_yticks(y_pos)
                ax.set_yticklabels([ALL_FEATURES[k] for k in sorted_idx], fontsize=9)
                ax.axvline(0, color="black", linewidth=0.8)
                ax.set_xlabel("SHAP Impact on 'Match' Decision", fontsize=9, weight="bold")
                plt.tight_layout()
                st.pyplot(fig)
            
            with st.expander("Computed Feature Vector (Aligned Feature Space)"):
                st.dataframe(df_feats)


if __name__ == "__main__":
    main()
