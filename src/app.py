#!/usr/bin/env python3
"""
src/app.py

Minimal Streamlit demonstration interface for Explainable Financial KPI Matching:
1. User provides a narrative sentence and candidate financial statement line item.
2. Computes Block A hand-crafted and Block B embedding features dynamically.
3. Runs inference through the trained Logistic Regression classifier.
4. Displays prediction label ('match', 'no_match', 'ambiguous') and class probabilities.
5. Renders a live SHAP waterfall / attribution plot explaining the decision.
"""

import streamlit as st
import numpy as np
import pandas as pd
import joblib
from pathlib import Path
import matplotlib.pyplot as plt
from rapidfuzz import fuzz
import re

st.set_page_config(
    page_title="XAI KPI-Check Demo",
    page_icon="📊",
    layout="wide"
)

SAVED_MODEL_PATH = Path("src/model/saved/logistic_regression_full.pkl")
TRAIN_SPLIT_PATH = Path("data/processed/train_split.csv")

BLOCK_A_FEATURES = [
    "numeric_value_match", "numeric_value_close", "keyword_overlap",
    "period_match", "string_similarity", "sentence_length", "line_item_name_length"
]
BLOCK_B_FEATURES = ["embedding_cosine_similarity"]
ALL_FEATURES = BLOCK_A_FEATURES + BLOCK_B_FEATURES

FINANCIAL_KEYWORDS = [
    "revenue", "net income", "operating income", "gross profit", "cost of sales",
    "margin", "ebitda", "assets", "liabilities", "cash", "debt", "earnings",
    "diluted", "share", "sales", "operating expenses", "cash flow", "tax"
]
NUM_EXTRACT_RE = re.compile(r"[-+]?(?:\d{1,3}(?:,\d{3})+|\d+)(?:\.\d+)?")
YEAR_PATTERN = re.compile(r"\b(20\d\d)\b")


@st.cache_resource
def load_model_and_explainer():
    """Loads trained pipeline and cached background data for SHAP."""
    if not SAVED_MODEL_PATH.exists():
        return None, None, None
    pipeline = joblib.load(SAVED_MODEL_PATH)
    train_df = pd.read_csv(TRAIN_SPLIT_PATH) if TRAIN_SPLIT_PATH.exists() else None
    
    # SHAP explainer
    explainer = None
    try:
        import shap
        scaler = pipeline.named_steps["scaler"]
        clf = pipeline.named_steps["clf"]
        if train_df is not None:
            bg_scaled = scaler.transform(train_df[ALL_FEATURES])
            explainer = shap.LinearExplainer(clf, bg_scaled, feature_names=ALL_FEATURES)
    except Exception as e:
        st.warning(f"SHAP explainer init warning: {e}")

    return pipeline, explainer, train_df


def extract_features_single(sentence: str, line_name: str, line_val: float):
    """Computes feature vector for a single pair."""
    # Numbers
    raw_nums = NUM_EXTRACT_RE.findall(sentence)
    nums = [float(n.replace(",", "")) for n in raw_nums if n.replace(",", "").replace(".", "").isdigit()]

    # Numeric match / close
    num_match = 0
    num_close = 0
    if line_val != 0:
        for num in nums:
            diff = abs(num - line_val) / max(abs(line_val), 1e-5)
            if diff <= 0.01:
                num_match = 1
                num_close = 1
                break
            elif diff <= 0.05:
                num_close = 1

    # Keyword overlap
    s_lower = sentence.lower()
    l_lower = line_name.lower()
    s_kws = {kw for kw in FINANCIAL_KEYWORDS if kw in s_lower}
    l_kws = {kw for kw in FINANCIAL_KEYWORDS if kw in l_lower}
    union = s_kws.union(l_kws)
    kw_overlap = len(s_kws.intersection(l_kws)) / len(union) if union else 0.0

    # Period match
    s_years = set(YEAR_PATTERN.findall(sentence))
    l_years = set(YEAR_PATTERN.findall(line_name))
    period_match = 1 if (s_years and l_years and s_years.intersection(l_years)) else 0

    # String similarity
    str_sim = fuzz.token_sort_ratio(sentence, line_name) / 100.0

    # Lengths
    sent_len = len(sentence)
    line_len = len(line_name)

    # Embedding similarity
    try:
        from sentence_transformers import SentenceTransformer
        encoder = SentenceTransformer("all-MiniLM-L6-v2")
        emb1 = encoder.encode([sentence], normalize_embeddings=True)[0]
        emb2 = encoder.encode([line_name], normalize_embeddings=True)[0]
        emb_sim = float(np.dot(emb1, emb2))
    except Exception:
        # Fallback character n-gram / fuzzy proxy
        emb_sim = float(fuzz.partial_ratio(sentence, line_name) / 100.0)

    feature_dict = {
        "numeric_value_match": num_match,
        "numeric_value_close": num_close,
        "keyword_overlap": round(kw_overlap, 4),
        "period_match": period_match,
        "string_similarity": round(str_sim, 4),
        "sentence_length": sent_len,
        "line_item_name_length": line_len,
        "embedding_cosine_similarity": round(emb_sim, 4)
    }
    return pd.DataFrame([feature_dict])


def main():
    st.title("📊 XAI KPI-Check: Explainable Financial KPI Matching")
    st.markdown("""
    Evaluate narrative financial claims against candidate audited financial statement line items.
    Trained on real SEC EDGAR 10-K filings with transparent **SHAP / LIME** attributions.
    """)

    pipeline, explainer, _ = load_model_and_explainer()

    if pipeline is None:
        st.error("Trained model not found at `src/model/saved/logistic_regression_full.pkl`. Please execute the training pipeline first.")
        return

    col1, col2 = st.columns([1, 1])

    with col1:
        st.subheader("1. Input Financial Pair")
        default_sentence = "Total net sales reached $416,161 million during the fiscal year 2025, driven by growth in iPhone and Services."
        sentence_input = st.text_area("Narrative KPI Sentence", value=default_sentence, height=120)

        line_col1, line_col2 = st.columns([2, 1])
        with line_col1:
            line_item_input = st.text_input("Candidate Line Item Name", value="Total net sales")
        with line_col2:
            candidate_val_input = st.number_input("Candidate Value ($M)", value=416161.0, format="%.2f")

        predict_btn = st.button("🔍 Match & Explain", type="primary", use_container_width=True)

    with col2:
        st.subheader("2. Prediction & Model Attribution")
        if predict_btn or sentence_input:
            df_feats = extract_features_single(sentence_input, line_item_input, candidate_val_input)
            
            # Predict
            classes = list(pipeline.classes_)
            probs = pipeline.predict_proba(df_feats)[0]
            pred_class = classes[np.argmax(probs)]
            pred_prob = np.max(probs)

            # Display badge
            badge_color = "#28a745" if pred_class == "match" else ("#dc3545" if pred_class == "no_match" else "#ffc107")
            st.markdown(
                f"""
                <div style="background-color: {badge_color}22; border-left: 6px solid {badge_color}; padding: 12px; border-radius: 4px; margin-bottom: 15px;">
                    <h3 style="margin: 0; color: {badge_color};">Prediction: <strong>{pred_class.upper()}</strong> ({pred_prob*100:.1f}% confidence)</h3>
                </div>
                """,
                unsafe_allow_html=True
            )

            # Probabilities breakdown
            st.write("**Class Probability Distribution:**")
            p_df = pd.DataFrame({"Class": classes, "Probability": [f"{p*100:.1f}%" for p in probs]})
            st.dataframe(p_df.T, use_container_width=True)

            # SHAP attribution
            if explainer is not None:
                scaler = pipeline.named_steps["scaler"]
                X_scaled = scaler.transform(df_feats)
                target_idx = classes.index("match") if "match" in classes else 0
                
                shap_raw = explainer.shap_values(X_scaled)
                if isinstance(shap_raw, list):
                    shap_target = shap_raw[target_idx][0]
                elif len(shap_raw.shape) == 3:
                    shap_target = shap_raw[0, :, target_idx]
                else:
                    shap_target = shap_raw[0]

                st.write("**Feature Attributions towards 'Match' (SHAP):**")
                fig, ax = plt.subplots(figsize=(8, 4))
                sorted_idx = np.argsort(shap_target)
                y_pos = np.arange(len(ALL_FEATURES))
                colors = ["#28a745" if v >= 0 else "#dc3545" for v in shap_target[sorted_idx]]
                
                ax.barh(y_pos, shap_target[sorted_idx], color=colors)
                ax.set_yticks(y_pos)
                ax.set_yticklabels([ALL_FEATURES[k] for k in sorted_idx], fontsize=9)
                ax.axvline(0, color="black", linewidth=0.8)
                ax.set_xlabel("SHAP Impact on 'Match' Log-Odds", fontsize=9, weight="bold")
                plt.tight_layout()
                st.pyplot(fig)
            
            with st.expander("View Raw Computed Features"):
                st.dataframe(df_feats)


if __name__ == "__main__":
    main()
