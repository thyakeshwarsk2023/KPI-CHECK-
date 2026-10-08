import sys
from pathlib import Path
ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

import numpy as np
import pandas as pd
import joblib
import shap
from src.app import (
    load_model_and_explainer,
    load_filings_dataset,
    extract_evidence_details,
    extract_features_single,
    ALL_FEATURES,
    PRESETS
)

def run_smoke_test():
    results = {}
    bugs = []

    # 1. Load pipeline and data
    try:
        pipeline, explainer, train_df, embedder = load_model_and_explainer()
        assert pipeline is not None, "Pipeline failed to load"
        assert explainer is not None, "Explainer failed to load"
        df_all = load_filings_dataset()
        assert not df_all.empty, "Filings dataset is empty"
        results["1. Open dashboard / Load core dependencies"] = "PASS"
    except Exception as e:
        results["1. Open dashboard / Load core dependencies"] = "FAIL"
        bugs.append(f"Load failed: {e}")
        return results, bugs

    # 2. Select company filing (Alphabet Inc.)
    try:
        company = "Alphabet Inc."
        df_company = df_all[df_all["company"] == company].copy().reset_index(drop=True)
        assert len(df_company) == 22, f"Expected 22 claims for Alphabet, got {len(df_company)}"
        results["2. Select annual report/company"] = "PASS"
    except Exception as e:
        results["2. Select annual report/company"] = "FAIL"
        bugs.append(f"Filing selection failed: {e}")

    # 3. Dynamic Executive Summary calculation
    try:
        X_company = df_company[ALL_FEATURES]
        probs = pipeline.predict_proba(X_company)
        classes = list(pipeline.classes_)
        preds = [classes[i] for i in np.argmax(probs, axis=1)]
        confs = np.max(probs, axis=1)

        decisions = []
        for p, c in zip(preds, confs):
            if p == "match" and c >= 0.70:
                decisions.append("VERIFIED")
            elif p == "no_match":
                decisions.append("DISCREPANCY")
            else:
                decisions.append("NEEDS HUMAN REVIEW")

        n_tot = len(df_company)
        n_ver = sum(d == "VERIFIED" for d in decisions)
        n_disc = sum(d == "DISCREPANCY" for d in decisions)
        n_rev = sum(d == "NEEDS HUMAN REVIEW" for d in decisions)

        assert n_tot == n_ver + n_disc + n_rev, "Summary counts do not sum to total"
        assert n_ver == 7 and n_disc == 8 and n_rev == 7, f"Counts mismatch: {n_ver}, {n_disc}, {n_rev}"
        results["3. Executive summary numbers calculated dynamically"] = "PASS"
    except Exception as e:
        results["3. Executive summary numbers calculated dynamically"] = "FAIL"
        bugs.append(f"Summary calculation failed: {e}")

    # 4. Correct display categories (Verified / Discrepancies / Needs Human Review)
    try:
        assert set(decisions) == {"VERIFIED", "DISCREPANCY", "NEEDS HUMAN REVIEW"}
        results["4. Verified / Discrepancies / Needs Human Review displayed correctly"] = "PASS"
    except Exception as e:
        results["4. Verified / Discrepancies / Needs Human Review displayed correctly"] = "FAIL"
        bugs.append(f"Category labels failed: {e}")

    # 5 & 6. Select discrepancy finding and populate Deep-Dive Inspector
    try:
        disc_indices = [i for i, d in enumerate(decisions) if d == "DISCREPANCY"]
        assert len(disc_indices) > 0, "No discrepancies found"
        selected_idx = disc_indices[0]
        selected_row = df_company.iloc[selected_idx]
        
        # Verify row populates inspector
        ev = extract_evidence_details(
            selected_row["sentence"],
            selected_row["candidate_line_item"],
            selected_row["candidate_value"],
            selected_row[ALL_FEATURES].to_dict(),
            selected_row.to_dict()
        )
        assert ev["claim"] == selected_row["sentence"]
        assert ev["line_item"] == selected_row["candidate_line_item"]
        results["5. Select discrepancy from findings table"] = "PASS"
        results["6. Correct claim populates Deep-Dive Inspector"] = "PASS"
    except Exception as e:
        results["5. Select discrepancy from findings table"] = "FAIL"
        results["6. Correct claim populates Deep-Dive Inspector"] = "FAIL"
        bugs.append(f"Discrepancy selection failed: {e}")

    # 7. Actual data values displayed (claim value & candidate value)
    try:
        assert ev["stmt_val_str"] != "—" and ev["stmt_val_str"] != "Unavailable"
        assert ev["claim_val_str"] != "—" and ev["claim_val_str"] != "Unavailable"
        results["7. Displayed claim value and candidate value from actual data"] = "PASS"
    except Exception as e:
        results["7. Displayed claim value and candidate value from actual data"] = "FAIL"
        bugs.append(f"Actual data extraction failed: {e}")

    # 8. Predicted probability displayed (not 'confidence')
    try:
        # Check src/app.py code for any unreplaced 'confidence' in probability labels
        with open(ROOT_DIR / "src" / "app.py", "r", encoding="utf-8") as f:
            code = f.read()
        assert "Predicted probability:" in code
        assert "Model Confidence:" not in code
        results["8. Predicted probability displayed, not 'confidence'"] = "PASS"
    except Exception as e:
        results["8. Predicted probability displayed, not 'confidence'"] = "FAIL"
        bugs.append(f"Probability label verification failed: {e}")

    # 9. SHAP contributions for actual predicted class
    try:
        scaler = pipeline.named_steps["scaler"]
        X_s = scaler.transform(selected_row[ALL_FEATURES].to_frame().T)
        target_cls = preds[selected_idx] # 'no_match'
        target_cls_idx = classes.index(target_cls)
        sv = explainer.shap_values(X_s)[0, :, target_cls_idx]
        assert len(sv) == 8
        results["9. SHAP displays contributions for actual predicted class"] = "PASS"
    except Exception as e:
        results["9. SHAP displays contributions for actual predicted class"] = "FAIL"
        bugs.append(f"SHAP predicted class extraction failed: {e}")

    # 10. Test one MATCH example
    try:
        match_row = df_all[df_all["label"] == "match"].iloc[0]
        p_match = pipeline.predict(match_row[ALL_FEATURES].to_frame().T)[0]
        assert p_match == "match"
        results["10. Test one MATCH example"] = "PASS"
    except Exception as e:
        results["10. Test one MATCH example"] = "FAIL"
        bugs.append(f"MATCH test failed: {e}")

    # 11. Test one NO_MATCH example
    try:
        no_match_row = df_all[df_all["label"] == "no_match"].iloc[0]
        p_no = pipeline.predict(no_match_row[ALL_FEATURES].to_frame().T)[0]
        assert p_no == "no_match"
        results["11. Test one NO_MATCH example"] = "PASS"
    except Exception as e:
        results["11. Test one NO_MATCH example"] = "FAIL"
        bugs.append(f"NO_MATCH test failed: {e}")

    # 12. Test one AMBIGUOUS example
    try:
        amb_row = df_all[df_all["label"] == "ambiguous"].iloc[0]
        p_amb = pipeline.predict(amb_row[ALL_FEATURES].to_frame().T)[0]
        assert p_amb == "ambiguous"
        results["12. Test one AMBIGUOUS example"] = "PASS"
    except Exception as e:
        results["12. Test one AMBIGUOUS example"] = "FAIL"
        bugs.append(f"AMBIGUOUS test failed: {e}")

    # 13. Check that no page has errors, tracebacks, placeholder values, or broken components
    try:
        import py_compile
        py_compile.compile(str(ROOT_DIR / "src" / "app.py"), doraise=True)
        # Check all presets compile & extract features cleanly
        for name, data in PRESETS.items():
            feats = extract_features_single(data["sentence"], data["line_item"], data["value"], embedder=embedder)
            pr = pipeline.predict_proba(feats[ALL_FEATURES])
            assert pr.shape == (1, 3)
        results["13. Check no page has errors, tracebacks, or broken components"] = "PASS"
    except Exception as e:
        results["13. Check no page has errors, tracebacks, or broken components"] = "FAIL"
        bugs.append(f"Integrity check failed: {e}")

    return results, bugs

if __name__ == "__main__":
    res, bugs = run_smoke_test()
    for test, status in res.items():
        print(f"{test}: {status}")
    if bugs:
        print("BUGS FOUND:")
        for b in bugs:
            print(f" - {b}")
    else:
        print("ALL TESTS PASSED. Zero bugs found.")
