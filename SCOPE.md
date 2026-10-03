# SCOPE.md — XAI KPI-Check Scope Guardrails

- **Task:** binary/3-class text-pair classification — given a narrative KPI sentence and a candidate financial statement line item, classify `match` / `no_match` / `ambiguous`.
- **Model:** sentence-pair embeddings → logistic regression (primary) + small MLP (secondary, for comparison). **Not** a fine-tuned transformer from scratch — no time for that.
- **XAI:** SHAP (KernelSHAP or LinearSHAP on the LR) + LIME, compared side by side, plus a manual explanation-quality check on a sample.
- **Benchmark:** your micro-F1 vs. KPI-Check's reported **73.00% micro-F1** (Hillebrand et al., IEEE BigData 2022, arXiv:2211.06112), clearly caveated as a different, smaller-scale setup.
- **Dataset:** real financial reports (SEC EDGAR 10-Ks, or NSE/BSE annual reports), you hand-label 100–200 sentence/line-item pairs.
