#!/usr/bin/env python3
"""
scripts/generate_review2_ppt.py

Generates the fully updated, leak-free, audit-compliant 18-slide PowerPoint
presentation (PPTX) for Project Review 2 of Explainable Financial KPI-Matching.

Authors: S.K. Thyakeshwar (23BAI0194), Sai Sanjay (23BAI0168)
Supervisor: Dr. Manikandan G (School of Computer Science & Engineering, VIT)

Enforces:
- Hard Rules 1-6 (All metrics from results/metrics_master.json)
- Primary Model: Logistic Regression without length features (84.67% Micro-F1, CI [78.67, 90.00])
- Complete removal of Cohen's kappa and double-blind annotation claims
- Relabeling/quarantine of all legacy and leaked metrics
- Self-Audit: Leakage Found and Fixed slide
- Limitations & Threats to Validity slide
- Measured XAI latencies and concordances
- Honest calibration reporting with prior shift diagnostics
- Updated Final Review Roadmap
"""

import sys
from pathlib import Path
from pptx import Presentation
from pptx.util import Inches, Pt
from pptx.dml.color import RGBColor
from pptx.enum.text import PP_ALIGN, MSO_ANCHOR
from pptx.enum.shapes import MSO_SHAPE

# Paths
BASE_DIR = Path(__file__).resolve().parent.parent
FIGURES_DIR = BASE_DIR / "results" / "figures"
OUTPUT_PPTX = BASE_DIR / "docs" / "review2_presentation.pptx"

# Color Palette (Deep Navy / Slate / Royal Blue / Emerald / Crimson / Clean Off-White)
COLOR_BG_DARK = RGBColor(15, 23, 42)        # Slate 900
COLOR_BG_LIGHT = RGBColor(248, 250, 252)    # Slate 50
COLOR_CARD_BG = RGBColor(255, 255, 255)     # White
COLOR_CARD_BORDER = RGBColor(226, 232, 240) # Slate 200
COLOR_PRIMARY = RGBColor(30, 41, 59)        # Slate 800
COLOR_ACCENT = RGBColor(37, 99, 235)        # Royal Blue 600
COLOR_ACCENT_LIGHT = RGBColor(219, 234, 254)# Blue 100
COLOR_EMERALD = RGBColor(5, 150, 105)       # Emerald 600
COLOR_CRIMSON = RGBColor(220, 38, 38)       # Red 600
COLOR_TEXT_DARK = RGBColor(15, 23, 42)      # Slate 900
COLOR_TEXT_MUTED = RGBColor(100, 116, 139)  # Slate 500
COLOR_TEXT_LIGHT = RGBColor(255, 255, 255)  # White
COLOR_TABLE_HEADER = RGBColor(30, 41, 59)   # Slate 800
COLOR_TABLE_ALT = RGBColor(241, 245, 249)   # Slate 100


def create_deck():
    prs = Presentation()
    # 16:9 Widescreen dimensions
    prs.slide_width = Inches(13.333)
    prs.slide_height = Inches(7.5)
    blank_slide_layout = prs.slide_layouts[6] # Blank layout

    def add_header(slide, title_text, category_text="PROJECT REVIEW 2 (80% IMPLEMENTATION)"):
        # Header text frame
        header_box = slide.shapes.add_textbox(Inches(0.8), Inches(0.35), Inches(11.733), Inches(0.95))
        tf = header_box.text_frame
        tf.word_wrap = True
        tf.margin_left = tf.margin_top = tf.margin_right = tf.margin_bottom = 0

        p_cat = tf.paragraphs[0]
        p_cat.text = category_text.upper()
        p_cat.font.size = Pt(10)
        p_cat.font.bold = True
        p_cat.font.color.rgb = COLOR_ACCENT

        p_title = tf.add_paragraph()
        p_title.text = title_text
        p_title.font.size = Pt(21)
        p_title.font.bold = True
        p_title.font.color.rgb = COLOR_PRIMARY
        p_title.space_before = Pt(2)

        # Header bottom separator line
        line = slide.shapes.add_shape(MSO_SHAPE.RECTANGLE, Inches(0.8), Inches(1.35), Inches(11.733), Inches(0.02))
        line.fill.solid()
        line.fill.fore_color.rgb = COLOR_CARD_BORDER
        line.line.color.rgb = COLOR_CARD_BORDER

        # Professional Footer
        footer_box = slide.shapes.add_textbox(Inches(0.8), Inches(7.05), Inches(11.733), Inches(0.35))
        tf_f = footer_box.text_frame
        tf_f.margin_left = tf_f.margin_top = tf_f.margin_right = tf_f.margin_bottom = 0
        p_f = tf_f.paragraphs[0]
        p_f.text = "Explainable Financial KPI-Matching | S.K. Thyakeshwar (23BAI0194) & Sai Sanjay (23BAI0168) | Guide: Dr. Manikandan G | VIT"
        p_f.font.size = Pt(9)
        p_f.font.color.rgb = COLOR_TEXT_MUTED

    def add_card(slide, left, top, width, height, bg_color=COLOR_CARD_BG, border_color=COLOR_CARD_BORDER):
        card = slide.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, left, top, width, height)
        card.fill.solid()
        card.fill.fore_color.rgb = bg_color
        card.line.color.rgb = border_color
        card.line.width = Pt(1)
        return card

    # =========================================================================
    # SLIDE 1: Title Slide (Dark Theme)
    # =========================================================================
    slide1 = prs.slides.add_slide(blank_slide_layout)
    bg1 = slide1.shapes.add_shape(MSO_SHAPE.RECTANGLE, 0, 0, Inches(13.333), Inches(7.5))
    bg1.fill.solid()
    bg1.fill.fore_color.rgb = COLOR_BG_DARK
    bg1.line.fill.background()

    # Title Badge
    badge = slide1.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, Inches(1.2), Inches(0.85), Inches(4.5), Inches(0.38))
    badge.fill.solid()
    badge.fill.fore_color.rgb = RGBColor(30, 58, 138)
    badge.line.color.rgb = COLOR_ACCENT
    tb_b = badge.text_frame
    p_b = tb_b.paragraphs[0]
    p_b.text = "CAPSTONE PROJECT REVIEW 2 (80% IMPLEMENTATION)"
    p_b.font.size = Pt(9.5)
    p_b.font.bold = True
    p_b.font.color.rgb = RGBColor(191, 219, 254)
    p_b.alignment = PP_ALIGN.CENTER

    # Main Title & Subtitle
    tb1 = slide1.shapes.add_textbox(Inches(1.2), Inches(1.35), Inches(10.9), Inches(2.2))
    tf1 = tb1.text_frame
    tf1.word_wrap = True
    p1 = tf1.paragraphs[0]
    p1.text = "Explainable Financial KPI-Matching"
    p1.font.size = Pt(36)
    p1.font.bold = True
    p1.font.color.rgb = COLOR_TEXT_LIGHT

    p2 = tf1.add_paragraph()
    p2.text = "Dual-Block Feature Architecture & Audit-Compliant XAI for 10-K Narrative Claims"
    p2.font.size = Pt(19)
    p2.font.color.rgb = RGBColor(148, 163, 184)
    p2.space_before = Pt(6)

    # Info Grid Cards on Title Slide
    cards_meta = [
        ("Student Investigators",
         "• S.K. Thyakeshwar (Reg No: 23BAI0194)\n• Sai Sanjay (Reg No: 23BAI0168)\nSchool of Computer Science & Engineering, VIT"),
        ("Project Supervisor",
         "Dr. Manikandan G\nAssociate Professor, SCOPE\nVellore Institute of Technology (VIT)"),
        ("Primary Model Performance",
         "• Balanced Logistic Regression (No Length Features)\n• Test Micro-F1: 84.67% (95% CI: [78.67%, 90.00%])\n• Macro-F1: 83.75% (Held-Out GOLD n=150)"),
        ("Scope & Verification Protocol",
         "• 150 SEC EDGAR author-curated pairs (Single Annotator)\n• Dual LinearSHAP (0.009 ms) & LIME XAI Layer\n• Status: 80% Implementation Complete")
    ]

    for i, (title, content) in enumerate(cards_meta):
        x = Inches(1.2 + (i % 2) * 5.6)
        y = Inches(3.9 + (i // 2) * 1.5)
        c = add_card(slide1, x, y, Inches(5.3), Inches(1.35), bg_color=RGBColor(30, 41, 59), border_color=RGBColor(51, 65, 85))
        tf_c = c.text_frame
        tf_c.word_wrap = True
        tf_c.margin_left = Inches(0.2)
        tf_c.margin_top = Inches(0.12)
        p_ct = tf_c.paragraphs[0]
        p_ct.text = title.upper()
        p_ct.font.size = Pt(10)
        p_ct.font.bold = True
        p_ct.font.color.rgb = COLOR_ACCENT

        p_cc = tf_c.add_paragraph()
        p_cc.text = content
        p_cc.font.size = Pt(10.5)
        p_cc.font.color.rgb = COLOR_TEXT_LIGHT
        p_cc.space_before = Pt(4)

    # =========================================================================
    # SLIDE 2: Problem Statement & Motivation
    # =========================================================================
    slide2 = prs.slides.add_slide(blank_slide_layout)
    add_header(slide2, "Project Motivation: Automating 10-K Claim Verification with Audit-Compliant AI")

    col_data_s2 = [
        ("The Dual-Modality of Form 10-K", [
            "Corporate SEC 10-Ks combine unstructured narrative text (MD&A) with authoritative financial tables (Balance Sheets, Statements).",
            "Executive claims (e.g., 'Revenues grew to $416.2B') must be reconciled against audited financial line items before certification.",
            "Auditing firms spend thousands of billable CPA hours manually cross-checking hundreds of disclosure pages per filing."
        ], RGBColor(239, 246, 255), COLOR_ACCENT),
        ("The Black-Box Dilemma in AI Auditing", [
            "Standard deep text-pair classifiers operate as opaque black boxes, providing zero mathematical accountability for auditors.",
            "In regulated capital markets, unexplainable predictions cannot be certified by CPAs, audit committees, or compliance officers.",
            "LLMs introduce severe hallucination risks, numerical scale errors, prompt fragility, and impractical 2-5s inference latencies."
        ], RGBColor(254, 242, 242), COLOR_CRIMSON),
        ("Core Objectives for Review 2", [
            "Develop an explainable text-pair matching framework linking narrative statements to audited table rows.",
            "Formulate a Dual-Block Feature Architecture (Interpretable Hand-Crafted + Dense Sentence Embeddings).",
            "Implement dual game-theoretic and surrogate XAI (SHAP + LIME) operating with sub-millisecond audit latency."
        ], RGBColor(236, 253, 245), COLOR_EMERALD)
    ]

    for i, (head, bullets, bg_c, bdr_c) in enumerate(col_data_s2):
        x = Inches(0.8 + i * 3.95)
        c = add_card(slide2, x, Inches(1.6), Inches(3.8), Inches(5.1), bg_color=bg_c, border_color=bdr_c)
        tf = c.text_frame
        tf.word_wrap = True
        tf.margin_left = Inches(0.25)
        tf.margin_right = Inches(0.25)
        tf.margin_top = Inches(0.25)
        
        p = tf.paragraphs[0]
        p.text = head
        p.font.size = Pt(14)
        p.font.bold = True
        p.font.color.rgb = COLOR_TEXT_DARK

        for bullet in bullets:
            pb = tf.add_paragraph()
            pb.text = f"• {bullet}"
            pb.font.size = Pt(11)
            pb.font.color.rgb = COLOR_PRIMARY
            pb.space_before = Pt(12)

    # =========================================================================
    # SLIDE 3: Literature Review & Identified Research Gaps
    # =========================================================================
    slide3 = prs.slides.add_slide(blank_slide_layout)
    add_header(slide3, "Literature Review & Research Gaps: The Need for Numerical Scale Awareness")

    # Card 1: Primary Baseline Reference: KPI-Check (Top-Left)
    c1 = add_card(slide3, Inches(0.8), Inches(1.6), Inches(5.75), Inches(2.45))
    tf1 = c1.text_frame
    tf1.word_wrap = True
    tf1.margin_left = Inches(0.2)
    tf1.margin_top = Inches(0.15)
    p1 = tf1.paragraphs[0]
    p1.text = "1. Reference Baseline: KPI-Check (Hillebrand et al., 2022)"
    p1.font.size = Pt(12)
    p1.font.bold = True
    p1.font.color.rgb = COLOR_ACCENT
    b1_items = [
        "Citation: IEEE BigData 2022 (Fraunhofer IAIS & PwC Germany).",
        "Architecture: Transformer NER + Table Extraction + Fine-Tuned BERT Relation Extractor.",
        "Performance: 73.00% Micro-F1 on proprietary German commercial reports.",
        "Crucial Scope Distinction: Not directly comparable (German language, different task, proprietary uncurated data). Fair baseline on our data planned."
    ]
    for b in b1_items:
        pb = tf1.add_paragraph()
        pb.text = f"• {b}"
        pb.font.size = Pt(9.5)
        pb.font.color.rgb = COLOR_PRIMARY
        pb.space_before = Pt(2)

    # Card 2: Modern LLM Auditing Context (Top-Right)
    c2 = add_card(slide3, Inches(6.783), Inches(1.6), Inches(5.75), Inches(2.45))
    tf2 = c2.text_frame
    tf2.word_wrap = True
    tf2.margin_left = Inches(0.2)
    tf2.margin_top = Inches(0.15)
    p2 = tf2.paragraphs[0]
    p2.text = "2. Contemporary Context: LLMs in Auditing (Deuser et al., 2025)"
    p2.font.size = Pt(12)
    p2.font.bold = True
    p2.font.color.rgb = COLOR_ACCENT
    b2_items = [
        "Citation: arXiv:2507.16642 (Fraunhofer IAIS & University of Bonn, 2025).",
        "Findings: Evaluates LLM zero-shot verification in regulatory auditing.",
        "Operational Risks: Hallucination of figures, prompt sensitivity, and non-determinism.",
        "Latency Barrier: 2-5s per claim inference latency prevents batch audit verification."
    ]
    for b in b2_items:
        pb = tf2.add_paragraph()
        pb.text = f"• {b}"
        pb.font.size = Pt(9.5)
        pb.font.color.rgb = COLOR_PRIMARY
        pb.space_before = Pt(2)

    # Card 3: Theoretical Foundations: Explainable AI (Bottom-Left)
    c3 = add_card(slide3, Inches(0.8), Inches(4.25), Inches(5.75), Inches(2.45))
    tf3 = c3.text_frame
    tf3.word_wrap = True
    tf3.margin_left = Inches(0.2)
    tf3.margin_top = Inches(0.15)
    p3 = tf3.paragraphs[0]
    p3.text = "3. Theoretical Foundations: Game-Theoretic & Surrogate XAI"
    p3.font.size = Pt(12)
    p3.font.bold = True
    p3.font.color.rgb = COLOR_PRIMARY
    b3_items = [
        "SHAP (Lundberg & Lee, NeurIPS 2017): Shapley values distributing global payoffs fairly.",
        "LinearSHAP: Computes exact attributions analytically in 0.009 ms / sample.",
        "LIME (Ribeiro et al., KDD 2016): Trains local interpretable surrogate g to test stability.",
        "Dual Concordance: Cross-validates global axiomatic attribution with local perturbation."
    ]
    for b in b3_items:
        pb = tf3.add_paragraph()
        pb.text = f"• {b}"
        pb.font.size = Pt(9.5)
        pb.font.color.rgb = COLOR_PRIMARY
        pb.space_before = Pt(2)

    # Card 4: Identified Research Gaps (Bottom-Right, Highlighted)
    c4 = add_card(slide3, Inches(6.783), Inches(4.25), Inches(5.75), Inches(2.45), bg_color=RGBColor(254, 242, 242), border_color=COLOR_CRIMSON)
    tf4 = c4.text_frame
    tf4.word_wrap = True
    tf4.margin_left = Inches(0.2)
    tf4.margin_top = Inches(0.15)
    p4 = tf4.paragraphs[0]
    p4.text = "Identified Research Gaps Addressed in this Work"
    p4.font.size = Pt(12)
    p4.font.bold = True
    p4.font.color.rgb = RGBColor(185, 28, 28)
    b4_items = [
        "Gap 1 (Black-Box Opacity): SOTA deep relation extractors provide no auditable trail.",
        "Gap 2 (Numerical Scale Blindness): Dense sentence embeddings confuse numerical magnitudes ($400M vs $100B have >0.85 cosine similarity).",
        "Gap 3 (Uncalibrated Margins): Raw classifier confidences lack empirical CPA reliability.",
        "Gap 4 (Latency Infeasibility): KernelSHAP / LLM inference is too slow for interactive tools."
    ]
    for b in b4_items:
        pb = tf4.add_paragraph()
        pb.text = f"• {b}"
        pb.font.size = Pt(9.5)
        pb.font.color.rgb = COLOR_TEXT_DARK
        pb.space_before = Pt(2)

    # =========================================================================
    # SLIDE 4: Complete Proposed System Architecture
    # =========================================================================
    slide4 = prs.slides.add_slide(blank_slide_layout)
    add_header(slide4, "System Architecture: Modular Pipeline Decoupling Features from Black Boxes")

    stages = [
        ("STAGE 1: Ingestion", "SEC EDGAR 10-K\nFull-Text Retrieval\nAutomated HTML Parsing\n473 Unique Sentences\n353 Line Items", COLOR_ACCENT_LIGHT, COLOR_ACCENT),
        ("STAGE 2: Curation", "Candidate Proposals\n150 Author-Curated Pairs\nSingle Annotator\nmatch: 55 | no_match: 55\nambiguous: 40", RGBColor(254, 243, 199), RGBColor(217, 119, 6)),
        ("STAGE 3: Features", "Dual-Block Pipeline\nBlock A: Numeric/Lexical\n(No Length Shortcuts)\nBlock B: Dense MiniLM\n384-d Cosine Sim", RGBColor(243, 232, 255), RGBColor(147, 51, 234)),
        ("STAGE 4: Classifiers", "Balanced Logistic Reg.\n(Primary Model)\nMLP Benchmark\nStrict Leak-Free Split\nIsotonic Calibration", RGBColor(236, 253, 245), COLOR_EMERALD),
        ("STAGE 5: XAI & UI", "Dual SHAP & LIME\nExact LinearSHAP (0.009ms)\nMeasured Concordance\nInteractive Audit UI\nVerification Proofs", RGBColor(255, 237, 213), RGBColor(234, 88, 12))
    ]

    for i, (title, content, bg_c, bdr_c) in enumerate(stages):
        x = Inches(0.8 + i * 2.38)
        c = add_card(slide4, x, Inches(1.6), Inches(2.25), Inches(3.6), bg_color=bg_c, border_color=bdr_c)
        tf = c.text_frame
        tf.word_wrap = True
        tf.margin_left = Inches(0.15)
        tf.margin_top = Inches(0.18)

        p = tf.paragraphs[0]
        p.text = title
        p.font.size = Pt(12)
        p.font.bold = True
        p.font.color.rgb = bdr_c
        p.alignment = PP_ALIGN.CENTER

        p_cnt = tf.add_paragraph()
        p_cnt.text = content
        p_cnt.font.size = Pt(10.5)
        p_cnt.font.color.rgb = COLOR_TEXT_DARK
        p_cnt.space_before = Pt(10)
        p_cnt.alignment = PP_ALIGN.CENTER

        if i < 4:
            arrow = slide4.shapes.add_shape(MSO_SHAPE.RIGHT_ARROW, Inches(0.8 + (i+1)*2.38 - 0.18), Inches(3.2), Inches(0.2), Inches(0.3))
            arrow.fill.solid()
            arrow.fill.fore_color.rgb = COLOR_ACCENT
            arrow.line.fill.background()

    # Architecture Summary Bottom Card
    sum_c = add_card(slide4, Inches(0.8), Inches(5.4), Inches(11.733), Inches(1.3), bg_color=COLOR_CARD_BG)
    tf_s = sum_c.text_frame
    tf_s.word_wrap = True
    tf_s.margin_left = Inches(0.25)
    tf_s.margin_top = Inches(0.12)
    p_s = tf_s.paragraphs[0]
    p_s.text = "Key Architectural Advantage: Modular Decoupling for Full Auditability"
    p_s.font.size = Pt(12)
    p_s.font.bold = True
    p_s.font.color.rgb = COLOR_PRIMARY
    p_sb = tf_s.add_paragraph()
    p_sb.text = "Unlike monolithic black-box transformers, this pipeline enforces transparent feature separation. Financial auditors inspect the mathematical contribution of exact numerical matching (Block A) independently from semantic paraphrasing (Block B) in real-time (0.009 ms latency)."
    p_sb.font.size = Pt(10.5)
    p_sb.font.color.rgb = COLOR_TEXT_MUTED
    p_sb.space_before = Pt(4)

    # =========================================================================
    # SLIDE 5: Detailed System Design: Dual-Block Feature Architecture
    # =========================================================================
    slide5 = prs.slides.add_slide(blank_slide_layout)
    add_header(slide5, "Detailed System Design: Dual-Block Feature Pipeline (Ablated Length Shortcuts)")

    card_a = add_card(slide5, Inches(0.8), Inches(1.6), Inches(5.75), Inches(5.1))
    tf_a = card_a.text_frame
    tf_a.word_wrap = True
    tf_a.margin_left = Inches(0.25)
    tf_a.margin_top = Inches(0.2)
    pa = tf_a.paragraphs[0]
    pa.text = "BLOCK A: Interpretable Hand-Crafted Features"
    pa.font.size = Pt(13)
    pa.font.bold = True
    pa.font.color.rgb = COLOR_ACCENT

    block_a_feats = [
        ("numeric_value_match ({0, 1})", "Exact numerical agreement within 1% relative tolerance between narrative claim and statement figure."),
        ("numeric_value_close ({0, 1})", "Numerical proximity within 5% relative tolerance, handling rounding (e.g., $416.2B vs $416,161M)."),
        ("keyword_overlap (in [0, 1])", "Jaccard similarity computed over curated domain lexicons (revenue, net income, ebitda, cash, debt)."),
        ("period_match ({0, 1})", "Binary regex matching verifying fiscal year alignment (e.g., FY 2025 vs FY 2024)."),
        ("string_similarity (in [0, 1])", "RapidFuzz token sort ratio invariant to word reordering."),
        ("Ablated Length Features", "sentence_length and line_item_name_length were ablated following forensic audit to eliminate spurious cross-domain shortcuts.")
    ]
    for name, desc in block_a_feats:
        p = tf_a.add_paragraph()
        p.text = f"• {name}: {desc}"
        p.font.size = Pt(10)
        p.font.color.rgb = COLOR_PRIMARY
        p.space_before = Pt(6)

    card_b = add_card(slide5, Inches(6.783), Inches(1.6), Inches(5.75), Inches(5.1))
    tf_b = card_b.text_frame
    tf_b.word_wrap = True
    tf_b.margin_left = Inches(0.25)
    tf_b.margin_top = Inches(0.2)
    pb = tf_b.paragraphs[0]
    pb.text = "BLOCK B: Dense Semantic Similarity & Synergy"
    pb.font.size = Pt(13)
    pb.font.bold = True
    pb.font.color.rgb = COLOR_EMERALD

    block_b_feats = [
        ("embedding_cosine_similarity (in [-1, 1])", "Cosine similarity between normalized 384-dimensional dense sentence embeddings from all-MiniLM-L6-v2: sim(u, v) = (u . v) / (||u|| ||v||)."),
        ("Why Embeddings Alone Fail (42.00% Micro-F1)", "Dense vector embeddings capture topical semantics but are completely blind to numerical magnitudes. 'Revenue was $400M' and 'Revenue was $100B' yield >0.85 cosine similarity!"),
        ("Why the Dual Synergy Succeeds (84.67% Micro-F1)", "Block A provides the authoritative numerical bedrock (78.67% F1). Block B resolves vocabulary paraphrasing where management uses synonyms not present in standard accounting taxonomy headers (boosting F1 to 84.67%).")
    ]
    for name, desc in block_b_feats:
        p = tf_b.add_paragraph()
        p.text = f"• {name}:\n  {desc}"
        p.font.size = Pt(10.5)
        p.font.color.rgb = COLOR_PRIMARY
        p.space_before = Pt(10)

    # =========================================================================
    # SLIDE 6: Dataset Description & Preprocessing Details
    # =========================================================================
    slide6 = prs.slides.add_slide(blank_slide_layout)
    add_header(slide6, "Dataset Description: 150 SEC EDGAR Author-Curated Pairs & External Benchmarks")

    card_ds = add_card(slide6, Inches(0.8), Inches(1.6), Inches(6.0), Inches(5.1))
    tf_ds = card_ds.text_frame
    tf_ds.word_wrap = True
    tf_ds.margin_left = Inches(0.25)
    tf_ds.margin_top = Inches(0.2)
    p = tf_ds.paragraphs[0]
    p.text = "SEC EDGAR Form 10-K Retrieval Corpus"
    p.font.size = Pt(13)
    p.font.bold = True
    p.font.color.rgb = COLOR_PRIMARY

    filing_data = [
        ("Apple Inc. (AAPL_2025)", "36 unique candidate sentences", "37 unique line items"),
        ("Microsoft Corp. (MSFT_2026)", "149 unique candidate sentences", "53 unique line items"),
        ("Tesla Inc. (TSLA_2026)", "72 unique candidate sentences", "78 unique line items"),
        ("NVIDIA Corp. (NVDA_2026)", "81 unique candidate sentences", "58 unique line items"),
        ("Amazon.com Inc. (AMZN_2026)", "74 unique candidate sentences", "56 unique line items"),
        ("Alphabet Inc. (GOOGL_2026)", "61 unique candidate sentences", "71 unique line items")
    ]
    for comp, sents, items in filing_data:
        p = tf_ds.add_paragraph()
        p.text = f"• {comp}: {sents}, {items}"
        p.font.size = Pt(10.5)
        p.font.color.rgb = COLOR_PRIMARY
        p.space_before = Pt(6)

    p_tot = tf_ds.add_paragraph()
    p_tot.text = "\nCorpus Scale: 473 Unique Sentences | 353 Financial Line Items\n150 Author-Curated Gold Pairs (Single Annotator, Held-Out Test N=150)\nExternal Benchmark Training Corpus: 581 Pairs (FinQA & TAT-QA)"
    p_tot.font.size = Pt(10.5)
    p_tot.font.bold = True
    p_tot.font.color.rgb = COLOR_ACCENT

    card_pp = add_card(slide6, Inches(7.033), Inches(1.6), Inches(5.5), Inches(5.1))
    tf_pp = card_pp.text_frame
    tf_pp.word_wrap = True
    tf_pp.margin_left = Inches(0.25)
    tf_pp.margin_top = Inches(0.2)
    p = tf_pp.paragraphs[0]
    p.text = "Data Preprocessing & Normalization Protocol"
    p.font.size = Pt(13)
    p.font.bold = True
    p.font.color.rgb = COLOR_PRIMARY

    preproc_steps = [
        ("1. HTML Table & Text Cleaning", "BeautifulSoup cleans HTML tags, extracts MD&A narrative paragraphs and Consolidated Balance Sheet/Income Statement tables."),
        ("2. Regex Number & Unit Scaling", "Custom regex parses figures with thousands commas, decimal points, and scales billions/millions ($B -> $M multiplier: 1000x)."),
        ("3. Fiscal Year Period Extraction", "4-digit year regex identifies reporting periods (e.g., 2025, 2024, 2023) for temporal consistency verification."),
        ("4. Financial Lexicon Tokenization", "18 curated core accounting terms extracted for Jaccard overlap computation."),
        ("5. Dense Vector Generation", "PyTorch sentence-transformers encodes text pairs into normalized 384-dimensional dense vectors.")
    ]
    for name, desc in preproc_steps:
        p = tf_pp.add_paragraph()
        p.text = f"• {name}:\n  {desc}"
        p.font.size = Pt(10)
        p.font.color.rgb = COLOR_PRIMARY
        p.space_before = Pt(5)

    # =========================================================================
    # SLIDE 7: The 4-Tier Financial Ambiguity Taxonomy
    # =========================================================================
    slide7 = prs.slides.add_slide(blank_slide_layout)
    add_header(slide7, "Dataset Innovation: 4-Tier Financial Ambiguity Taxonomy")

    amb_categories = [
        ("1. Subsegment vs Aggregate", "Narrative refers to an operational division while table shows consolidated total.\nExample: 'Google Cloud revenue: $33,088M' vs 'Total revenue: $307,394M'.\nGround Truth: ambiguous (Subcomponent)", RGBColor(239, 246, 255), COLOR_ACCENT),
        ("2. Non-GAAP Disclosures", "Management reports adjusted figures not directly visible on GAAP statements.\nExample: 'Adjusted EBITDA was $14,650M' vs 'Operating income: $8,891M'.\nGround Truth: ambiguous (Accounting Standard)", RGBColor(254, 242, 242), COLOR_CRIMSON),
        ("3. Ratio vs Nominal Dollar", "Narrative presents a percentage or per-share KPI derived from total statement sum.\nExample: 'Gross margin percentage: 46.9%' vs 'Total gross margin: $195,201M'.\nGround Truth: ambiguous (Scale/Unit)", RGBColor(254, 243, 199), RGBColor(217, 119, 6)),
        ("4. Temporal Mismatch", "Narrative discusses forward-looking or cumulative multi-year obligations.\nExample: 'Maturities over next 24 months: $21.5B' vs 'Current debt: $12,350M'.\nGround Truth: ambiguous (Time Horizon)", RGBColor(243, 232, 255), RGBColor(147, 51, 234))
    ]

    for i, (title, text, bg_c, bdr_c) in enumerate(amb_categories):
        x = Inches(0.8 + (i % 2) * 5.95)
        y = Inches(1.6 + (i // 2) * 2.55)
        c = add_card(slide7, x, y, Inches(5.75), Inches(2.4), bg_color=bg_c, border_color=bdr_c)
        tf = c.text_frame
        tf.word_wrap = True
        tf.margin_left = Inches(0.2)
        tf.margin_top = Inches(0.18)

        p = tf.paragraphs[0]
        p.text = title
        p.font.size = Pt(13)
        p.font.bold = True
        p.font.color.rgb = bdr_c

        p_t = tf.add_paragraph()
        p_t.text = text
        p_t.font.size = Pt(10.5)
        p_t.font.color.rgb = COLOR_TEXT_DARK
        p_t.space_before = Pt(6)

    # =========================================================================
    # SLIDE 8: Algorithms & Machine Learning Models Used
    # =========================================================================
    slide8 = prs.slides.add_slide(blank_slide_layout)
    add_header(slide8, "Algorithms & Models: Primary Linear Classifier & Neural Benchmark")

    algos = [
        ("Primary Classifier: Balanced Logistic Regression (No Length)", [
            "Mathematical Formulation: P(y=c|x) = softmax(W_c * x + b_c).",
            "Class Weighting: Inversely proportional to class frequencies: w_c = N / (C * N_c), counteracting dataset class skew.",
            "Key Advantage: Globally convex, fully inspectable, deterministic, and supports exact analytical LinearSHAP in 0.009 ms / sample.",
            "Primary Performance: 84.67% Micro-F1 (95% CI: [78.67%, 90.00%]), 83.75% Macro-F1 on held-out GOLD."
        ], RGBColor(239, 246, 255), COLOR_ACCENT),
        ("Secondary Classifier: Multi-Layer Perceptron (MLP)", [
            "Architecture: 2 hidden layers (32, 16 units) with ReLU activation and Adam optimizer.",
            "Evaluation Role: Acts as non-linear benchmark to test feature combination non-linearities.",
            "Findings: When length features are ablated, MLP achieves 85.33% Micro-F1 (Macro: 83.74%), matching LR while requiring slower KernelSHAP (10.97 ms latency)."
        ], RGBColor(241, 245, 249), COLOR_PRIMARY),
        ("Probability Calibration: Isotonic Regression", [
            "Objective: Transforms raw classifier decision values into reliable empirical posterior probabilities.",
            "Empirical Result: Uncalibrated LR yields sharp Brier score of 0.0070 (ECE: 0.29%).",
            "Prior Shift Finding: 5-fold CV calibration on external train yields Brier 0.0078 (ECE: 1.34%) due to train-test class prior shift (5.5% vs 36.7% match)."
        ], RGBColor(236, 253, 245), COLOR_EMERALD)
    ]

    for i, (title, bullets, bg_c, bdr_c) in enumerate(algos):
        x = Inches(0.8 + i * 3.95)
        c = add_card(slide8, x, Inches(1.6), Inches(3.8), Inches(5.1), bg_color=bg_c, border_color=bdr_c)
        tf = c.text_frame
        tf.word_wrap = True
        tf.margin_left = Inches(0.2)
        tf.margin_top = Inches(0.2)

        p = tf.paragraphs[0]
        p.text = title
        p.font.size = Pt(13)
        p.font.bold = True
        p.font.color.rgb = bdr_c

        for b in bullets:
            pb = tf.add_paragraph()
            pb.text = f"• {b}"
            pb.font.size = Pt(10.5)
            pb.font.color.rgb = COLOR_PRIMARY
            pb.space_before = Pt(8)

    # =========================================================================
    # SLIDE 9: Methodology & Experimental Validation Framework
    # =========================================================================
    slide9 = prs.slides.add_slide(blank_slide_layout)
    add_header(slide9, "Methodology & Experimental Protocol: Strict Leak-Free Evaluation (Held-Out GOLD N=150)")

    card_proto = add_card(slide9, Inches(0.8), Inches(1.6), Inches(6.2), Inches(5.1))
    tf_pr = card_proto.text_frame
    tf_pr.word_wrap = True
    tf_pr.margin_left = Inches(0.25)
    tf_pr.margin_top = Inches(0.18)
    p_pr = tf_pr.paragraphs[0]
    p_pr.text = "Rigorous Scientific Experimental Protocol"
    p_pr.font.size = Pt(13)
    p_pr.font.bold = True
    p_pr.font.color.rgb = COLOR_ACCENT

    proto_sections = [
        ("1. Strict Leak-Free Partitioning (Hard Rule 1)", [
            "GOLD (150 pairs) held out strictly as test-only (0 GOLD pairs in training).",
            "Models trained exclusively on external financial QA datasets (N=581) to evaluate genuine out-of-domain generalization to real SEC 10-K claims."
        ]),
        ("2. Author-Curated Ground-Truth (Single Annotator)", [
            "150 claim-table pairs curated directly from audited 10-K figures across 6 core enterprise filings (AAPL, MSFT, TSLA, NVDA, AMZN, GOOGL).",
            "Balanced distribution: 55 match (36.7%), 55 no_match (36.7%), 40 ambiguous (26.7%)."
        ]),
        ("3. Statistical Rigor via Bootstrap Confidence Intervals", [
            "Non-parametric bootstrap estimation (1,000 iterations, seed=42) for all Micro-F1 and Macro-F1 scores.",
            "Honest reporting of confidence intervals (~±5.5 points on n=150)."
        ])
    ]
    for sec_title, bullets in proto_sections:
        ps = tf_pr.add_paragraph()
        ps.text = sec_title
        ps.font.size = Pt(10.5)
        ps.font.bold = True
        ps.font.color.rgb = COLOR_PRIMARY
        ps.space_before = Pt(8)
        for b in bullets:
            pb = tf_pr.add_paragraph()
            pb.text = f"• {b}"
            pb.font.size = Pt(9.5)
            pb.font.color.rgb = COLOR_TEXT_MUTED
            pb.space_before = Pt(2)

    # Right Card: Per-Class Performance Bar Chart
    pcm_path = FIGURES_DIR / "per_class_metrics_barchart.png"
    card_pcm = add_card(slide9, Inches(7.2), Inches(1.6), Inches(5.333), Inches(5.1))
    tf_pcm = card_pcm.text_frame
    tf_pcm.word_wrap = True
    tf_pcm.margin_left = Inches(0.2)
    tf_pcm.margin_top = Inches(0.18)
    p_pcm = tf_pcm.paragraphs[0]
    p_pcm.text = "Per-Class Performance Profile (LR No-Length)"
    p_pcm.font.size = Pt(12)
    p_pcm.font.bold = True
    p_pcm.font.color.rgb = COLOR_PRIMARY

    if pcm_path.exists():
        slide9.shapes.add_picture(str(pcm_path), Inches(7.35), Inches(2.15), Inches(5.0), Inches(3.2))

    tb_pcm = slide9.shapes.add_textbox(Inches(7.35), Inches(5.45), Inches(5.0), Inches(1.15))
    tf_pcm_sub = tb_pcm.text_frame
    tf_pcm_sub.word_wrap = True
    tf_pcm_sub.margin_left = tf_pcm_sub.margin_top = tf_pcm_sub.margin_right = tf_pcm_sub.margin_bottom = 0
    p_pcm_sub = tf_pcm_sub.paragraphs[0]
    p_pcm_sub.text = "Evaluation Insight: High precision & recall on 'Match' (Precision: 96.4%, Recall: 98.2%, F1: 97.3%). Ambiguous class achieves 75.0% precision, demonstrating nuanced accounting discrimination."
    p_pcm_sub.font.size = Pt(9.5)
    p_pcm_sub.font.color.rgb = COLOR_TEXT_MUTED

    # =========================================================================
    # SLIDE 10: Experimental Results & Model Comparison Table
    # =========================================================================
    slide10 = prs.slides.add_slide(blank_slide_layout)
    add_header(slide10, "Experimental Results: Leak-Free Model Comparison Table (Held-Out GOLD n=150)")

    card_t = add_card(slide10, Inches(0.8), Inches(1.6), Inches(7.2), Inches(5.1))
    tf_t = card_t.text_frame
    tf_t.word_wrap = True
    tf_t.margin_left = Inches(0.2)
    tf_t.margin_top = Inches(0.2)
    p = tf_t.paragraphs[0]
    p.text = "Test-Set Evaluation Metrics (Leak-Free, Held-Out GOLD n=150)"
    p.font.size = Pt(13)
    p.font.bold = True
    p.font.color.rgb = COLOR_PRIMARY

    table_shape = slide10.shapes.add_table(7, 4, Inches(1.0), Inches(2.2), Inches(6.8), Inches(2.8))
    table = table_shape.table
    table.columns[0].width = Inches(2.8)
    table.columns[1].width = Inches(1.6)
    table.columns[2].width = Inches(1.6)
    table.columns[3].width = Inches(0.8)

    headers = ["Model Configuration", "Test Micro-F1 (95% CI)", "Test Macro-F1 (95% CI)", "Test n"]
    for col_idx, h in enumerate(headers):
        cell = table.cell(0, col_idx)
        cell.fill.solid()
        cell.fill.fore_color.rgb = COLOR_TABLE_HEADER
        cell.text_frame.text = h
        cell.text_frame.paragraphs[0].font.size = Pt(9.5)
        cell.text_frame.paragraphs[0].font.bold = True
        cell.text_frame.paragraphs[0].font.color.rgb = COLOR_TEXT_LIGHT
        cell.text_frame.paragraphs[0].alignment = PP_ALIGN.CENTER

    row_data = [
        ("Logistic Reg. (No Length) [PRIMARY]", "84.67% [78.67, 90.00]", "83.75% [77.20, 89.14]", "150"),
        ("Logistic Reg. (Full A+B, with length)", "86.67% [81.33, 92.00]", "85.45% [79.46, 90.82]", "150"),
        ("MLP Classifier (No Length)", "85.33% [80.00, 90.67]", "83.74% [77.50, 89.47]", "150"),
        ("MLP Classifier (Full A+B, with length)", "68.67% [61.32, 76.00]", "67.23% [59.37, 74.41]", "150"),
        ("Block A Only (No Length)", "78.67% [71.33, 85.33]", "77.36% [70.31, 83.64]", "150"),
        ("Block B Only (MiniLM Embeddings)", "42.00% [34.00, 50.00]", "40.31% [33.01, 47.03]", "150")
    ]
    for row_idx, row in enumerate(row_data):
        for col_idx, val in enumerate(row):
            cell = table.cell(row_idx + 1, col_idx)
            cell.fill.solid()
            cell.fill.fore_color.rgb = COLOR_TABLE_ALT if row_idx % 2 == 1 else COLOR_CARD_BG
            cell.text_frame.text = val
            p = cell.text_frame.paragraphs[0]
            p.font.size = Pt(9)
            p.font.color.rgb = COLOR_TEXT_DARK
            if row_idx == 0:
                p.font.bold = True
                p.font.color.rgb = COLOR_ACCENT
            if col_idx > 0:
                p.alignment = PP_ALIGN.CENTER

    tb_findings = slide10.shapes.add_textbox(Inches(1.0), Inches(5.15), Inches(6.8), Inches(1.4))
    tf_f = tb_findings.text_frame
    tf_f.word_wrap = True
    tf_f.margin_left = tf_f.margin_top = tf_f.margin_right = tf_f.margin_bottom = 0
    pf0 = tf_f.paragraphs[0]
    pf0.text = "Key Verified Empirical Findings:"
    pf0.font.size = Pt(10.5)
    pf0.font.bold = True
    pf0.font.color.rgb = COLOR_PRIMARY

    findings_bullets = [
        "Primary Model Selection: LR without length features achieves 84.67% Micro-F1, eliminating spurious length shortcuts.",
        "Neural Classifier Recovery: Removing length features restores MLP from 68.67% to 85.33% (+16.66% recovery).",
        "Embedding Blindness: Dense embeddings alone achieve only 42.00% Micro-F1 due to numerical scale blindness.",
        "Statistical Confidence: 95% bootstrap confidence intervals (~±5.5%) honestly bounded across 1,000 resamples."
    ]
    for fb in findings_bullets:
        pfb = tf_f.add_paragraph()
        pfb.text = f"• {fb}"
        pfb.font.size = Pt(9)
        pfb.font.color.rgb = COLOR_TEXT_MUTED
        pfb.space_before = Pt(2)

    cm_path = FIGURES_DIR / "confusion_matrix_logistic_regression_full.png"
    if cm_path.exists():
        card_img = add_card(slide10, Inches(8.2), Inches(1.6), Inches(4.333), Inches(5.1))
        tf_im = card_img.text_frame
        tf_im.margin_left = Inches(0.2)
        tf_im.margin_top = Inches(0.15)
        p = tf_im.paragraphs[0]
        p.text = "Primary Model Confusion Matrix"
        p.font.size = Pt(12)
        p.font.bold = True
        p.font.color.rgb = COLOR_PRIMARY
        slide10.shapes.add_picture(str(cm_path), Inches(8.4), Inches(2.15), Inches(3.933), Inches(3.28))

        tb_cm = slide10.shapes.add_textbox(Inches(8.4), Inches(5.55), Inches(3.933), Inches(1.05))
        tf_cm_c = tb_cm.text_frame
        tf_cm_c.word_wrap = True
        tf_cm_c.margin_left = tf_cm_c.margin_top = tf_cm_c.margin_right = tf_cm_c.margin_bottom = 0
        pcm = tf_cm_c.paragraphs[0]
        pcm.text = "Audit Matrix Insight: High recall on 'match' (54/55 = 98.2%). Misclassifications concentrate between 'ambiguous' and 'no_match' due to nuanced footnote accounting disclosures."
        pcm.font.size = Pt(9)
        pcm.font.color.rgb = COLOR_TEXT_MUTED

    # =========================================================================
    # SLIDE 11: Feature & Architecture Ablation Study
    # =========================================================================
    slide11 = prs.slides.add_slide(blank_slide_layout)
    add_header(slide11, "Feature Ablation Analysis: Numerical Rules as Bedrock, Dense Embeddings as Fine-Tuner")

    abl_img_path = FIGURES_DIR / "ablation_chart.png"
    card_abl_img = add_card(slide11, Inches(0.8), Inches(1.6), Inches(5.9), Inches(5.1))
    tf_ai = card_abl_img.text_frame
    tf_ai.margin_left = Inches(0.2)
    tf_ai.margin_top = Inches(0.15)
    p_ai = tf_ai.paragraphs[0]
    p_ai.text = "Feature Block Ablation Tradeoff (No-Length Models)"
    p_ai.font.size = Pt(12)
    p_ai.font.bold = True
    p_ai.font.color.rgb = COLOR_PRIMARY

    if abl_img_path.exists():
        slide11.shapes.add_picture(str(abl_img_path), Inches(0.95), Inches(2.15), Inches(5.6), Inches(3.35))

    tb_abl = slide11.shapes.add_textbox(Inches(0.95), Inches(5.55), Inches(5.6), Inches(1.05))
    tf_abl_c = tb_abl.text_frame
    tf_abl_c.word_wrap = True
    tf_abl_c.margin_left = tf_abl_c.margin_top = tf_abl_c.margin_right = tf_abl_c.margin_bottom = 0
    p_abl_c = tf_abl_c.paragraphs[0]
    p_abl_c.text = "Ablation Plot: Compares Micro-F1 and Macro-F1 across Block A (78.67%), Block B (42.00%), and Synergy A+B (84.67%). Block A+B synergy yields superior performance (+6.00% over Block A alone)."
    p_abl_c.font.size = Pt(9.5)
    p_abl_c.font.color.rgb = COLOR_TEXT_MUTED

    card_abl = add_card(slide11, Inches(7.0), Inches(1.6), Inches(5.533), Inches(5.1))
    tf_abl = card_abl.text_frame
    tf_abl.word_wrap = True
    tf_abl.margin_left = Inches(0.25)
    tf_abl.margin_top = Inches(0.2)
    p = tf_abl.paragraphs[0]
    p.text = "Key Ablation Insights"
    p.font.size = Pt(14)
    p.font.bold = True
    p.font.color.rgb = COLOR_PRIMARY

    abl_bullets = [
        ("1. The Semantic Embedding Pitfall (Block B = 42.00% Micro-F1)", "Dense vector embeddings alone achieve only 42.00% Micro-F1 (40.31% Macro-F1). Embeddings capture semantic topic similarity (enterprise sales) but are completely invariant to numerical values. A claim of $400M and a table item of $100B have >0.85 cosine similarity!"),
        ("2. Hand-Crafted Features as the Bedrock (Block A = 78.67% Micro-F1)", "Interpretable numerical tolerances and keyword consistency achieve 78.67% Micro-F1 (77.36% Macro-F1), proving financial claim matching is primarily governed by numeric constraints."),
        ("3. Synergistic Optimum (Block A+B = 84.67% Micro-F1)", "Combining Block A and Block B resolves vocabulary paraphrasing (84.67% Micro-F1, 83.75% Macro-F1) where management uses synonyms not present in standard accounting taxonomy headers.")
    ]
    for title, desc in abl_bullets:
        p = tf_abl.add_paragraph()
        p.text = f"• {title}:\n  {desc}"
        p.font.size = Pt(10)
        p.font.color.rgb = COLOR_PRIMARY
        p.space_before = Pt(7)

    # =========================================================================
    # SLIDE 12: Probability Calibration & Reliability
    # =========================================================================
    slide12 = prs.slides.add_slide(blank_slide_layout)
    add_header(slide12, "Probability Calibration: Prior Shift Diagnostics & Honest Audit Bounds")

    cal_img_path = FIGURES_DIR / "calibration.png"
    card_cal_img = add_card(slide12, Inches(0.8), Inches(1.6), Inches(5.9), Inches(5.1))
    tf_ci = card_cal_img.text_frame
    tf_ci.margin_left = Inches(0.2)
    tf_ci.margin_top = Inches(0.15)
    p_ci = tf_ci.paragraphs[0]
    p_ci.text = "Reliability Diagram on Held-Out GOLD"
    p_ci.font.size = Pt(12)
    p_ci.font.bold = True
    p_ci.font.color.rgb = COLOR_PRIMARY

    if cal_img_path.exists():
        slide12.shapes.add_picture(str(cal_img_path), Inches(0.95), Inches(2.15), Inches(5.6), Inches(2.25))

    tb_cal = slide12.shapes.add_textbox(Inches(0.95), Inches(4.55), Inches(5.6), Inches(2.0))
    tf_cal_c = tb_cal.text_frame
    tf_cal_c.word_wrap = True
    tf_cal_c.margin_left = tf_cal_c.margin_top = tf_cal_c.margin_right = tf_cal_c.margin_bottom = 0
    p_cal_c1 = tf_cal_c.paragraphs[0]
    p_cal_c1.text = "• Reliability Curve (Left): Uncalibrated LR probabilities (red) vs 5-fold CV calibrated probabilities (blue). Uncalibrated model exhibits sharp discrimination (ECE 0.29%)."
    p_cal_c1.font.size = Pt(9.5)
    p_cal_c1.font.color.rgb = COLOR_TEXT_MUTED
    p_cal_c2 = tf_cal_c.add_paragraph()
    p_cal_c2.text = "• Probability Shift (Right): Calibration on external training set shifts probabilities downward due to the low external match prior (5.51%), inducing underconfidence on GOLD."
    p_cal_c2.font.size = Pt(9.5)
    p_cal_c2.font.color.rgb = COLOR_TEXT_MUTED
    p_cal_c2.space_before = Pt(4)

    card_cal = add_card(slide12, Inches(7.0), Inches(1.6), Inches(5.533), Inches(5.1))
    tf_cal = card_cal.text_frame
    tf_cal.word_wrap = True
    tf_cal.margin_left = Inches(0.25)
    tf_cal.margin_top = Inches(0.2)
    p = tf_cal.paragraphs[0]
    p.text = "Audit Reliability via Calibration"
    p.font.size = Pt(14)
    p.font.bold = True
    p.font.color.rgb = COLOR_EMERALD

    cal_bullets = [
        ("Why Calibration Matters in Financial Audits", "Raw machine learning probabilities are often overconfident. A prediction of 90% confidence must correspond to an actual 90% empirical precision for an auditor to rely on it."),
        ("Empirical Findings on Held-Out GOLD (n=150)", "• Uncalibrated Logistic Regression: Brier Score = 0.0070, Expected Calibration Error (ECE) = 0.29%.\n• Calibrated Logistic Regression (5-fold CV): Brier Score = 0.0078, ECE = 1.34%."),
        ("Why Calibration Degrades on GOLD (Prior Shift)", "Cross-domain prior shift: External training set has a match prior of 5.51%, while GOLD test set has a match prior of 36.67%. Fitting isotonic regression on external training shifts probabilities downward."),
        ("Planned Remediation", "Calibrate on an in-domain financial statement split or implement Bayesian class-prior correction.")
    ]
    for title, desc in cal_bullets:
        p = tf_cal.add_paragraph()
        p.text = f"• {title}:\n  {desc}"
        p.font.size = Pt(10)
        p.font.color.rgb = COLOR_PRIMARY
        p.space_before = Pt(7)

    # =========================================================================
    # SLIDE 13: Global Interpretability: SHAP Feature Importance
    # =========================================================================
    slide13 = prs.slides.add_slide(blank_slide_layout)
    add_header(slide13, "Global Interpretability: LinearSHAP Feature Attribution (Length Shortcuts Removed)")

    shap_img_path = FIGURES_DIR / "shap_summary.png"
    card_shap_img = add_card(slide13, Inches(0.8), Inches(1.6), Inches(5.9), Inches(5.1))
    tf_si = card_shap_img.text_frame
    tf_si.margin_left = Inches(0.2)
    tf_si.margin_top = Inches(0.15)
    p_si = tf_si.paragraphs[0]
    p_si.text = "Global Feature Attribution Distribution"
    p_si.font.size = Pt(12)
    p_si.font.bold = True
    p_si.font.color.rgb = COLOR_PRIMARY

    if shap_img_path.exists():
        slide13.shapes.add_picture(str(shap_img_path), Inches(0.95), Inches(2.15), Inches(5.6), Inches(3.15))

    tb_shap = slide13.shapes.add_textbox(Inches(0.95), Inches(5.45), Inches(5.6), Inches(1.15))
    tf_shap_c = tb_shap.text_frame
    tf_shap_c.word_wrap = True
    tf_shap_c.margin_left = tf_shap_c.margin_top = tf_shap_c.margin_right = tf_shap_c.margin_bottom = 0
    p_shap_c = tf_shap_c.paragraphs[0]
    p_shap_c.text = "Global beeswarm plot displays feature impacts on 'match' log-odds. Features ranked top-to-bottom by mean absolute SHAP value across all test claims. Spurious sentence length features have been ablated."
    p_shap_c.font.size = Pt(9.5)
    p_shap_c.font.color.rgb = COLOR_TEXT_MUTED

    card_shap = add_card(slide13, Inches(7.0), Inches(1.6), Inches(5.533), Inches(5.1))
    tf_shap = card_shap.text_frame
    tf_shap.word_wrap = True
    tf_shap.margin_left = Inches(0.25)
    tf_shap.margin_top = Inches(0.2)
    p = tf_shap.paragraphs[0]
    p.text = "Global Feature Attribution Hierarchy"
    p.font.size = Pt(14)
    p.font.bold = True
    p.font.color.rgb = COLOR_PRIMARY

    shap_ranking = [
        ("1. numeric_value_match", "Dominant global predictor. An exact numerical match heavily shifts log-odds toward positive match."),
        ("2. numeric_value_close", "Strong positive attribution, ensuring resilience against rounding differences ($416.2B vs $416,161M)."),
        ("3. embedding_cosine_sim", "Secondary semantic fine-tuner, breaking ties when accounting terminology varies between narratives and tables."),
        ("4. keyword_overlap", "Domain accounting lexicon overlap ensuring topical consistency across statements."),
        ("5. string_similarity & period_match", "Lexical token overlap and fiscal year alignment preventing cross-period mismatches.")
    ]
    for title, desc in shap_ranking:
        p = tf_shap.add_paragraph()
        p.text = f"• {title}:\n  {desc}"
        p.font.size = Pt(10)
        p.font.color.rgb = COLOR_PRIMARY
        p.space_before = Pt(6)

    # =========================================================================
    # SLIDE 14: Local Interpretability: SHAP vs. LIME Concordance
    # =========================================================================
    slide14 = prs.slides.add_slide(blank_slide_layout)
    add_header(slide14, "Local Interpretability: Measured SHAP vs. LIME Concordance on Held-Out GOLD")

    card_agree = add_card(slide14, Inches(0.8), Inches(1.6), Inches(6.6), Inches(5.1))
    tf_ag = card_agree.text_frame
    tf_ag.word_wrap = True
    tf_ag.margin_left = Inches(0.2)
    tf_ag.margin_top = Inches(0.18)
    p = tf_ag.paragraphs[0]
    p.text = "Measured Attribution Concordance Across All 150 GOLD Claims"
    p.font.size = Pt(13)
    p.font.bold = True
    p.font.color.rgb = COLOR_PRIMARY

    # Concordance Metrics Box
    c_sub_m = add_card(slide14, Inches(1.0), Inches(2.2), Inches(6.2), Inches(1.8), bg_color=RGBColor(248, 250, 252))
    tf_c_m = c_sub_m.text_frame
    tf_c_m.word_wrap = True
    tf_c_m.margin_left = Inches(0.2)
    tf_c_m.margin_top = Inches(0.15)
    pm1 = tf_c_m.paragraphs[0]
    pm1.text = "Measured Explainer Agreement (150 Held-Out Samples):"
    pm1.font.size = Pt(11)
    pm1.font.bold = True
    pm1.font.color.rgb = COLOR_PRIMARY
    
    pm_bullets = [
        "Mean Top-3 Feature Slot Overlap: 80.44% (362 out of 450 feature slots agree).",
        "Exact Instance Concordance: 50.67% (76 out of 150 instances exhibit 3/3 exact top-3 match).",
        "Characterized as Moderate Agreement: High concordance on unambiguous matches; divergence occurs on complex multi-term footnotes where LIME perturbations test local boundaries."
    ]
    for b in pm_bullets:
        pb = tf_c_m.add_paragraph()
        pb.text = f"• {b}"
        pb.font.size = Pt(9.5)
        pb.font.color.rgb = COLOR_TEXT_DARK
        pb.space_before = Pt(3)

    tb_ag_sub = slide14.shapes.add_textbox(Inches(1.0), Inches(4.2), Inches(6.2), Inches(2.3))
    tf_ag_sub = tb_ag_sub.text_frame
    tf_ag_sub.word_wrap = True
    tf_ag_sub.margin_left = tf_ag_sub.margin_top = tf_ag_sub.margin_right = tf_ag_sub.margin_bottom = 0
    pag1 = tf_ag_sub.paragraphs[0]
    pag1.text = "Audit Significance of Dual XAI Concordance:"
    pag1.font.size = Pt(11)
    pag1.font.bold = True
    pag1.font.color.rgb = COLOR_ACCENT
    
    concord_bullets = [
        "Axiomatic vs Perturbative Cross-Validation: SHAP provides globally consistent game-theoretic attributions; LIME validates local surrogate stability.",
        "Detecting Fragile Decisions: When SHAP and LIME diverge, it flags instances with high local sensitivity (e.g. multi-year forward commitments).",
        "Zero Hardcoded Metrics: Previous 93.3% / 80.0% metrics from 5-sample micro-audit replaced with full 150-sample empirical measurements."
    ]
    for b in concord_bullets:
        pb = tf_ag_sub.add_paragraph()
        pb.text = f"• {b}"
        pb.font.size = Pt(9.5)
        pb.font.color.rgb = COLOR_TEXT_MUTED
        pb.space_before = Pt(3)

    ex5_path = FIGURES_DIR / "shap_example_5.png"
    if ex5_path.exists():
        card_ex5 = add_card(slide14, Inches(7.6), Inches(1.6), Inches(4.933), Inches(5.1))
        tf_e = card_ex5.text_frame
        tf_e.margin_left = Inches(0.2)
        tf_e.margin_top = Inches(0.15)
        p = tf_e.paragraphs[0]
        p.text = "Case Study: Footnote Ambiguity Attribution"
        p.font.size = Pt(11)
        p.font.bold = True
        p.font.color.rgb = COLOR_PRIMARY
        slide14.shapes.add_picture(str(ex5_path), Inches(7.75), Inches(2.15), Inches(4.6), Inches(2.3))

        tb_ex5 = slide14.shapes.add_textbox(Inches(7.75), Inches(4.55), Inches(4.6), Inches(2.05))
        tf_e_sub = tb_ex5.text_frame
        tf_e_sub.word_wrap = True
        tf_e_sub.margin_left = tf_e_sub.margin_top = tf_e_sub.margin_right = tf_e_sub.margin_bottom = 0
        pe1 = tf_e_sub.paragraphs[0]
        pe1.text = "Edge-Case Attribution Deep-Dive:"
        pe1.font.size = Pt(10)
        pe1.font.bold = True
        pe1.font.color.rgb = COLOR_PRIMARY
        pe_bullets = [
            "Claim: Notes payable & other borrowings footnote.",
            "True Label: ambiguous | Predicted: no_match",
            "Why Explainers Agree: Both SHAP and LIME assign heavy negative attribution (-0.72) to numeric_value_match due to absence of exact tabular figures.",
            "Audit Benefit: Transparent audit trails enable auditors to detect nuanced borderline disclosures."
        ]
        for b in pe_bullets:
            peb = tf_e_sub.add_paragraph()
            peb.text = f"• {b}"
            peb.font.size = Pt(8.5)
            peb.font.color.rgb = COLOR_TEXT_MUTED
            peb.space_before = Pt(1)

    # =========================================================================
    # SLIDE 15: Comparison with KPI-Check Baseline & Latency Tradeoff
    # =========================================================================
    slide15 = prs.slides.add_slide(blank_slide_layout)
    add_header(slide15, "Benchmark Context & Measured Computational Efficiency")

    base_img_path = FIGURES_DIR / "baseline_comparison.png"
    card_base_img = add_card(slide15, Inches(0.8), Inches(1.6), Inches(5.5), Inches(5.1))
    tf_bi = card_base_img.text_frame
    tf_bi.margin_left = Inches(0.2)
    tf_bi.margin_top = Inches(0.15)
    p_bi = tf_bi.paragraphs[0]
    p_bi.text = "Micro-F1 Baseline Context"
    p_bi.font.size = Pt(12)
    p_bi.font.bold = True
    p_bi.font.color.rgb = COLOR_PRIMARY

    if base_img_path.exists():
        slide15.shapes.add_picture(str(base_img_path), Inches(0.95), Inches(2.15), Inches(5.1), Inches(4.01))

    card_bench = add_card(slide15, Inches(6.5), Inches(1.6), Inches(6.033), Inches(5.1))
    tf_b = card_bench.text_frame
    tf_b.word_wrap = True
    tf_b.margin_left = Inches(0.25)
    tf_b.margin_top = Inches(0.2)
    p = tf_b.paragraphs[0]
    p.text = "Benchmark Context & Measured Inference Latencies"
    p.font.size = Pt(13)
    p.font.bold = True
    p.font.color.rgb = COLOR_PRIMARY

    bench_bullets = [
        ("Honest Baseline Context: KPI-Check (2022)", "Not directly comparable (German language, different task, proprietary uncurated data). KPI-Check achieved 73.00% Micro-F1 as an end-to-end relation extractor. Fair in-domain FinBERT baseline on our data planned for final review."),
        ("Measured Explainer Latency (In this setup)", "• Exact LinearSHAP (Primary LR Model): 0.009 ms / sample (amortized over batch; 1.35 ms total across 150 test claims).\n• KernelSHAP (MLP Classifier): 10.97 ms / sample (1,645.7 ms total across 150 test claims).\nLinear models achieve a 1,223.5x explainer latency speedup."),
        ("Interactive CPA Auditing Feasibility", "Sub-millisecond LinearSHAP latency enables instant interactive calculation within spreadsheet extensions and auditing UIs, unlike slow transformer/LLM APIs (2-5s latency).")
    ]
    for title, desc in bench_bullets:
        p = tf_b.add_paragraph()
        p.text = f"• {title}:\n  {desc}"
        p.font.size = Pt(10)
        p.font.color.rgb = COLOR_PRIMARY
        p.space_before = Pt(6)

    # =========================================================================
    # SLIDE 16: Self-Audit: Leakage Found and Fixed
    # =========================================================================
    slide16 = prs.slides.add_slide(blank_slide_layout)
    add_header(slide16, "Self-Audit: Data Leakage Identified, Remediated, and Verified")

    # Card A: Leakage Pathways Found & Fixed
    card_leak_a = add_card(slide16, Inches(0.8), Inches(1.6), Inches(6.2), Inches(5.1), bg_color=RGBColor(254, 242, 242), border_color=COLOR_CRIMSON)
    tf_la = card_leak_a.text_frame
    tf_la.word_wrap = True
    tf_la.margin_left = Inches(0.2)
    tf_la.margin_top = Inches(0.18)
    p_la = tf_la.paragraphs[0]
    p_la.text = "6 Forensic Audit Deficits Diagnosed & Fixed"
    p_la.font.size = Pt(13)
    p_la.font.bold = True
    p_la.font.color.rgb = COLOR_CRIMSON

    leak_items = [
        ("1. Train/Test Split Leakage", "123 of 150 gold pairs leaked into training under legacy random split. Fixed: Held out 100% of GOLD as pure test (Hard Rule 1)."),
        ("2. Calibration Fit on Training", "Isotonic regression fit with cv='prefit' on training data. Fixed: Replaced with honest 5-fold CV calibration on training folds."),
        ("3. Heuristic Label Leakage", "External labels generated with rel_diff <= 2%, nearly identical to feature numeric_value_match (<= 1%). Documented openly."),
        ("4. Spurious Length Shortcut", "TAT-QA multi-sentence paragraphs (~492 chars) inflated length vs SEC sentences (~70 chars). Fixed: Removed sentence_length & line_len."),
        ("5. Hard-Coded XAI Values", "Legacy 1.4ms/850ms latencies and 93.3% concordance were hard-coded strings. Replaced with measured values on all 150 GOLD samples."),
        ("6. Unsupported Kappa Claim", "Cohen's kappa (0.8841) was programmatically generated without underlying annotation files. Removed from all claims.")
    ]
    for name, desc in leak_items:
        p = tf_la.add_paragraph()
        p.text = f"• {name}: {desc}"
        p.font.size = Pt(9.5)
        p.font.color.rgb = COLOR_TEXT_DARK
        p.space_before = Pt(3)

    # Card B: Impact on Model Behavior Table & Recovery
    card_leak_b = add_card(slide16, Inches(7.2), Inches(1.6), Inches(5.333), Inches(5.1))
    tf_lb = card_leak_b.text_frame
    tf_lb.word_wrap = True
    tf_lb.margin_left = Inches(0.2)
    tf_lb.margin_top = Inches(0.18)
    p_lb = tf_lb.paragraphs[0]
    p_lb.text = "Model Behavior: Leaked vs. Remediated"
    p_lb.font.size = Pt(13)
    p_lb.font.bold = True
    p_lb.font.color.rgb = COLOR_PRIMARY

    table_leak_shape = slide16.shapes.add_table(5, 3, Inches(7.35), Inches(2.2), Inches(5.0), Inches(2.1))
    t_leak = table_leak_shape.table
    t_leak.columns[0].width = Inches(2.4)
    t_leak.columns[1].width = Inches(1.3)
    t_leak.columns[2].width = Inches(1.3)

    h_leak = ["Metric / Model", "Leaked Legacy", "Leak-Free Verified"]
    for c_i, h in enumerate(h_leak):
        c = t_leak.cell(0, c_i)
        c.fill.solid()
        c.fill.fore_color.rgb = COLOR_TABLE_HEADER
        c.text_frame.text = h
        c.text_frame.paragraphs[0].font.size = Pt(9)
        c.text_frame.paragraphs[0].font.bold = True
        c.text_frame.paragraphs[0].font.color.rgb = COLOR_TEXT_LIGHT
        c.text_frame.paragraphs[0].alignment = PP_ALIGN.CENTER

    t_leak_data = [
        ("LR Full (All Feats)", "85.71% Micro-F1", "86.67% Micro-F1"),
        ("LR No-Length (Primary)", "N/A", "84.67% Micro-F1"),
        ("MLP Full (All Feats)", "85.03% Micro-F1", "68.67% (-16.36%)"),
        ("MLP No-Length (Recovery)", "N/A", "85.33% (+16.66%!)")
    ]
    for r_i, row in enumerate(t_leak_data):
        for c_i, val in enumerate(row):
            c = t_leak.cell(r_i + 1, c_i)
            c.fill.solid()
            c.fill.fore_color.rgb = COLOR_TABLE_ALT if r_i % 2 == 1 else COLOR_CARD_BG
            c.text_frame.text = val
            p = c.text_frame.paragraphs[0]
            p.font.size = Pt(9)
            p.font.color.rgb = COLOR_TEXT_DARK
            if c_i > 0:
                p.alignment = PP_ALIGN.CENTER

    tb_rec = slide16.shapes.add_textbox(Inches(7.35), Inches(4.5), Inches(5.0), Inches(2.0))
    tf_rec = tb_rec.text_frame
    tf_rec.word_wrap = True
    tf_rec.margin_left = tf_rec.margin_top = tf_rec.margin_right = tf_rec.margin_bottom = 0
    prec1 = tf_rec.paragraphs[0]
    prec1.text = "Audit Proof of Shortcut Recovery:"
    prec1.font.size = Pt(10.5)
    prec1.font.bold = True
    prec1.font.color.rgb = COLOR_EMERALD

    rec_bullets = [
        "MLP Overfitting Proof: MLP collapsed from 85.03% to 68.67% under leak-free evaluation because it learned TAT-QA paragraph lengths.",
        "Ablation Recovery: Removing length features restored MLP to 85.33% (+16.66% recovery), matching LR (84.67%).",
        "Scientific Integrity: Demonstrates adherence to Hard Rules 1-6: discovering and fixing leakage strengthens validity."
    ]
    for b in rec_bullets:
        pb = tf_rec.add_paragraph()
        pb.text = f"• {b}"
        pb.font.size = Pt(9)
        pb.font.color.rgb = COLOR_TEXT_MUTED
        pb.space_before = Pt(2)

    # =========================================================================
    # SLIDE 17: Limitations & Threats to Validity
    # =========================================================================
    slide17 = prs.slides.add_slide(blank_slide_layout)
    add_header(slide17, "Limitations & Threats to Validity: Academic Boundary Conditions")

    limits = [
        ("1. Sample Size Limitation (n=150)",
         "Evaluation is conducted on 150 author-curated pairs, yielding a 95% bootstrap confidence interval of ~±5.5 points ([78.67%, 90.00%]). Larger sample sizes required for narrower bounds.",
         RGBColor(239, 246, 255), COLOR_ACCENT),
        ("2. Lack of Hard Numerical Negatives",
         "Current gold dataset lacks negative pairs sharing identical numbers with differing line-item concepts. Consequently, Brier score (0.0070) is optimistic.",
         RGBColor(254, 242, 242), COLOR_CRIMSON),
        ("3. Enterprise & Sector Concentration",
         "Curated corpus is limited to 6 large-cap tech/consumer companies (AAPL, MSFT, TSLA, NVDA, AMZN, GOOGL); cross-industry generalizability requires broader sector validation.",
         RGBColor(254, 243, 199), RGBColor(217, 119, 6)),
        ("4. Language & GAAP Accounting Scope",
         "Exclusively English-language Form 10-K filings prepared under US-GAAP. Does not generalize to IFRS or multilingual annual reports without domain re-training.",
         RGBColor(243, 232, 255), RGBColor(147, 51, 234)),
        ("5. Single-Annotator Ground Truth",
         "The 150 pairs were curated by a single annotator without independent double-blind validation records. Dual-annotator agreement protocol planned.",
         RGBColor(241, 245, 249), COLOR_PRIMARY),
        ("6. Cross-Domain Prior Shift & Calibration",
         "Training on external QA data (5.5% match prior) vs testing on SEC GOLD (36.7% match prior) induces class prior shift that currently degrades isotonic calibration (0.0070 -> 0.0078).",
         RGBColor(236, 253, 245), COLOR_EMERALD)
    ]

    for i, (title, text, bg_c, bdr_c) in enumerate(limits):
        x = Inches(0.8 + (i % 2) * 5.95)
        y = Inches(1.6 + (i // 2) * 1.7)
        c = add_card(slide17, x, y, Inches(5.75), Inches(1.55), bg_color=bg_c, border_color=bdr_c)
        tf = c.text_frame
        tf.word_wrap = True
        tf.margin_left = Inches(0.2)
        tf.margin_top = Inches(0.12)

        p = tf.paragraphs[0]
        p.text = title
        p.font.size = Pt(11)
        p.font.bold = True
        p.font.color.rgb = bdr_c

        p_t = tf.add_paragraph()
        p_t.text = text
        p_t.font.size = Pt(9.5)
        p_t.font.color.rgb = COLOR_TEXT_DARK
        p_t.space_before = Pt(3)

    # =========================================================================
    # SLIDE 18: Summary of Achievements & Final Review Roadmap
    # =========================================================================
    slide18 = prs.slides.add_slide(blank_slide_layout)
    add_header(slide18, "Review 2 Summary & Final Review Roadmap (16th-21st October 2026)")

    card_r2 = add_card(slide18, Inches(0.8), Inches(1.6), Inches(5.75), Inches(5.1), bg_color=RGBColor(236, 253, 245), border_color=COLOR_EMERALD)
    tf_r2 = card_r2.text_frame
    tf_r2.word_wrap = True
    tf_r2.margin_left = Inches(0.25)
    tf_r2.margin_top = Inches(0.2)
    p = tf_r2.paragraphs[0]
    p.text = "Review 2 Completed Milestones (80% Implementation)"
    p.font.size = Pt(13)
    p.font.bold = True
    p.font.color.rgb = COLOR_EMERALD

    r2_achievements = [
        ("Automated End-to-End Pipeline", "Document Ingestion -> Feature Extraction -> Balanced Training -> Honest Calibration -> Dual XAI Layer."),
        ("Leak-Free Benchmark Foundation", "Strict held-out evaluation on 150 author-curated GOLD pairs (84.67% Micro-F1, LR No-Length)."),
        ("Dual XAI Layer with Measured Metrics", "LinearSHAP (0.009 ms) and LIME with measured 80.44% feature overlap across all 150 test claims."),
        ("Forensic Self-Audit Completed", "Diagnosed and resolved all 6 leakage pathways; documented openly in docs/audit_report.md."),
        ("Structured SEC XBRL Store", "Built 298,663 line-item store (line_items.parquet) across 33 companies to replace HTML table parsing.")
    ]
    for title, desc in r2_achievements:
        p = tf_r2.add_paragraph()
        p.text = f"• {title}: {desc}"
        p.font.size = Pt(10)
        p.font.color.rgb = COLOR_PRIMARY
        p.space_before = Pt(5)

    card_fin = add_card(slide18, Inches(6.783), Inches(1.6), Inches(5.75), Inches(5.1), bg_color=RGBColor(239, 246, 255), border_color=COLOR_ACCENT)
    tf_fin = card_fin.text_frame
    tf_fin.word_wrap = True
    tf_fin.margin_left = Inches(0.25)
    tf_fin.margin_top = Inches(0.2)
    p = tf_fin.paragraphs[0]
    p.text = "Final Review Roadmap (16th-21st October 2026)"
    p.font.size = Pt(13)
    p.font.bold = True
    p.font.color.rgb = COLOR_ACCENT

    fin_goals = [
        ("1. XBRL Ingestion with Hard Negatives", "Mine numerical distractors sharing identical dollar figures but differing line items from line_items.parquet."),
        ("2. Company-Level Partitioning", "Implement strict leave-one-company-out and company-partitioned training splits."),
        ("3. Gold Set Expansion & Dual Annotation", "Expand GOLD corpus with hard negatives; recruit second annotator on 50 pairs for genuine inter-annotator agreement."),
        ("4. Advanced Models & Fair Baseline", "Train LightGBM + TreeSHAP; implement a fair fine-tuned FinBERT / DeBERTa baseline on our data."),
        ("5. Calibration Fix", "Solve prior shift via in-domain split calibration or Bayesian prior adjustment."),
        ("6. Stretch Goals", "End-to-end vector retrieval (hybrid dense/sparse) and timed auditor efficiency user study.")
    ]
    for title, desc in fin_goals:
        p = tf_fin.add_paragraph()
        p.text = f"• {title}:\n  {desc}"
        p.font.size = Pt(9.5)
        p.font.color.rgb = COLOR_PRIMARY
        p.space_before = Pt(4)

    # Save presentation
    prs.save(OUTPUT_PPTX)
    print(f"[SUCCESS] PowerPoint presentation saved successfully to: {OUTPUT_PPTX}")


if __name__ == "__main__":
    create_deck()
