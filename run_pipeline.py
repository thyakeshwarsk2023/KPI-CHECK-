#!/usr/bin/env python3
"""
run_pipeline.py

Master execution script for the XAI KPI-Check project.
Runs the complete pipeline end-to-end or individual stages:
  1. Extraction (fetch + parse SEC reports)
  2. Features (Block A hand-crafted + Block B embedding features)
  3. Model (train classifiers + calibration)
  4. XAI (SHAP + LIME interpretability layer)
  5. Evaluation (baseline comparison + ablation study)

Usage:
  python run_pipeline.py --all
  python run_pipeline.py --stage [features|train|calibrate|xai|eval]
"""

import sys
import argparse
import subprocess
import logging
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

STAGES = {
    "extraction": [
        "src/extraction/fetch_reports.py",
        "src/extraction/parse_reports.py"
    ],
    "features": [
        "src/features/build_features.py"
    ],
    "train": [
        "src/model/train.py"
    ],
    "calibrate": [
        "src/model/calibration.py"
    ],
    "xai": [
        "src/xai/shap_explain.py",
        "src/xai/lime_explain.py"
    ],
    "eval": [
        "src/eval/compare_to_baseline.py",
        "src/eval/ablation_table.py",
        "src/eval/cross_validation.py",
        "src/eval/inter_annotator_agreement.py",
        "src/eval/per_class_metrics.py"
    ]
}


def run_script(script_path: str):
    """Executes a Python script in a subprocess and monitors completion."""
    logger.info(f"==> Running: {script_path}")
    res = subprocess.run([sys.executable, script_path], capture_output=False)
    if res.returncode != 0:
        logger.error(f"Execution failed for {script_path} (exit code: {res.returncode})")
        sys.exit(res.returncode)
    logger.info(f"==> Successfully completed: {script_path}\n")


def main():
    parser = argparse.ArgumentParser(description="Master pipeline runner for XAI KPI-Check.")
    parser.add_argument("--all", action="store_true", help="Run the entire pipeline from features through evaluation.")
    parser.add_argument("--stage", choices=list(STAGES.keys()), help="Run a specific stage.")
    args = parser.parse_args()

    if not args.all and not args.stage:
        parser.print_help()
        print("\nDefaulting to running all core experimental stages (features -> train -> calibrate -> xai -> eval)...")
        stages_to_run = ["features", "train", "calibrate", "xai", "eval"]
    elif args.all:
        stages_to_run = list(STAGES.keys())
    else:
        stages_to_run = [args.stage]

    print("=" * 75)
    print(f"{'XAI KPI-CHECK: PIPELINE EXECUTION':^75}")
    print("=" * 75)

    for stage in stages_to_run:
        logger.info(f"--- Starting Stage: [{stage.upper()}] ---")
        for script in STAGES[stage]:
            run_script(script)

    print("=" * 75)
    print(f"{'ALL REQUESTED PIPELINE STAGES COMPLETED SUCCESSFULLY':^75}")
    print("=" * 75)


if __name__ == "__main__":
    main()
