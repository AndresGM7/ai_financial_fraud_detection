#!/usr/bin/env python
"""
run_pipeline.py
───────────────
Master entry point — generates data, trains all models, evaluates,
and writes dashboard-ready results to data/results/.

Usage
─────
    python run_pipeline.py                  # full run
    python run_pipeline.py --skip-data-gen  # reuse existing CSV/Parquet
    python run_pipeline.py --llm-samples 50 # more LLM evaluations
"""

from __future__ import annotations

import argparse
import subprocess
import sys


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Fraud Detection Pipeline Runner",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--data-path",
        default="data/raw/labeled_transactions.csv",
        help="Path to labeled transaction CSV",
    )
    parser.add_argument(
        "--llm-samples",
        type=int,
        default=30,
        help="Number of transactions to evaluate through the LLM agent",
    )
    parser.add_argument(
        "--skip-data-gen",
        action="store_true",
        help="Skip synthetic data generation (reuse existing files)",
    )
    args = parser.parse_args()

    SEP = "=" * 60
    print(f"\n{SEP}")
    print("  FINANCIAL FRAUD DETECTION -- FULL PIPELINE")
    print(SEP)

    # Step 1: Generate synthetic data
    if not args.skip_data_gen:
        print("\n▶ Generating synthetic transaction data…")
        subprocess.run(
            [sys.executable, "-m", "src.data_generator"],
            check=True,
        )
    else:
        print("\n▶ Skipping data generation (--skip-data-gen)")

    # Step 2: Train models & evaluate
    print("\n▶ Running training pipeline…")
    from src.training_pipeline import run
    run(data_path=args.data_path, llm_samples=args.llm_samples)


if __name__ == "__main__":
    main()

