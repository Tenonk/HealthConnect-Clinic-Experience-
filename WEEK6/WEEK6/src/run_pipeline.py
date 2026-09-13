#!/usr/bin/env python3
"""
CLI entry point for HealthConnect Integrated ML Pipeline (Week 6).

Usage:
  python -m src.run_pipeline train
  python -m src.run_pipeline validate
  python -m src.run_pipeline infer --input path/to/data.csv
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

# Ensure project root is on path
ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.pipeline import HealthConnectPipeline
from src.utils import setup_logging


def main():
    parser = argparse.ArgumentParser(description="HealthConnect ML Pipeline – Week 6")
    parser.add_argument(
        "command",
        choices=["train", "validate", "infer"],
        help="Pipeline command to execute",
    )
    parser.add_argument(
        "--input",
        type=str,
        default=None,
        help="Input CSV for inference",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="data/processed/predictions.csv",
        help="Output path for predictions",
    )
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="Path to config YAML",
    )
    parser.add_argument(
        "--retrain",
        action="store_true",
        default=True,
        help="Force retraining of models",
    )
    args = parser.parse_args()

    setup_logging(level="INFO")
    pipe = HealthConnectPipeline(config_path=args.config)

    if args.command == "train":
        results = pipe.run_training_pipeline(retrain=args.retrain)
        print("\n=== Training Results ===")
        print(json.dumps(results["models"], indent=2))
        print(f"Features: {len(results['feature_names'])}")
        print(f"Train size: {results['n_train']}, Test size: {results['n_test']}")

    elif args.command == "validate":
        checks = pipe.validate_integration()
        print("\n=== Integration Validation ===")
        for k, v in checks.items():
            print(f"  [{'PASS' if v else 'FAIL'}] {k}")
        all_ok = all(checks.values())
        sys.exit(0 if all_ok else 1)

    elif args.command == "infer":
        if not args.input:
            print("--input is required for infer command")
            sys.exit(1)
        import pandas as pd

        df = pd.read_csv(args.input)
        preds = pipe.run_inference(df)
        out_path = Path(args.output)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        preds.to_csv(out_path, index=False)
        print(f"Predictions saved to {out_path}")
        print(preds.head())


if __name__ == "__main__":
    main()
