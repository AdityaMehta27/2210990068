"""
Main Pipeline Runner
=====================
Runs the full pipeline end-to-end:
  1. Generate dataset
  2. Train & evaluate all six models
  3. Generate all paper figures

Usage:
  python run_pipeline.py
  python run_pipeline.py --skip-data-gen   (if dataset already exists)
"""

import argparse
import sys
from pathlib import Path

BASE_DIR = Path(__file__).parent
sys.path.insert(0, str(BASE_DIR))


def main():
    parser = argparse.ArgumentParser(description="Crop Yield Prediction Pipeline")
    parser.add_argument("--skip-data-gen", action="store_true",
                        help="Skip dataset generation (use existing CSV)")
    parser.add_argument("--skip-training", action="store_true",
                        help="Skip model training (use existing .pkl files)")
    args = parser.parse_args()

    data_csv = BASE_DIR / "data" / "agro_climatic_dataset.csv"

    # ── Step 1: Generate Dataset ─────────────────────────────────────────
    if not args.skip_data_gen:
        print("\n" + "=" * 60)
        print("  STEP 1/3 — Generating Dataset")
        print("=" * 60)
        from data.generate_dataset import build_dataset
        df = build_dataset(n_records=2000)
        df.to_csv(data_csv, index=False)
        print(f"  Saved → {data_csv}  ({df.shape[0]} records, {df.shape[1]} columns)")
    else:
        if not data_csv.exists():
            print(f"ERROR: Dataset not found at {data_csv}")
            print("Remove --skip-data-gen or run: python data/generate_dataset.py")
            sys.exit(1)
        print(f"\nSkipping data generation — using {data_csv}")

    # ── Step 2: Train & Evaluate ─────────────────────────────────────────
    if not args.skip_training:
        print("\n" + "=" * 60)
        print("  STEP 2/3 — Training & Evaluating Models")
        print("=" * 60)
        from models.train import main as train_main
        train_main(data_csv)
    else:
        print("\nSkipping model training — using existing .pkl files")

    # ── Step 3: Visualise ────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("  STEP 3/3 — Generating Figures")
    print("=" * 60)
    from models.visualize import main as viz_main
    viz_main()

    print("\n" + "=" * 60)
    print("  PIPELINE COMPLETE")
    print(f"  Results → {BASE_DIR / 'results'}/")
    print("=" * 60)


if __name__ == "__main__":
    main()
