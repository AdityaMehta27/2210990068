"""
Training & Evaluation Script
==============================
Trains all six ML models, evaluates them on the held-out test set
(2019–2022), and saves results to results/.

Metrics (Section IV-D):
  MAE, RMSE, R², MAPE, 5-Fold CV-R²

Usage:
  python train.py
  python train.py --data path/to/custom.csv
"""

import argparse
import json
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import cross_val_score

# ── Local imports ────────────────────────────────────────────────────────────
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from data.preprocess import load_and_preprocess
from models.models   import get_models, SCALED_MODELS

RESULTS_DIR = Path(__file__).parent.parent / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)


# ─────────────────────────────────────────────────────────────────────────────
# Metric helpers
# ─────────────────────────────────────────────────────────────────────────────

def mape(y_true, y_pred) -> float:
    y_true, y_pred = np.asarray(y_true), np.asarray(y_pred)
    mask = y_true != 0
    return float(np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100)


def evaluate(y_true, y_pred) -> dict:
    return {
        "MAE":  float(mean_absolute_error(y_true, y_pred)),
        "RMSE": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "R2":   float(r2_score(y_true, y_pred)),
        "MAPE": mape(y_true, y_pred),
    }


# ─────────────────────────────────────────────────────────────────────────────
# Train + evaluate one model
# ─────────────────────────────────────────────────────────────────────────────

def run_model(name, model, splits, cv_folds=5):
    print(f"\n{'─'*55}")
    print(f"  Training : {name}")
    print(f"{'─'*55}")

    # Choose scaled or raw features
    if name in SCALED_MODELS:
        X_tr = splits["X_train_scaled"]
        X_te = splits["X_test_scaled"]
    else:
        X_tr = splits["X_train"]
        X_te = splits["X_test"]

    y_tr = splits["y_train"]
    y_te = splits["y_test"]

    # ── 5-Fold CV on training data ───────────────────────────────────────
    cv_scores = cross_val_score(
        model, X_tr, y_tr,
        cv=cv_folds, scoring="r2", n_jobs=-1
    )
    cv_mean = float(cv_scores.mean())
    cv_std  = float(cv_scores.std())
    print(f"  CV R²   : {cv_mean:.4f} ± {cv_std:.4f}")

    # ── Final fit on full training split ────────────────────────────────
    model.fit(X_tr, y_tr)

    # ── Test-set evaluation ──────────────────────────────────────────────
    y_pred  = model.predict(X_te)
    metrics = evaluate(y_te, y_pred)
    metrics["CV_R2_mean"] = cv_mean
    metrics["CV_R2_std"]  = cv_std

    print(f"  MAE     : {metrics['MAE']:.4f}")
    print(f"  RMSE    : {metrics['RMSE']:.4f}")
    print(f"  R²      : {metrics['R2']:.4f}")
    print(f"  MAPE    : {metrics['MAPE']:.2f}%")

    # ── Save model ───────────────────────────────────────────────────────
    model_path = RESULTS_DIR / f"{name.replace(' ', '_').replace('(', '').replace(')', '')}.pkl"
    with open(model_path, "wb") as f:
        pickle.dump(model, f)

    # ── Save predictions ─────────────────────────────────────────────────
    pred_df = pd.DataFrame({"y_true": y_te.values, "y_pred": y_pred})
    pred_df.to_csv(
        RESULTS_DIR / f"predictions_{name.replace(' ', '_')}.csv",
        index=False
    )

    return metrics, y_pred


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main(data_csv: Path):
    print("=" * 55)
    print("  Crop Yield Prediction — Model Training")
    print("=" * 55)

    splits = load_and_preprocess(data_csv)
    models = get_models()

    all_metrics   = {}
    all_preds     = {}

    for name, model in models.items():
        metrics, preds = run_model(name, model, splits)
        all_metrics[name] = metrics
        all_preds[name]   = preds

    # ── Summary table ────────────────────────────────────────────────────
    print("\n\n" + "=" * 75)
    print("  RESULTS SUMMARY  (Test set 2019–2022)")
    print("=" * 75)
    rows = []
    for name, m in all_metrics.items():
        rows.append({
            "Model":        name,
            "MAE":          round(m["MAE"],  4),
            "RMSE":         round(m["RMSE"], 4),
            "R²":           round(m["R2"],   4),
            "MAPE (%)":     round(m["MAPE"], 2),
            "CV R² (mean)": round(m["CV_R2_mean"], 4),
            "CV R² (std)":  round(m["CV_R2_std"],  4),
        })
    results_df = pd.DataFrame(rows).sort_values("R²", ascending=False)
    print(results_df.to_string(index=False))

    # ── Save results ─────────────────────────────────────────────────────
    results_df.to_csv(RESULTS_DIR / "model_comparison.csv", index=False)
    with open(RESULTS_DIR / "metrics.json", "w") as f:
        json.dump(all_metrics, f, indent=2)

    print(f"\nResults saved → {RESULTS_DIR}/")
    return results_df, all_metrics, splits, models, all_preds


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train crop yield prediction models")
    parser.add_argument(
        "--data", type=Path,
        default=Path(__file__).parent.parent / "data" / "agro_climatic_dataset.csv",
        help="Path to the agro-climatic CSV dataset"
    )
    args = parser.parse_args()
    main(args.data)
