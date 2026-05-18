"""
Visualisation Script
======================
Reproduces all figures from the paper:
  Fig 2  – Actual vs. Predicted scatter plots (top-3 models)
  Fig 3  – Feature importance bar chart (Random Forest)
  Fig 4  – Residual histograms (Ridge, XGBoost, Gradient Boosting)
  Fig 5  – 5-Fold CV-R² with error bars
  Fig 6  – Crop-specific Random Forest R² and RMSE

Usage:
  python visualize.py          (runs after train.py has been executed)
"""

import json
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from sklearn.metrics import r2_score
from sklearn.ensemble import RandomForestRegressor

# ── Paths ────────────────────────────────────────────────────────────────────
BASE_DIR    = Path(__file__).parent.parent
RESULTS_DIR = BASE_DIR / "results"
FIGS_DIR    = RESULTS_DIR / "figures"
FIGS_DIR.mkdir(parents=True, exist_ok=True)

plt.rcParams.update({
    "font.family":  "DejaVu Sans",
    "font.size":    11,
    "axes.spines.top":    False,
    "axes.spines.right":  False,
})


# ─────────────────────────────────────────────────────────────────────────────
# Helper
# ─────────────────────────────────────────────────────────────────────────────

def load_predictions():
    preds = {}
    for f in RESULTS_DIR.glob("predictions_*.csv"):
        name = f.stem.replace("predictions_", "").replace("_", " ")
        preds[name] = pd.read_csv(f)
    return preds


def load_metrics():
    with open(RESULTS_DIR / "metrics.json") as f:
        return json.load(f)


# ─────────────────────────────────────────────────────────────────────────────
# Fig 2 – Actual vs. Predicted scatter
# ─────────────────────────────────────────────────────────────────────────────

def plot_scatter(preds, metrics):
    top3 = sorted(metrics, key=lambda m: metrics[m]["R2"], reverse=True)[:3]
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    for ax, name in zip(axes, top3):
        df  = preds.get(name) or preds.get(name.replace(" ", "_"))
        if df is None:
            continue
        r2  = metrics[name]["R2"]
        ax.scatter(df["y_true"], df["y_pred"], alpha=0.4, s=20,
                   color="#2563EB", edgecolors="none")
        lims = [df[["y_true", "y_pred"]].min().min() - 0.1,
                df[["y_true", "y_pred"]].max().max() + 0.1]
        ax.plot(lims, lims, "k--", lw=1.5, label="Perfect prediction")
        z   = np.polyfit(df["y_true"], df["y_pred"], 1)
        p   = np.poly1d(z)
        ax.plot(sorted(df["y_true"]), p(sorted(df["y_true"])),
                "r-", lw=1.5, label="Fitted regression")
        ax.set_xlim(lims); ax.set_ylim(lims)
        ax.set_xlabel("Actual Yield (t/ha)")
        ax.set_ylabel("Predicted Yield (t/ha)")
        ax.set_title(f"{name}\nR² = {r2:.4f}")
        ax.legend(fontsize=8)

    plt.suptitle("Fig 2 – Actual vs. Predicted Yield (Top-3 Models)",
                 fontsize=13, fontweight="bold", y=1.01)
    plt.tight_layout()
    fig.savefig(FIGS_DIR / "fig2_scatter.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("Saved fig2_scatter.png")


# ─────────────────────────────────────────────────────────────────────────────
# Fig 3 – Feature Importance
# ─────────────────────────────────────────────────────────────────────────────

def plot_feature_importance(splits):
    rf_path = RESULTS_DIR / "Random_Forest.pkl"
    if not rf_path.exists():
        print("Random Forest model not found – skipping Fig 3")
        return

    with open(rf_path, "rb") as f:
        rf: RandomForestRegressor = pickle.load(f)

    feature_names = list(splits["X_train"].columns)
    importances   = rf.feature_importances_
    idx           = np.argsort(importances)[-10:]   # top 10

    fig, ax = plt.subplots(figsize=(9, 6))
    bars = ax.barh(
        [feature_names[i] for i in idx],
        importances[idx],
        color="#10B981", edgecolor="white", height=0.6
    )
    ax.bar_label(bars, fmt="%.4f", padding=3, fontsize=9)
    ax.set_xlabel("Mean Decrease in Impurity")
    ax.set_title("Fig 3 – Top-10 Feature Importances (Random Forest)",
                 fontweight="bold")
    plt.tight_layout()
    fig.savefig(FIGS_DIR / "fig3_feature_importance.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("Saved fig3_feature_importance.png")


# ─────────────────────────────────────────────────────────────────────────────
# Fig 4 – Residual Histograms
# ─────────────────────────────────────────────────────────────────────────────

def plot_residuals(preds):
    models_to_plot = ["Ridge Regression", "XGBoost", "Gradient Boosting"]
    colors         = ["#EF4444", "#F59E0B", "#10B981"]

    fig, axes = plt.subplots(1, 3, figsize=(15, 4), sharey=True)
    for ax, name, color in zip(axes, models_to_plot, colors):
        df = preds.get(name) or preds.get(name.replace(" ", "_"))
        if df is None:
            ax.set_title(f"{name}\n(no data)")
            continue
        residuals = df["y_pred"] - df["y_true"]
        mu, sigma = residuals.mean(), residuals.std()
        ax.hist(residuals, bins=30, color=color, alpha=0.75, edgecolor="white")
        ax.axvline(0, color="black", lw=1.5, ls="--")
        ax.set_xlabel("Residual (t/ha)")
        ax.set_ylabel("Count" if ax is axes[0] else "")
        ax.set_title(f"{name}\nμ={mu:.3f}, σ={sigma:.3f}")

    plt.suptitle("Fig 4 – Residual Distributions", fontweight="bold")
    plt.tight_layout()
    fig.savefig(FIGS_DIR / "fig4_residuals.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("Saved fig4_residuals.png")


# ─────────────────────────────────────────────────────────────────────────────
# Fig 5 – CV-R² bar chart with error bars
# ─────────────────────────────────────────────────────────────────────────────

def plot_cv_r2(metrics):
    names  = list(metrics.keys())
    means  = [metrics[n]["CV_R2_mean"] for n in names]
    stds   = [metrics[n]["CV_R2_std"]  for n in names]
    colors = ["#6366F1", "#F59E0B", "#10B981", "#EF4444", "#3B82F6", "#8B5CF6"]

    fig, ax = plt.subplots(figsize=(10, 5))
    bars = ax.bar(names, means, yerr=stds, capsize=5,
                  color=colors, edgecolor="white", width=0.55,
                  error_kw={"ecolor": "black", "elinewidth": 1.5})
    ax.bar_label(bars, fmt="%.4f", padding=4, fontsize=9)
    ax.set_ylabel("5-Fold CV R²")
    ax.set_ylim(0, 1.05)
    ax.set_title("Fig 5 – 5-Fold CV-R² with ± Std Dev", fontweight="bold")
    plt.xticks(rotation=20, ha="right")
    plt.tight_layout()
    fig.savefig(FIGS_DIR / "fig5_cv_r2.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("Saved fig5_cv_r2.png")


# ─────────────────────────────────────────────────────────────────────────────
# Fig 6 – Crop-specific Random Forest performance
# ─────────────────────────────────────────────────────────────────────────────

def plot_crop_specific(splits):
    from sklearn.metrics import mean_squared_error

    rf_path = RESULTS_DIR / "Random_Forest.pkl"
    if not rf_path.exists():
        print("Random Forest model not found – skipping Fig 6")
        return

    with open(rf_path, "rb") as f:
        rf: RandomForestRegressor = pickle.load(f)

    # Re-load raw CSV for crop labels
    data_csv = BASE_DIR / "data" / "agro_climatic_dataset.csv"
    df_full  = pd.read_csv(data_csv)
    test_mask = df_full["Year"].between(2019, 2022)
    df_test   = df_full[test_mask].copy()

    X_te   = splits["X_test"].copy()
    y_te   = splits["y_test"].copy()
    crops  = df_test["Crop"].values[:len(y_te)]

    y_pred = rf.predict(X_te)

    crop_list = sorted(set(crops))
    r2s, rmses = [], []
    for crop in crop_list:
        mask = crops == crop
        if mask.sum() == 0:
            continue
        r2s.append(r2_score(y_te.values[mask], y_pred[mask]))
        rmses.append(np.sqrt(mean_squared_error(y_te.values[mask], y_pred[mask])))

    x  = np.arange(len(crop_list))
    w  = 0.35
    fig, ax1 = plt.subplots(figsize=(10, 5))
    ax2 = ax1.twinx()

    b1 = ax1.bar(x - w/2, r2s,  width=w, label="R²",   color="#3B82F6", alpha=0.85)
    b2 = ax2.bar(x + w/2, rmses, width=w, label="RMSE", color="#F59E0B", alpha=0.85)

    ax1.set_ylabel("R²",        color="#3B82F6")
    ax2.set_ylabel("RMSE (t/ha)", color="#F59E0B")
    ax1.set_xticks(x); ax1.set_xticklabels(crop_list)
    ax1.set_ylim(0, 1)
    ax1.set_title("Fig 6 – Crop-Specific Random Forest Performance", fontweight="bold")
    lines  = [b1, b2]
    labels = ["R²", "RMSE"]
    ax1.legend(lines, labels, loc="lower right")
    plt.tight_layout()
    fig.savefig(FIGS_DIR / "fig6_crop_specific.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("Saved fig6_crop_specific.png")


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    import sys
    sys.path.insert(0, str(BASE_DIR))
    from data.preprocess import load_and_preprocess

    data_csv = BASE_DIR / "data" / "agro_climatic_dataset.csv"
    if not data_csv.exists():
        print("Dataset not found. Run: python data/generate_dataset.py")
        return

    splits  = load_and_preprocess(data_csv)
    preds   = load_predictions()
    metrics = load_metrics()

    if not preds:
        print("No prediction files found. Run: python models/train.py first.")
        return

    print("\nGenerating figures...")
    plot_scatter(preds, metrics)
    plot_feature_importance(splits)
    plot_residuals(preds)
    plot_cv_r2(metrics)
    plot_crop_specific(splits)
    print(f"\nAll figures saved → {FIGS_DIR}/")


if __name__ == "__main__":
    main()
