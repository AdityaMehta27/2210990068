# Crop Yield Prediction Using Machine Learning

> **Paper:** *Crop Yield Prediction Using Machine Learning: A Comparative Experimental Study of Six Algorithms on an Integrated Indian Agro-Climatic Dataset*  
> **Authors:** Aditya Mehta, Dr. Gurpreet Singh — Chitkara University, Punjab, India

---

## Overview

This repository contains all code, data generation scripts, and evaluation pipelines for the paper. Six ML algorithms are benchmarked on a 2,000-record integrated Indian agro-climatic dataset covering **6 crops**, **15 states**, and **26 years (1997–2022)**.

| Model | R² | RMSE | MAPE |
|---|---|---|---|
| **Gradient Boosting** | **0.9247** | **0.2664** | **4.31%** |
| XGBoost | 0.9228 | 0.2699 | 4.44% |
| Decision Tree | 0.7955 | 0.4392 | 6.86% |
| Random Forest | 0.7615 | 0.4743 | 7.87% |
| SVR (RBF) | 0.6353 | 0.5865 | 9.65% |
| Ridge Regression | 0.4341 | 0.7307 | 12.59% |

---

## Repository Structure

```
crop_yield_prediction/
├── data/
│   ├── generate_dataset.py      # Generates the 2,000-record agro-climatic dataset
│   └── preprocess.py            # Encoding, scaling, temporal train/test split
├── models/
│   ├── models.py                # All six ML model definitions (exact hyperparameters)
│   ├── train.py                 # Training, CV evaluation, metric computation
│   └── visualize.py             # Reproduces all paper figures (Fig 2–6)
├── results/                     # Auto-created: CSVs, .pkl models, figures/
├── run_pipeline.py              # One-command full pipeline runner
├── requirements.txt
└── README.md
```

---

## Quickstart

### 1. Install dependencies
```bash
pip install -r requirements.txt
```

### 2. Run the full pipeline
```bash
python run_pipeline.py
```

This will:
1. Generate the synthetic agro-climatic dataset (`data/agro_climatic_dataset.csv`)
2. Train and evaluate all 6 models, saving metrics to `results/`
3. Generate all paper figures to `results/figures/`

### 3. Run individual steps
```bash
# Generate dataset only
python data/generate_dataset.py

# Train models (requires dataset)
python models/train.py

# Generate figures (requires trained models)
python models/visualize.py
```

---

## Dataset

The dataset is programmatically generated (`data/generate_dataset.py`) to replicate the parameter distributions from:
- Ministry of Agriculture and Farmers' Welfare (GoI)
- India Meteorological Department (IMD)
- National Bureau of Soil Survey and Land Use Planning (NBSS&LUP)

### Features (20 total)

| Domain | Features |
|---|---|
| Climate | Rainfall (mm), Tmax (°C), Tmin (°C), Humidity (%), Solar Radiation |
| Soil | pH, Organic Carbon, N, P, K (kg/ha), Texture |
| Management | Irrigation (binary), Fertilizer (kg/ha), Area Sown (lakh ha) |
| Derived Indices | GDD, VPD (kPa), Peak NDVI, Rainfall CI |
| Categorical | Crop Type, State |

**Target:** Crop yield (t/ha)

### Temporal Split (prevents data leakage)
- **Train:** 1997–2018 (n = 1,681)
- **Test:** 2019–2022 (n = 319)

---

## Models & Hyperparameters

| Model | Key Parameters |
|---|---|
| Ridge Regression | α = 1.0 |
| Decision Tree | max_depth=8, min_samples_leaf=10 |
| Random Forest | n_estimators=300, max_features='sqrt', min_samples_leaf=5 |
| SVR (RBF) | C=10, ε=0.05 |
| XGBoost | n_estimators=400, lr=0.05, max_depth=6, L1/L2 regularisation |
| Gradient Boosting | n_estimators=300, lr=0.05, max_depth=5, subsample=0.8 |

All experiments use `random_state = 42` for reproducibility.

---

## Results

After running the pipeline, `results/` will contain:

- `model_comparison.csv` — full metrics table (Table IV of the paper)
- `metrics.json` — metrics in JSON format
- `predictions_<model>.csv` — per-sample predictions for each model
- `<model>.pkl` — serialised trained models
- `figures/fig2_scatter.png` — Actual vs. Predicted (top-3 models)
- `figures/fig3_feature_importance.png` — Top-10 feature importances
- `figures/fig4_residuals.png` — Residual distributions
- `figures/fig5_cv_r2.png` — 5-Fold CV-R² comparison
- `figures/fig6_crop_specific.png` — Crop-specific R² and RMSE

---

## Key Findings

1. **Gradient Boosting** achieves R² = 0.9247, RMSE = 0.2664 t/ha — best across all metrics.
2. **Ridge Regression** confirms linear models are inadequate (R² = 0.4341).
3. **Top-3 features:** Crop Type (0.2927), Rainfall (0.1853), Peak NDVI (0.1681) — jointly 64.6% of variance.
4. **NDVI** is the highest-leverage zero-cost feature for operational deployment.

---

## Citation

If you use this code, please cite:

```
A. Mehta and G. Singh, "Crop Yield Prediction Using Machine Learning: A Comparative 
Experimental Study of Six Algorithms on an Integrated Indian Agro-Climatic Dataset," 
Chitkara University, Punjab, India, 2024.
```

---

## License

This project is released for academic and research use.
