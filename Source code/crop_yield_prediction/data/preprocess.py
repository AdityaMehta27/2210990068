"""
Preprocessing Pipeline
========================
Handles encoding, scaling, and temporal train-test split as described in
Section III-C of the paper.

Temporal split:
  Train : 1997–2018  (n ≈ 1,681)
  Test  : 2019–2022  (n ≈ 319)
"""

import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split

FEATURE_COLS = [
    "Crop", "State",
    "Cumulative_Rainfall_mm", "Max_Temperature_C", "Min_Temperature_C",
    "Relative_Humidity_pct", "Solar_Radiation_MJ",
    "Soil_pH", "Organic_Carbon_pct", "Nitrogen_kg_ha",
    "Phosphorus_kg_ha", "Potassium_kg_ha", "Soil_Texture",
    "Irrigation", "Fertilizer_kg_ha", "Area_Sown_lakh_ha",
    "GDD", "VPD_kPa", "Peak_NDVI", "Rainfall_CI",
]

TARGET_COL  = "Yield_t_ha"
YEAR_COL    = "Year"
TRAIN_YEARS = (1997, 2018)
TEST_YEARS  = (2019, 2022)

# Features that need StandardScaler (for Ridge & SVR)
SCALE_COLS = [
    "Cumulative_Rainfall_mm", "Max_Temperature_C", "Min_Temperature_C",
    "Relative_Humidity_pct", "Solar_Radiation_MJ",
    "Soil_pH", "Organic_Carbon_pct", "Nitrogen_kg_ha",
    "Phosphorus_kg_ha", "Potassium_kg_ha",
    "Fertilizer_kg_ha", "Area_Sown_lakh_ha",
    "GDD", "VPD_kPa", "Peak_NDVI", "Rainfall_CI",
]


class Preprocessor:
    """End-to-end preprocessing: encode → temporal split → optional scale."""

    def __init__(self):
        self.label_encoders: dict[str, LabelEncoder] = {}
        self.scaler = StandardScaler()

    def fit_transform(self, df: pd.DataFrame):
        df = df.copy()

        # ── 1. Encode categoricals ──────────────────────────────────────
        for col in ["Crop", "State", "Soil_Texture"]:
            le = LabelEncoder()
            df[col] = le.fit_transform(df[col].astype(str))
            self.label_encoders[col] = le

        # ── 2. Temporal split ───────────────────────────────────────────
        train_mask = df[YEAR_COL].between(*TRAIN_YEARS)
        test_mask  = df[YEAR_COL].between(*TEST_YEARS)

        X = df[FEATURE_COLS]
        y = df[TARGET_COL]

        X_train, y_train = X[train_mask], y[train_mask]
        X_test,  y_test  = X[test_mask],  y[test_mask]

        # ── 3. Fit scaler on train only ─────────────────────────────────
        self.scaler.fit(X_train[SCALE_COLS])

        return X_train, X_test, y_train, y_test

    def get_scaled(self, X_train, X_test):
        """Return copies with continuous features standardised (for Ridge/SVR)."""
        X_tr = X_train.copy()
        X_te = X_test.copy()
        X_tr[SCALE_COLS] = self.scaler.transform(X_train[SCALE_COLS])
        X_te[SCALE_COLS] = self.scaler.transform(X_test[SCALE_COLS])
        return X_tr, X_te


def load_and_preprocess(csv_path: str | Path):
    """Convenience wrapper used by train.py and evaluate.py."""
    df = pd.read_csv(csv_path)
    prep = Preprocessor()
    X_train, X_test, y_train, y_test = prep.fit_transform(df)
    X_train_scaled, X_test_scaled    = prep.get_scaled(X_train, X_test)

    print(f"Train : {X_train.shape[0]} records  ({TRAIN_YEARS[0]}–{TRAIN_YEARS[1]})")
    print(f"Test  : {X_test.shape[0]}  records  ({TEST_YEARS[0]}–{TEST_YEARS[1]})")

    return {
        "X_train":        X_train,
        "X_test":         X_test,
        "X_train_scaled": X_train_scaled,
        "X_test_scaled":  X_test_scaled,
        "y_train":        y_train,
        "y_test":         y_test,
        "preprocessor":   prep,
    }


if __name__ == "__main__":
    data_dir = Path(__file__).parent
    splits   = load_and_preprocess(data_dir / "agro_climatic_dataset.csv")
    print("Feature columns :", list(splits["X_train"].columns))
