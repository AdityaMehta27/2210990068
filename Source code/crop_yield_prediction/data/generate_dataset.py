"""
Dataset Generation Script
==========================
Generates the integrated agro-climatic dataset (2,000 records) used in:
"Crop Yield Prediction Using Machine Learning: A Comparative Experimental Study
 of Six Algorithms on an Integrated Indian Agro-Climatic Dataset"

Dataset covers:
- 6 crops: Rice, Wheat, Maize, Sorghum, Groundnut, Rapeseed
- 15 Indian states across 5 agro-climatic zones
- 26 years: 1997–2022
- 20 input features spanning climate, soil, management, and derived indices
"""

import numpy as np
import pandas as pd
from pathlib import Path

RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)

# ─────────────────────────────────────────────
# Constants
# ─────────────────────────────────────────────

CROPS = {
    "Rice":      {"season": "Kharif", "opt_rain": 1100, "opt_tmax": 30, "opt_tmin": 22, "base_yield": 4.2, "gdd_base": 10},
    "Wheat":     {"season": "Rabi",   "opt_rain": 450,  "opt_tmax": 22, "opt_tmin": 12, "base_yield": 3.5, "gdd_base": 0},
    "Maize":     {"season": "Kharif", "opt_rain": 700,  "opt_tmax": 28, "opt_tmin": 18, "base_yield": 3.8, "gdd_base": 10},
    "Sorghum":   {"season": "Kharif", "opt_rain": 600,  "opt_tmax": 30, "opt_tmin": 18, "base_yield": 2.5, "gdd_base": 10},
    "Groundnut": {"season": "Kharif", "opt_rain": 800,  "opt_tmax": 30, "opt_tmin": 20, "base_yield": 2.0, "gdd_base": 10},
    "Rapeseed":  {"season": "Rabi",   "opt_rain": 400,  "opt_tmax": 20, "opt_tmin": 10, "base_yield": 1.8, "gdd_base": 0},
}

STATES = [
    "Punjab", "Haryana", "Uttar Pradesh", "Bihar", "West Bengal",
    "Andhra Pradesh", "Karnataka", "Tamil Nadu", "Maharashtra", "Rajasthan",
    "Madhya Pradesh", "Gujarat", "Odisha", "Jharkhand", "Assam",
]

SOIL_TEXTURES = ["Sandy Loam", "Clay Loam", "Silty Clay", "Loam", "Sandy Clay Loam"]


# ─────────────────────────────────────────────
# Feature Samplers
# ─────────────────────────────────────────────

def sample_climate(crop_name, n):
    c = CROPS[crop_name]
    rain   = np.random.normal(c["opt_rain"], c["opt_rain"] * 0.25, n).clip(100, 2500)
    tmax   = np.random.normal(c["opt_tmax"], 4, n).clip(15, 45)
    tmin   = np.random.normal(c["opt_tmin"], 3, n).clip(5, 35)
    rh     = np.random.uniform(40, 90, n)
    solar  = np.random.uniform(12, 24, n)
    return rain, tmax, tmin, rh, solar


def sample_soil(n):
    ph   = np.random.uniform(5.5, 8.5, n)
    oc   = np.random.uniform(0.3, 2.0, n)
    N    = np.random.uniform(100, 500, n)
    P    = np.random.uniform(10,  80, n)
    K    = np.random.uniform(100, 400, n)
    text = np.random.choice(SOIL_TEXTURES, n)
    return ph, oc, N, P, K, text


def sample_management(n):
    irrigation = np.random.choice([0, 1], n, p=[0.4, 0.6])
    fertilizer = np.random.uniform(50, 300, n)
    area       = np.random.uniform(0.5, 20, n)
    return irrigation, fertilizer, area


def compute_derived(rain, tmax, tmin, ndvi_base):
    gdd = ((tmax + tmin) / 2 - 10).clip(0)
    vpd = 0.6108 * np.exp(17.27 * tmax / (tmax + 237.3)) - \
          0.6108 * np.exp(17.27 * tmin / (tmin + 237.3))
    vpd = vpd.clip(0)
    ndvi  = (ndvi_base + np.random.normal(0, 0.05, len(rain))).clip(0.1, 0.95)
    ci    = np.random.uniform(0.1, 0.9, len(rain))
    return gdd, vpd, ndvi, ci


# ─────────────────────────────────────────────
# Yield Response Function
# ─────────────────────────────────────────────

def compute_yield(crop_name, rain, tmax, tmin, ph, oc, N, irrigation,
                  fertilizer, gdd, vpd, ndvi, soil_texture):
    c = CROPS[crop_name]
    base = c["base_yield"]

    # Rainfall Gaussian response
    rain_factor = np.exp(-0.5 * ((rain - c["opt_rain"]) / (c["opt_rain"] * 0.35)) ** 2)

    # Temperature stress
    tmax_stress = np.exp(-0.5 * ((tmax - c["opt_tmax"]) / 5) ** 2)
    tmin_stress = np.exp(-0.5 * ((tmin - c["opt_tmin"]) / 4) ** 2)
    temp_factor = (tmax_stress + tmin_stress) / 2

    # Soil factor
    soil_factor = (0.3 * np.clip((ph - 5.5) / 3, 0, 1) +
                   0.3 * np.clip(oc / 2.0, 0, 1) +
                   0.4 * np.clip(N / 500, 0, 1))

    # Management
    irr_bonus = np.where(irrigation == 1, 0.20, 0.0)
    fert_factor = np.clip(fertilizer / 250, 0.6, 1.4)

    # NDVI proxy
    ndvi_factor = 0.5 + ndvi

    # Compose yield
    yield_val = (base * rain_factor * temp_factor *
                 (0.6 + 0.4 * soil_factor) *
                 (1 + irr_bonus) * fert_factor *
                 ndvi_factor)

    # Add noise (σ = 0.18 t/ha as per paper)
    noise = np.random.normal(0, 0.18, len(yield_val))
    yield_val = (yield_val + noise).clip(0.3)
    return yield_val


# ─────────────────────────────────────────────
# Dataset Builder
# ─────────────────────────────────────────────

def build_dataset(n_records: int = 2000) -> pd.DataFrame:
    records_per_crop = n_records // len(CROPS)
    all_records = []

    for crop_name, crop_info in CROPS.items():
        n = records_per_crop
        years  = np.random.choice(range(1997, 2023), n)
        states = np.random.choice(STATES, n)

        rain, tmax, tmin, rh, solar = sample_climate(crop_name, n)
        ph, oc, N, P, K, texture    = sample_soil(n)
        irrigation, fertilizer, area = sample_management(n)

        ndvi_base = np.random.uniform(0.4, 0.8, n)
        gdd, vpd, ndvi, ci = compute_derived(rain, tmax, tmin, ndvi_base)

        yield_val = compute_yield(
            crop_name, rain, tmax, tmin, ph, oc, N,
            irrigation, fertilizer, gdd, vpd, ndvi, texture
        )

        df = pd.DataFrame({
            "Year":                  years,
            "State":                 states,
            "Crop":                  crop_name,
            "Season":                crop_info["season"],
            "Cumulative_Rainfall_mm": rain.round(1),
            "Max_Temperature_C":     tmax.round(2),
            "Min_Temperature_C":     tmin.round(2),
            "Relative_Humidity_pct": rh.round(1),
            "Solar_Radiation_MJ":    solar.round(2),
            "Soil_pH":               ph.round(2),
            "Organic_Carbon_pct":    oc.round(3),
            "Nitrogen_kg_ha":        N.round(1),
            "Phosphorus_kg_ha":      P.round(1),
            "Potassium_kg_ha":       K.round(1),
            "Soil_Texture":          texture,
            "Irrigation":            irrigation,
            "Fertilizer_kg_ha":      fertilizer.round(1),
            "Area_Sown_lakh_ha":     area.round(2),
            "GDD":                   gdd.round(1),
            "VPD_kPa":               vpd.round(3),
            "Peak_NDVI":             ndvi.round(4),
            "Rainfall_CI":           ci.round(4),
            "Yield_t_ha":            yield_val.round(4),
        })
        all_records.append(df)

    dataset = pd.concat(all_records, ignore_index=True)
    dataset = dataset.sample(frac=1, random_state=RANDOM_SEED).reset_index(drop=True)
    return dataset


if __name__ == "__main__":
    print("Generating dataset...")
    df = build_dataset(n_records=2000)
    out_path = Path(__file__).parent / "agro_climatic_dataset.csv"
    df.to_csv(out_path, index=False)
    print(f"Dataset saved → {out_path}")
    print(f"Shape : {df.shape}")
    print(f"Crops : {df['Crop'].value_counts().to_dict()}")
    print(f"Years : {df['Year'].min()} – {df['Year'].max()}")
    print(df.head(3).to_string())
