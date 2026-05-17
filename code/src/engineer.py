"""
engineer.py
-----------
Creates a focused, non-redundant feature set for typhoid prediction.

Selected features (based on the epidemiological literature):
  - precip_lag1       : 1-month lagged precipitation (primary driver of waterborne disease)
  - temp_mean_lag1    : 1-month lagged mean temperature (affects pathogen survival)
  - humidity_lag1     : 1-month lagged humidity (correlated with faecal contamination)
  - flood_lag1        : 1-month lagged flood events (contamination pathways)
  - precip_roll3      : 3-month rolling mean precipitation (seasonal accumulation)
  - temp_roll3        : 3-month rolling mean temperature
  - monsoon           : Binary flag for June–September
  - month_sin/cos     : Circular encoding of month (avoids ordinal artefacts)
  - cases_lag1        : 1-month lagged case count (auto-regressive component)
  - log_cases         : Log-transformed target variable

Log transformation rationale:
  Typhoid case counts are heavily right-skewed.  A log1p transform stabilises
  variance and makes the regression residuals closer to Gaussian, improving
  both model performance and the validity of linear-assumption based metrics.
"""

import numpy as np
import pandas as pd


# Features used for modelling (no raw cases, no redundant lags)
FEATURE_COLS = [
    "precip_lag1",
    "temp_mean_lag1",
    "humidity_lag1",
    "flood_lag1",
    "precip_roll3",
    "temp_roll3",
    "monsoon",
    "month_sin",
    "month_cos",
    "cases_lag1",
]

TARGET_COL    = "log_cases"    # log1p-transformed typhoid cases
TARGET_RAW    = "cases"        # original scale (used for back-transformation in evaluation)


class FeatureEngineer:
    """Build a clean, epidemiologically-informed feature matrix."""

    def create_features(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        df["date_dt"] = pd.to_datetime(df["date_ad"])
        df = df.sort_values(["district", "date_dt"]).reset_index(drop=True)

        g = df.groupby("district", sort=False)

        # ── Lag features (1-month) ──────────────────────────────────────
        df["precip_lag1"]    = g["precip"].shift(1)
        df["temp_mean_lag1"] = g["temp_mean"].shift(1)
        df["humidity_lag1"]  = g["humidity"].shift(1)
        df["flood_lag1"]     = g["flood_events"].shift(1)
        df["cases_lag1"]     = g["cases"].shift(1)

        # ── Rolling means (3-month, computed on raw values pre-shift) ───
        df["precip_roll3"] = g["precip"].transform(lambda x: x.shift(1).rolling(3).mean())
        df["temp_roll3"]   = g["temp_mean"].transform(lambda x: x.shift(1).rolling(3).mean())

        # ── Temporal features ───────────────────────────────────────────
        df["month"]     = df["date_dt"].dt.month
        df["monsoon"]   = df["month"].between(6, 9).astype(int)
        # Circular encoding prevents January and December from appearing distant
        df["month_sin"] = np.sin(2 * np.pi * df["month"] / 12)
        df["month_cos"] = np.cos(2 * np.pi * df["month"] / 12)

        # ── Log-transform target ─────────────────────────────────────────
        df[TARGET_COL] = np.log1p(df[TARGET_RAW])

        # ── Drop rows created incomplete by lags ─────────────────────────
        df = df.dropna(subset=FEATURE_COLS + [TARGET_COL]).reset_index(drop=True)

        return df

    @staticmethod
    def get_feature_list() -> list:
        return list(FEATURE_COLS)
