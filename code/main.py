"""
main.py
-------
Entry point for the Typhoid Incidence Prediction pipeline.

Pipeline stages
---------------
1.  Load all three source datasets
2.  Preprocess: normalise districts, convert BS calendar, handle outliers,
    rename columns to short aliases
3.  Merge into a single panel DataFrame
4.  Feature engineering: lag features, rolling means, temporal indicators,
    log-transform of target variable
5.  Save descriptive statistics and lag-correlation table
6.  Time-series split (strictly chronological – last 12 months = test set)
7.  Train Random Forest, XGBoost, and Sequential (LSTM/MLP) models
8.  Build Weighted Ensemble
9.  Evaluate all models; save performance table
10. Generate 5 publication-quality figures
11. Generate academic analysis report
12. Persist trained models and scalers

Run
---
    python main.py
"""

import os
import warnings

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

# ── Ensure output directories exist ─────────────────────────────────────
for _d in ["output/models", "output/tables", "output/figures", "output/analysis"]:
    os.makedirs(_d, exist_ok=True)

from src.loader    import DataLoader
from src.processor import DataProcessor
from src.engineer  import FeatureEngineer, FEATURE_COLS, TARGET_COL, TARGET_RAW
from src.models    import TyphoidModels, HAS_TF
from src.evaluator import Evaluator
from src.visualizer import Visualizer
from src.reporter   import Reporter


# ═══════════════════════════════════════════════════════════════════════ #
#  STEP 1 — Load raw data                                                 #
# ═══════════════════════════════════════════════════════════════════════ #
print("[1/9] Loading raw datasets...")
loader = DataLoader()
raw    = loader.load_all()
print(f"      Typhoid rows  : {len(raw['typhoid']):,}")
print(f"      Climate rows  : {len(raw['climate']):,}")
print(f"      Flood rows    : {len(raw['flood']):,}")


# ═══════════════════════════════════════════════════════════════════════ #
#  STEP 2 & 3 — Preprocess and merge                                      #
# ═══════════════════════════════════════════════════════════════════════ #
print("[2/9] Preprocessing and merging datasets...")
proc = DataProcessor()

df_typhoid = proc.process_typhoid(raw["typhoid"])
df_climate = proc.process_climate(raw["climate"])
df_flood   = proc.process_flood(raw["flood"])

merged = proc.merge_datasets(df_typhoid, df_climate, df_flood)
print(f"      Merged panel  : {merged.shape[0]:,} rows × {merged.shape[1]} columns")
print(f"      Date range    : {merged['date_ad'].min()} to {merged['date_ad'].max()}")
print(f"      Districts     : {merged['district'].nunique()} unique")


# ═══════════════════════════════════════════════════════════════════════ #
#  STEP 4 — Feature engineering                                            #
# ═══════════════════════════════════════════════════════════════════════ #
print("[3/9] Engineering features...")
eng  = FeatureEngineer()
df   = eng.create_features(merged)
feats = eng.get_feature_list()
print(f"      Rows after lag drop : {len(df):,}")
print(f"      Features            : {feats}")


# ═══════════════════════════════════════════════════════════════════════ #
#  STEP 5 — Descriptive statistics and lag-correlation table               #
# ═══════════════════════════════════════════════════════════════════════ #
print("[4/9] Saving descriptive statistics...")

desc_cols = feats + [TARGET_RAW, TARGET_COL]
df[desc_cols].describe().round(3).to_csv("output/tables/descriptive_statistics.csv")

# Lag-correlation table: correlation of each feature with log_cases
lag_corr = (
    df[feats + [TARGET_COL]]
    .corr()[TARGET_COL]
    .drop(TARGET_COL)
    .reset_index()
    .rename(columns={"index": "Feature", TARGET_COL: "Pearson_r_with_log_cases"})
    .sort_values("Pearson_r_with_log_cases", ascending=False)
)
lag_corr.to_csv("output/tables/lag_correlation.csv", index=False)
print("      Saved: descriptive_statistics.csv, lag_correlation.csv")


# ═══════════════════════════════════════════════════════════════════════ #
#  STEP 6 — Time-series split (strictly chronological)                     #
# ═══════════════════════════════════════════════════════════════════════ #
print("[5/9] Splitting into train / test sets...")

df = df.sort_values("date_dt").reset_index(drop=True)
unique_dates = df["date_dt"].sort_values().unique()

# Reserve last 12 months as the test set
split_idx  = len(unique_dates) - 12
split_date = unique_dates[split_idx]

train = df[df["date_dt"] <  split_date].copy()
test  = df[df["date_dt"] >= split_date].copy()

X_train, y_train = train[feats], train[TARGET_COL]
X_test,  y_test  = test[feats],  test[TARGET_COL]
y_test_raw        = test[TARGET_RAW].values     # original case counts for evaluation

print(f"      Training rows : {len(train):,}  ({train['date_ad'].min()} – {train['date_ad'].max()})")
print(f"      Testing rows  : {len(test):,}  ({test['date_ad'].min()} – {test['date_ad'].max()})")


# ═══════════════════════════════════════════════════════════════════════ #
#  STEP 7 — Train models                                                   #
# ═══════════════════════════════════════════════════════════════════════ #
print("[6/9] Training models...")
models = TyphoidModels()

# Random Forest
print("      Training Random Forest...")
models.train_rf(X_train, y_train)
rf_pred_log = models.predict_rf(X_test)

# XGBoost (use last 10% of training set as early-stopping validation)
print("      Training XGBoost...")
val_idx  = int(len(X_train) * 0.9)
X_tr_xgb, X_val_xgb = X_train.iloc[:val_idx], X_train.iloc[val_idx:]
y_tr_xgb, y_val_xgb = y_train.iloc[:val_idx], y_train.iloc[val_idx:]
models.train_xgb(X_tr_xgb, y_tr_xgb, X_val_xgb, y_val_xgb)
xgb_pred_log = models.predict_xgb(X_test)

# Sequential model (LSTM or MLP fallback)
seq_label = "LSTM" if HAS_TF else "MLP (fallback, TF not available)"
print(f"      Training {seq_label}...")
models.train_sequential(X_train, y_train, n_steps=3, epochs=50)

if HAS_TF:
    seq_pred_log = models.predict_sequential(X_test, n_steps=3)
    # LSTM returns n_steps fewer points; align test labels accordingly
    y_test_aligned_log = y_test.values[3:]
    y_test_aligned_raw = y_test_raw[3:]
    rf_pred_log_al     = rf_pred_log[3:]
    xgb_pred_log_al    = xgb_pred_log[3:]
else:
    seq_pred_log       = models.predict_sequential(X_test, n_steps=3)
    y_test_aligned_log = y_test.values
    y_test_aligned_raw = y_test_raw
    rf_pred_log_al     = rf_pred_log
    xgb_pred_log_al    = xgb_pred_log

# Weighted ensemble (XGBoost up-weighted; see models.py for rationale)
ensemble_log = TyphoidModels.weighted_ensemble(
    rf_pred_log_al, xgb_pred_log_al, seq_pred_log
)

print("      All models trained.")


# ═══════════════════════════════════════════════════════════════════════ #
#  STEP 8 — Evaluate                                                       #
# ═══════════════════════════════════════════════════════════════════════ #
print("[7/9] Evaluating all models...")
evaluator = Evaluator()

results = []
for name, pred_log in [
    ("Random Forest",    rf_pred_log_al),
    ("XGBoost",          xgb_pred_log_al),
    (seq_label,          seq_pred_log),
    ("Weighted Ensemble",ensemble_log),
]:
    results.append(evaluator.evaluate(y_test_aligned_raw, pred_log, name))

metrics_df = evaluator.save_metrics(results)
print("\n" + metrics_df.to_string(index=False) + "\n")

best_model_name = metrics_df.loc[metrics_df["RMSE"].idxmin(), "Model"]
print(f"      Best model by RMSE: {best_model_name}")


# ═══════════════════════════════════════════════════════════════════════ #
#  STEP 9 — Visualisations                                                 #
# ═══════════════════════════════════════════════════════════════════════ #
print("[8/9] Generating publication-quality figures...")
viz = Visualizer()

# Fig 1: National time series
viz.plot_time_series(merged)

# Fig 2: Correlation heatmap (pass the fully-featured DataFrame)
viz.plot_correlation(df)

# Fig 3: Feature importance from XGBoost
viz.plot_feature_importance(models.xgb, feats)

# Fig 4: Actual vs Predicted (all four models in a panel)
predictions_dict = {
    "Random Forest":    rf_pred_log_al,
    "XGBoost":          xgb_pred_log_al,
    seq_label:          seq_pred_log,
    "Weighted Ensemble":ensemble_log,
}
viz.plot_actual_vs_predicted(y_test_aligned_raw, predictions_dict)

# Fig 5: Seasonal boxplot
viz.plot_seasonal_boxplot(df)

print("      Saved 5 figures to output/figures/")


# ═══════════════════════════════════════════════════════════════════════ #
#  STEP 10 — Academic report                                               #
# ═══════════════════════════════════════════════════════════════════════ #
print("[9/9] Generating academic analysis report...")

# Compute correlations for the report (actual data values)
corr_mat      = df[feats + [TARGET_COL]].corr()
corr_precip   = float(corr_mat.loc["precip_lag1",   TARGET_COL]) if "precip_lag1"   in corr_mat else None
corr_humidity = float(corr_mat.loc["humidity_lag1", TARGET_COL]) if "humidity_lag1" in corr_mat else None

reporter    = Reporter()
report_path = reporter.generate_report(
    metrics_df, best_model_name, corr_precip, corr_humidity
)
print(f"      Report saved : {report_path}")

# ── Persist trained models ──────────────────────────────────────────────
models.save_models()
print("      Models saved to output/models/")

print("\n" + "=" * 60)
print("  PIPELINE COMPLETE")
print(f"  Best model     : {best_model_name}")
print(f"  Figures saved  : output/figures/   (5 files)")
print(f"  Tables saved   : output/tables/    (3 files)")
print(f"  Models saved   : output/models/")
print(f"  Analysis saved : {report_path}")
print("=" * 60 + "\n")
