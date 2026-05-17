"""
paper_main.py
-------------
Main research pipeline for the four publication figures (Fig. 4.3 / 4.4 /
National Trend / Ecological Analysis) and the headline metrics CSV that
the website renders from. Sister script to main.py — same models, same
data, slightly different figure outputs aimed at the paper rather than
the slides.

Run from the repo root::

    python code/paper_main.py

Outputs land in:
    code/output/figures/    publication PNGs
    code/output/analysis/   performance_metrics.csv + research_analysis.txt
"""

import os
import sys
import warnings

# This file lives at  code/paper_main.py.
# Data lives at        data/                (repo root)
# Outputs land at      code/output/...
_HERE = os.path.dirname(os.path.abspath(__file__))                    # code/
_REPO_ROOT = os.path.abspath(os.path.join(_HERE, ".."))               # repo root

DATA_PATH = os.path.join(_REPO_ROOT, "data")
sys.path.insert(0, _HERE)        # allow `from src.loader import ...`

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

# ── Ensure output directories exist ─────────────────────────────────────
BASE_DIR = _HERE
FIG_DIR = os.path.join(BASE_DIR, "output", "figures")
ANALYSIS_DIR = os.path.join(BASE_DIR, "output", "analysis")

for _d in [FIG_DIR, ANALYSIS_DIR]:
    os.makedirs(_d, exist_ok=True)

from src.loader    import DataLoader
from src.processor import DataProcessor
from src.engineer  import FeatureEngineer, FEATURE_COLS, TARGET_COL, TARGET_RAW
from src.models    import TyphoidModels, HAS_TF
from src.evaluator import Evaluator
from src.visualizer import Visualizer
from src.reporter   import Reporter

# 1. Load & Process
print(f"[1/5] Loading real data from CSV files in: {DATA_PATH}")
loader = DataLoader()
raw    = loader.load_all()
proc = DataProcessor()

# Process typhoid (Disable aggressive outlier clipping for research peaks)
typhoid_cleaned = proc.process_typhoid(raw["typhoid"])
# Note: process_typhoid internally uses 3.0 IQR factor. 
# We'll trust it for now but ensure national aggregates are preserved.

merged = proc.merge_datasets(typhoid_cleaned, 
                             proc.process_climate(raw["climate"]), 
                             proc.process_flood(raw["flood"]))

eng  = FeatureEngineer()
df   = eng.create_features(merged)
feats = eng.get_feature_list()
df = df.sort_values("date_dt").reset_index(drop=True)

# 2. Split (Latest 12 months for testing)
split_idx = len(df["date_dt"].unique()) - 12
split_date = df["date_dt"].unique()[split_idx]
train, test = df[df["date_dt"] < split_date], df[df["date_dt"] >= split_date]
X_train, y_train, X_test, y_test = train[feats], train[TARGET_COL], test[feats], test[TARGET_COL]
y_test_raw = test[TARGET_RAW].values

# 3. Train Models for Figure 4.4 comparison
print("[2/5] Training Models on real data...")
models = TyphoidModels()
models.train_rf(X_train, y_train)
rf_pred = models.predict_rf(X_test)

val_idx = int(len(X_train) * 0.9)
models.train_xgb(X_train.iloc[:val_idx], y_train.iloc[:val_idx], X_train.iloc[val_idx:], y_train.iloc[val_idx:])
xgb_pred = models.predict_xgb(X_test)

models.train_sequential(X_train, y_train, n_steps=3, epochs=50)
seq_pred = models.predict_sequential(X_test, n_steps=3)

# Align for metrics
if HAS_TF:
    y_test_al, rf_al, xgb_al, seq_al = y_test_raw[3:], rf_pred[3:], xgb_pred[3:], seq_pred
    seq_label = "LSTM"
else:
    y_test_al, rf_al, xgb_al, seq_al = y_test_raw, rf_pred, xgb_pred, seq_pred
    seq_label = "MLP"

# 4. Evaluate
evaluator = Evaluator()
results = [
    evaluator.evaluate(y_test_al, rf_al, "Random Forest"),
    evaluator.evaluate(y_test_al, xgb_al, "XGBoost"),
    evaluator.evaluate(y_test_al, seq_al, seq_label)
]
metrics_df = evaluator.save_metrics(results, path=os.path.join(ANALYSIS_DIR, "performance_metrics.csv"))

# 5. EXACT VISUALIZATIONS (Using real data)
print("[3/5] Generating figures with exact style using real data...")
viz = Visualizer(output_dir=FIG_DIR)

# Generate the 4 core figures
viz.plot_correlation_heatmap(df)
viz.plot_model_comparison(metrics_df)
viz.plot_annual_trend(merged)
viz.plot_ecology(merged)

# 6. Report
print("[4/5] Updating analysis report...")
reporter = Reporter(output_dir=ANALYSIS_DIR)
corr_mat = df[feats + [TARGET_COL]].corr()
reporter.generate_report(
    metrics_df, "XGBoost", 
    corr_precip=float(corr_mat.loc["precip_lag1", TARGET_COL]), 
    corr_humidity=float(corr_mat.loc["humidity_lag1", TARGET_COL]),
    title="Predicting Typhoid Incidence in Nepal using Climate-Driven Machine Learning",
    doc_type="Research Paper"
)

print("\n[5/5] SUCCESS: Research figures generated from real data in research_paper/figures/")
