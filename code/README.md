# Replication code — Typhoid Prediction in Nepal

This folder contains the **exact production pipeline** that generates
every number, table, and figure rendered on the website. It is in-tree
so that visitors can reproduce the results without chasing private
upstream folders.

## Quick start (3 commands, ~2 min)

From the **repo root**:

```bash
pip install -r code/requirements.txt
python code/main.py          # full pipeline (RF + XGB + LSTM/MLP)
# or
python code/paper_main.py    # paper-style pipeline (3 models, paper figures)
```

Both scripts read the **same** four CSVs from `data/` at the repo root.

## Layout

```text
code/
├── README.md         (this file)
├── requirements.txt  pinned versions matching the website
├── main.py           full pipeline + weighted ensemble
├── paper_main.py     3-model paper pipeline + paper figures
└── src/
    ├── loader.py     load CSVs from <repo>/data/ (env-var overridable)
    ├── processor.py  district normalisation + BS→AD calendar conversion
    ├── engineer.py   lag features, rolling means, log-transform target
    ├── models.py     RF / XGBoost / MLP / LSTM training + persistence
    ├── evaluator.py  RMSE / MAE / R² / MAPE
    ├── visualizer.py figure rendering (paper + slide variants)
    └── reporter.py   text-format academic analysis report
```

## What you get

After a clean run, `code/output/` will contain:

```text
code/output/
├── models/           rf_model.pkl, xgb_model.pkl, mlp_model.pkl (or lstm_model.h5), scalers
├── figures/          publication PNGs
├── tables/           descriptive_statistics.csv, lag_correlation.csv, performance_metrics.csv
└── analysis/         research_analysis.txt
```

`code/output/` is **gitignored** — the pipeline is reproducible from
source, so persisting binary artifacts in the repo would be both wasteful
(the RF pickle alone is ~16 MB) and a maintenance trap (pickles aren't
portable across scikit-learn versions).

## Replicating the website's headline numbers

Running `python code/paper_main.py` against the committed `data/` CSVs
will land within rounding of the headline numbers on the rendered site:

| Model         | RMSE    | MAE    | R²     | MAPE (%) |
|---------------|--------:|-------:|-------:|---------:|
| Random Forest | 131.43  | 72.10  | 0.8563 | 41.48    |
| **XGBoost**   | **129.07** | **69.50** | **0.8614** | 44.64    |
| MLP fallback  | 128.07  | 71.45  | 0.8635 | 43.50    |

XGBoost is the **operational choice** — lowest MAE, interpretable
gain-based feature importance, graceful missing-value handling,
reproducible under a fixed random seed (42, as set in `models.py`).

## Pointing at different data

`loader.py` reads `<repo>/data/` by default; set `TYPHOID_DATA_DIR` to
override:

```bash
TYPHOID_DATA_DIR=/path/to/new/data python code/paper_main.py
```

## Citing

If you use this pipeline, please cite the underlying thesis — see the
[How to cite](../index.qmd#how-to-cite) block on the rendered site.
