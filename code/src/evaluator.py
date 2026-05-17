"""
evaluator.py
------------
Computes standard regression metrics for epidemic forecasting models.

Metrics:
  RMSE  – penalises large errors most heavily (sensitive to extreme outliers)
  MAE   – robust average absolute error (interpretable in case-count units)
  R²    – proportion of variance explained by the model
  MAPE  – percentage error (useful for comparing across districts/scales),
           computed only on non-zero observed values to avoid division by zero.

All metrics are computed in the *original* case-count space (after back-
transforming log1p predictions) so that results are directly interpretable
in the thesis and can be compared to epidemiological baselines in the
literature.
"""

import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score


class Evaluator:
    """Model performance evaluator for typhoid incidence forecasts."""

    def compute_mape(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        y_true, y_pred = np.array(y_true), np.array(y_pred)
        mask = y_true > 0                              # guard against division by zero
        if mask.sum() == 0:
            return np.nan
        return float(np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100)

    def evaluate(self, y_true: np.ndarray, y_pred_log: np.ndarray,
                 model_name: str) -> dict:
        """
        Back-transform log1p predictions and compute all four metrics.

        Parameters
        ----------
        y_true      : observed case counts in original scale
        y_pred_log  : predicted log1p(cases) – function applies expm1 internally
        model_name  : label for the result row
        """
        y_true  = np.array(y_true)
        y_pred  = np.expm1(np.array(y_pred_log))   # back-transform to raw counts
        y_pred  = np.clip(y_pred, 0, None)          # prevent negative predictions

        rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
        mae  = float(mean_absolute_error(y_true, y_pred))
        r2   = float(r2_score(y_true, y_pred))
        mape = self.compute_mape(y_true, y_pred)

        return {
            "Model":     model_name,
            "RMSE":      round(rmse,  2),
            "MAE":       round(mae,   2),
            "R2":        round(r2,    4),
            "MAPE (%)":  round(mape,  2),
        }

    def save_metrics(self, results: list,
                     path: str = "performance_metrics.csv") -> pd.DataFrame:
        import os
        parent = os.path.dirname(path)
        if parent:
            os.makedirs(parent, exist_ok=True)
        df = pd.DataFrame(results)
        df.to_csv(path, index=False)
        return df
