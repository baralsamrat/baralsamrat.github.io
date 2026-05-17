"""
models.py
---------
Implements three predictive architectures and a weighted ensemble:
  1. Random Forest (RF)  – strong baseline with low variance
  2. XGBoost (XGB)       – optimised gradient boosting (typically best performer)
  3. LSTM / MLP fallback – temporal sequential model

All models are trained on log1p-transformed typhoid cases.  Predictions are
back-transformed with np.expm1() before evaluation so that metrics (RMSE, MAE,
MAPE) are expressed in the original case-count scale for interpretability.

Time-series integrity:
  Splitting is strictly chronological.  No future information is used in any
  training fold.  The MLP fallback uses the same scaled features as LSTM
  and also back-transforms its predictions before returning them.
"""

import pickle
import numpy as np
import pandas as pd

from sklearn.ensemble import RandomForestRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler
from xgboost import XGBRegressor

# ── Optional TensorFlow / Keras ──────────────────────────────────────────
try:
    import tensorflow as tf
    _KERAS = tf.keras
    HAS_TF = True
except Exception:
    try:
        import keras as _KERAS
        HAS_TF = True
    except Exception:
        HAS_TF = False


class TyphoidModels:
    """Train, predict, and persist all typhoid prediction models."""

    def __init__(self):
        # Random Forest – conservative, interpretable baseline
        self.rf = RandomForestRegressor(
            n_estimators=300,
            max_depth=10,
            min_samples_leaf=5,
            random_state=42,
            n_jobs=-1,
        )
        # XGBoost – tuned to reduce overfitting on small epidemiological panels
        self.xgb = XGBRegressor(
            n_estimators=300,
            learning_rate=0.05,
            max_depth=4,
            subsample=0.8,
            colsample_bytree=0.8,
            reg_alpha=0.1,
            reg_lambda=1.0,
            random_state=42,
            verbosity=0,
        )
        self.scaler_X = StandardScaler()
        self.scaler_y = StandardScaler()
        self.seq_model = None   # LSTM or MLP

    # ── Random Forest ────────────────────────────────────────────────────
    def train_rf(self, X_train: pd.DataFrame, y_train: pd.Series):
        self.rf.fit(X_train, y_train)
        return self.rf

    def predict_rf(self, X_test: pd.DataFrame) -> np.ndarray:
        return self.rf.predict(X_test)

    # ── XGBoost ──────────────────────────────────────────────────────────
    def train_xgb(self, X_train: pd.DataFrame, y_train: pd.Series,
                  X_val=None, y_val=None):
        eval_set = [(X_val, y_val)] if X_val is not None else []
        self.xgb.fit(
            X_train, y_train,
            eval_set=eval_set,
            verbose=False,
        )
        return self.xgb

    def predict_xgb(self, X_test: pd.DataFrame) -> np.ndarray:
        return self.xgb.predict(X_test)

    # ── Sequential model (LSTM or MLP fallback) ──────────────────────────
    def _build_mlp(self):
        """Scikit-learn MLP as a portable fallback when TF is unavailable."""
        return MLPRegressor(
            hidden_layer_sizes=(128, 64, 32),
            activation="relu",
            solver="adam",
            learning_rate_init=0.001,
            max_iter=500,
            early_stopping=True,
            validation_fraction=0.1,
            random_state=42,
        )

    def _build_lstm(self, n_features: int):
        """Two-layer stacked LSTM with dropout for temporal sequence learning."""
        model = _KERAS.Sequential([
            _KERAS.layers.LSTM(64, return_sequences=True,
                               input_shape=(3, n_features)),
            _KERAS.layers.Dropout(0.2),
            _KERAS.layers.LSTM(32),
            _KERAS.layers.Dropout(0.2),
            _KERAS.layers.Dense(1),
        ])
        model.compile(optimizer=_KERAS.optimizers.Adam(1e-3), loss="mse")
        return model

    def train_sequential(self, X_train: pd.DataFrame, y_train: pd.Series,
                         n_steps: int = 3, epochs: int = 50):
        """
        Scale features and target, then train either LSTM or MLP.

        Data leakage prevention: the StandardScaler is fitted *only* on the
        training partition and later applied identically to the test partition.
        """
        X_sc = self.scaler_X.fit_transform(X_train)
        y_sc = self.scaler_y.fit_transform(y_train.values.reshape(-1, 1)).ravel()

        if HAS_TF:
            X3d, y3d = self._make_sequences(X_sc, y_sc, n_steps)
            self.seq_model = self._build_lstm(X_train.shape[1])
            self.seq_model.fit(
                X3d, y3d,
                epochs=epochs,
                batch_size=32,
                verbose=0,
            )
        else:
            self.seq_model = self._build_mlp()
            self.seq_model.fit(X_sc, y_sc)

        return self.seq_model

    def predict_sequential(self, X_test: pd.DataFrame, n_steps: int = 3):
        """Return back-transformed predictions in log1p scale."""
        X_sc = self.scaler_X.transform(X_test)

        if HAS_TF:
            X3d, _ = self._make_sequences(X_sc, np.zeros(len(X_sc)), n_steps)
            y_sc = self.seq_model.predict(X3d, verbose=0).ravel()
        else:
            y_sc = self.seq_model.predict(X_sc)

        return self.scaler_y.inverse_transform(y_sc.reshape(-1, 1)).ravel()

    @staticmethod
    def _make_sequences(X, y, n_steps):
        Xs, ys = [], []
        for i in range(len(X) - n_steps):
            Xs.append(X[i: i + n_steps])
            ys.append(y[i + n_steps])
        return np.array(Xs), np.array(ys)

    # ── Weighted Ensemble ─────────────────────────────────────────────────
    @staticmethod
    def weighted_ensemble(rf_pred, xgb_pred, seq_pred,
                          w_rf=0.25, w_xgb=0.50, w_seq=0.25) -> np.ndarray:
        """
        XGBoost is up-weighted because it consistently achieves lower RMSE on
        epidemiological count-data panels.  Weights are set a priori and not
        optimised on the test set to prevent data leakage.
        """
        return w_rf * rf_pred + w_xgb * xgb_pred + w_seq * seq_pred

    # ── Persistence ───────────────────────────────────────────────────────
    def save_models(self, out_dir: str = "output/models"):
        with open(f"{out_dir}/rf_model.pkl",  "wb") as f:
            pickle.dump(self.rf, f)
        with open(f"{out_dir}/xgb_model.pkl", "wb") as f:
            pickle.dump(self.xgb, f)
        if self.seq_model is not None:
            if HAS_TF:
                self.seq_model.save(f"{out_dir}/lstm_model.h5")
            else:
                with open(f"{out_dir}/mlp_model.pkl", "wb") as f:
                    pickle.dump(self.seq_model, f)
        # Also save scalers for future inference
        with open(f"{out_dir}/scaler_X.pkl", "wb") as f:
            pickle.dump(self.scaler_X, f)
        with open(f"{out_dir}/scaler_y.pkl", "wb") as f:
            pickle.dump(self.scaler_y, f)
