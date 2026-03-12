"""LSTM-based AQI forecaster using TensorFlow / Keras."""

import json
import logging
import pickle
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

from config.constants import (
    DATA_PLOTS_DIR,
    LSTM_BATCH_SIZE,
    LSTM_EARLY_STOPPING_PATIENCE,
    LSTM_EPOCHS,
    LSTM_SEQUENCE_LENGTH,
    LSTM_UNITS,
    MODELS_SAVED_DIR,
)

logger = logging.getLogger(__name__)


class LSTMForecaster:
    """Sequence-to-one LSTM model trained with Keras."""

    def __init__(
        self,
        city: str,
        sequence_length: int = LSTM_SEQUENCE_LENGTH,
        units: int = LSTM_UNITS,
        epochs: int = LSTM_EPOCHS,
        batch_size: int = LSTM_BATCH_SIZE,
    ) -> None:
        self.city = city
        self.sequence_length = sequence_length
        self.units = units
        self.epochs = epochs
        self.batch_size = batch_size
        self.model = None
        self.feature_scaler = None
        self.target_scaler = None
        self._model_dir = Path(MODELS_SAVED_DIR)
        self._model_dir.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # Model architecture
    # ------------------------------------------------------------------
    def build_model(self, n_features: int) -> None:
        """Construct a single-layer LSTM network.

        Args:
            n_features: Number of input features per time step.
        """
        try:
            from tensorflow.keras.models import Sequential
            from tensorflow.keras.layers import LSTM, Dense
        except ImportError:
            logger.error("TensorFlow not installed.")
            raise

        model = Sequential([
            LSTM(self.units, input_shape=(self.sequence_length, n_features),
                 return_sequences=False),
            Dense(1),
        ])
        model.compile(optimizer="adam", loss="mse")
        self.model = model
        logger.info("Built LSTM model with %d units, seq_len=%d, features=%d.",
                     self.units, self.sequence_length, n_features)

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------
    def train(self, df: pd.DataFrame) -> None:
        """Fit the LSTM on a feature-engineered DataFrame.

        Args:
            df: DataFrame with numeric features and ``AQI`` target.
        """
        from sklearn.preprocessing import StandardScaler, MinMaxScaler
        try:
            from tensorflow.keras.callbacks import EarlyStopping
        except ImportError:
            logger.error("TensorFlow not installed.")
            raise

        numeric = df.select_dtypes(include=[np.number]).dropna(axis=1)
        if "AQI" not in numeric.columns:
            raise ValueError("DataFrame must contain 'AQI' column.")

        features = numeric.drop(columns=["AQI"])
        target = numeric[["AQI"]]

        self.feature_scaler = StandardScaler()
        self.target_scaler = MinMaxScaler()

        X_scaled = self.feature_scaler.fit_transform(features)
        y_scaled = self.target_scaler.fit_transform(target)

        X_seq, y_seq = self._create_sequences(X_scaled, y_scaled)

        split = int(len(X_seq) * 0.8)
        X_train, X_val = X_seq[:split], X_seq[split:]
        y_train, y_val = y_seq[:split], y_seq[split:]

        if self.model is None:
            self.build_model(X_train.shape[2])

        early_stop = EarlyStopping(
            monitor="val_loss",
            patience=LSTM_EARLY_STOPPING_PATIENCE,
            restore_best_weights=True,
        )

        history = self.model.fit(
            X_train, y_train,
            validation_data=(X_val, y_val),
            epochs=self.epochs,
            batch_size=self.batch_size,
            callbacks=[early_stop],
            verbose=1,
        )

        # Save artefacts
        self.model.save(str(self._model_dir / f"lstm_{self.city}.h5"))
        with open(self._model_dir / f"lstm_scaler_{self.city}.pkl", "wb") as fh:
            pickle.dump({
                "feature_scaler": self.feature_scaler,
                "target_scaler": self.target_scaler,
            }, fh)

        self._plot_loss(history)
        logger.info("LSTM training complete for %s.", self.city)

    # ------------------------------------------------------------------
    # Prediction
    # ------------------------------------------------------------------
    def predict(self, df: pd.DataFrame, horizon_hours: int = 24) -> np.ndarray:
        """Generate AQI predictions.

        Args:
            df: Recent data (must contain the same feature columns as training).
            horizon_hours: Number of forward steps.

        Returns:
            1-D array of predicted AQI values.
        """
        if self.model is None:
            self._load_model()
        if self.model is None:
            logger.error("No LSTM model available for %s.", self.city)
            return np.array([])

        numeric = df.select_dtypes(include=[np.number]).dropna(axis=1)
        features = numeric.drop(columns=["AQI"], errors="ignore")

        # Align feature columns with scaler
        expected = self.feature_scaler.feature_names_in_ if hasattr(
            self.feature_scaler, "feature_names_in_"
        ) else features.columns
        features = features.reindex(columns=expected, fill_value=0)

        X_scaled = self.feature_scaler.transform(features)

        predictions: list[float] = []
        window = X_scaled[-self.sequence_length:]

        for _ in range(horizon_hours):
            inp = window.reshape(1, self.sequence_length, -1)
            pred_scaled = self.model.predict(inp, verbose=0)
            pred = self.target_scaler.inverse_transform(pred_scaled).flatten()[0]
            predictions.append(float(pred))
            # Slide window (repeat last prediction row as proxy)
            new_row = window[-1].copy()
            window = np.vstack([window[1:], new_row])

        return np.array(predictions)

    # ------------------------------------------------------------------
    # Evaluation
    # ------------------------------------------------------------------
    def evaluate(self, test_df: pd.DataFrame) -> dict:
        """Compute RMSE, MAE, MAPE on a held-out test set.

        Args:
            test_df: Test-split DataFrame.

        Returns:
            Dict with RMSE, MAE, MAPE.
        """
        preds = self.predict(test_df, horizon_hours=len(test_df))
        actual = test_df["AQI"].values[-len(preds):]
        n = min(len(actual), len(preds))
        actual, preds = actual[:n], preds[:n]

        if n == 0:
            return {"RMSE": np.nan, "MAE": np.nan, "MAPE": np.nan}

        rmse = float(np.sqrt(np.mean((actual - preds) ** 2)))
        mae = float(np.mean(np.abs(actual - preds)))
        mape = float(np.mean(np.abs((actual - preds) / np.clip(actual, 1, None))) * 100)
        return {"RMSE": rmse, "MAE": mae, "MAPE": mape}

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------
    def _create_sequences(
        self, X: np.ndarray, y: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        Xs, ys = [], []
        for i in range(self.sequence_length, len(X)):
            Xs.append(X[i - self.sequence_length : i])
            ys.append(y[i])
        return np.array(Xs), np.array(ys)

    def _load_model(self) -> None:
        model_path = self._model_dir / f"lstm_{self.city}.h5"
        scaler_path = self._model_dir / f"lstm_scaler_{self.city}.pkl"
        if model_path.exists() and scaler_path.exists():
            try:
                from tensorflow.keras.models import load_model
                self.model = load_model(str(model_path))
                with open(scaler_path, "rb") as fh:
                    scalers = pickle.load(fh)
                self.feature_scaler = scalers["feature_scaler"]
                self.target_scaler = scalers["target_scaler"]
                logger.info("Loaded LSTM model from %s", model_path)
            except Exception:
                logger.error("Failed to load LSTM model.", exc_info=True)

    def _plot_loss(self, history) -> None:
        try:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt
            plots = Path(DATA_PLOTS_DIR)
            plots.mkdir(parents=True, exist_ok=True)
            fig, ax = plt.subplots()
            ax.plot(history.history["loss"], label="train")
            ax.plot(history.history.get("val_loss", []), label="val")
            ax.set_title(f"LSTM Loss — {self.city}")
            ax.set_xlabel("Epoch")
            ax.set_ylabel("MSE")
            ax.legend()
            fig.tight_layout()
            fig.savefig(plots / f"lstm_loss_{self.city}.png", dpi=150)
            plt.close(fig)
        except Exception:
            logger.debug("Could not plot LSTM loss.", exc_info=True)
