"""Inverse-RMSE weighted ensemble of TFT and LSTM forecasters."""

import json
import logging
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

from config.constants import MODELS_SAVED_DIR
from models.tft_model import TFTForecaster
from models.lstm_model import LSTMForecaster

logger = logging.getLogger(__name__)


class EnsembleForecaster:
    """Combine TFT and LSTM predictions using inverse-RMSE weighting."""

    def __init__(self, tft: TFTForecaster, lstm: LSTMForecaster) -> None:
        self.tft = tft
        self.lstm = lstm
        self.weight_tft: float = 0.5
        self.weight_lstm: float = 0.5
        self.city = tft.city
        self._weights_path = Path(MODELS_SAVED_DIR) / f"ensemble_weights_{self.city}.json"
        self._weights_path.parent.mkdir(parents=True, exist_ok=True)
        self._load_weights()

    def compute_weights(self, val_df: pd.DataFrame) -> tuple[float, float]:
        """Evaluate both models and derive weights.

        Args:
            val_df: Validation DataFrame.

        Returns:
            (weight_tft, weight_lstm) tuple.
        """
        tft_metrics = self.tft.evaluate(val_df)
        lstm_metrics = self.lstm.evaluate(val_df)

        rmse_tft = tft_metrics.get("RMSE", np.inf)
        rmse_lstm = lstm_metrics.get("RMSE", np.inf)

        if rmse_tft == 0 and rmse_lstm == 0:
            self.weight_tft = 0.5
            self.weight_lstm = 0.5
        elif rmse_tft == 0:
            self.weight_tft = 1.0
            self.weight_lstm = 0.0
        elif rmse_lstm == 0:
            self.weight_tft = 0.0
            self.weight_lstm = 1.0
        else:
            inv_tft = 1.0 / rmse_tft
            inv_lstm = 1.0 / rmse_lstm
            total = inv_tft + inv_lstm
            self.weight_tft = inv_tft / total
            self.weight_lstm = inv_lstm / total

        # Persist
        with open(self._weights_path, "w") as fh:
            json.dump({"weight_tft": self.weight_tft,
                        "weight_lstm": self.weight_lstm}, fh)

        logger.info("Ensemble weights: TFT=%.3f, LSTM=%.3f",
                     self.weight_tft, self.weight_lstm)
        return self.weight_tft, self.weight_lstm

    def predict(self, df: pd.DataFrame, horizon_hours: int = 24) -> dict:
        """Generate ensemble forecast.

        Args:
            df: Recent data for prediction.
            horizon_hours: Forecast horizon.

        Returns:
            Dict with datetime, aqi_p10, aqi_p50, aqi_p90, model_used.
        """
        tft_preds = self.tft.predict(df, horizon_hours)
        lstm_preds = self.lstm.predict(df, horizon_hours)

        tft_p50 = np.array(tft_preds.get("aqi_p50", []))
        lstm_arr = np.array(lstm_preds) if isinstance(lstm_preds, np.ndarray) else np.array(lstm_preds)

        n = min(len(tft_p50), len(lstm_arr))
        if n == 0:
            # Fall back to whichever is available
            if len(tft_p50) > 0:
                return {**tft_preds, "model_used": "tft_only"}
            if len(lstm_arr) > 0:
                dt = tft_preds.get("datetime", [])
                if not dt:
                    last_dt = df.index[-1] if isinstance(df.index, pd.DatetimeIndex) else pd.Timestamp.utcnow()
                    dt = pd.date_range(last_dt + pd.Timedelta(hours=1),
                                       periods=len(lstm_arr), freq="h").tolist()
                return {
                    "datetime": dt[:len(lstm_arr)],
                    "aqi_p10": lstm_arr.tolist(),
                    "aqi_p50": lstm_arr.tolist(),
                    "aqi_p90": lstm_arr.tolist(),
                    "model_used": "lstm_only",
                }
            return {"datetime": [], "aqi_p10": [], "aqi_p50": [], "aqi_p90": [],
                    "model_used": "none"}

        tft_p50, lstm_arr = tft_p50[:n], lstm_arr[:n]
        blended = self.weight_tft * tft_p50 + self.weight_lstm * lstm_arr

        return {
            "datetime": tft_preds.get("datetime", [])[:n],
            "aqi_p10": tft_preds.get("aqi_p10", blended.tolist())[:n],
            "aqi_p50": blended.tolist(),
            "aqi_p90": tft_preds.get("aqi_p90", blended.tolist())[:n],
            "model_used": "ensemble",
        }

    def _load_weights(self) -> None:
        if self._weights_path.exists():
            try:
                with open(self._weights_path) as fh:
                    w = json.load(fh)
                self.weight_tft = w["weight_tft"]
                self.weight_lstm = w["weight_lstm"]
                logger.info("Loaded ensemble weights: TFT=%.3f, LSTM=%.3f",
                             self.weight_tft, self.weight_lstm)
            except Exception:
                logger.warning("Could not load ensemble weights; using 0.5/0.5.")
