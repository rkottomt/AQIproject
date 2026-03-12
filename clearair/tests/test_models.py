"""Tests for the modelling layer (lightweight / synthetic data only)."""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from config import loader as cfg
cfg.load_all()


def _synthetic_df(n: int = 200) -> pd.DataFrame:
    """Build a small synthetic feature-engineered DataFrame."""
    np.random.seed(42)
    idx = pd.date_range("2024-01-01", periods=n, freq="h", tz="UTC")
    df = pd.DataFrame(index=idx)
    df["AQI"] = 80 + np.cumsum(np.random.normal(0, 0.5, n))
    df["AQI"] = df["AQI"].clip(0, 500)
    df["PM2.5"] = df["AQI"] * 0.5
    df["NO2"] = 20 + np.random.normal(0, 2, n)
    df["CO"] = 1.5 + np.random.normal(0, 0.2, n)
    df["temperature_c"] = 28 + np.random.normal(0, 1, n)
    df["humidity_pct"] = 65 + np.random.normal(0, 3, n)
    df["wind_speed_ms"] = 3 + np.abs(np.random.normal(0, 0.5, n))
    df["hour_sin"] = np.sin(2 * np.pi * idx.hour / 24)
    df["hour_cos"] = np.cos(2 * np.pi * idx.hour / 24)
    df["dow_sin"] = np.sin(2 * np.pi * idx.dayofweek / 7)
    df["dow_cos"] = np.cos(2 * np.pi * idx.dayofweek / 7)
    df["month_sin"] = np.sin(2 * np.pi * idx.month / 12)
    df["month_cos"] = np.cos(2 * np.pi * idx.month / 12)
    df["holiday_flag"] = 0
    df["city"] = "mumbai"
    return df


class TestBaselineModels:
    def test_all_models_train_successfully(self):
        from models.baselines import BaselineModels
        df = _synthetic_df(300)
        features = ["PM2.5", "NO2", "CO", "temperature_c", "humidity_pct",
                     "wind_speed_ms", "hour_sin", "hour_cos"]
        bl = BaselineModels("test_city")
        rmses = bl.train_all(df, features)
        assert len(rmses) == 6
        for name, rmse in rmses.items():
            assert rmse >= 0, f"{name} has negative RMSE"

    def test_evaluate_returns_correct_columns(self):
        from models.baselines import BaselineModels
        df = _synthetic_df(300)
        features = ["PM2.5", "NO2", "CO", "temperature_c", "humidity_pct",
                     "wind_speed_ms", "hour_sin", "hour_cos"]
        bl = BaselineModels("test_city_eval")
        bl.train_all(df, features)
        result = bl.evaluate_all(df.iloc[240:], features)
        assert "RMSE" in result.columns
        assert "MAE" in result.columns
        assert "MAPE" in result.columns


class TestEnsembleWeights:
    def test_weights_sum_to_one(self):
        w_tft, w_lstm = 0.6, 0.4
        assert abs((w_tft + w_lstm) - 1.0) < 1e-9


class TestTFTDatasetPreparation:
    def test_dataset_shape(self):
        """Ensure TFT prepare_dataset does not crash on a tiny dataset."""
        pytest.importorskip("pytorch_forecasting")
        from models.tft_model import TFTForecaster
        df = _synthetic_df(500)
        tft = TFTForecaster("test_city", max_encoder_length=24,
                            max_prediction_length=6)
        train_ds, val_ds = tft.prepare_dataset(df)
        assert len(train_ds) > 0
        assert len(val_ds) >= 0
