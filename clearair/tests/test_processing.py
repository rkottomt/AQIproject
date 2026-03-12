"""Tests for the processing layer."""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from config import loader as cfg
cfg.load_all()


def _make_hourly_df(n_hours: int = 720, base_aqi: float = 80) -> pd.DataFrame:
    """Create a synthetic hourly DataFrame for testing."""
    np.random.seed(42)
    idx = pd.date_range("2024-01-01", periods=n_hours, freq="h", tz="UTC")
    df = pd.DataFrame(index=idx)
    df["AQI"] = base_aqi + np.cumsum(np.random.normal(0, 1, n_hours))
    df["AQI"] = df["AQI"].clip(0, 500)
    df["PM2.5"] = df["AQI"] * 0.5 + np.random.normal(0, 2, n_hours)
    df["NO2"] = 20 + np.random.normal(0, 3, n_hours)
    df["CO"] = 1.5 + np.random.normal(0, 0.3, n_hours)
    df["temperature_c"] = 28 + np.random.normal(0, 2, n_hours)
    df["humidity_pct"] = 65 + np.random.normal(0, 5, n_hours)
    df["wind_speed_ms"] = 3 + np.abs(np.random.normal(0, 1, n_hours))
    df["city"] = "mumbai"
    return df


class TestFeatureEngineer:
    def test_creates_lag_columns(self):
        from processing.features import FeatureEngineer
        fe = FeatureEngineer()
        df = _make_hourly_df()
        result = fe.transform(df, "mumbai")
        assert "AQI_lag_1h" in result.columns
        assert "AQI_lag_24h" in result.columns
        assert "PM2.5_lag_168h" in result.columns

    def test_creates_cyclic_encodings(self):
        from processing.features import FeatureEngineer
        fe = FeatureEngineer()
        df = _make_hourly_df()
        result = fe.transform(df, "mumbai")
        for col in ["hour_sin", "hour_cos", "dow_sin", "dow_cos",
                     "month_sin", "month_cos"]:
            assert col in result.columns
        assert result["hour_sin"].between(-1, 1).all()

    def test_creates_rolling_stats(self):
        from processing.features import FeatureEngineer
        fe = FeatureEngineer()
        df = _make_hourly_df()
        result = fe.transform(df, "mumbai")
        assert "AQI_rolling_mean_24h" in result.columns
        assert "AQI_rolling_std_24h" in result.columns

    def test_creates_holiday_and_season(self):
        from processing.features import FeatureEngineer
        fe = FeatureEngineer()
        df = _make_hourly_df()
        result = fe.transform(df, "mumbai")
        assert "holiday_flag" in result.columns
        assert "season" in result.columns


class TestDataValidator:
    def test_catches_aqi_out_of_range(self):
        from processing.validator import DataValidator
        df = _make_hourly_df(n_hours=100)
        df.loc[df.index[0], "AQI"] = 600
        v = DataValidator()
        is_valid, errors = v.validate(df)
        assert not is_valid
        assert any("outside [0, 500]" in e for e in errors)

    def test_catches_large_gaps(self):
        from processing.validator import DataValidator
        df = _make_hourly_df(n_hours=200)
        # Drop 10 consecutive hours to create a gap
        df = df.drop(df.index[50:60])
        v = DataValidator()
        is_valid, errors = v.validate(df)
        assert any("gaps" in e.lower() for e in errors)

    def test_passes_valid_data(self):
        from processing.validator import DataValidator
        df = _make_hourly_df(n_hours=800)
        v = DataValidator()
        is_valid, errors = v.validate(df)
        assert is_valid
        assert len(errors) == 0


class TestMerger:
    def test_produces_one_row_per_hour(self):
        from processing.merger import DataMerger
        merger = DataMerger()
        # Provide pre-built DataFrames
        aq_rows = []
        idx = pd.date_range("2024-01-01", periods=48, freq="h", tz="UTC")
        for dt in idx:
            aq_rows.append({
                "datetime": dt, "city": "mumbai",
                "lat": 19.076, "lon": 72.877,
                "parameter": "PM2.5", "value": 42.0,
                "unit": "µg/m³", "source": "test",
            })
        aq_df = pd.DataFrame(aq_rows)

        result = merger.merge_all(
            "mumbai", "2024-01-01", "2024-01-02", aq_df=aq_df,
        )
        # Should be hourly
        assert not result.empty
        diffs = result.index.to_series().diff().dropna()
        assert (diffs == pd.Timedelta(hours=1)).all()
