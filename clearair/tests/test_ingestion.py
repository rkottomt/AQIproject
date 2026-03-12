"""Tests for the data ingestion layer."""

import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from config import loader as cfg

cfg.load_all()

MUMBAI_CFG = cfg.get_city("mumbai")
CHICAGO_CFG = cfg.get_city("chicago")


class TestOpenAQFetcher:
    """Tests for ingestion.openaq.OpenAQFetcher."""

    def test_returns_standardized_columns(self):
        from ingestion.openaq import OpenAQFetcher

        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = {
            "results": [
                {
                    "date": {"utc": "2024-01-01T00:00:00Z"},
                    "parameter": "pm25",
                    "value": 42.0,
                    "unit": "µg/m³",
                },
            ]
        }

        with patch("ingestion.openaq.requests.get", return_value=mock_resp):
            fetcher = OpenAQFetcher("mumbai", MUMBAI_CFG)
            # Bypass cache
            with patch.object(Path, "exists", return_value=False):
                df = fetcher.fetch("2024-01-01", "2024-01-02")

        expected_cols = {"datetime", "city", "lat", "lon", "parameter",
                         "value", "unit", "source"}
        assert expected_cols.issubset(set(df.columns))


class TestOpenMeteoFetcher:
    """Tests for ingestion.openmeteo.OpenMeteoFetcher."""

    def test_returns_all_weather_variables(self):
        from ingestion.openmeteo import OpenMeteoFetcher

        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = {
            "hourly": {
                "time": ["2024-01-01T00:00", "2024-01-01T01:00"],
                "temperature_2m": [25.0, 25.5],
                "relative_humidity_2m": [70, 72],
                "wind_speed_10m": [3.5, 4.0],
                "wind_direction_10m": [180, 190],
                "precipitation": [0.0, 0.1],
                "uv_index": [3, 4],
                "surface_pressure": [1013, 1012],
                "visibility": [10000, 9800],
            }
        }

        with patch("ingestion.openmeteo.requests.get", return_value=mock_resp):
            fetcher = OpenMeteoFetcher("mumbai", MUMBAI_CFG)
            with patch.object(Path, "exists", return_value=False):
                df = fetcher.fetch("2024-01-01", "2024-01-02")

        assert "temperature_c" in df.columns
        assert "wind_speed_ms" in df.columns
        assert len(df) == 2


class TestEPAFetcher:
    """Tests for ingestion.epa.EPAFetcher."""

    def test_skips_non_epa_cities(self):
        from ingestion.epa import EPAFetcher

        fetcher = EPAFetcher("mumbai", MUMBAI_CFG)
        df = fetcher.fetch("2024-01-01", "2024-01-31")
        assert df.empty

    def test_fetches_for_epa_cities(self):
        from ingestion.epa import EPAFetcher

        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = {
            "Data": [
                {
                    "date_local": "2024-01-01",
                    "arithmetic_mean": 12.5,
                    "units_of_measure": "µg/m³",
                }
            ]
        }

        with patch("ingestion.epa.requests.get", return_value=mock_resp):
            fetcher = EPAFetcher("chicago", CHICAGO_CFG)
            with patch.object(Path, "exists", return_value=False):
                df = fetcher.fetch("2024-01-01", "2024-01-31")

        assert not df.empty


class TestMERRA2URLGeneration:
    """Tests for ingestion.nasa_merra2.MERRA2Fetcher URL builder."""

    def test_url_contains_variable_and_bbox(self):
        from ingestion.nasa_merra2 import MERRA2Fetcher

        url = MERRA2Fetcher._build_url(
            lat_min=18.5, lat_max=19.5,
            lon_min=72.3, lon_max=73.3,
            start_date="2024-01-01", end_date="2024-01-31",
            variable="DUSMASS",
        )
        assert "DUSMASS" in url
        assert "18.5" in url
        assert "MERRA2" in url
