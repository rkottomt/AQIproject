"""Fetcher for the Open-Meteo weather API (forecast + historical archive)."""

import logging
import os
from typing import Any

import pandas as pd
import requests

from ingestion.base import BaseFetcher

logger = logging.getLogger(__name__)

OPENMETEO_BASE = os.getenv("OPENMETEO_BASE_URL", "https://api.open-meteo.com/v1")
ARCHIVE_BASE = "https://archive-api.open-meteo.com/v1/archive"

HOURLY_VARS = [
    "temperature_2m",
    "relative_humidity_2m",
    "wind_speed_10m",
    "wind_direction_10m",
    "precipitation",
    "uv_index",
    "surface_pressure",
    "visibility",
]


class OpenMeteoFetcher(BaseFetcher):
    """Retrieve hourly meteorological variables for a city's lat/lon."""

    SOURCE_NAME = "openmeteo"

    def fetch(self, start_date: str, end_date: str) -> pd.DataFrame:
        """Fetch weather data from the Open-Meteo archive API.

        Args:
            start_date: ISO date ``YYYY-MM-DD``.
            end_date:   ISO date ``YYYY-MM-DD``.

        Returns:
            Wide-format DataFrame with one column per weather variable.
        """
        cache_file = self._cache_path(
            "weather", f"{self.city_key}_{start_date}_{end_date}.parquet"
        )
        if cache_file.exists():
            logger.info("Loading cached weather from %s", cache_file)
            return pd.read_parquet(cache_file)

        params: dict[str, Any] = {
            "latitude": self.lat,
            "longitude": self.lon,
            "start_date": start_date,
            "end_date": end_date,
            "hourly": ",".join(HOURLY_VARS),
            "timezone": "UTC",
        }

        try:
            resp = self._retry_request(
                requests.get, ARCHIVE_BASE, params=params, timeout=60,
            )
            data = resp.json()
        except Exception:
            logger.error("Open-Meteo request failed for %s", self.city_key,
                         exc_info=True)
            return pd.DataFrame()

        hourly = data.get("hourly", {})
        times = hourly.get("time", [])
        if not times:
            logger.warning("No hourly data returned for %s.", self.city_key)
            return pd.DataFrame()

        records: dict[str, list] = {"datetime": times}
        for var in HOURLY_VARS:
            records[var] = hourly.get(var, [None] * len(times))

        df = pd.DataFrame(records)
        df["datetime"] = pd.to_datetime(df["datetime"], utc=True)
        df["city"] = self.city_key
        df["lat"] = self.lat
        df["lon"] = self.lon
        df["source"] = self.SOURCE_NAME

        # Rename to standard weather column names
        rename_map = {
            "temperature_2m": "temperature_c",
            "relative_humidity_2m": "humidity_pct",
            "wind_speed_10m": "wind_speed_ms",
            "wind_direction_10m": "wind_direction_deg",
            "precipitation": "precipitation_mm",
        }
        df.rename(columns=rename_map, inplace=True)

        df.to_parquet(cache_file, index=False)
        logger.info("Cached %d weather rows → %s", len(df), cache_file)
        return df
