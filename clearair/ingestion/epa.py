"""Fetcher for the US EPA Air Quality System (AQS) daily data API."""

import logging
import os
from typing import Any

import pandas as pd
import requests

from config.constants import EPA_BASE_URL
from ingestion.base import BaseFetcher

logger = logging.getLogger(__name__)


class EPAFetcher(BaseFetcher):
    """Pull daily pollutant data from the EPA AQS API for US cities only.

    Only runs for cities where ``has_epa_data`` is True in the city config.
    """

    SOURCE_NAME = "epa"
    PARAMETERS = ["88101", "88502", "42602", "42401", "44201", "42101"]
    PARAM_NAME_MAP = {
        "88101": "PM2.5",
        "88502": "PM2.5",
        "42602": "NO2",
        "42401": "SO2",
        "44201": "O3",
        "42101": "CO",
    }

    def __init__(self, city_key: str, config: dict) -> None:
        super().__init__(city_key, config)
        self.api_key: str = os.getenv("EPA_API_KEY", "")
        self.has_epa_data: bool = config.get("has_epa_data", False)
        self.state_code: str = config.get("epa_state_code", "")
        self.county_code: str = config.get("epa_county_code", "")

    def fetch(self, start_date: str, end_date: str) -> pd.DataFrame:
        """Fetch daily AQS data for the configured state/county.

        Args:
            start_date: ISO date ``YYYY-MM-DD``.
            end_date:   ISO date ``YYYY-MM-DD``.

        Returns:
            Standardised DataFrame.
        """
        if not self.has_epa_data:
            logger.info("Skipping EPA fetch for %s (no EPA data configured).",
                        self.city_key)
            return pd.DataFrame(
                columns=["datetime", "city", "lat", "lon",
                          "parameter", "value", "unit", "source"]
            )

        cache_file = self._cache_path(
            "epa", f"{self.city_key}_{start_date}_{end_date}.parquet"
        )
        if cache_file.exists():
            logger.info("Loading cached EPA data from %s", cache_file)
            return pd.read_parquet(cache_file)

        all_rows: list[dict[str, Any]] = []
        begin = start_date.replace("-", "")
        end = end_date.replace("-", "")

        for param_code in self.PARAMETERS:
            url = f"{EPA_BASE_URL}/dailyData/byCounty"
            params = {
                "email": "clearair@example.com",
                "key": self.api_key,
                "param": param_code,
                "bdate": begin,
                "edate": end,
                "state": self.state_code,
                "county": self.county_code,
            }

            try:
                resp = self._retry_request(
                    requests.get, url, params=params, timeout=60,
                )
                data = resp.json()
            except Exception:
                logger.error(
                    "EPA request failed for param %s, %s",
                    param_code, self.city_key, exc_info=True,
                )
                continue

            for rec in data.get("Data", []):
                all_rows.append({
                    "datetime": rec.get("date_local"),
                    "city": self.city_key,
                    "lat": self.lat,
                    "lon": self.lon,
                    "parameter": self.PARAM_NAME_MAP.get(param_code,
                                                         param_code),
                    "value": rec.get("arithmetic_mean"),
                    "unit": rec.get("units_of_measure", "µg/m³"),
                    "source": self.SOURCE_NAME,
                })

        if not all_rows:
            logger.warning("No EPA data for %s.", self.city_key)
            return pd.DataFrame(
                columns=["datetime", "city", "lat", "lon",
                          "parameter", "value", "unit", "source"]
            )

        df = pd.DataFrame(all_rows)
        df = self._standardize(df)
        if self._validate(df):
            df.to_parquet(cache_file, index=False)
            logger.info("Cached %d EPA rows → %s", len(df), cache_file)
        return df
