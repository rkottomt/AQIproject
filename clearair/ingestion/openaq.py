"""Fetcher for the OpenAQ v3 API."""

import logging
import os
from typing import Any

import pandas as pd
import requests

from config.constants import (
    OPENAQ_BASE_URL,
    OPENAQ_PAGE_LIMIT,
    OPENAQ_PARAM_MAP,
)
from ingestion.base import BaseFetcher

logger = logging.getLogger(__name__)


class OpenAQFetcher(BaseFetcher):
    """Pull pollutant measurements from OpenAQ v3 for configured location IDs."""

    SOURCE_NAME = "openaq"
    PARAMETERS = ["pm25", "pm10", "no2", "so2", "o3", "co"]

    def __init__(self, city_key: str, config: dict) -> None:
        super().__init__(city_key, config)
        self.api_key: str = os.getenv("OPENAQ_API_KEY", "")
        self.location_ids: list[int] = config.get("openaq_location_ids", [])

    def fetch(self, start_date: str, end_date: str) -> pd.DataFrame:
        """Fetch measurements for every configured location between dates.

        Args:
            start_date: ISO date string ``YYYY-MM-DD``.
            end_date:   ISO date string ``YYYY-MM-DD``.

        Returns:
            Standardised DataFrame with pollutant readings.
        """
        cache_file = self._cache_path(
            f"openaq", f"{self.city_key}_{start_date}_{end_date}.parquet"
        )
        if cache_file.exists():
            logger.info("Loading cached OpenAQ data from %s", cache_file)
            return pd.read_parquet(cache_file)

        all_rows: list[dict[str, Any]] = []
        headers = {"X-API-Key": self.api_key} if self.api_key else {}

        for loc_id in self.location_ids:
            page = 1
            while True:
                url = (
                    f"{OPENAQ_BASE_URL}/locations/{loc_id}/measurements"
                )
                params = {
                    "date_from": start_date,
                    "date_to": end_date,
                    "limit": OPENAQ_PAGE_LIMIT,
                    "page": page,
                }
                try:
                    resp = self._retry_request(
                        requests.get, url, params=params, headers=headers,
                        timeout=30,
                    )
                    data = resp.json()
                except Exception:
                    logger.error(
                        "OpenAQ request failed for location %d, page %d",
                        loc_id, page, exc_info=True,
                    )
                    break

                results = data.get("results", [])
                if not results:
                    break

                for rec in results:
                    param_raw = rec.get("parameter", {})
                    param_name = param_raw if isinstance(param_raw, str) else param_raw.get("name", "")
                    std_name = OPENAQ_PARAM_MAP.get(param_name, param_name)
                    all_rows.append({
                        "datetime": rec.get("date", {}).get("utc")
                                    if isinstance(rec.get("date"), dict)
                                    else rec.get("date"),
                        "city": self.city_key,
                        "lat": self.lat,
                        "lon": self.lon,
                        "parameter": std_name,
                        "value": rec.get("value"),
                        "unit": rec.get("unit", "µg/m³"),
                        "source": self.SOURCE_NAME,
                    })

                if len(results) < OPENAQ_PAGE_LIMIT:
                    break
                page += 1

        if not all_rows:
            logger.warning("No OpenAQ data retrieved for %s.", self.city_key)
            return pd.DataFrame(
                columns=["datetime", "city", "lat", "lon",
                          "parameter", "value", "unit", "source"]
            )

        df = pd.DataFrame(all_rows)
        df = self._standardize(df)
        if self._validate(df):
            df.to_parquet(cache_file, index=False)
            logger.info("Cached %d OpenAQ rows → %s", len(df), cache_file)
        return df
