"""Fetcher for TomTom Traffic Flow API."""

import logging
import os
from datetime import datetime
from typing import Any

import numpy as np
import pandas as pd
import requests

from config.constants import PEAK_HOURS, TOMTOM_BASE_URL
from ingestion.base import BaseFetcher

logger = logging.getLogger(__name__)


def _grid_coords(lat: float, lon: float, n: int = 5,
                 delta: float = 0.02) -> list[tuple[float, float]]:
    """Generate *n* evenly-spaced coordinate pairs in a small grid."""
    offsets = np.linspace(-delta, delta, max(int(np.sqrt(n)), 2))
    coords = []
    for dlat in offsets:
        for dlon in offsets:
            coords.append((lat + dlat, lon + dlon))
            if len(coords) >= n:
                return coords
    return coords[:n]


class TomTomFetcher(BaseFetcher):
    """Retrieve real-time traffic flow data from TomTom and derive congestion metrics."""

    SOURCE_NAME = "tomtom"

    def __init__(self, city_key: str, config: dict) -> None:
        super().__init__(city_key, config)
        self.api_key: str = os.getenv("TOMTOM_API_KEY", "")

    def fetch(self, start_date: str, end_date: str) -> pd.DataFrame:
        """Fetch current traffic snapshot for the city.

        TomTom only provides real-time data, so *start_date*/*end_date* are
        recorded but the actual query hits the live endpoint.

        Args:
            start_date: Stored for logging/caching.
            end_date:   Stored for logging/caching.

        Returns:
            DataFrame with congestion_index, peak_hour_flag, avg_speed_kmh.
        """
        cache_file = self._cache_path(
            "traffic", f"{self.city_key}_{start_date}_{end_date}.parquet"
        )
        if cache_file.exists():
            logger.info("Loading cached traffic from %s", cache_file)
            return pd.read_parquet(cache_file)

        coords = _grid_coords(self.lat, self.lon)
        speeds: list[float] = []
        free_flows: list[float] = []

        for lat, lon in coords:
            url = TOMTOM_BASE_URL
            params = {
                "key": self.api_key,
                "point": f"{lat},{lon}",
            }
            try:
                resp = self._retry_request(
                    requests.get, url, params=params, timeout=15,
                )
                data = resp.json()
                seg = data.get("flowSegmentData", {})
                speeds.append(seg.get("currentSpeed", 0))
                free_flows.append(seg.get("freeFlowSpeed", 1))
            except Exception:
                logger.warning("TomTom failed for point (%.4f, %.4f)",
                               lat, lon, exc_info=True)

        now = datetime.utcnow()
        avg_speed = float(np.mean(speeds)) if speeds else 0.0
        avg_ff = float(np.mean(free_flows)) if free_flows else 1.0
        congestion = 1.0 - (avg_speed / avg_ff) if avg_ff else 0.0
        peak_flag = 1 if now.hour in PEAK_HOURS else 0

        row = {
            "datetime": pd.Timestamp(now, tz="UTC"),
            "city": self.city_key,
            "congestion_index": max(0.0, min(1.0, congestion)),
            "peak_hour_flag": peak_flag,
            "avg_speed_kmh": avg_speed,
        }
        df = pd.DataFrame([row])
        df.to_parquet(cache_file, index=False)
        logger.info("Cached traffic snapshot → %s", cache_file)
        return df
