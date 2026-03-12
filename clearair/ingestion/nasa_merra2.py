"""Fetcher for NASA MERRA-2 aerosol diagnostics via Giovanni / OPeNDAP."""

import logging
import os
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import requests

from config.constants import MERRA2_DATASET, MERRA2_VARIABLES
from ingestion.base import BaseFetcher

logger = logging.getLogger(__name__)


class MERRA2Fetcher(BaseFetcher):
    """Download MERRA-2 aerosol fields and interpolate to the city's coordinates."""

    SOURCE_NAME = "merra2"

    def __init__(self, city_key: str, config: dict) -> None:
        super().__init__(city_key, config)
        self.token: str = os.getenv("NASA_EARTHDATA_TOKEN", "")

    @staticmethod
    def _build_url(
        lat_min: float, lat_max: float,
        lon_min: float, lon_max: float,
        start_date: str, end_date: str,
        variable: str,
    ) -> str:
        """Build a Giovanni/OPeNDAP request URL for a MERRA-2 variable."""
        base = (
            "https://goldsmr4.gesdisc.eosdis.nasa.gov/opendap"
            f"/MERRA2/{MERRA2_DATASET}.5.12.4"
        )
        return (
            f"{base}?{variable}"
            f"[{start_date}:{end_date}]"
            f"[{lat_min}:{lat_max}]"
            f"[{lon_min}:{lon_max}]"
        )

    def fetch(self, start_date: str, end_date: str) -> pd.DataFrame:
        """Fetch MERRA-2 aerosol diagnostics for the city.

        Args:
            start_date: ISO date ``YYYY-MM-DD``.
            end_date:   ISO date ``YYYY-MM-DD``.

        Returns:
            Standardised DataFrame with aerosol columns.
        """
        cache_nc = self._cache_path(
            "merra2", f"{self.city_key}_{start_date}_{end_date}.nc"
        )
        cache_parquet = cache_nc.with_suffix(".parquet")

        if cache_parquet.exists():
            logger.info("Loading cached MERRA-2 from %s", cache_parquet)
            return pd.read_parquet(cache_parquet)

        delta = 0.5
        lat_min, lat_max = self.lat - delta, self.lat + delta
        lon_min, lon_max = self.lon - delta, self.lon + delta

        headers = {"Authorization": f"Bearer {self.token}"} if self.token else {}

        var_frames: list[pd.DataFrame] = []

        for var in MERRA2_VARIABLES:
            url = self._build_url(
                lat_min, lat_max, lon_min, lon_max,
                start_date, end_date, var,
            )
            try:
                resp = self._retry_request(
                    requests.get, url, headers=headers, timeout=120,
                )
                # Attempt xarray open from bytes if netCDF
                try:
                    import xarray as xr
                    tmp_path = self._cache_path("merra2", f"_tmp_{var}.nc")
                    tmp_path.write_bytes(resp.content)
                    ds = xr.open_dataset(tmp_path)
                    # Interpolate to city point
                    point = ds.interp(lat=self.lat, lon=self.lon, method="nearest")
                    series = point[var].to_series()
                    var_frames.append(
                        series.resample("1h").interpolate(method="linear")
                        .ffill(limit=3).to_frame(name=var)
                    )
                    ds.close()
                    tmp_path.unlink(missing_ok=True)
                except Exception:
                    logger.warning("xarray parsing failed for %s", var,
                                   exc_info=True)
            except Exception:
                logger.error("MERRA-2 download failed for %s / %s",
                             self.city_key, var, exc_info=True)

        if not var_frames:
            logger.warning("No MERRA-2 data for %s.", self.city_key)
            return pd.DataFrame()

        df = pd.concat(var_frames, axis=1)
        df = df.reset_index()
        df.rename(columns={"index": "datetime"}, inplace=True)
        df["datetime"] = pd.to_datetime(df["datetime"], utc=True)
        df["city"] = self.city_key
        df["lat"] = self.lat
        df["lon"] = self.lon
        df["source"] = self.SOURCE_NAME

        df.to_parquet(cache_parquet, index=False)
        logger.info("Cached %d MERRA-2 rows → %s", len(df), cache_parquet)
        return df
