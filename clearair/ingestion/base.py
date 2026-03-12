"""Abstract base class shared by every data fetcher."""

import logging
import os
import time
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Optional

import pandas as pd

from config.constants import MAX_API_RETRIES, RETRY_BASE_DELAY_S

logger = logging.getLogger(__name__)


class BaseFetcher(ABC):
    """Common interface for all external-data fetchers.

    Subclasses must implement ``fetch`` and may override ``_standardize``.
    """

    def __init__(self, city_key: str, config: dict) -> None:
        self.city_key = city_key
        self.config = config
        self.lat: float = config["lat"]
        self.lon: float = config["lon"]
        self._ensure_dirs()

    # ------------------------------------------------------------------
    # Abstract
    # ------------------------------------------------------------------
    @abstractmethod
    def fetch(self, start_date: str, end_date: str) -> pd.DataFrame:
        """Fetch data for *city_key* between ISO-formatted dates (inclusive).

        Args:
            start_date: ``YYYY-MM-DD``
            end_date:   ``YYYY-MM-DD``

        Returns:
            Standardised :class:`~pandas.DataFrame`.
        """

    # ------------------------------------------------------------------
    # Standardisation
    # ------------------------------------------------------------------
    def _standardize(self, df: pd.DataFrame) -> pd.DataFrame:
        """Ensure the DataFrame has the canonical column set.

        Expected columns:
            datetime, city, lat, lon, parameter, value, unit, source
        """
        required = {"datetime", "city", "lat", "lon", "parameter", "value",
                     "unit", "source"}
        missing = required - set(df.columns)
        if missing:
            raise ValueError(f"Standardised DF missing columns: {missing}")
        df["datetime"] = pd.to_datetime(df["datetime"], utc=True)
        return df[list(required)]

    # ------------------------------------------------------------------
    # Validation
    # ------------------------------------------------------------------
    def _validate(self, df: pd.DataFrame) -> bool:
        """Quick sanity check — no critical nulls and numeric values."""
        if df.empty:
            logger.warning("Validation: DataFrame is empty.")
            return False
        critical = ["datetime", "city", "parameter"]
        for col in critical:
            if df[col].isnull().any():
                logger.warning("Validation: nulls found in '%s'.", col)
                return False
        if not pd.api.types.is_numeric_dtype(df["value"]):
            logger.warning("Validation: 'value' column is not numeric.")
            return False
        return True

    # ------------------------------------------------------------------
    # HTTP helpers
    # ------------------------------------------------------------------
    @staticmethod
    def _retry_request(
        request_fn,
        *args,
        max_retries: int = MAX_API_RETRIES,
        base_delay: float = RETRY_BASE_DELAY_S,
        **kwargs,
    ):
        """Execute *request_fn* with exponential back-off on 429 / 5xx."""
        import requests

        last_exc: Optional[Exception] = None
        for attempt in range(max_retries + 1):
            try:
                resp = request_fn(*args, **kwargs)
                if resp.status_code == 429 or resp.status_code >= 500:
                    raise requests.HTTPError(
                        f"HTTP {resp.status_code}", response=resp
                    )
                resp.raise_for_status()
                return resp
            except requests.RequestException as exc:
                last_exc = exc
                delay = base_delay * (2 ** attempt)
                logger.warning(
                    "Request failed (attempt %d/%d): %s — retrying in %.1fs",
                    attempt + 1, max_retries + 1, exc, delay,
                )
                time.sleep(delay)
        raise RuntimeError(
            f"All {max_retries + 1} request attempts failed."
        ) from last_exc

    # ------------------------------------------------------------------
    # Directory helpers
    # ------------------------------------------------------------------
    @staticmethod
    def _ensure_dirs() -> None:
        for d in ("data/raw", "data/processed", "data/plots", "models/saved"):
            Path(d).mkdir(parents=True, exist_ok=True)

    @staticmethod
    def _cache_path(subdir: str, filename: str) -> Path:
        path = Path("data/raw") / subdir
        path.mkdir(parents=True, exist_ok=True)
        return path / filename
