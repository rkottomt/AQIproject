"""Merge air-quality, weather, traffic, and satellite data into a single wide table."""

import logging
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

from config.constants import AQI_BREAKPOINTS, DATA_RAW_DIR

logger = logging.getLogger(__name__)


def _concentration_to_aqi(value: float, breakpoints: list[tuple[float, float, int, int]]) -> float:
    """Convert a single pollutant concentration to a sub-index using EPA breakpoints."""
    for c_lo, c_hi, i_lo, i_hi in breakpoints:
        if c_lo <= value <= c_hi:
            return ((i_hi - i_lo) / (c_hi - c_lo)) * (value - c_lo) + i_lo
    return np.nan


class DataMerger:
    """Combines data from every source into a single hourly wide-format DataFrame."""

    POLLUTANTS = ["PM2.5", "PM10", "NO2", "SO2", "O3", "CO"]

    def merge_all(
        self,
        city: str,
        start: str,
        end: str,
        aq_df: Optional[pd.DataFrame] = None,
        weather_df: Optional[pd.DataFrame] = None,
        traffic_df: Optional[pd.DataFrame] = None,
        merra2_df: Optional[pd.DataFrame] = None,
    ) -> pd.DataFrame:
        """Merge all available data sources for *city* into a single wide table.

        Args:
            city: City key (must match config).
            start: ISO start date.
            end: ISO end date.
            aq_df: Pre-loaded air-quality data (if None, tries parquet cache).
            weather_df: Pre-loaded weather data.
            traffic_df: Pre-loaded traffic data.
            merra2_df: Pre-loaded MERRA-2 data.

        Returns:
            Wide DataFrame indexed by ``datetime`` at 1-hour resolution.
        """
        raw = Path(DATA_RAW_DIR)

        # --- air quality (long → wide) --------------------------------
        if aq_df is None:
            aq_df = self._load_parquets(raw, "openaq", city, start, end)
            epa = self._load_parquets(raw, "epa", city, start, end)
            if not epa.empty:
                aq_df = pd.concat([aq_df, epa], ignore_index=True)

        if not aq_df.empty:
            aq_df["datetime"] = pd.to_datetime(aq_df["datetime"], utc=True)
            aq_wide = (
                aq_df
                .pivot_table(index="datetime", columns="parameter",
                             values="value", aggfunc="mean")
                .resample("1h").mean()
            )
        else:
            aq_wide = pd.DataFrame()

        # --- weather ---------------------------------------------------
        if weather_df is None:
            weather_df = self._load_parquets(raw, "weather", city, start, end)

        if not weather_df.empty:
            weather_df["datetime"] = pd.to_datetime(weather_df["datetime"], utc=True)
            weather_df = weather_df.set_index("datetime").resample("1h").mean(numeric_only=True)
        else:
            weather_df = pd.DataFrame()

        # --- traffic ---------------------------------------------------
        if traffic_df is None:
            traffic_df = self._load_parquets(raw, "traffic", city, start, end)

        if not traffic_df.empty:
            traffic_df["datetime"] = pd.to_datetime(traffic_df["datetime"], utc=True)
            traffic_df = traffic_df.set_index("datetime").resample("1h").mean(numeric_only=True)
        else:
            traffic_df = pd.DataFrame()

        # --- MERRA-2 ---------------------------------------------------
        if merra2_df is None:
            merra2_df = self._load_parquets(raw, "merra2", city, start, end)

        if not merra2_df.empty:
            merra2_df["datetime"] = pd.to_datetime(merra2_df["datetime"], utc=True)
            merra2_df = merra2_df.set_index("datetime").resample("1h").mean(numeric_only=True)
        else:
            merra2_df = pd.DataFrame()

        # --- outer join all --------------------------------------------
        frames = [f for f in [aq_wide, weather_df, traffic_df, merra2_df]
                  if not f.empty]
        if not frames:
            logger.warning("No data available for %s (%s — %s).", city, start, end)
            return pd.DataFrame()

        merged = frames[0]
        for f in frames[1:]:
            merged = merged.join(f, how="outer")

        # --- compute composite AQI ------------------------------------
        merged["AQI"] = self._compute_aqi(merged)

        merged["city"] = city
        logger.info("Merged %d rows for %s.", len(merged), city)
        return merged

    def _compute_aqi(self, df: pd.DataFrame) -> pd.Series:
        """Compute overall AQI as the max sub-index across available pollutants."""
        sub_indices = pd.DataFrame(index=df.index)
        for pollutant, bps in AQI_BREAKPOINTS.items():
            if pollutant in df.columns:
                sub_indices[pollutant] = df[pollutant].apply(
                    lambda v: _concentration_to_aqi(v, bps) if pd.notna(v) else np.nan
                )
        if sub_indices.empty:
            return pd.Series(np.nan, index=df.index)
        return sub_indices.max(axis=1)

    @staticmethod
    def _load_parquets(raw: Path, subdir: str, city: str,
                       start: str, end: str) -> pd.DataFrame:
        """Load all parquet files matching the city/date pattern from a subdirectory."""
        folder = raw / subdir
        if not folder.exists():
            return pd.DataFrame()
        files = list(folder.glob(f"{city}*.parquet"))
        if not files:
            return pd.DataFrame()
        dfs = [pd.read_parquet(f) for f in files]
        return pd.concat(dfs, ignore_index=True)
