"""Feature engineering pipeline for air-quality forecasting."""

import logging
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.preprocessing import StandardScaler

from config import loader as cfg
from config.constants import DATA_PLOTS_DIR

logger = logging.getLogger(__name__)


def _ensure_plots_dir() -> Path:
    p = Path(DATA_PLOTS_DIR)
    p.mkdir(parents=True, exist_ok=True)
    return p


class FeatureEngineer:
    """Adds lag, rolling, cyclic, and calendar features to merged data."""

    LAG_HOURS = [1, 6, 12, 24, 168]
    ROLLING_WINDOWS = [24, 168]
    TARGET_POLLUTANTS = ["AQI", "PM2.5", "NO2", "CO"]

    def transform(self, df: pd.DataFrame, city: str) -> pd.DataFrame:
        """Apply all feature-engineering steps in sequence.

        Args:
            df: Wide-format DataFrame indexed by datetime (1 h resolution).
            city: City key used for holiday detection.

        Returns:
            Feature-enriched DataFrame with NaN-target rows dropped.
        """
        df = df.copy()
        if not isinstance(df.index, pd.DatetimeIndex):
            if "datetime" in df.columns:
                df = df.set_index("datetime")
            else:
                raise ValueError("DataFrame must have a datetime index or column.")

        df = self._add_lags(df)
        df = self._add_rolling(df)
        df = self._add_cyclic(df)
        df = self._add_holiday(df, city)
        df = self._add_season(df)

        before = len(df)
        df = df.dropna(subset=["AQI"])
        logger.info("Dropped %d rows with NaN AQI (of %d).", before - len(df), before)
        return df

    # ------------------------------------------------------------------
    # Lag features
    # ------------------------------------------------------------------
    def _add_lags(self, df: pd.DataFrame) -> pd.DataFrame:
        for col in self.TARGET_POLLUTANTS:
            if col not in df.columns:
                continue
            for h in self.LAG_HOURS:
                df[f"{col}_lag_{h}h"] = df[col].shift(h)
        return df

    # ------------------------------------------------------------------
    # Rolling statistics
    # ------------------------------------------------------------------
    def _add_rolling(self, df: pd.DataFrame) -> pd.DataFrame:
        for col in self.TARGET_POLLUTANTS:
            if col not in df.columns:
                continue
            for w in self.ROLLING_WINDOWS:
                df[f"{col}_rolling_mean_{w}h"] = df[col].rolling(w, min_periods=1).mean()
                if w == 24:
                    df[f"{col}_rolling_std_{w}h"] = df[col].rolling(w, min_periods=1).std()
        return df

    # ------------------------------------------------------------------
    # Cyclic time encodings
    # ------------------------------------------------------------------
    @staticmethod
    def _add_cyclic(df: pd.DataFrame) -> pd.DataFrame:
        idx = df.index
        df["hour_sin"] = np.sin(2 * np.pi * idx.hour / 24)
        df["hour_cos"] = np.cos(2 * np.pi * idx.hour / 24)
        df["dow_sin"] = np.sin(2 * np.pi * idx.dayofweek / 7)
        df["dow_cos"] = np.cos(2 * np.pi * idx.dayofweek / 7)
        df["month_sin"] = np.sin(2 * np.pi * idx.month / 12)
        df["month_cos"] = np.cos(2 * np.pi * idx.month / 12)
        return df

    # ------------------------------------------------------------------
    # Holiday flag
    # ------------------------------------------------------------------
    @staticmethod
    def _add_holiday(df: pd.DataFrame, city: str) -> pd.DataFrame:
        try:
            import holidays as hol_lib
            city_cfg = cfg.get_city(city)
            cc = city_cfg.get("country_code", "US")
            country_holidays = hol_lib.country_holidays(cc)
            df["holiday_flag"] = df.index.map(
                lambda dt: 1 if dt.date() in country_holidays else 0
            )
        except Exception:
            logger.warning("Holiday detection failed for %s; filling zeros.", city)
            df["holiday_flag"] = 0
        return df

    # ------------------------------------------------------------------
    # Season
    # ------------------------------------------------------------------
    @staticmethod
    def _add_season(df: pd.DataFrame) -> pd.DataFrame:
        month = df.index.month
        conditions = [
            month.isin([3, 4, 5]),
            month.isin([6, 7, 8]),
            month.isin([9, 10, 11]),
            month.isin([12, 1, 2]),
        ]
        choices = ["spring", "summer", "autumn", "winter"]
        df["season"] = np.select(conditions, choices, default="unknown")
        return df

    # ------------------------------------------------------------------
    # Feature importance
    # ------------------------------------------------------------------
    def get_top_features(
        self,
        df: pd.DataFrame,
        target: str = "AQI",
        n: int = 10,
        city: str = "unknown",
    ) -> list[str]:
        """Rank features by ExtraTreesRegressor importance.

        Args:
            df: Feature-engineered DataFrame.
            target: Target column name.
            n: Number of top features to return.
            city: City key (for plot filenames).

        Returns:
            List of top *n* feature names.
        """
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import seaborn as sns

        numeric = df.select_dtypes(include=[np.number])
        X = numeric.drop(columns=[target], errors="ignore").dropna(axis=1)
        y = df.loc[X.index, target].dropna()
        common = X.index.intersection(y.index)
        X, y = X.loc[common], y.loc[common]

        if X.empty or len(X) < 10:
            logger.warning("Insufficient data for feature importance.")
            return list(X.columns[:n])

        model = ExtraTreesRegressor(n_estimators=100, random_state=42, n_jobs=-1)
        model.fit(X, y)

        importance = pd.Series(model.feature_importances_, index=X.columns)
        importance = importance.sort_values(ascending=False)
        top = importance.head(n).index.tolist()

        plots = _ensure_plots_dir()

        # Bar chart
        fig, ax = plt.subplots(figsize=(10, 6))
        importance.head(n).plot.barh(ax=ax)
        ax.set_title(f"Feature Importance — {city}")
        ax.set_xlabel("Importance")
        fig.tight_layout()
        fig.savefig(plots / f"feature_importance_{city}.png", dpi=150)
        plt.close(fig)

        # Correlation heatmap (top features)
        corr = X[top].corr()
        fig, ax = plt.subplots(figsize=(10, 8))
        sns.heatmap(corr, annot=True, fmt=".2f", ax=ax, cmap="coolwarm")
        ax.set_title(f"Correlation Heatmap — {city}")
        fig.tight_layout()
        fig.savefig(plots / f"correlation_heatmap_{city}.png", dpi=150)
        plt.close(fig)

        # Pairplot of top 6
        pair_cols = top[:6] + [target]
        pair_cols = [c for c in pair_cols if c in df.columns]
        pair_df = df[pair_cols].dropna().head(500)
        if len(pair_df) > 10:
            g = sns.pairplot(pair_df, diag_kind="kde")
            g.fig.suptitle(f"Pairplot — {city}", y=1.02)
            g.savefig(plots / f"pairplot_{city}.png", dpi=100)
            plt.close("all")

        logger.info("Top %d features for %s: %s", n, city, top)
        return top
