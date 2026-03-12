"""Data-quality validation checks before model training."""

import logging

import pandas as pd

logger = logging.getLogger(__name__)


class DataValidator:
    """Run a suite of sanity checks on a merged/feature-engineered DataFrame."""

    def validate(self, df: pd.DataFrame) -> tuple[bool, list[str]]:
        """Execute all validation checks.

        Args:
            df: DataFrame with a datetime index and an ``AQI`` column.

        Returns:
            Tuple of (is_valid, list_of_error_messages).
        """
        errors: list[str] = []

        if df.empty:
            return False, ["DataFrame is empty."]

        if not isinstance(df.index, pd.DatetimeIndex):
            errors.append("Index is not a DatetimeIndex.")
            return False, errors

        # Duplicate timestamps
        dupes = df.index.duplicated().sum()
        if dupes:
            errors.append(f"Found {dupes} duplicate datetime entries.")

        # AQI range
        if "AQI" in df.columns:
            out_of_range = ((df["AQI"] < 0) | (df["AQI"] > 500)).sum()
            if out_of_range:
                errors.append(f"{out_of_range} AQI values outside [0, 500].")
        else:
            errors.append("Missing 'AQI' column.")

        # Gaps > 6 hours
        diffs = df.index.to_series().diff()
        big_gaps = (diffs > pd.Timedelta(hours=6)).sum()
        if big_gaps:
            errors.append(f"Found {big_gaps} gaps > 6 hours in datetime index.")

        # Minimum span
        span_days = (df.index.max() - df.index.min()).days
        if span_days < 30:
            errors.append(
                f"Only {span_days} days of data present (minimum 30 required)."
            )

        is_valid = len(errors) == 0
        if is_valid:
            logger.info("Validation passed (%d rows, %d days).", len(df), span_days)
        else:
            for msg in errors:
                logger.warning("Validation issue: %s", msg)

        return is_valid, errors
