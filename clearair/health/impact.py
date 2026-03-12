"""WHO/EPA BenMAP-inspired health impact calculations."""

import logging
import math
from typing import Optional

import numpy as np

from config.constants import (
    AQI_BREAKPOINTS,
    AQI_CATEGORIES,
    AVG_HOSPITAL_COST_USD,
    HOSPITAL_ADMISSIONS_PER_100K,
    MORTALITY_BASELINE_RATE,
    PM25_BETA,
    RESPIRATORY_CASES_PER_100K,
    VALUE_OF_STATISTICAL_LIFE_USD,
)

logger = logging.getLogger(__name__)


class HealthImpactCalculator:
    """Concentration-response health impact estimator."""

    VALUE_OF_STATISTICAL_LIFE_USD = VALUE_OF_STATISTICAL_LIFE_USD

    # ------------------------------------------------------------------
    # Conversion helpers
    # ------------------------------------------------------------------
    @staticmethod
    def aqi_to_pm25(aqi: float) -> float:
        """Convert AQI value to approximate PM2.5 concentration (µg/m³).

        Uses EPA PM2.5 breakpoint table in reverse.
        """
        for c_lo, c_hi, i_lo, i_hi in AQI_BREAKPOINTS["PM2.5"]:
            if i_lo <= aqi <= i_hi:
                return (aqi - i_lo) / (i_hi - i_lo) * (c_hi - c_lo) + c_lo
        return 0.0

    @staticmethod
    def pm25_to_aqi(pm25: float) -> float:
        """Convert PM2.5 concentration to AQI."""
        for c_lo, c_hi, i_lo, i_hi in AQI_BREAKPOINTS["PM2.5"]:
            if c_lo <= pm25 <= c_hi:
                return (i_hi - i_lo) / (c_hi - c_lo) * (pm25 - c_lo) + i_lo
        return 0.0

    @staticmethod
    def aqi_to_category(aqi: float) -> str:
        """Map an AQI value to its descriptive category label."""
        for label, (lo, hi, _colour) in AQI_CATEGORIES.items():
            if lo <= aqi <= hi:
                return label
        return "Hazardous" if aqi > 300 else "Good"

    # ------------------------------------------------------------------
    # Core calculation
    # ------------------------------------------------------------------
    def compute_health_impact(
        self,
        aqi_before: float,
        aqi_after: float,
        population: int,
        exposure_days: int = 365,
    ) -> dict:
        """Estimate averted health outcomes from an AQI change.

        Args:
            aqi_before: Baseline AQI.
            aqi_after: Post-intervention AQI.
            population: Exposed population count.
            exposure_days: Duration of exposure (default 1 year).

        Returns:
            Dict with reduction metrics, averted outcomes, and economic value.
        """
        aqi_reduction = aqi_before - aqi_after
        pm25_before = self.aqi_to_pm25(aqi_before)
        pm25_after = self.aqi_to_pm25(aqi_after)
        delta_pm25 = pm25_before - pm25_after

        # Premature mortality (Pope et al. log-linear model)
        # RR = exp(beta * delta_pm25); attributable fraction = 1 - exp(-beta * delta)
        if delta_pm25 > 0:
            af = 1 - math.exp(-PM25_BETA * delta_pm25)
        else:
            af = -(1 - math.exp(-PM25_BETA * abs(delta_pm25)))

        annual_fraction = exposure_days / 365.0
        avoided_deaths = population * af * MORTALITY_BASELINE_RATE * annual_fraction

        # Morbidity
        avoided_hospital = (HOSPITAL_ADMISSIONS_PER_100K / 100_000) * aqi_reduction * population * annual_fraction
        avoided_respiratory = (RESPIRATORY_CASES_PER_100K / 100_000) * aqi_reduction * population * annual_fraction

        # Economic valuation
        economic = (
            abs(avoided_deaths) * self.VALUE_OF_STATISTICAL_LIFE_USD
            + abs(avoided_hospital) * AVG_HOSPITAL_COST_USD
        )

        return {
            "aqi_reduction": round(aqi_reduction, 2),
            "pm25_reduction_ug_m3": round(delta_pm25, 2),
            "avoided_premature_deaths_per_year": round(avoided_deaths, 1),
            "avoided_hospital_admissions": round(avoided_hospital, 0),
            "avoided_respiratory_cases": round(avoided_respiratory, 0),
            "economic_value_usd": round(economic, 0),
            "economic_value_formatted": self._format_usd(economic),
        }

    # ------------------------------------------------------------------
    @staticmethod
    def _format_usd(value: float) -> str:
        abs_val = abs(value)
        sign = "-" if value < 0 else ""
        if abs_val >= 1e9:
            return f"{sign}${abs_val / 1e9:.1f}B"
        if abs_val >= 1e6:
            return f"{sign}${abs_val / 1e6:.1f}M"
        if abs_val >= 1e3:
            return f"{sign}${abs_val / 1e3:.0f}K"
        return f"{sign}${abs_val:.0f}"
