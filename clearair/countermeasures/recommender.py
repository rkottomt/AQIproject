"""Score and rank countermeasures for a given city and budget."""

import logging
from datetime import datetime, timedelta
from typing import Any, Optional

import numpy as np
import pandas as pd

from countermeasures.causal import CausalAttributor
from countermeasures.library import CountermeasureLibrary
from health.impact import HealthImpactCalculator
from models.ensemble import EnsembleForecaster

logger = logging.getLogger(__name__)


def _current_season() -> str:
    month = datetime.utcnow().month
    if month in (3, 4, 5):
        return "spring"
    if month in (6, 7, 8):
        return "summer"
    if month in (9, 10, 11):
        return "autumn"
    return "winter"


class CountermeasureRecommender:
    """Rank countermeasures by estimated AQI impact, health benefit, and cost."""

    def __init__(
        self,
        library: CountermeasureLibrary,
        forecaster: EnsembleForecaster,
        health_calculator: HealthImpactCalculator,
        city_config: dict,
    ) -> None:
        self.library = library
        self.forecaster = forecaster
        self.health_calc = health_calculator
        self.city_config = city_config

    def recommend(
        self,
        city: str,
        budget_tier: str,
        current_df: pd.DataFrame,
        season: Optional[str] = None,
        top_n: int = 3,
    ) -> list[dict[str, Any]]:
        """Generate ranked countermeasure recommendations.

        Args:
            city: City key.
            budget_tier: ``low``, ``medium``, ``high``, or ``all``.
            current_df: Most-recent data for forecast context.
            season: Override season filter (defaults to current season).
            top_n: Number of recommendations to return.

        Returns:
            Sorted list of recommendation dicts.
        """
        season = season or _current_season()
        population = self.city_config.get("population", 1_000_000)

        all_cms = self.library.get_all()
        candidates = []

        for key, cm in all_cms.items():
            # Budget filter
            if budget_tier.lower() != "all" and cm.get("cost_tier") != budget_tier.lower():
                continue

            # Season filter
            suitable = cm.get("suitable_seasons", ["all"])
            if "all" not in suitable and season not in suitable:
                continue

            # Build counterfactual: apply multipliers to feature values
            modified_df = current_df.copy()
            for feat, mult in cm.get("affects_features", {}).items():
                if feat in modified_df.columns:
                    modified_df[feat] = modified_df[feat] * (1 + mult)

            # Forecast baseline and counterfactual
            baseline = self.forecaster.predict(current_df, horizon_hours=24)
            modified = self.forecaster.predict(modified_df, horizon_hours=24)

            base_p50 = np.mean(baseline.get("aqi_p50", [0])) or 1
            mod_p50 = np.mean(modified.get("aqi_p50", [0]))
            pct_change = ((mod_p50 - base_p50) / base_p50) * 100

            base_p10 = np.mean(baseline.get("aqi_p10", [0]))
            base_p90 = np.mean(baseline.get("aqi_p90", [0]))
            mod_p10 = np.mean(modified.get("aqi_p10", [0]))
            mod_p90 = np.mean(modified.get("aqi_p90", [0]))

            # Health impact
            health = self.health_calc.compute_health_impact(
                aqi_before=base_p50, aqi_after=mod_p50,
                population=population,
            )

            cost_weight = cm.get("cost_tier_weight", 1.0)
            avoided_deaths = health.get("avoided_premature_deaths_per_year", 0)
            score = (abs(pct_change) * max(avoided_deaths, 0.01)) / cost_weight

            # Optimal window: next 30-day period starting after lag
            lag_days = cm.get("typical_lag_days", 7)
            window_start = datetime.utcnow() + timedelta(days=lag_days)
            window_end = window_start + timedelta(days=30)

            candidates.append({
                "key": key,
                "display_name": cm.get("display_name", key),
                "description": cm.get("description", ""),
                "pct_change": round(pct_change, 2),
                "aqi_p10_change": round(mod_p10 - base_p10, 2),
                "aqi_p90_change": round(mod_p90 - base_p90, 2),
                "health_impact": health,
                "optimal_window_start": window_start.isoformat(),
                "optimal_window_end": window_end.isoformat(),
                "cost_tier": cm.get("cost_tier"),
                "score": round(score, 4),
                "recommendation_reason": (
                    f"Expected {abs(pct_change):.1f}% AQI reduction with "
                    f"{avoided_deaths:.0f} avoided premature deaths/year."
                ),
            })

        candidates.sort(key=lambda c: c["score"], reverse=True)
        return candidates[:top_n]

    def evaluate_historical(
        self,
        city: str,
        countermeasure_key: str,
        event_start: str,
        event_end: str,
        df: pd.DataFrame,
    ) -> dict[str, Any]:
        """Run causal attribution on a past countermeasure event.

        Args:
            city: City key.
            countermeasure_key: Countermeasure identifier.
            event_start: ISO datetime of event start.
            event_end: ISO datetime of event end.
            df: Historical data covering the event period.

        Returns:
            Causal evaluation dict.
        """
        attributor = CausalAttributor(df, city)
        result = attributor.estimate_ate(countermeasure_key, event_start, event_end)

        aqi_before = df.loc[df.index < event_start, "AQI"].mean() if "AQI" in df.columns else np.nan
        aqi_after = df.loc[(df.index >= event_start) & (df.index <= event_end), "AQI"].mean() if "AQI" in df.columns else np.nan
        pct = ((aqi_after - aqi_before) / aqi_before * 100) if aqi_before else np.nan

        return {
            "countermeasure_key": countermeasure_key,
            "ate": result["ate"],
            "pct_change": round(pct, 2) if not np.isnan(pct) else None,
            "p_value": result["p_value"],
            "ci_lower": result["ci_lower"],
            "ci_upper": result["ci_upper"],
            "was_significant": result["p_value"] < 0.05 if not np.isnan(result["p_value"]) else None,
        }
