"""Tests for the countermeasure engine and health impact calculator."""

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from config import loader as cfg
cfg.load_all()


class TestCountermeasureLibrary:
    def test_loads_all_countermeasures(self):
        from countermeasures.library import CountermeasureLibrary
        lib = CountermeasureLibrary()
        all_cms = lib.get_all()
        assert len(all_cms) >= 7
        assert "construction_dust_control" in all_cms

    def test_get_by_cost_tier_filters_correctly(self):
        from countermeasures.library import CountermeasureLibrary
        lib = CountermeasureLibrary()

        low = lib.get_by_cost_tier("low")
        for cm in low:
            assert cm["cost_tier"] == "low"

        high = lib.get_by_cost_tier("high")
        for cm in high:
            assert cm["cost_tier"] == "high"

    def test_get_affected_features(self):
        from countermeasures.library import CountermeasureLibrary
        lib = CountermeasureLibrary()
        feats = lib.get_affected_features("construction_dust_control")
        assert "PM2.5" in feats
        assert feats["PM2.5"] < 0  # should reduce PM2.5


class TestHealthImpactCalculator:
    def test_avoided_deaths_positive_for_aqi_reduction(self):
        from health.impact import HealthImpactCalculator
        calc = HealthImpactCalculator()
        result = calc.compute_health_impact(
            aqi_before=150, aqi_after=100,
            population=20_000_000,
        )
        assert result["avoided_premature_deaths_per_year"] > 0
        assert result["aqi_reduction"] == 50

    def test_economic_value_formatted_correctly(self):
        from health.impact import HealthImpactCalculator
        calc = HealthImpactCalculator()
        result = calc.compute_health_impact(
            aqi_before=200, aqi_after=100,
            population=10_000_000,
        )
        fmt = result["economic_value_formatted"]
        assert fmt.startswith("$")
        assert any(c in fmt for c in ["B", "M", "K"])

    def test_aqi_category_mapping(self):
        from health.impact import HealthImpactCalculator
        calc = HealthImpactCalculator()
        assert calc.aqi_to_category(25) == "Good"
        assert calc.aqi_to_category(75) == "Moderate"
        assert calc.aqi_to_category(125) == "Unhealthy for Sensitive Groups"
        assert calc.aqi_to_category(175) == "Unhealthy"
        assert calc.aqi_to_category(250) == "Very Unhealthy"
        assert calc.aqi_to_category(400) == "Hazardous"

    def test_pm25_aqi_roundtrip(self):
        from health.impact import HealthImpactCalculator
        calc = HealthImpactCalculator()
        for aqi in [25, 75, 120, 175]:
            pm25 = calc.aqi_to_pm25(aqi)
            back = calc.pm25_to_aqi(pm25)
            assert abs(back - aqi) < 1.0

    def test_recommender_low_budget_scores(self):
        """Low-cost measures should appear when budget_tier is 'low'."""
        from countermeasures.library import CountermeasureLibrary
        lib = CountermeasureLibrary()
        low = lib.get_by_cost_tier("low")
        assert len(low) >= 1
        keys = [c["key"] for c in low]
        assert "construction_dust_control" in keys or "traffic_management" in keys
