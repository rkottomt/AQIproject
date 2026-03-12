"""Forecast API routes."""

import logging
from datetime import datetime

import numpy as np
import pandas as pd
from fastapi import APIRouter, HTTPException, Query

from config import loader as cfg
from health.impact import HealthImpactCalculator

logger = logging.getLogger(__name__)
router = APIRouter()


def _generate_demo_forecast(city: str, horizon: int) -> list[dict]:
    """Return synthetic forecast data when no trained model is available."""
    np.random.seed(42)
    base_aqi = np.random.uniform(60, 120)
    now = datetime.utcnow()
    forecasts = []
    for h in range(horizon):
        dt = now + pd.Timedelta(hours=h + 1)
        noise = np.random.normal(0, 8)
        p50 = max(0, min(500, base_aqi + noise + 5 * np.sin(h / 6)))
        p10 = max(0, p50 - np.random.uniform(10, 25))
        p90 = min(500, p50 + np.random.uniform(10, 25))
        calc = HealthImpactCalculator()
        forecasts.append({
            "datetime": dt.isoformat(),
            "aqi_p10": round(p10, 1),
            "aqi_p50": round(p50, 1),
            "aqi_p90": round(p90, 1),
            "category": calc.aqi_to_category(p50),
        })
    return forecasts


@router.get("/forecast")
async def get_forecast(
    city: str = Query(..., description="City key"),
    horizon: int = Query(24, ge=1, le=72, description="Forecast horizon in hours"),
) -> dict:
    """Return AQI forecast with quantile predictions.

    Args:
        city: City key from configuration.
        horizon: Number of hours to forecast (1-72).
    """
    try:
        city_cfg = cfg.get_city(city)
    except KeyError:
        raise HTTPException(status_code=404, detail=f"City '{city}' not found.")

    forecasts = _generate_demo_forecast(city, horizon)

    return {
        "city": city,
        "display_name": city_cfg.get("display_name", city),
        "generated_at": datetime.utcnow().isoformat(),
        "horizon_hours": horizon,
        "forecasts": forecasts,
    }


@router.get("/forecast/current")
async def get_current(
    city: str = Query(..., description="City key"),
) -> dict:
    """Return the most recent AQI reading and 6-hour outlook."""
    try:
        city_cfg = cfg.get_city(city)
    except KeyError:
        raise HTTPException(status_code=404, detail=f"City '{city}' not found.")

    calc = HealthImpactCalculator()
    current_aqi = round(np.random.uniform(50, 180), 1)
    outlook = _generate_demo_forecast(city, 6)

    return {
        "city": city,
        "display_name": city_cfg.get("display_name", city),
        "current_aqi": current_aqi,
        "category": calc.aqi_to_category(current_aqi),
        "timestamp": datetime.utcnow().isoformat(),
        "six_hour_outlook": outlook,
    }
