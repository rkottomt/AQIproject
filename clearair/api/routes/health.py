"""Health impact API routes."""

import logging

from fastapi import APIRouter, HTTPException, Query

from config import loader as cfg
from health.impact import HealthImpactCalculator

logger = logging.getLogger(__name__)
router = APIRouter()


@router.get("/health-impact")
async def get_health_impact(
    city: str = Query(..., description="City key"),
    aqi_before: float = Query(..., ge=0, le=500),
    aqi_after: float = Query(..., ge=0, le=500),
    exposure_days: int = Query(365, ge=1, le=3650),
) -> dict:
    """Compute health impact of an AQI change for a city's population.

    The endpoint automatically uses the configured population for the city.
    """
    try:
        city_cfg = cfg.get_city(city)
    except KeyError:
        raise HTTPException(status_code=404, detail=f"City '{city}' not found.")

    population = city_cfg.get("population", 1_000_000)
    calc = HealthImpactCalculator()
    result = calc.compute_health_impact(
        aqi_before=aqi_before,
        aqi_after=aqi_after,
        population=population,
        exposure_days=exposure_days,
    )
    result["city"] = city
    result["population"] = population
    return result
