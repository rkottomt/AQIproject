"""City management API routes."""

import logging
from datetime import datetime

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from config import loader as cfg

logger = logging.getLogger(__name__)
router = APIRouter()


class CityConfig(BaseModel):
    """Schema for adding a new city."""
    display_name: str
    lat: float = Field(..., ge=-90, le=90)
    lon: float = Field(..., ge=-180, le=180)
    timezone: str
    population: int = Field(..., ge=0)
    country_code: str = Field(..., min_length=2, max_length=2)
    openaq_location_ids: list[int] = []
    has_epa_data: bool = False
    epa_state_code: str = ""
    epa_county_code: str = ""
    traffic_source: str = "tomtom"


@router.get("/cities")
async def list_cities() -> dict:
    """Return all configured cities with display names and coordinates."""
    all_cities = cfg.get_all_cities()
    items = []
    for key, c in all_cities.items():
        items.append({
            "key": key,
            "display_name": c.get("display_name", key),
            "lat": c.get("lat"),
            "lon": c.get("lon"),
            "country_code": c.get("country_code"),
            "population": c.get("population"),
        })
    return {"cities": items}


@router.post("/cities/add")
async def add_city(key: str, config: CityConfig) -> dict:
    """Register a new city in the configuration.

    Validates the payload, appends to cities.yaml, and updates the
    in-memory cache.
    """
    existing = cfg.get_all_cities()
    if key in existing:
        raise HTTPException(status_code=409,
                            detail=f"City '{key}' already exists.")
    cfg.add_city(key, config.model_dump())
    return {"message": f"City '{key}' added successfully.", "key": key}


@router.get("/cities/{city_key}/status")
async def city_status(city_key: str) -> dict:
    """Return operational status for a single city."""
    try:
        city_cfg = cfg.get_city(city_key)
    except KeyError:
        raise HTTPException(status_code=404,
                            detail=f"City '{city_key}' not found.")

    return {
        "city": city_key,
        "display_name": city_cfg.get("display_name", city_key),
        "last_updated": datetime.utcnow().isoformat(),
        "data_coverage_days": 0,
        "models_trained": False,
        "latest_aqi": None,
        "data_quality_score": None,
    }
