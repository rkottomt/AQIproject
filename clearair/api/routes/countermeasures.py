"""Countermeasure API routes."""

import logging
from typing import Optional

from fastapi import APIRouter, HTTPException, Query

from config import loader as cfg
from countermeasures.library import CountermeasureLibrary

logger = logging.getLogger(__name__)
router = APIRouter()


@router.get("/countermeasures/list")
async def list_countermeasures() -> dict:
    """Return all available countermeasures with metadata."""
    lib = CountermeasureLibrary()
    return {"countermeasures": lib.get_all()}


@router.get("/countermeasures/recommend")
async def recommend_countermeasures(
    city: str = Query(..., description="City key"),
    budget: str = Query("all", description="Cost tier: low, medium, high, all"),
    top_n: int = Query(3, ge=1, le=10),
) -> dict:
    """Return ranked countermeasure recommendations.

    Uses synthetic scoring when full models are not yet trained.
    """
    try:
        city_cfg = cfg.get_city(city)
    except KeyError:
        raise HTTPException(status_code=404, detail=f"City '{city}' not found.")

    lib = CountermeasureLibrary()
    all_cms = lib.get_all()

    # Simplified scoring without trained model
    recs = []
    for key, cm in all_cms.items():
        if budget.lower() != "all" and cm.get("cost_tier") != budget.lower():
            continue
        affects = cm.get("affects_features", {})
        impact_score = sum(abs(v) for v in affects.values())
        cost_weight = cm.get("cost_tier_weight", 1.0)
        score = impact_score / cost_weight

        recs.append({
            "key": key,
            "display_name": cm.get("display_name"),
            "description": cm.get("description"),
            "cost_tier": cm.get("cost_tier"),
            "typical_lag_days": cm.get("typical_lag_days"),
            "score": round(score, 4),
        })

    recs.sort(key=lambda r: r["score"], reverse=True)
    return {"city": city, "budget_tier": budget, "recommendations": recs[:top_n]}


@router.get("/countermeasures/evaluate")
async def evaluate_countermeasure(
    city: str = Query(..., description="City key"),
    measure: str = Query(..., description="Countermeasure key"),
    start_date: str = Query(..., description="Event start ISO date"),
    end_date: str = Query(..., description="Event end ISO date"),
) -> dict:
    """Evaluate historical impact of a countermeasure using causal inference."""
    try:
        cfg.get_city(city)
    except KeyError:
        raise HTTPException(status_code=404, detail=f"City '{city}' not found.")

    lib = CountermeasureLibrary()
    try:
        lib.get(measure)
    except KeyError:
        raise HTTPException(status_code=404,
                            detail=f"Countermeasure '{measure}' not found.")

    return {
        "city": city,
        "countermeasure_key": measure,
        "event_start": start_date,
        "event_end": end_date,
        "ate": None,
        "p_value": None,
        "ci_lower": None,
        "ci_upper": None,
        "note": "Full causal evaluation requires historical data in the database.",
    }
