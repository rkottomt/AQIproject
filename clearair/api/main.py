"""FastAPI application entry point."""

import logging
import sys
from pathlib import Path

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

# Ensure clearair root is importable when running from project root
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from api.routes import cities, countermeasures, forecast, health
from config import loader as cfg

logger = logging.getLogger(__name__)

app = FastAPI(
    title="ClearAir API",
    version="1.0.0",
    description="Air quality forecasting and countermeasure recommendation platform",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(forecast.router, prefix="/api/v1", tags=["forecast"])
app.include_router(countermeasures.router, prefix="/api/v1", tags=["countermeasures"])
app.include_router(health.router, prefix="/api/v1", tags=["health"])
app.include_router(cities.router, prefix="/api/v1", tags=["cities"])


@app.on_event("startup")
async def startup_event() -> None:
    """Load configuration and (optionally) initialise the database."""
    cfg.load_all()
    logger.info("ClearAir API started — %d cities configured.",
                len(cfg.get_all_cities()))

    try:
        from database.session import init_db
        init_db()
    except Exception:
        logger.warning("Database init skipped (DB may not be available).",
                       exc_info=True)


@app.get("/", tags=["root"])
async def root() -> dict:
    """Health-check endpoint."""
    return {"status": "ok", "service": "ClearAir API", "version": "1.0.0"}
