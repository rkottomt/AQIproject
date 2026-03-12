"""APScheduler job definitions for periodic data refresh and model retraining."""

import logging
from datetime import datetime, timedelta

from apscheduler.schedulers.background import BackgroundScheduler

from config import loader as cfg
from config.constants import (
    LSTM_RETRAIN_CRON_HOUR,
    MIN_DAYS_FOR_LSTM,
    MIN_DAYS_FOR_TFT,
    REALTIME_REFRESH_INTERVAL_HOURS,
    TFT_RETRAIN_CRON_HOUR,
)

logger = logging.getLogger(__name__)


# -----------------------------------------------------------------------
# Job 1 — Real-time data refresh (every hour)
# -----------------------------------------------------------------------
def refresh_realtime_data() -> None:
    """Fetch the last 2 hours of AQ, weather, and traffic data for every city."""
    from ingestion.openaq import OpenAQFetcher
    from ingestion.openmeteo import OpenMeteoFetcher
    from ingestion.traffic import TomTomFetcher

    cities = cfg.get_all_cities()
    end = datetime.utcnow().strftime("%Y-%m-%d")
    start = (datetime.utcnow() - timedelta(hours=2)).strftime("%Y-%m-%d")

    for key, city_cfg in cities.items():
        try:
            if city_cfg.get("openaq_location_ids"):
                OpenAQFetcher(key, city_cfg).fetch(start, end)
            OpenMeteoFetcher(key, city_cfg).fetch(start, end)
            TomTomFetcher(key, city_cfg).fetch(start, end)
            logger.info("Refreshed real-time data for %s.", key)
        except Exception:
            logger.error("Real-time refresh failed for %s.", key, exc_info=True)


# -----------------------------------------------------------------------
# Job 2 — LSTM retrain (daily at 2 AM UTC)
# -----------------------------------------------------------------------
def retrain_lstm() -> None:
    """Retrain LSTM models for cities with sufficient data."""
    from models.lstm_model import LSTMForecaster
    from processing.features import FeatureEngineer
    from processing.merger import DataMerger

    cities = cfg.get_all_cities()
    end = datetime.utcnow().strftime("%Y-%m-%d")
    start = (datetime.utcnow() - timedelta(days=90)).strftime("%Y-%m-%d")

    merger = DataMerger()
    fe = FeatureEngineer()

    for key, city_cfg in cities.items():
        try:
            merged = merger.merge_all(key, start, end)
            if merged.empty or len(merged) < MIN_DAYS_FOR_LSTM * 24:
                logger.info("Skipping LSTM retrain for %s (insufficient data).", key)
                continue
            df = fe.transform(merged, key)
            lstm = LSTMForecaster(key)
            lstm.train(df)
            metrics = lstm.evaluate(df.iloc[int(len(df) * 0.8):])
            logger.info("LSTM retrained for %s — RMSE=%.2f", key, metrics.get("RMSE", float("nan")))
        except Exception:
            logger.error("LSTM retrain failed for %s.", key, exc_info=True)


# -----------------------------------------------------------------------
# Job 3 — TFT retrain (weekly, Sunday 3 AM UTC)
# -----------------------------------------------------------------------
def retrain_tft() -> None:
    """Retrain TFT models and recompute ensemble weights."""
    from models.tft_model import TFTForecaster
    from models.lstm_model import LSTMForecaster
    from models.ensemble import EnsembleForecaster
    from processing.features import FeatureEngineer
    from processing.merger import DataMerger

    cities = cfg.get_all_cities()
    end = datetime.utcnow().strftime("%Y-%m-%d")
    start = (datetime.utcnow() - timedelta(days=365)).strftime("%Y-%m-%d")

    merger = DataMerger()
    fe = FeatureEngineer()

    for key, city_cfg in cities.items():
        try:
            merged = merger.merge_all(key, start, end)
            if merged.empty or len(merged) < MIN_DAYS_FOR_TFT * 24:
                logger.info("Skipping TFT retrain for %s (insufficient data).", key)
                continue
            df = fe.transform(merged, key)
            tft = TFTForecaster(key)
            train_ds, val_ds = tft.prepare_dataset(df)
            tft.train(train_ds, val_ds)

            # Recompute ensemble weights
            lstm = LSTMForecaster(key)
            ensemble = EnsembleForecaster(tft, lstm)
            val_df = df.iloc[int(len(df) * 0.8):]
            ensemble.compute_weights(val_df)
            logger.info("TFT retrained and ensemble updated for %s.", key)
        except Exception:
            logger.error("TFT retrain failed for %s.", key, exc_info=True)


# -----------------------------------------------------------------------
# Scheduler factory
# -----------------------------------------------------------------------
def start_scheduler() -> BackgroundScheduler:
    """Create and start the background scheduler with all three jobs.

    Returns:
        Running ``BackgroundScheduler`` instance.
    """
    scheduler = BackgroundScheduler()

    scheduler.add_job(
        refresh_realtime_data,
        "interval",
        hours=REALTIME_REFRESH_INTERVAL_HOURS,
        id="refresh_realtime",
        name="Real-time data refresh",
    )

    scheduler.add_job(
        retrain_lstm,
        "cron",
        hour=LSTM_RETRAIN_CRON_HOUR,
        id="retrain_lstm",
        name="Daily LSTM retrain",
    )

    scheduler.add_job(
        retrain_tft,
        "cron",
        day_of_week="sun",
        hour=TFT_RETRAIN_CRON_HOUR,
        id="retrain_tft",
        name="Weekly TFT retrain",
    )

    scheduler.start()
    logger.info("Background scheduler started with %d jobs.", len(scheduler.get_jobs()))
    return scheduler
