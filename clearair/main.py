#!/usr/bin/env python3
"""ClearAir — command-line interface.

Usage examples::

    python main.py ingest   --city mumbai --start 2020-01-01 --end 2025-01-01
    python main.py process  --city mumbai --start 2020-01-01 --end 2025-01-01
    python main.py train    --city mumbai --model all
    python main.py recommend --city mumbai --budget medium
    python main.py serve
"""

import argparse
import logging
import sys
from pathlib import Path

# Ensure clearair root is on sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent))

from config import loader as cfg
from config.constants import DATA_PROCESSED_DIR

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(module)s | %(message)s",
)
logger = logging.getLogger(__name__)


# -----------------------------------------------------------------------
# Sub-command implementations
# -----------------------------------------------------------------------

def cmd_ingest(args: argparse.Namespace) -> None:
    """Run data ingestion for the specified city and sources."""
    city_cfg = cfg.get_city(args.city)
    sources = args.sources or ["openaq", "openmeteo", "epa", "traffic", "merra2"]

    if "openaq" in sources and city_cfg.get("openaq_location_ids"):
        from ingestion.openaq import OpenAQFetcher
        logger.info("Ingesting OpenAQ for %s …", args.city)
        OpenAQFetcher(args.city, city_cfg).fetch(args.start, args.end)

    if "openmeteo" in sources:
        from ingestion.openmeteo import OpenMeteoFetcher
        logger.info("Ingesting Open-Meteo for %s …", args.city)
        OpenMeteoFetcher(args.city, city_cfg).fetch(args.start, args.end)

    if "epa" in sources and city_cfg.get("has_epa_data"):
        from ingestion.epa import EPAFetcher
        logger.info("Ingesting EPA for %s …", args.city)
        EPAFetcher(args.city, city_cfg).fetch(args.start, args.end)

    if "traffic" in sources:
        from ingestion.traffic import TomTomFetcher
        logger.info("Ingesting TomTom traffic for %s …", args.city)
        TomTomFetcher(args.city, city_cfg).fetch(args.start, args.end)

    if "merra2" in sources:
        from ingestion.nasa_merra2 import MERRA2Fetcher
        logger.info("Ingesting MERRA-2 for %s …", args.city)
        MERRA2Fetcher(args.city, city_cfg).fetch(args.start, args.end)

    logger.info("Ingestion complete for %s.", args.city)


def cmd_process(args: argparse.Namespace) -> None:
    """Merge and feature-engineer data for a city."""
    from processing.merger import DataMerger
    from processing.features import FeatureEngineer
    from processing.validator import DataValidator

    merger = DataMerger()
    merged = merger.merge_all(args.city, args.start, args.end)
    if merged.empty:
        logger.error("No data to process for %s.", args.city)
        return

    fe = FeatureEngineer()
    df = fe.transform(merged, args.city)

    validator = DataValidator()
    is_valid, errors = validator.validate(df)
    if not is_valid:
        for e in errors:
            logger.warning("Validation: %s", e)

    out = Path(DATA_PROCESSED_DIR)
    out.mkdir(parents=True, exist_ok=True)
    out_path = out / f"{args.city}_{args.start}_{args.end}.parquet"
    df.to_parquet(out_path)
    logger.info("Processed data saved to %s (%d rows).", out_path, len(df))


def cmd_train(args: argparse.Namespace) -> None:
    """Train forecasting models for a city."""
    import pandas as pd
    from processing.features import FeatureEngineer

    processed_dir = Path(DATA_PROCESSED_DIR)
    files = list(processed_dir.glob(f"{args.city}_*.parquet"))
    if not files:
        logger.error("No processed data found for %s. Run 'process' first.", args.city)
        return
    df = pd.read_parquet(files[-1])

    fe = FeatureEngineer()
    top_features = fe.get_top_features(df, target="AQI", n=15, city=args.city)

    models_to_train = args.model if args.model != "all" else ["tft", "lstm", "baselines"]
    if isinstance(models_to_train, str):
        models_to_train = [models_to_train]

    if "baselines" in models_to_train:
        from models.baselines import BaselineModels
        bl = BaselineModels(args.city)
        bl.train_all(df, top_features)

    if "lstm" in models_to_train:
        from models.lstm_model import LSTMForecaster
        lstm = LSTMForecaster(args.city)
        lstm.train(df)

    if "tft" in models_to_train:
        from models.tft_model import TFTForecaster
        tft = TFTForecaster(args.city)
        try:
            train_ds, val_ds = tft.prepare_dataset(df)
            tft.train(train_ds, val_ds)
        except Exception:
            logger.error("TFT training failed (data may be insufficient).",
                         exc_info=True)

    logger.info("Training complete for %s.", args.city)


def cmd_evaluate(args: argparse.Namespace) -> None:
    """Evaluate a historical countermeasure event using causal inference."""
    import pandas as pd
    from countermeasures.recommender import CountermeasureRecommender
    from countermeasures.library import CountermeasureLibrary
    from health.impact import HealthImpactCalculator
    from models.tft_model import TFTForecaster
    from models.lstm_model import LSTMForecaster
    from models.ensemble import EnsembleForecaster

    processed_dir = Path(DATA_PROCESSED_DIR)
    files = list(processed_dir.glob(f"{args.city}_*.parquet"))
    if not files:
        logger.error("No processed data for %s.", args.city)
        return
    df = pd.read_parquet(files[-1])

    city_cfg = cfg.get_city(args.city)
    lib = CountermeasureLibrary()
    tft = TFTForecaster(args.city)
    lstm = LSTMForecaster(args.city)
    ensemble = EnsembleForecaster(tft, lstm)
    health = HealthImpactCalculator()
    rec = CountermeasureRecommender(lib, ensemble, health, city_cfg)

    result = rec.evaluate_historical(
        args.city, args.countermeasure, args.start, args.end, df,
    )
    for k, v in result.items():
        print(f"  {k}: {v}")


def cmd_recommend(args: argparse.Namespace) -> None:
    """Print ranked countermeasure recommendations."""
    import pandas as pd
    from countermeasures.recommender import CountermeasureRecommender
    from countermeasures.library import CountermeasureLibrary
    from health.impact import HealthImpactCalculator
    from models.tft_model import TFTForecaster
    from models.lstm_model import LSTMForecaster
    from models.ensemble import EnsembleForecaster

    city_cfg = cfg.get_city(args.city)
    lib = CountermeasureLibrary()
    tft = TFTForecaster(args.city)
    lstm = LSTMForecaster(args.city)
    ensemble = EnsembleForecaster(tft, lstm)
    health = HealthImpactCalculator()
    recommender = CountermeasureRecommender(lib, ensemble, health, city_cfg)

    # Use an empty DataFrame as context when no data is available
    processed_dir = Path(DATA_PROCESSED_DIR)
    files = list(processed_dir.glob(f"{args.city}_*.parquet"))
    if files:
        current_df = pd.read_parquet(files[-1]).tail(168)
    else:
        current_df = pd.DataFrame()

    recs = recommender.recommend(
        args.city, args.budget, current_df, top_n=args.top_n,
    )
    for i, r in enumerate(recs, 1):
        print(f"\n--- #{i} {r['display_name']} (score {r['score']}) ---")
        print(f"  AQI change: {r['pct_change']}%")
        print(f"  Cost tier:  {r['cost_tier']}")
        print(f"  Window:     {r['optimal_window_start'][:10]} — {r['optimal_window_end'][:10]}")
        print(f"  Reason:     {r['recommendation_reason']}")


def cmd_serve(args: argparse.Namespace) -> None:
    """Start FastAPI server with APScheduler."""
    import uvicorn
    from scheduler.jobs import start_scheduler

    scheduler = start_scheduler()
    try:
        uvicorn.run(
            "api.main:app",
            host=args.host,
            port=args.port,
            reload=False,
        )
    finally:
        scheduler.shutdown()


# -----------------------------------------------------------------------
# Argument parser
# -----------------------------------------------------------------------

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="clearair",
        description="ClearAir — Adaptive Air Quality Intelligence Platform",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    # ingest
    p_ingest = sub.add_parser("ingest", help="Fetch raw data from external APIs")
    p_ingest.add_argument("--city", required=True)
    p_ingest.add_argument("--start", required=True, help="YYYY-MM-DD")
    p_ingest.add_argument("--end", required=True, help="YYYY-MM-DD")
    p_ingest.add_argument("--sources", nargs="+",
                          choices=["openaq", "openmeteo", "epa", "traffic", "merra2"],
                          default=None, help="Sources to fetch (default: all)")

    # process
    p_proc = sub.add_parser("process", help="Merge and feature-engineer data")
    p_proc.add_argument("--city", required=True)
    p_proc.add_argument("--start", required=True)
    p_proc.add_argument("--end", required=True)

    # train
    p_train = sub.add_parser("train", help="Train forecasting models")
    p_train.add_argument("--city", required=True)
    p_train.add_argument("--model", default="all",
                         choices=["tft", "lstm", "baselines", "all"])

    # evaluate
    p_eval = sub.add_parser("evaluate", help="Evaluate a countermeasure event")
    p_eval.add_argument("--city", required=True)
    p_eval.add_argument("--countermeasure", required=True)
    p_eval.add_argument("--start", required=True)
    p_eval.add_argument("--end", required=True)

    # recommend
    p_rec = sub.add_parser("recommend", help="Get countermeasure recommendations")
    p_rec.add_argument("--city", required=True)
    p_rec.add_argument("--budget", default="all", choices=["low", "medium", "high", "all"])
    p_rec.add_argument("--top_n", type=int, default=3)

    # serve
    p_serve = sub.add_parser("serve", help="Start API + scheduler")
    p_serve.add_argument("--host", default="0.0.0.0")
    p_serve.add_argument("--port", type=int, default=8000)

    return parser


def main() -> None:
    cfg.load_all()
    parser = build_parser()
    args = parser.parse_args()

    dispatch = {
        "ingest": cmd_ingest,
        "process": cmd_process,
        "train": cmd_train,
        "evaluate": cmd_evaluate,
        "recommend": cmd_recommend,
        "serve": cmd_serve,
    }
    dispatch[args.command](args)


if __name__ == "__main__":
    main()
