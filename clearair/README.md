# ClearAir — Adaptive Air Quality Intelligence Platform

ClearAir is an end-to-end air quality forecasting and countermeasure recommendation
platform. It ingests multi-source environmental data, produces probabilistic AQI
forecasts using deep-learning ensembles, evaluates countermeasure interventions via
causal inference, and quantifies health impact using WHO/EPA-derived models.

## Architecture

```
┌─────────────────────────────────────────────────────────────────────┐
│                        ClearAir Platform                            │
├───────────┬──────────────┬──────────────┬──────────────┬────────────┤
│  Ingest   │  Processing  │   Models     │ Countermeas. │  Health    │
│           │              │              │              │            │
│ OpenAQ    │ Merger       │ TFT (PyTorch)│ Library      │ BenMAP     │
│ Open-Meteo│ Features     │ LSTM (TF)    │ Causal (DoWhy│ Conc-Resp  │
│ EPA AQS   │ Validator    │ Baselines    │ Recommender  │ Economic   │
│ TomTom    │              │ Ensemble     │              │ Valuation  │
│ MERRA-2   │              │              │              │            │
├───────────┴──────────────┴──────────────┴──────────────┴────────────┤
│                           FastAPI  (/api/v1)                        │
├─────────────────────────────────────────────────────────────────────┤
│                     Streamlit Dashboard (:8501)                     │
├─────────────────────────────────────────────────────────────────────┤
│  PostgreSQL │  APScheduler (hourly refresh, daily/weekly retrain)   │
└─────────────────────────────────────────────────────────────────────┘
```

## Prerequisites

- Python 3.11+
- PostgreSQL 15+ (or Docker)
- (Optional) API keys for OpenAQ, TomTom, NASA Earthdata, EPA AQS

## Quick Start — Docker

```bash
cd clearair
cp .env.example .env        # edit with your API keys
docker-compose up --build
```

- **API**: http://localhost:8000/docs
- **Dashboard**: http://localhost:8501

## Manual Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

# Create and configure database
export DATABASE_URL=postgresql://clearair:password@localhost:5432/clearair
python -c "from database.session import init_db; init_db()"

# Start API + scheduler
python main.py serve
```

## Environment Variables

Create a `.env` file (see `.env.example`):

| Variable              | Description                                       | Required |
| --------------------- | ------------------------------------------------- | -------- |
| `DATABASE_URL`        | PostgreSQL connection string                      | Yes      |
| `OPENAQ_API_KEY`      | OpenAQ v3 API key (https://openaq.org)            | Optional |
| `TOMTOM_API_KEY`      | TomTom Traffic API key (https://developer.tomtom.com) | Optional |
| `NASA_EARTHDATA_TOKEN`| NASA Earthdata token (https://urs.earthdata.nasa.gov) | Optional |
| `EPA_API_KEY`         | EPA AQS API key (https://aqs.epa.gov/aqsweb/documents/data_api.html) | Optional |
| `OPENMETEO_BASE_URL`  | Open-Meteo base URL (default provided)            | No       |

## CLI Reference

```bash
# Ingest data
python main.py ingest --city mumbai --start 2020-01-01 --end 2025-01-01
python main.py ingest --city mumbai --start 2024-01-01 --end 2024-12-31 --sources openaq openmeteo

# Process (merge + feature engineering)
python main.py process --city mumbai --start 2020-01-01 --end 2025-01-01

# Train models
python main.py train --city mumbai --model all
python main.py train --city mumbai --model lstm

# Evaluate a countermeasure event
python main.py evaluate --city mumbai --countermeasure construction_dust_control \
    --start 2024-03-01 --end 2024-04-15

# Get recommendations
python main.py recommend --city mumbai --budget medium --top_n 5

# Start API server + scheduler
python main.py serve --host 0.0.0.0 --port 8000
```

## API Endpoints

| Method | Path                              | Description                              |
| ------ | --------------------------------- | ---------------------------------------- |
| GET    | `/api/v1/forecast`                | AQI forecast (city, horizon)             |
| GET    | `/api/v1/forecast/current`        | Current AQI + 6h outlook                 |
| GET    | `/api/v1/countermeasures/list`    | All countermeasures                      |
| GET    | `/api/v1/countermeasures/recommend` | Ranked recommendations                 |
| GET    | `/api/v1/countermeasures/evaluate`| Causal attribution of past event         |
| GET    | `/api/v1/health-impact`           | Health impact of AQI change              |
| GET    | `/api/v1/cities`                  | List configured cities                   |
| POST   | `/api/v1/cities/add`              | Register a new city                      |
| GET    | `/api/v1/cities/{key}/status`     | City operational status                  |

Full interactive docs at `/docs` (Swagger) or `/redoc`.

## Dashboard Pages

1. **City Overview** — Current AQI gauge, pollutant breakdown, 72h forecast, station map
2. **Forecast Explorer** — Adjustable horizon, weather overlays, model comparison
3. **Countermeasure Planner** — Budget-filtered recommendations, before/after forecast, health impact
4. **Historical Analysis** — AQI time-series, countermeasure event overlays, causal attribution

## Adding a New City

Edit `config/cities.yaml`:

```yaml
delhi:
  display_name: "Delhi, India"
  lat: 28.6139
  lon: 77.2090
  timezone: "Asia/Kolkata"
  population: 32941000
  openaq_location_ids: [1234, 1235]
  has_epa_data: false
  traffic_source: "tomtom"
  country_code: "IN"
```

Or use the API:

```bash
curl -X POST "http://localhost:8000/api/v1/cities/add?key=delhi" \
  -H "Content-Type: application/json" \
  -d '{"display_name":"Delhi, India","lat":28.6139,"lon":77.2090,"timezone":"Asia/Kolkata","population":32941000,"country_code":"IN"}'
```

## Adding a New Countermeasure

Edit `config/countermeasures.yaml`:

```yaml
water_mist_cannons:
  display_name: "Water Mist Cannons"
  description: "Deploy mobile mist cannons at pollution hotspots"
  affects_features:
    PM2.5: -0.20
    PM10: -0.25
  cost_tier: low
  cost_tier_weight: 1.5
  typical_lag_days: 1
  suitable_seasons: ["all"]
  suitable_city_types: ["urban"]
```

## Retraining Models

Models retrain automatically via the scheduler:
- **LSTM**: Daily at 2 AM UTC (requires 30+ days of data)
- **TFT**: Weekly on Sunday at 3 AM UTC (requires 90+ days of data)

Manual retraining:

```bash
python main.py train --city mumbai --model lstm
python main.py train --city mumbai --model tft
```

## Running Tests

```bash
cd clearair
pytest tests/ -v
```

## Project Structure

```
clearair/
├── main.py                    # CLI entry point
├── config/                    # YAML configs + loader + constants
├── ingestion/                 # Data fetchers (OpenAQ, Open-Meteo, EPA, TomTom, MERRA-2)
├── processing/                # Merger, feature engineering, validation
├── models/                    # TFT, LSTM, baselines, ensemble
├── countermeasures/           # Library, causal inference, recommender
├── health/                    # WHO/EPA health impact calculator
├── api/                       # FastAPI application + route modules
├── dashboard/                 # Streamlit multi-page app + components
├── scheduler/                 # APScheduler background jobs
├── database/                  # SQLAlchemy ORM + session management
└── tests/                     # Pytest test suite
```

## License

MIT
