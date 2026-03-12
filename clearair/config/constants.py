"""Project-wide constants — no hard-coded magic numbers elsewhere."""

# ---------------------------------------------------------------------------
# EPA AQI breakpoints  (parameter → list of (C_lo, C_hi, I_lo, I_hi))
# Source: EPA Technical Assistance Document, 2024
# ---------------------------------------------------------------------------
AQI_BREAKPOINTS: dict[str, list[tuple[float, float, int, int]]] = {
    "PM2.5": [
        (0.0, 12.0, 0, 50),
        (12.1, 35.4, 51, 100),
        (35.5, 55.4, 101, 150),
        (55.5, 150.4, 151, 200),
        (150.5, 250.4, 201, 300),
        (250.5, 500.4, 301, 500),
    ],
    "PM10": [
        (0, 54, 0, 50),
        (55, 154, 51, 100),
        (155, 254, 101, 150),
        (255, 354, 151, 200),
        (355, 424, 201, 300),
        (425, 604, 301, 500),
    ],
    "NO2": [
        (0, 53, 0, 50),
        (54, 100, 51, 100),
        (101, 360, 101, 150),
        (361, 649, 151, 200),
        (650, 1249, 201, 300),
        (1250, 2049, 301, 500),
    ],
    "SO2": [
        (0, 35, 0, 50),
        (36, 75, 51, 100),
        (76, 185, 101, 150),
        (186, 304, 151, 200),
        (305, 604, 201, 300),
        (605, 1004, 301, 500),
    ],
    "O3": [
        (0.000, 0.054, 0, 50),
        (0.055, 0.070, 51, 100),
        (0.071, 0.085, 101, 150),
        (0.086, 0.105, 151, 200),
        (0.106, 0.200, 201, 300),
    ],
    "CO": [
        (0.0, 4.4, 0, 50),
        (4.5, 9.4, 51, 100),
        (9.5, 12.4, 101, 150),
        (12.5, 15.4, 151, 200),
        (15.5, 30.4, 201, 300),
        (30.5, 50.4, 301, 500),
    ],
}

# AQI category labels and colour-hex codes
AQI_CATEGORIES: dict[str, tuple[int, int, str]] = {
    "Good":                            (0,   50,  "#00E400"),
    "Moderate":                        (51,  100, "#FFFF00"),
    "Unhealthy for Sensitive Groups":  (101, 150, "#FF7E00"),
    "Unhealthy":                       (151, 200, "#FF0000"),
    "Very Unhealthy":                  (201, 300, "#8F3F97"),
    "Hazardous":                       (301, 500, "#7E0023"),
}

# Health impact parameters (WHO / EPA BenMAP)
VALUE_OF_STATISTICAL_LIFE_USD = 7_400_000
MORTALITY_BASELINE_RATE = 0.00785       # annual all-cause mortality per person
PM25_BETA = 0.0057                       # Pope et al. concentration-response slope
HOSPITAL_ADMISSIONS_PER_100K = 0.8       # per 1-unit AQI reduction
RESPIRATORY_CASES_PER_100K = 4.2         # per 1-unit AQI reduction
AVG_HOSPITAL_COST_USD = 25_000

# API rate-limit / retry settings
MAX_API_RETRIES = 3
RETRY_BASE_DELAY_S = 1.0

# Model hyper-parameter defaults
TFT_MAX_ENCODER_LENGTH = 168
TFT_MAX_PREDICTION_LENGTH = 72
TFT_MAX_EPOCHS = 50
TFT_EARLY_STOPPING_PATIENCE = 5

LSTM_SEQUENCE_LENGTH = 24
LSTM_UNITS = 100
LSTM_EPOCHS = 500
LSTM_BATCH_SIZE = 32
LSTM_EARLY_STOPPING_PATIENCE = 20

# Scheduler intervals
REALTIME_REFRESH_INTERVAL_HOURS = 1
LSTM_RETRAIN_CRON_HOUR = 2          # 2 AM UTC daily
TFT_RETRAIN_CRON_HOUR = 3           # 3 AM UTC weekly (Sunday)

# Minimum data requirements
MIN_DAYS_FOR_LSTM = 30
MIN_DAYS_FOR_TFT = 90

# OpenAQ parameter name mapping
OPENAQ_PARAM_MAP: dict[str, str] = {
    "pm25": "PM2.5",
    "pm10": "PM10",
    "no2": "NO2",
    "so2": "SO2",
    "o3": "O3",
    "co": "CO",
}

# Data directories (relative to project root)
DATA_RAW_DIR = "data/raw"
DATA_PROCESSED_DIR = "data/processed"
DATA_PLOTS_DIR = "data/plots"
MODELS_SAVED_DIR = "models/saved"

# Peak traffic hours
PEAK_HOURS: list[int] = [7, 8, 9, 17, 18, 19]

# OpenAQ API
OPENAQ_BASE_URL = "https://api.openaq.io/v3"
OPENAQ_PAGE_LIMIT = 1000

# EPA AQS API
EPA_BASE_URL = "https://aqs.epa.gov/data/api"

# TomTom Traffic API
TOMTOM_BASE_URL = "https://api.tomtom.com/traffic/services/4/flowSegmentData/absolute/10/json"

# NASA MERRA-2
MERRA2_DATASET = "M2T1NXAER"
MERRA2_VARIABLES = ["DUSMASS", "OCSMASS", "BCSMASS", "TOTSCATAU"]
