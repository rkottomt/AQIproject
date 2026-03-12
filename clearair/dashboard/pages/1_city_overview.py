"""Page 1 — City Overview: current AQI, pollutant breakdown, forecast, and map."""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

import numpy as np
import pandas as pd
import streamlit as st

from dashboard.components.aqi_gauge import render_aqi_gauge
from dashboard.components.forecast_chart import render_forecast_chart
from dashboard.components.map_view import render_map

st.header("City Overview")

city = st.session_state.get("selected_city", "mumbai")
display = st.session_state.get("city_display", city)
api = st.session_state.get("api_base", "http://localhost:8000/api/v1")

# ---- Fetch data (demo fallback) --------------------------------------
try:
    import requests as _req
    current = _req.get(f"{api}/forecast/current", params={"city": city}, timeout=5).json()
    forecast_resp = _req.get(f"{api}/forecast", params={"city": city, "horizon": 72}, timeout=5).json()
except Exception:
    current = {"current_aqi": np.random.uniform(60, 160), "category": "Moderate"}
    forecast_resp = {"forecasts": []}

current_aqi = current.get("current_aqi", 85)

# ---- Layout -----------------------------------------------------------
left, right = st.columns([4, 6])

with left:
    render_aqi_gauge(current_aqi, display, delta_24h=np.random.uniform(-10, 10))
    st.subheader("Pollutant Breakdown")
    pollutants = {"PM2.5": 42, "PM10": 68, "NO2": 30, "SO2": 12, "O3": 55, "CO": 8}
    poll_df = pd.DataFrame({"Pollutant": pollutants.keys(), "Value": pollutants.values()})
    st.bar_chart(poll_df.set_index("Pollutant"))
    st.caption(f"Last updated: {current.get('timestamp', 'N/A')}")

with right:
    fc = forecast_resp.get("forecasts", [])
    if fc:
        fc_df = pd.DataFrame(fc)
        fc_df["datetime"] = pd.to_datetime(fc_df["datetime"])
        render_forecast_chart(fc_df, title="72-Hour AQI Forecast")
    else:
        st.info("Forecast data not yet available — run `python main.py serve` first.")

    render_map(
        city, [
            {"lat": 19.08, "lon": 72.88, "aqi": 75, "name": "Station A", "dominant_pollutant": "PM2.5"},
            {"lat": 19.07, "lon": 72.87, "aqi": 110, "name": "Station B", "dominant_pollutant": "O3"},
        ],
    )

# ---- Metrics row ------------------------------------------------------
st.subheader("Key Metrics")
m1, m2, m3, m4 = st.columns(4)
m1.metric("PM2.5", "42 µg/m³", delta="-3 vs yesterday")
m2.metric("Wind Speed", "4.2 m/s", delta="+0.8")
m3.metric("Humidity", "68%", delta="-5%")
m4.metric("Temperature", "31 °C", delta="+1 °C")
