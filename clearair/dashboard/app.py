"""ClearAir Streamlit dashboard — main entry point."""

import sys
from pathlib import Path

# Ensure clearair package is importable
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import streamlit as st

st.set_page_config(
    page_title="ClearAir",
    layout="wide",
    page_icon="\U0001F32C\uFE0F",
    initial_sidebar_state="expanded",
)

API_BASE = "http://localhost:8000/api/v1"

# ---- Sidebar ----------------------------------------------------------
st.sidebar.title("ClearAir")
st.sidebar.caption("Adaptive Air Quality Intelligence")

try:
    import requests
    resp = requests.get(f"{API_BASE}/cities", timeout=5)
    cities_data = resp.json().get("cities", []) if resp.ok else []
except Exception:
    cities_data = [
        {"key": "mumbai", "display_name": "Mumbai, India"},
        {"key": "chicago", "display_name": "Chicago, USA"},
        {"key": "beijing", "display_name": "Beijing, China"},
        {"key": "london", "display_name": "London, UK"},
        {"key": "new_york", "display_name": "New York, USA"},
    ]

city_names = {c["key"]: c["display_name"] for c in cities_data}
selected_city = st.sidebar.selectbox(
    "Select City",
    options=list(city_names.keys()),
    format_func=lambda k: city_names[k],
)
st.session_state["selected_city"] = selected_city
st.session_state["city_display"] = city_names.get(selected_city, selected_city)
st.session_state["api_base"] = API_BASE

import datetime
date_range = st.sidebar.date_input(
    "Date Range",
    value=(datetime.date.today() - datetime.timedelta(days=30), datetime.date.today()),
)
st.session_state["date_range"] = date_range

if st.sidebar.button("Refresh Data"):
    st.cache_data.clear()
    st.rerun()

# ---- Main page --------------------------------------------------------
st.title(f"ClearAir — {st.session_state['city_display']}")
st.markdown(
    "Welcome to ClearAir, an adaptive air quality intelligence platform. "
    "Use the sidebar to select a city and navigate to the different analysis pages."
)

col1, col2, col3, col4 = st.columns(4)
col1.metric("Current AQI", "—", help="Will populate once data is ingested")
col2.metric("PM2.5", "—")
col3.metric("Wind Speed", "—")
col4.metric("Temperature", "—")

st.info(
    "Navigate to the pages in the sidebar to explore forecasts, "
    "countermeasure plans, and historical analysis."
)
