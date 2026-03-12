"""Page 4 — Historical Analysis: time-series, countermeasure events, causal attribution."""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

import datetime

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

st.header("Historical Analysis")

city = st.session_state.get("selected_city", "mumbai")

# ---- Controls ---------------------------------------------------------
col1, col2 = st.columns(2)
with col1:
    start = st.date_input("Start Date", datetime.date(2024, 1, 1))
with col2:
    end = st.date_input("End Date", datetime.date(2024, 12, 31))

show_events = st.toggle("Overlay Countermeasure Events", value=True)

# ---- Synthetic historical data ----------------------------------------
np.random.seed(7)
dates = pd.date_range(start, end, freq="D")
aqi = 80 + np.cumsum(np.random.normal(0, 3, len(dates)))
aqi = np.clip(aqi, 0, 500)

fig = go.Figure()
fig.add_trace(go.Scatter(x=dates, y=aqi, mode="lines", name="AQI",
                         line=dict(color="#2980B9")))

# AQI category background bands
for threshold, colour, label in [
    (50, "rgba(0,228,0,0.08)", "Good"),
    (100, "rgba(255,255,0,0.08)", "Moderate"),
    (150, "rgba(255,126,0,0.08)", "USG"),
    (200, "rgba(255,0,0,0.08)", "Unhealthy"),
]:
    fig.add_hrect(y0=threshold - 50, y1=threshold,
                  fillcolor=colour, line_width=0,
                  annotation_text=label, annotation_position="top left")

if show_events:
    events = [
        ("Construction Dust Control", "2024-03-01", "2024-04-15"),
        ("Traffic Management", "2024-07-10", "2024-08-20"),
    ]
    for name, es, ee in events:
        fig.add_vrect(x0=es, x1=ee, fillcolor="rgba(155,89,182,0.15)",
                      line_width=1, line_dash="dot",
                      annotation_text=name, annotation_position="top left")

fig.update_layout(title="Historical AQI", xaxis_title="Date",
                  yaxis_title="AQI", height=450)
st.plotly_chart(fig, use_container_width=True)

# ---- Impact table -----------------------------------------------------
st.subheader("Countermeasure Impact History")
impact_df = pd.DataFrame({
    "Countermeasure": ["Construction Dust Control", "Traffic Management"],
    "Period": ["Mar–Apr 2024", "Jul–Aug 2024"],
    "AQI Before": [128, 105],
    "AQI After": [95, 88],
    "% Change": [-25.8, -16.2],
    "Causal p-value": [0.012, 0.045],
})
st.dataframe(impact_df, use_container_width=True)

# ---- Causal attribution chart -----------------------------------------
st.subheader("Causal Attribution")
attr = pd.DataFrame({
    "Source": ["Countermeasure", "Weather Improvement", "Seasonal"],
    "Contribution (%)": [60, 25, 15],
})
fig2 = go.Figure(go.Bar(x=attr["Source"], y=attr["Contribution (%)"],
                         marker_color=["#27AE60", "#3498DB", "#E67E22"]))
fig2.update_layout(yaxis_title="Contribution (%)", height=300)
st.plotly_chart(fig2, use_container_width=True)

# ---- Model RMSE comparison -------------------------------------------
st.subheader("Model RMSE Comparison")
rmse_df = pd.DataFrame({
    "Model": ["TFT", "LSTM", "Ensemble", "RandomForest", "MLP", "LinearReg"],
    "RMSE": [12.3, 14.1, 11.5, 18.5, 16.8, 22.0],
})
fig3 = go.Figure(go.Bar(
    y=rmse_df["Model"], x=rmse_df["RMSE"],
    orientation="h",
    marker_color=["#27AE60", "#E74C3C", "#8E44AD", "#3498DB", "#3498DB", "#3498DB"],
))
fig3.update_layout(xaxis_title="RMSE", height=300)
st.plotly_chart(fig3, use_container_width=True)
