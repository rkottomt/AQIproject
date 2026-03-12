"""AQI gauge visualisation component."""

from typing import Optional

import plotly.graph_objects as go
import streamlit as st

AQI_COLORS = [
    {"range": [0, 50], "color": "#00E400"},
    {"range": [51, 100], "color": "#FFFF00"},
    {"range": [101, 150], "color": "#FF7E00"},
    {"range": [151, 200], "color": "#FF0000"},
    {"range": [201, 300], "color": "#8F3F97"},
    {"range": [301, 500], "color": "#7E0023"},
]

AQI_LABELS = {
    (0, 50): "Good",
    (51, 100): "Moderate",
    (101, 150): "Unhealthy for Sensitive Groups",
    (151, 200): "Unhealthy",
    (201, 300): "Very Unhealthy",
    (301, 500): "Hazardous",
}


def _aqi_category(value: float) -> str:
    for (lo, hi), label in AQI_LABELS.items():
        if lo <= value <= hi:
            return label
    return "Hazardous" if value > 300 else "Good"


def render_aqi_gauge(
    aqi_value: float,
    city_name: str,
    delta_24h: Optional[float] = None,
) -> None:
    """Render a coloured AQI gauge with category label.

    Args:
        aqi_value: Current AQI reading.
        city_name: City display name for the title.
        delta_24h: Change vs. 24 hours ago (arrow direction).
    """
    fig = go.Figure(go.Indicator(
        mode="gauge+number+delta" if delta_24h is not None else "gauge+number",
        value=aqi_value,
        delta={"reference": aqi_value - (delta_24h or 0),
               "increasing": {"color": "#FF0000"},
               "decreasing": {"color": "#00E400"}} if delta_24h else None,
        title={"text": city_name, "font": {"size": 18}},
        gauge={
            "axis": {"range": [0, 500], "tickwidth": 1},
            "bar": {"color": "#333"},
            "steps": AQI_COLORS,
            "threshold": {
                "line": {"color": "black", "width": 4},
                "thickness": 0.75,
                "value": aqi_value,
            },
        },
    ))
    fig.update_layout(height=280, margin=dict(t=50, b=10, l=30, r=30))
    st.plotly_chart(fig, use_container_width=True)
    st.caption(f"Category: **{_aqi_category(aqi_value)}**")
