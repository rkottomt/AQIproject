"""Forecast time-series chart with confidence bands."""

from datetime import datetime

import pandas as pd
import plotly.graph_objects as go
import streamlit as st


AQI_THRESHOLDS = [
    (50, "Good", "#00E400"),
    (100, "Moderate", "#FFFF00"),
    (150, "USG", "#FF7E00"),
    (200, "Unhealthy", "#FF0000"),
]


def render_forecast_chart(
    forecast_df: pd.DataFrame,
    title: str = "AQI Forecast",
) -> None:
    """Plot median forecast with p10/p90 confidence band.

    Args:
        forecast_df: DataFrame with columns ``datetime``, ``aqi_p10``,
                     ``aqi_p50``, ``aqi_p90``.
        title: Chart title.
    """
    if forecast_df.empty:
        st.warning("No forecast data available.")
        return

    fig = go.Figure()

    # Confidence band
    fig.add_trace(go.Scatter(
        x=forecast_df["datetime"],
        y=forecast_df["aqi_p90"],
        mode="lines",
        line=dict(width=0),
        showlegend=False,
    ))
    fig.add_trace(go.Scatter(
        x=forecast_df["datetime"],
        y=forecast_df["aqi_p10"],
        mode="lines",
        line=dict(width=0),
        fill="tonexty",
        fillcolor="rgba(52, 152, 219, 0.2)",
        name="80% CI",
    ))

    # Median line
    fig.add_trace(go.Scatter(
        x=forecast_df["datetime"],
        y=forecast_df["aqi_p50"],
        mode="lines",
        line=dict(color="#2980B9", width=2),
        name="Median (p50)",
    ))

    # AQI threshold reference lines
    for threshold, label, colour in AQI_THRESHOLDS:
        fig.add_hline(
            y=threshold, line_dash="dot",
            line_color=colour, opacity=0.5,
            annotation_text=label,
            annotation_position="top left",
        )

    # "Now" line
    now = datetime.utcnow()
    fig.add_vline(x=now, line_dash="dash", line_color="grey",
                  annotation_text="Now")

    fig.update_layout(
        title=title,
        xaxis_title="Date / Time (UTC)",
        yaxis_title="AQI",
        yaxis=dict(range=[0, max(300, forecast_df["aqi_p90"].max() * 1.1)]),
        height=420,
        margin=dict(t=50, b=40),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    )
    st.plotly_chart(fig, use_container_width=True)
