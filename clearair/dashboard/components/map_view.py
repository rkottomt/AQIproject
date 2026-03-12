"""Pydeck-based map component for monitoring stations."""

from typing import Any

import pydeck as pdk
import streamlit as st


def _aqi_colour(aqi: float) -> list[int]:
    """Map AQI to an RGBA colour."""
    if aqi <= 50:
        return [0, 228, 0, 180]
    if aqi <= 100:
        return [255, 255, 0, 180]
    if aqi <= 150:
        return [255, 126, 0, 180]
    if aqi <= 200:
        return [255, 0, 0, 180]
    if aqi <= 300:
        return [143, 63, 151, 180]
    return [126, 0, 35, 180]


def render_map(
    city: str,
    station_data: list[dict[str, Any]],
    lat: float = 19.076,
    lon: float = 72.877,
) -> None:
    """Render a dark-themed map with AQI-coloured station markers.

    Args:
        city: City key (for labelling).
        station_data: List of dicts with keys ``lat``, ``lon``, ``aqi``,
                      ``name``, ``dominant_pollutant``.
        lat: Centre latitude.
        lon: Centre longitude.
    """
    if not station_data:
        st.info("No station data available for the map.")
        return

    for s in station_data:
        s["color"] = _aqi_colour(s.get("aqi", 0))
        s["radius"] = max(200, s.get("aqi", 50) * 5)

    layer = pdk.Layer(
        "ScatterplotLayer",
        data=station_data,
        get_position=["lon", "lat"],
        get_color="color",
        get_radius="radius",
        pickable=True,
    )

    tooltip = {
        "html": "<b>{name}</b><br/>AQI: {aqi}<br/>Dominant: {dominant_pollutant}",
        "style": {"color": "white"},
    }

    view = pdk.ViewState(latitude=lat, longitude=lon, zoom=10, pitch=0)

    st.pydeck_chart(
        pdk.Deck(
            layers=[layer],
            initial_view_state=view,
            tooltip=tooltip,
            map_style="mapbox://styles/mapbox/dark-v10",
        )
    )
