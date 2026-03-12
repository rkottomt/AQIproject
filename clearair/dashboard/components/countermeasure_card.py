"""Styled countermeasure recommendation card component."""

from typing import Any

import streamlit as st

TIER_BADGE = {
    "low": "\U0001F7E2 Low",
    "medium": "\U0001F7E1 Medium",
    "high": "\U0001F534 High",
}


def render_countermeasure_card(rec: dict[str, Any], rank: int) -> None:
    """Render a single countermeasure recommendation as a styled card.

    Args:
        rec: Recommendation dict from the recommender / API.
        rank: Rank position (1-based).
    """
    tier = rec.get("cost_tier", "medium")
    badge = TIER_BADGE.get(tier, tier)

    with st.container():
        st.markdown(f"### #{rank}  {rec.get('display_name', rec.get('key', ''))}")
        st.markdown(f"_{rec.get('description', '')}_")

        cols = st.columns([1, 1, 1])

        pct = rec.get("pct_change")
        if pct is not None:
            cols[0].metric("AQI Change", f"{pct:+.1f}%",
                           delta=f"p10 {rec.get('aqi_p10_change', '—')} / p90 {rec.get('aqi_p90_change', '—')}")
        else:
            cols[0].metric("Impact Score", f"{rec.get('score', 0):.2f}")

        health = rec.get("health_impact", {})
        if health:
            cols[1].metric("Avoided Deaths/yr",
                           f"{health.get('avoided_premature_deaths_per_year', 0):.0f}")
            cols[2].metric("Economic Value",
                           health.get("economic_value_formatted", "—"))
        else:
            cols[1].metric("Cost Tier", badge)
            cols[2].metric("Lag Days", rec.get("typical_lag_days", "—"))

        window_start = rec.get("optimal_window_start")
        window_end = rec.get("optimal_window_end")
        if window_start and window_end:
            st.caption(f"Optimal window: {window_start[:10]} — {window_end[:10]}")

        score = rec.get("score", 0)
        if score:
            st.progress(min(score / 2.0, 1.0))

        st.divider()
