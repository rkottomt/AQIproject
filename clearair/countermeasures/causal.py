"""Causal inference for estimating the ATE of countermeasures on AQI."""

import logging
from typing import Any, Optional

import numpy as np
import pandas as pd

from countermeasures.library import CountermeasureLibrary

logger = logging.getLogger(__name__)


class CausalAttributor:
    """Uses DoWhy to estimate the average treatment effect of a countermeasure."""

    WEATHER_CONFOUNDERS = [
        "temperature_c", "humidity_pct", "wind_speed_ms",
        "wind_direction_deg", "precipitation_mm",
        "boundary_layer_height_m", "uv_index",
    ]
    TRAFFIC_CONFOUNDERS = ["congestion_index", "peak_hour_flag", "avg_speed_kmh"]

    def __init__(self, df: pd.DataFrame, city: str) -> None:
        self.df = df.copy()
        self.city = city
        self.library = CountermeasureLibrary()

    def build_causal_graph(self, countermeasure_key: str) -> str:
        """Build a DOT-format DAG for the given countermeasure.

        Args:
            countermeasure_key: Key from countermeasures.yaml.

        Returns:
            DOT-format string.
        """
        affected = self.library.get_affected_features(countermeasure_key)
        lines = ["digraph {"]

        for feat in affected:
            lines.append(f'  "{countermeasure_key}" -> "{feat}";')
            lines.append(f'  "{feat}" -> "AQI";')

        for conf in self.WEATHER_CONFOUNDERS:
            if conf in self.df.columns:
                lines.append(f'  "{conf}" -> "AQI";')

        for conf in self.TRAFFIC_CONFOUNDERS:
            if conf in self.df.columns:
                lines.append(f'  "{conf}" -> "AQI";')

        lines.append("}")
        return "\n".join(lines)

    def estimate_ate(
        self,
        countermeasure_key: str,
        event_start: str,
        event_end: str,
    ) -> dict[str, Any]:
        """Estimate the Average Treatment Effect using DoWhy.

        Args:
            countermeasure_key: Which countermeasure to evaluate.
            event_start: ISO datetime for the start of the intervention.
            event_end: ISO datetime for the end of the intervention.

        Returns:
            Dict with ate, p_value, ci_lower, ci_upper.
        """
        df = self.df.copy()

        # Binary treatment indicator
        start = pd.Timestamp(event_start, tz="UTC")
        end = pd.Timestamp(event_end, tz="UTC")
        idx = df.index if isinstance(df.index, pd.DatetimeIndex) else pd.to_datetime(df["datetime"])
        df["treatment"] = ((idx >= start) & (idx <= end)).astype(int)

        # Confounders present in the data
        confounders = [c for c in self.WEATHER_CONFOUNDERS + self.TRAFFIC_CONFOUNDERS
                       if c in df.columns]

        if "AQI" not in df.columns:
            return {"ate": np.nan, "p_value": np.nan,
                    "ci_lower": np.nan, "ci_upper": np.nan}

        try:
            import dowhy
            from dowhy import CausalModel

            graph = self.build_causal_graph(countermeasure_key)

            model = CausalModel(
                data=df[["treatment", "AQI"] + confounders].dropna(),
                treatment="treatment",
                outcome="AQI",
                graph=graph,
            )

            identified = model.identify_effect(proceed_when_unidentifiable=True)
            estimate = model.estimate_effect(
                identified,
                method_name="backdoor.propensity_score_matching",
            )

            ate = float(estimate.value)
            ci = estimate.get_confidence_intervals() if hasattr(estimate, "get_confidence_intervals") else (np.nan, np.nan)
            if isinstance(ci, tuple) and len(ci) == 2:
                ci_lower, ci_upper = float(ci[0]), float(ci[1])
            else:
                ci_lower, ci_upper = np.nan, np.nan

            return {
                "ate": ate,
                "p_value": np.nan,  # DoWhy does not always expose p-value
                "ci_lower": ci_lower,
                "ci_upper": ci_upper,
            }

        except ImportError:
            logger.error("DoWhy is not installed — using naive difference-in-means.")
            return self._naive_ate(df, start, end)
        except Exception:
            logger.error("Causal estimation failed.", exc_info=True)
            return self._naive_ate(df, start, end)

    def plot_causal_graph(self, countermeasure_key: str, save_path: str) -> None:
        """Render the DAG as an image.

        Args:
            countermeasure_key: Countermeasure identifier.
            save_path: Path to save the PNG.
        """
        try:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt
            import networkx as nx

            dot = self.build_causal_graph(countermeasure_key)
            G = nx.DiGraph()
            for line in dot.split("\n"):
                line = line.strip().rstrip(";")
                if "->" in line:
                    parts = line.split("->")
                    src = parts[0].strip().strip('"')
                    dst = parts[1].strip().strip('"')
                    G.add_edge(src, dst)

            pos = nx.spring_layout(G, seed=42)
            fig, ax = plt.subplots(figsize=(12, 8))
            nx.draw(G, pos, ax=ax, with_labels=True, node_color="#AED6F1",
                    edge_color="#5D6D7E", node_size=2000, font_size=8,
                    arrows=True, arrowsize=15)
            ax.set_title(f"Causal Graph — {countermeasure_key}")
            fig.tight_layout()
            from pathlib import Path as P
            P(save_path).parent.mkdir(parents=True, exist_ok=True)
            fig.savefig(save_path, dpi=150)
            plt.close(fig)
            logger.info("Causal graph saved to %s", save_path)
        except Exception:
            logger.warning("Failed to plot causal graph.", exc_info=True)

    # ------------------------------------------------------------------
    @staticmethod
    def _naive_ate(df: pd.DataFrame, start, end) -> dict:
        idx = df.index if isinstance(df.index, pd.DatetimeIndex) else pd.to_datetime(df.get("datetime"))
        during = df.loc[(idx >= start) & (idx <= end), "AQI"]
        outside = df.loc[(idx < start) | (idx > end), "AQI"]
        ate = float(during.mean() - outside.mean()) if len(during) and len(outside) else np.nan
        return {"ate": ate, "p_value": np.nan, "ci_lower": np.nan, "ci_upper": np.nan}
