"""Countermeasure catalogue backed by countermeasures.yaml."""

import logging
from typing import Any

from config import loader as cfg

logger = logging.getLogger(__name__)


class CountermeasureLibrary:
    """Provides read-only access to the countermeasure catalogue."""

    def __init__(self) -> None:
        self._data: dict[str, dict] = cfg.get_all_countermeasures()

    def get_all(self) -> dict[str, dict]:
        """Return the full countermeasure mapping."""
        return self._data

    def get(self, key: str) -> dict[str, Any]:
        """Return a single countermeasure by key.

        Args:
            key: Countermeasure identifier (e.g. ``construction_dust_control``).

        Returns:
            Config dict for the countermeasure.

        Raises:
            KeyError: If the key is not found.
        """
        if key not in self._data:
            raise KeyError(f"Countermeasure '{key}' not found in catalogue.")
        return self._data[key]

    def get_by_cost_tier(self, tier: str) -> list[dict[str, Any]]:
        """Filter countermeasures by cost tier.

        Args:
            tier: One of ``low``, ``medium``, ``high``.

        Returns:
            List of countermeasure dicts matching the tier.
        """
        tier = tier.lower()
        results = []
        for key, cm in self._data.items():
            if cm.get("cost_tier", "").lower() == tier:
                results.append({"key": key, **cm})
        return results

    def get_affected_features(self, key: str) -> dict[str, float]:
        """Return the {feature → multiplier} map for a countermeasure.

        Args:
            key: Countermeasure identifier.

        Returns:
            Dict mapping feature names to their effect multipliers.
        """
        cm = self.get(key)
        return cm.get("affects_features", {})
