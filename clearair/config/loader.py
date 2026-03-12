"""Centralised configuration loader.

Reads cities.yaml and countermeasures.yaml once and exposes convenience
accessors used throughout the codebase.
"""

import os
import logging
from pathlib import Path
from typing import Any

import yaml

logger = logging.getLogger(__name__)

_CONFIG_DIR = Path(__file__).resolve().parent
_cities: dict[str, dict] = {}
_countermeasures: dict[str, dict] = {}


def _load_yaml(path: Path) -> dict:
    with open(path, "r") as fh:
        return yaml.safe_load(fh)


def load_all() -> None:
    """Load both YAML config files into module-level caches."""
    global _cities, _countermeasures

    cities_path = _CONFIG_DIR / "cities.yaml"
    cm_path = _CONFIG_DIR / "countermeasures.yaml"

    _cities = _load_yaml(cities_path) or {}
    _countermeasures = _load_yaml(cm_path) or {}

    logger.info("Loaded %d cities, %d countermeasures from config.",
                len(_cities), len(_countermeasures))


def get_city(key: str) -> dict[str, Any]:
    """Return config dict for a single city, raising KeyError if not found."""
    if not _cities:
        load_all()
    if key not in _cities:
        raise KeyError(f"City '{key}' not found in cities.yaml")
    return _cities[key]


def get_all_cities() -> dict[str, dict]:
    """Return the full cities config mapping."""
    if not _cities:
        load_all()
    return _cities


def get_countermeasure(key: str) -> dict[str, Any]:
    """Return config dict for a single countermeasure."""
    if not _countermeasures:
        load_all()
    if key not in _countermeasures:
        raise KeyError(f"Countermeasure '{key}' not in countermeasures.yaml")
    return _countermeasures[key]


def get_all_countermeasures() -> dict[str, dict]:
    """Return the full countermeasures config mapping."""
    if not _countermeasures:
        load_all()
    return _countermeasures


def add_city(key: str, config: dict[str, Any]) -> None:
    """Append a new city to cities.yaml and update the in-memory cache."""
    _cities[key] = config
    cities_path = _CONFIG_DIR / "cities.yaml"
    with open(cities_path, "w") as fh:
        yaml.dump(_cities, fh, default_flow_style=False, sort_keys=False)
    logger.info("Added city '%s' to config.", key)
