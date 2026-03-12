"""Tests for the FastAPI endpoints."""

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


@pytest.fixture
def client():
    """Create a synchronous TestClient for the FastAPI app."""
    from fastapi.testclient import TestClient
    from api.main import app
    return TestClient(app)


class TestForecastEndpoint:
    def test_returns_predictions(self, client):
        resp = client.get("/api/v1/forecast", params={"city": "mumbai", "horizon": 72})
        assert resp.status_code == 200
        data = resp.json()
        assert data["city"] == "mumbai"
        assert len(data["forecasts"]) == 72

    def test_current_endpoint(self, client):
        resp = client.get("/api/v1/forecast/current", params={"city": "mumbai"})
        assert resp.status_code == 200
        assert "current_aqi" in resp.json()

    def test_unknown_city_returns_404(self, client):
        resp = client.get("/api/v1/forecast", params={"city": "atlantis"})
        assert resp.status_code == 404


class TestCountermeasureEndpoint:
    def test_recommend_returns_ranked_list(self, client):
        resp = client.get("/api/v1/countermeasures/recommend",
                          params={"city": "mumbai", "budget": "low", "top_n": 3})
        assert resp.status_code == 200
        recs = resp.json()["recommendations"]
        assert len(recs) <= 3

    def test_list_all_countermeasures(self, client):
        resp = client.get("/api/v1/countermeasures/list")
        assert resp.status_code == 200
        cms = resp.json()["countermeasures"]
        assert len(cms) >= 5


class TestHealthEndpoint:
    def test_returns_economic_value(self, client):
        resp = client.get("/api/v1/health-impact",
                          params={"city": "mumbai", "aqi_before": 150,
                                  "aqi_after": 100, "exposure_days": 365})
        assert resp.status_code == 200
        data = resp.json()
        assert "economic_value_usd" in data
        assert data["economic_value_usd"] > 0


class TestCitiesEndpoint:
    def test_returns_configured_cities(self, client):
        resp = client.get("/api/v1/cities")
        assert resp.status_code == 200
        cities = resp.json()["cities"]
        keys = [c["key"] for c in cities]
        assert "mumbai" in keys
        assert "chicago" in keys

    def test_add_city_validates_fields(self, client):
        resp = client.post("/api/v1/cities/add",
                           params={"key": "test_city"},
                           json={
                               "display_name": "Test City",
                               "lat": 10.0, "lon": 20.0,
                               "timezone": "UTC",
                               "population": 500000,
                               "country_code": "XX",
                           })
        assert resp.status_code == 200
        assert "added" in resp.json()["message"].lower()
