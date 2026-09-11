"""Read-only API: endpoints reuse the scoring pipeline and stay offline-safe."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest
from fastapi.testclient import TestClient

from api.main import app


@pytest.fixture(autouse=True)
def _force_demo_panel(monkeypatch):
    demo = Path(__file__).resolve().parents[1] / "data" / "demo" / "panel_wide.csv"
    frame = pd.read_csv(demo)
    monkeypatch.setattr("api.main._load_panel", lambda: (frame, True))
    yield


@pytest.fixture()
def client():
    with TestClient(app) as c:
        yield c


def test_health_ok(client):
    resp = client.get("/health")
    assert resp.status_code == 200
    assert resp.json()["status"] == "ok"
    assert resp.json()["using_demo_data"] is True


def test_index_lists_endpoints(client):
    resp = client.get("/")
    assert resp.status_code == 200
    body = resp.json()
    assert "health" in body["endpoints"]
    assert "risk" in body["endpoints"]
    assert "history" in body["endpoints"]


def test_risk_returns_full_slice(client):
    resp = client.get("/api/risk/USA")
    assert resp.status_code == 200
    body = resp.json()
    assert body["iso3"] == "USA"
    assert body["risk_score"] is not None
    assert body["risk_band"] in {"Low", "Moderate", "Elevated", "High", "Severe"}
    assert isinstance(body["trend"], dict)
    assert "direction" in body["trend"]
    assert isinstance(body["pillars"], list) and body["pillars"]
    assert "pillar" in body["pillars"][0]
    assert isinstance(body["top_drivers"], list) and body["top_drivers"]
    assert "contribution_points" in body["top_drivers"][0]
    assert body["provenance"]["using_demo_data"] is True
    assert body["provenance"]["data_verified"] is False


def test_risk_specific_year_and_commentary(client):
    resp = client.get("/api/risk/USA?year=2021&include_commentary=true")
    assert resp.status_code == 200
    body = resp.json()
    assert body["year"] == 2021
    assert isinstance(body["commentary"], str) and "Risk Score" in body["commentary"]


def test_risk_unknown_country_404(client):
    resp = client.get("/api/risk/ZZZ")
    assert resp.status_code == 404


def test_history_returns_scored_years(client):
    resp = client.get("/api/risk/USA/history")
    assert resp.status_code == 200
    body = resp.json()
    assert body["iso3"] == "USA"
    assert body["history"]
    first = body["history"][0]
    assert "year" in first and "risk_score" in first and "risk_band" in first
    years = [h["year"] for h in body["history"]]
    assert years == sorted(years)
