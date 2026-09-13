"""Read-only API tests for the current-year-only public contract."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest
from fastapi.testclient import TestClient

from api.main import app
from src.runtime.year_policy import CURRENT_YEAR


@pytest.fixture(autouse=True)
def _force_demo_panel(monkeypatch):
    demo = Path(__file__).resolve().parents[1] / "data" / "demo" / "panel_wide.csv"
    frame = pd.read_csv(demo)
    frame["year"] = pd.to_numeric(frame["year"], errors="coerce")
    frame = frame.loc[frame["year"].eq(CURRENT_YEAR)].copy()
    monkeypatch.setattr("api.main._load_panel", lambda: frame)
    yield


@pytest.fixture()
def client():
    with TestClient(app) as c:
        yield c


def test_health_ok(client):
    resp = client.get("/health")
    assert resp.status_code == 200
    assert resp.json()["status"] == "ok"
    assert resp.json()["current_year"] == CURRENT_YEAR


def test_index_lists_endpoints(client):
    resp = client.get("/")
    assert resp.status_code == 200
    body = resp.json()
    assert "health" in body["endpoints"]
    assert "risk" in body["endpoints"]
    assert body["current_year"] == CURRENT_YEAR


def test_risk_returns_full_slice(client):
    resp = client.get("/api/risk/USA")
    assert resp.status_code == 200
    body = resp.json()
    assert body["iso3"] == "USA"
    assert body["year"] == CURRENT_YEAR
    assert body["risk_score"] is not None
    assert body["risk_band"] in {"Low", "Moderate", "Elevated", "High", "Severe"}
    assert isinstance(body["trend"], dict)
    assert "direction" in body["trend"]
    assert isinstance(body["pillars"], list) and body["pillars"]
    assert "pillar" in body["pillars"][0]
    assert isinstance(body["top_drivers"], list) and body["top_drivers"]
    assert "contribution_points" in body["top_drivers"][0]


def test_historical_year_is_rejected(client):
    old_year = CURRENT_YEAR - 1
    resp = client.get(f"/api/risk/USA?year={old_year}")
    assert resp.status_code == 400
    assert str(CURRENT_YEAR) in resp.json()["detail"]


def test_current_year_commentary(client):
    resp = client.get(f"/api/risk/USA?year={CURRENT_YEAR}&include_commentary=true")
    assert resp.status_code == 200
    body = resp.json()
    assert body["year"] == CURRENT_YEAR
    assert isinstance(body["commentary"], str) and "Risk Score" in body["commentary"]


def test_risk_unknown_country_404(client):
    resp = client.get("/api/risk/ZZZ")
    assert resp.status_code == 404


def test_history_returns_current_year_only(client):
    resp = client.get("/api/risk/USA/history")
    assert resp.status_code == 200
    body = resp.json()
    assert body["iso3"] == "USA"
    assert body["history"]
    years = [h["year"] for h in body["history"]]
    assert years == [CURRENT_YEAR]
