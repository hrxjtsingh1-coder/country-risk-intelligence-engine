"""Eurostat Statistics API adapter. Public; no API key required."""
from __future__ import annotations

from datetime import date
from typing import Any

from .common import fetched_at, session

BASE_URL = "https://ec.europa.eu/eurostat/api/dissemination/statistics/1.0/data"
CURRENT_YEAR = date.today().year


def fetch_dataset(dataset: str, *, params: dict[str, Any] | None = None) -> dict[str, Any]:
    r = session().get(f"{BASE_URL}/{dataset}", params=params or {}, timeout=90)
    r.raise_for_status()
    return {"fetched_at": fetched_at(), "url": r.url, "data": r.json()}


def smoke_test() -> dict[str, Any]:
    result = fetch_dataset(
        "prc_hicp_midx",
        params={"geo": "DE", "coicop": "CP00", "unit": "I15.2015", "sinceTimePeriod": f"{CURRENT_YEAR}-01"},
    )
    data = result.get("data")
    return {"source": "Eurostat", "ok": isinstance(data, dict) and bool(data), "endpoint": result["url"], "year": CURRENT_YEAR}
