"""Eurostat dissemination API adapter. Public endpoint; no key required."""
from __future__ import annotations

from typing import Any

from .common import fetched_at, session

BASE_URL = "https://ec.europa.eu/eurostat/api/dissemination/statistics/1.0/data"


def fetch_dataset(dataset: str, *, params: dict[str, Any] | None = None) -> dict[str, Any]:
    url = f"{BASE_URL}/{dataset}"
    response = session().get(url, params=params or {}, timeout=90)
    response.raise_for_status()
    return {"fetched_at": fetched_at(), "url": response.url, "data": response.json()}


def smoke_test() -> dict[str, Any]:
    result = fetch_dataset("prc_hicp_midx", params={"geo": "DE", "coicop": "CP00", "unit": "I15.2015", "lang": "en"})
    data = result.get("data")
    ok = isinstance(data, dict) and bool(data.get("value"))
    return {"source": "Eurostat", "ok": ok, "endpoint": result["url"], "detail": "Germany HICP index returned" if ok else "No HICP observations returned"}
