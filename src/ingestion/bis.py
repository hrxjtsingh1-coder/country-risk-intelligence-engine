"""BIS Statistics SDMX REST API adapter."""
from __future__ import annotations

from typing import Any

from .common import fetched_at, session

BASE_URL = "https://stats.bis.org/api/v2/data/dataflow/BIS"
ACCEPT = "application/vnd.sdmx.data+json;version=1.0.0"


def fetch_data(dataflow: str, version: str, key: str, *, start_period: str | None = None, end_period: str | None = None) -> dict[str, Any]:
    url = f"{BASE_URL}/{dataflow}/{version}/{key}"
    params: dict[str, str] = {}
    if start_period:
        params["startPeriod"] = start_period
    if end_period:
        params["endPeriod"] = end_period
    response = session().get(url, params=params, headers={"Accept": ACCEPT}, timeout=90)
    response.raise_for_status()
    return {"fetched_at": fetched_at(), "url": response.url, "data": response.json()}


def smoke_test() -> dict[str, Any]:
    result = fetch_data("WS_CREDIT_GAP", "1.0", "Q.IN.P.A.C", start_period="2020-Q1", end_period="2024-Q4")
    data = result.get("data")
    return {"source": "BIS", "ok": bool(data), "endpoint": result["url"], "detail": "India credit-to-GDP gap returned" if data else "No BIS data returned"}
