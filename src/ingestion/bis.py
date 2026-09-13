"""BIS Statistics SDMX REST API adapter. Public."""
from __future__ import annotations

from datetime import date
from typing import Any

from .common import fetched_at, session

BASE_URL = "https://stats.bis.org/api/v2/data/dataflow/BIS"
ACCEPT = "application/vnd.sdmx.data+json;version=1.0.0"
CURRENT_YEAR = date.today().year


def fetch_data(dataflow: str, version: str, key: str, *, start_period: str | None = None, end_period: str | None = None) -> dict[str, Any]:
    url = f"{BASE_URL}/{dataflow}/{version}/{key}"
    params: dict[str, str] = {}
    if start_period:
        params["startPeriod"] = start_period
    if end_period:
        params["endPeriod"] = end_period
    r = session().get(url, params=params, headers={"Accept": ACCEPT}, timeout=90)
    r.raise_for_status()
    return {"fetched_at": fetched_at(), "url": r.url, "data": r.json()}


def smoke_test() -> dict[str, Any]:
    quarter = ((date.today().month - 1) // 3) + 1
    period = f"{CURRENT_YEAR}-Q{quarter}"
    result = fetch_data("WS_CREDIT_GAP", "1.0", "Q.IN.P.A.C", start_period=period, end_period=period)
    return {"source": "BIS", "ok": bool(result.get("data")), "endpoint": result["url"], "year": CURRENT_YEAR}
