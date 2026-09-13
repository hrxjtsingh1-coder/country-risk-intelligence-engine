"""IMF DataMapper API v2 adapter. Public endpoint; no key required."""
from __future__ import annotations

from typing import Any

from .common import fetched_at, session

BASE_URL = "https://www.imf.org/external/datamapper/api/v2"


def fetch_series(indicator: str, countries: str | None = None, periods: str | None = None) -> dict[str, Any]:
    parts = [indicator]
    if countries:
        parts.append(countries.upper())
    url = "/".join([BASE_URL, *parts])
    params = {"periods": periods} if periods else None
    response = session().get(url, params=params, timeout=60)
    response.raise_for_status()
    return {"fetched_at": fetched_at(), "url": response.url, "data": response.json()}


def smoke_test() -> dict[str, Any]:
    result = fetch_series("NGDP_RPCH", "IND")
    data = result.get("data") or {}
    values = data.get("values", {}) if isinstance(data, dict) else {}
    ok = bool(values.get("NGDP_RPCH") or values)
    return {"source": "IMF", "ok": ok, "endpoint": result["url"], "detail": "Real GDP growth series returned" if ok else "No series values returned"}
