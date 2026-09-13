"""IMF DataMapper API adapter. Public; no API key required for these endpoints."""
from __future__ import annotations

from typing import Any

from .common import get_json

BASE_URL = "https://www.imf.org/external/datamapper/api/v2"


def fetch_indicator(indicator: str, country: str | None = None, periods: str | None = None) -> dict[str, Any]:
    url = f"{BASE_URL}/{indicator}"
    if country:
        url = f"{url}/{country.upper()}"
    params = {"periods": periods} if periods else None
    return {"data": get_json(url, params=params), "url": url}


def smoke_test() -> dict[str, Any]:
    result = fetch_indicator("NGDP_RPCH", "IND")
    data = result.get("data")
    return {"source": "IMF", "ok": isinstance(data, dict) and bool(data), "endpoint": result["url"]}
