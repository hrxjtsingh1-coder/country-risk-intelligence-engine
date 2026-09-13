"""World Bank Indicators API adapter. Public; no API key required."""
from __future__ import annotations

from typing import Any

from .common import get_json

BASE_URL = "https://api.worldbank.org/v2/country/{country}/indicator/{indicator}"


def fetch_indicator(country: str, indicator: str, start: int | None = None, end: int | None = None) -> list[dict[str, Any]]:
    params: dict[str, Any] = {"format": "json", "per_page": 1000}
    if start is not None and end is not None:
        params["date"] = f"{start}:{end}"
    payload = get_json(BASE_URL.format(country=country.upper(), indicator=indicator), params=params)
    if not isinstance(payload, list) or len(payload) < 2 or not isinstance(payload[1], list):
        return []
    return payload[1]


def smoke_test() -> dict[str, Any]:
    rows = fetch_indicator("IND", "NY.GDP.MKTP.CD", 2020, 2024)
    return {"source": "World Bank", "ok": bool(rows), "rows": len(rows)}
