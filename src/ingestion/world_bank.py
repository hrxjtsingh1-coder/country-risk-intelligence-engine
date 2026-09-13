"""World Bank Indicators API adapter. Public; no API key required."""
from __future__ import annotations

from typing import Any

from src.runtime.year_policy import CURRENT_YEAR, validate_current_year_range

from .common import get_json

BASE_URL = "https://api.worldbank.org/v2/country/{country}/indicator/{indicator}"


def fetch_indicator(country: str, indicator: str, start: int | None = None, end: int | None = None) -> list[dict[str, Any]]:
    params: dict[str, Any] = {"format": "json", "per_page": 1000}
    if start is not None and end is not None:
        start, end = validate_current_year_range(start, end)
        params["date"] = f"{start}:{end}"
    elif start is not None or end is not None:
        raise ValueError(f"start and end must both be current year {CURRENT_YEAR}")
    payload = get_json(BASE_URL.format(country=country.upper(), indicator=indicator), params=params)
    if not isinstance(payload, list) or len(payload) < 2 or not isinstance(payload[1], list):
        return []
    # The production contract is current-year only. The upstream API may return
    # older observations when no date filter is supplied, so enforce the policy
    # again at the adapter boundary.
    return [row for row in payload[1] if str(row.get("date", "")) == str(CURRENT_YEAR)]


def smoke_test() -> dict[str, Any]:
    """Probe the public API while explicitly requesting the current year."""
    payload = get_json(
        BASE_URL.format(country="IND", indicator="SP.POP.TOTL"),
        params={"format": "json", "per_page": 100, "date": f"{CURRENT_YEAR}:{CURRENT_YEAR}"},
    )
    rows = payload[1] if isinstance(payload, list) and len(payload) >= 2 else None
    current_rows = [r for r in rows if str(r.get("date", "")) == str(CURRENT_YEAR)] if isinstance(rows, list) else []
    return {
        "source": "World Bank",
        "ok": bool(current_rows),
        "rows": len(current_rows),
        "year": CURRENT_YEAR,
        "endpoint": BASE_URL.format(country="IND", indicator="SP.POP.TOTL"),
    }
