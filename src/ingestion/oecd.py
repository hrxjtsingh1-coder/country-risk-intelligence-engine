"""OECD SDMX REST API adapter. Public endpoint; no API key required."""
from __future__ import annotations

from typing import Any

from .common import fetched_at, session

BASE_URL = "https://sdmx.oecd.org/public/rest"


def fetch_data(agency: str, dataflow: str, version: str = "latest", key: str = "all", *, start_period: str | None = None, end_period: str | None = None, accept: str = "application/vnd.sdmx.data+csv;version=2.0.0") -> dict[str, Any]:
    url = f"{BASE_URL}/data/{agency},{dataflow},{version}/{key}"
    params: dict[str, str] = {"dimension_at_observation": "AllDimensions"}
    if start_period:
        params["startPeriod"] = start_period
    if end_period:
        params["endPeriod"] = end_period
    response = session().get(url, params=params, headers={"Accept": accept}, timeout=90)
    response.raise_for_status()
    return {"fetched_at": fetched_at(), "url": response.url, "body": response.text, "content_type": response.headers.get("content-type", "")}


def fetch_dataflow(agency: str, dataflow: str, version: str = "latest") -> dict[str, Any]:
    url = f"{BASE_URL}/dataflow/{agency}/{dataflow}/{version}"
    response = session().get(url, headers={"Accept": "application/vnd.sdmx.structure+csv;version=2.0.0"}, timeout=60)
    response.raise_for_status()
    return {"fetched_at": fetched_at(), "url": response.url, "body": response.text, "content_type": response.headers.get("content-type", "")}


def smoke_test() -> dict[str, Any]:
    result = fetch_data("OECD.SDD.NAD", "DSD_NASU@DF_INDICATOR", "latest", "all", start_period="2020", end_period="2020")
    ok = bool(result.get("body", "").strip())
    return {"source": "OECD", "ok": ok, "endpoint": result["url"], "detail": "OECD SUT indicator response returned" if ok else "Empty OECD response"}
