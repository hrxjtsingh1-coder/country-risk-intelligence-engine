"""Read-only public API for the Country Risk Intelligence Engine.

Thin HTTP wrapper over the exact same scoring pipeline the dashboard runs —
no re-implemented models, no server-side mutations. Run from the repo root:

    uvicorn api.main:app

Panel source: data/processed/panel_wide.csv when a pipeline run exists,
otherwise the demo fixture (never silently mixed; every payload carries a
`using_demo_data` flag). The API stays offline-friendly and deterministic so
CI exercises it end to end.
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Any

import pandas as pd
import yaml
from fastapi import FastAPI, HTTPException, Query

from src.commentary.generate_commentary import generate_report
from src.scoring.risk_score import peer_percentile, score_panel, top_drivers

ROOT = Path(__file__).resolve().parents[1]
PROCESSED_PANEL = ROOT / "data" / "processed" / "panel_wide.csv"
DEMO_PANEL = ROOT / "data" / "demo" / "panel_wide.csv"
PROCESSED_MANIFEST = ROOT / "data" / "processed" / "manifest.json"


def _load_countries_cfg() -> dict[str, Any]:
    path = ROOT / "config" / "countries.yaml"
    if not path.exists():
        return {}
    raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    return raw if isinstance(raw, dict) else {}


_COUNTRIES_CFG = _load_countries_cfg()
_COUNTRY_NAMES = {
    str(c.get("iso3", "")): str(c.get("name", "")) for c in _COUNTRIES_CFG.get("countries", []) if isinstance(c, dict)
}


def _country_label(iso3: str) -> str:
    return _COUNTRY_NAMES.get(str(iso3).upper(), str(iso3).upper())


@lru_cache(maxsize=1)
def _load_panel() -> tuple[pd.DataFrame, bool]:
    if PROCESSED_PANEL.exists():
        return pd.read_csv(PROCESSED_PANEL), False
    if DEMO_PANEL.exists():
        return pd.read_csv(DEMO_PANEL), True
    raise RuntimeError("No panel available: missing data/processed and data/demo fixtures.")


@lru_cache(maxsize=1)
def _scoring() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    panel, _ = _load_panel()
    return score_panel(panel)


def _using_demo_data() -> bool:
    return _load_panel()[1]


def _to_float(value: object) -> float | None:
    try:
        if value is None or pd.isna(value):
            return None
        return round(float(value), 4)
    except (TypeError, ValueError):
        return None


def _scored_years(scores: pd.DataFrame, iso3: str) -> list[int]:
    subset = scores[scores["country_iso3"].astype(str).eq(str(iso3).upper())]
    subset = subset[pd.to_numeric(subset["risk_score"], errors="coerce").notna()]
    return sorted({int(y) for y in pd.to_numeric(subset["year"], errors="coerce").dropna()})


def _slice_row(scores: pd.DataFrame, iso3: str, year: int) -> pd.Series:
    matches = scores[
        scores["country_iso3"].astype(str).eq(str(iso3).upper())
        & pd.to_numeric(scores["year"], errors="coerce").eq(int(year))
    ]
    if matches.empty or pd.isna(matches.iloc[0].get("risk_score")):
        raise HTTPException(status_code=404, detail=f"No scored data for {iso3} in {year}.")
    return matches.iloc[0]


def _pillars(pillar_scores: pd.DataFrame, iso3: str, year: int) -> list[dict]:
    if not isinstance(pillar_scores, pd.DataFrame) or pillar_scores.empty:
        return []
    frame = pillar_scores[
        pillar_scores["country_iso3"].astype(str).eq(str(iso3).upper())
        & pd.to_numeric(pillar_scores["year"], errors="coerce").eq(int(year))
    ]
    rows = []
    for _, r in frame.iterrows():
        rows.append(
            {
                "pillar": str(r.get("pillar", "")),
                "score": _to_float(r.get("pillar_score")),
                "band": str(r.get("pillar_band", "") or ""),
                "weight": _to_float(r.get("pillar_weight")),
            }
        )
    rows.sort(key=lambda p: -p["score"] if p["score"] is not None else 1)
    return rows


def _drivers(drivers: pd.DataFrame, iso3: str, year: int) -> list[dict]:
    frame = top_drivers(drivers, iso3.upper(), int(year), n=6)
    rows = []
    for _, r in frame.iterrows():
        raw = _to_float(r.get("raw_value"))
        pts = _to_float(r.get("weighted_contribution"))
        rows.append(
            {
                "code": str(r.get("indicator_code", "")),
                "label": str(r.get("label", "") or r.get("indicator_code", "")),
                "category": str(r.get("category", "") or ""),
                "pillar": str(r.get("pillar", "") or ""),
                "contribution_points": (pts * 100) if pts is not None else None,
                "raw_value": raw,
                "unit": str(r.get("unit", "") or ""),
            }
        )
    return rows


def _provenance_block() -> dict:
    using_demo = _using_demo_data()
    manifest_hash = ""
    if PROCESSED_MANIFEST.exists():
        import hashlib

        manifest_hash = hashlib.sha256(PROCESSED_MANIFEST.read_bytes()).hexdigest()[:8]
    return {
        "using_demo_data": using_demo,
        "data_verified": (not using_demo) and bool(manifest_hash),
        "manifest_hash": manifest_hash,
    }


def _commentary(iso3: str, year: int, scores: pd.DataFrame, drivers: pd.DataFrame) -> str:
    return generate_report(
        country_name=_country_label(iso3),
        country_iso3=str(iso3).upper(),
        year=int(year),
        scores=scores,
        drivers=drivers,
        scenario_result=None,
        peer_group=None,
        sources=[],
        generated_at="api",
    )


app = FastAPI(
    title="Country Risk Intelligence Engine API",
    version="0.1.0",
    description="Thin read-only wrapper over the engine's scoring pipeline.",
)


@app.get("/")
def index() -> dict:
    return {
        "name": "Country Risk Intelligence Engine API",
        "endpoints": {
            "health": "/health",
            "risk": "/api/risk/{iso3}?year=",
            "history": "/api/risk/{iso3}/history",
        },
        "docs": "/docs",
    }


@app.get("/health")
def health() -> dict:
    return {"status": "ok", "using_demo_data": _using_demo_data()}


@app.get("/api/risk/{iso3}")
def risk(
    iso3: str,
    year: int | None = Query(default=None, description="Risk year; omitting uses the latest scored year."),
    include_commentary: bool = Query(default=False, description="Attach the analyst commentary block."),
) -> dict:
    scores, drivers, pillar_scores = _scoring()
    years = _scored_years(scores, iso3)
    if not years:
        raise HTTPException(status_code=404, detail=f"No scored data for {iso3}.")
    resolved = int(year) if year is not None else years[-1]
    if resolved not in years:
        raise HTTPException(status_code=404, detail=f"No scored data for {iso3} in {resolved}.")
    row = _slice_row(scores, iso3, resolved)

    peer = peer_percentile(scores, iso3.upper(), resolved)
    payload: dict[str, Any] = {
        "country": _country_label(iso3),
        "iso3": str(iso3).upper(),
        "year": resolved,
        "risk_score": _to_float(row.get("risk_score")),
        "risk_band": str(row.get("risk_band", "") or ""),
        "data_completeness": _to_float(row.get("data_completeness")),
        "trend": {
            "direction": str(row.get("trend_direction", "") or "") if pd.notna(row.get("trend_direction")) else "",
            "1y_delta": _to_float(row.get("trend_1y_delta")),
            "2y_delta": _to_float(row.get("trend_2y_delta")),
        },
        "pillars": _pillars(pillar_scores, iso3, resolved),
        "top_drivers": _drivers(drivers, iso3, resolved),
        "peer_standing": {
            "percentile": peer.get("percentile"),
            "riskier_share": peer.get("riskier_share"),
            "rank": peer.get("rank"),
            "n": int(peer.get("n", 0) or 0),
            "group_name": peer.get("group_name"),
        },
        "provenance": _provenance_block(),
    }
    if include_commentary:
        payload["commentary"] = _commentary(iso3, resolved, scores, drivers)
    return payload


@app.get("/api/risk/{iso3}/history")
def history(iso3: str) -> dict:
    scores, _, _ = _scoring()
    years = _scored_years(scores, iso3)
    if not years:
        raise HTTPException(status_code=404, detail=f"No scored data for {iso3}.")
    subset = scores[scores["country_iso3"].astype(str).eq(str(iso3).upper())]
    rows = []
    for y in years:
        r = subset[pd.to_numeric(subset["year"], errors="coerce").eq(y)].iloc[0]
        rows.append(
            {
                "year": y,
                "risk_score": _to_float(r.get("risk_score")),
                "risk_band": str(r.get("risk_band", "") or ""),
                "data_completeness": _to_float(r.get("data_completeness")),
                "trend_direction": str(r.get("trend_direction", "") or "")
                if pd.notna(r.get("trend_direction"))
                else "",
                "trend_1y_delta": _to_float(r.get("trend_1y_delta")),
            }
        )
    return {
        "country": _country_label(iso3),
        "iso3": str(iso3).upper(),
        "history": rows,
    }
