"""Transparent cross-sectional country-risk scoring engine."""
from __future__ import annotations
from pathlib import Path
import numpy as np
import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[2]
CONFIG_PATH = ROOT / "config" / "indicators.yaml"


def _records() -> list[dict]:
    cfg = yaml.safe_load(CONFIG_PATH.read_text(encoding="utf-8")) or {}
    return [r for r in cfg.get("indicators", []) if isinstance(r, dict) and r.get("code")]


def _indicator_map() -> dict[str, dict]:
    records = {str(r["code"]): r for r in _records()}
    active = [float(r.get("weight", 0) or 0) for r in records.values() if float(r.get("weight", 0) or 0) > 0]
    if not active or not np.isclose(sum(active), 1.0, atol=1e-9):
        raise ValueError("Configured positive indicator weights must sum to 1.0.")
    return records


def _band(score: float) -> str:
    if score < 20:
        return "Low"
    if score < 40:
        return "Moderate"
    if score < 60:
        return "Elevated"
    if score < 80:
        return "High"
    return "Severe"


def _zscore(series: pd.Series) -> pd.Series:
    s = pd.to_numeric(series, errors="coerce")
    std = s.std(ddof=0)
    if pd.isna(std) or std == 0:
        return pd.Series(0.0, index=s.index)
    return (s - s.mean()) / std


def score_panel(panel: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    if panel is None or panel.empty:
        return pd.DataFrame(columns=["country_iso3", "year", "risk_score", "risk_band", "data_completeness"]), pd.DataFrame()
    cfg = _indicator_map()
    if "country_iso3" not in panel.columns or "year" not in panel.columns:
        raise ValueError("Panel must contain country_iso3 and year columns.")

    working = panel.copy()
    working["country_iso3"] = working["country_iso3"].astype(str).str.upper().str.strip()
    working["year"] = pd.to_numeric(working["year"], errors="coerce")
    working = working.dropna(subset=["country_iso3", "year"]).copy()
    working["year"] = working["year"].astype(int)

    frames: list[pd.DataFrame] = []
    for code, meta in cfg.items():
        weight = float(meta.get("weight", 0) or 0)
        if weight <= 0 or code not in working.columns:
            continue
        values = pd.to_numeric(working[code], errors="coerce")
        direction = float(meta.get("risk_direction", 1) or 1)
        z_risk = working.assign(_value=values).groupby("year")["_value"].transform(_zscore) * direction
        frames.append(pd.DataFrame({
            "country_iso3": working["country_iso3"],
            "year": working["year"],
            "indicator_code": code,
            "category": meta.get("category", "Macro"),
            "label": meta.get("label", code),
            "raw_value": values,
            "unit": meta.get("unit", ""),
            "risk_direction": direction,
            "weight": weight,
            "z_risk": z_risk,
            "weighted_contribution": z_risk * weight,
            "available": values.notna(),
        }))
    if not frames:
        raise ValueError("No configured positive-weight indicators were found in the panel.")

    drivers = pd.concat(frames, ignore_index=True)
    keys = ["country_iso3", "year"]
    available_weight = drivers["weight"].where(drivers["available"], 0.0).groupby([drivers[k] for k in keys]).transform("sum")
    drivers["available_weight"] = available_weight
    valid = drivers["available"] & drivers["available_weight"].gt(0)
    signal = drivers["weighted_contribution"].where(valid, 0.0).groupby([drivers[k] for k in keys]).transform("sum") / drivers["available_weight"].replace(0, np.nan)
    signal = signal.fillna(0.0)
    score = (50.0 + 18.0 * signal).clip(0.0, 100.0)
    completeness = (drivers["available_weight"] / sum(float(m.get("weight", 0) or 0) for m in cfg.values())).clip(0.0, 1.0)

    scores = pd.DataFrame({"country_iso3": drivers["country_iso3"], "year": drivers["year"], "risk_score": score, "data_completeness": completeness})
    scores = scores.groupby(keys, as_index=False).first()
    scores["risk_band"] = scores["risk_score"].map(_band)
    scores = scores[["country_iso3", "year", "risk_score", "risk_band", "data_completeness"]].sort_values(keys).reset_index(drop=True)
    drivers = drivers[["country_iso3", "year", "indicator_code", "category", "label", "raw_value", "unit", "risk_direction", "weight", "z_risk", "weighted_contribution"]].copy()
    return scores, drivers


def top_drivers(drivers: pd.DataFrame, country_iso3: str, year: int, n: int = 6) -> pd.DataFrame:
    if drivers is None or drivers.empty:
        return pd.DataFrame()
    sub = drivers[(drivers["country_iso3"].astype(str).str.upper() == str(country_iso3).upper()) & (pd.to_numeric(drivers["year"], errors="coerce") == int(year))].copy()
    return sub.reindex(sub["weighted_contribution"].abs().sort_values(ascending=False).index).head(int(n)).reset_index(drop=True)
