"""
Transparent cross-sectional country-risk scoring engine.

Method:
1. Read indicator metadata/weights from config/indicators.yaml.
2. For every year and indicator, calculate a cross-sectional z-score.
3. Flip the sign where a higher raw value means lower risk.
4. Multiply by configured weights.
5. Renormalize over indicators actually observed for the country-year.
6. Convert the weighted signal to a 0–100 score around a neutral midpoint.

This is intentionally deterministic and inspectable.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[2]
CONFIG_PATH = ROOT / "config" / "indicators.yaml"


def _load_config() -> dict:
    with open(CONFIG_PATH, encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def _records() -> list[dict]:
    cfg = _load_config()
    records = cfg.get("indicators", []) if isinstance(cfg, dict) else []
    return [r for r in records if isinstance(r, dict)]


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


def _indicator_map() -> dict[str, dict]:
    return {
        str(r["code"]): r
        for r in _records()
        if r.get("code")
    }


def _zscore(series: pd.Series) -> pd.Series:
    numeric = pd.to_numeric(series, errors="coerce")
    mean = numeric.mean()
    std = numeric.std(ddof=0)

    if pd.isna(std) or std == 0:
        return pd.Series(0.0, index=series.index)

    return (numeric - mean) / std


def score_panel(panel: pd.DataFrame):
    if panel is None or panel.empty:
        return (
            pd.DataFrame(
                columns=[
                    "country_iso3",
                    "year",
                    "risk_score",
                    "risk_band",
                    "data_completeness",
                ]
            ),
            pd.DataFrame(
                columns=[
                    "country_iso3",
                    "year",
                    "indicator_code",
                    "category",
                    "z_risk",
                    "weighted_contribution",
                    "label",
                ]
            ),
        )

    cfg = _indicator_map()

    working = panel.copy()

    if "country_iso3" not in working.columns or "year" not in working.columns:
        raise ValueError("Panel must contain country_iso3 and year columns.")

    working["country_iso3"] = working["country_iso3"].astype(str).str.upper()
    working["year"] = pd.to_numeric(working["year"], errors="coerce").astype(int)

    driver_frames = []

    for code, meta in cfg.items():
        if code not in working.columns:
            continue

        values = pd.to_numeric(working[code], errors="coerce")
        risk_direction = float(meta.get("risk_direction", 1))
        weight = float(meta.get("weight", 0))

        z = (
            working.assign(_value=values)
            .groupby("year")["_value"]
            .transform(_zscore)
        )

        z_risk = z * risk_direction
        contribution = z_risk * weight

        driver_frames.append(
            pd.DataFrame(
                {
                    "country_iso3": working["country_iso3"],
                    "year": working["year"],
                    "indicator_code": code,
                    "category": meta.get("category", "Macro"),
                    "label": meta.get("label", code),
                    "z_risk": z_risk,
                    "weighted_contribution": contribution,
                    "_available": values.notna(),
                    "_weight": weight,
                }
            )
        )

    if not driver_frames:
        raise ValueError("No configured indicators were found in the panel.")

    drivers = pd.concat(driver_frames, ignore_index=True)

    # Normalize by the available configured weight for each country-year.
    group_keys = ["country_iso3", "year"]

    available_weight = (
        drivers["_weight"]
        .where(drivers["_available"], 0.0)
        .groupby([drivers[k] for k in group_keys])
        .transform("sum")
    )

    drivers["_available_weight"] = available_weight

    valid = drivers["_available"] & drivers["_available_weight"].gt(0)

    # A cross-sectional z-score has 0 as the panel midpoint. Mapping it through
    # a bounded logistic-like transform produces an interpretable 0–100 scale.
    normalized_signal = (
        drivers["weighted_contribution"]
        .where(valid, 0.0)
        .groupby([drivers[k] for k in group_keys])
        .transform("sum")
        / drivers["_available_weight"].replace(0, np.nan)
    )

    normalized_signal = normalized_signal.fillna(0.0)

    score = 50.0 + 18.0 * normalized_signal
    score = score.clip(0.0, 100.0)

    completeness = (
        drivers["_available_weight"]
        / sum(float(m.get("weight", 0)) for m in cfg.values())
    ).clip(0.0, 1.0)

    scores = working[["country_iso3", "year"]].copy()
    scores["risk_score"] = score
    scores["risk_band"] = scores["risk_score"].map(_band)
    scores["data_completeness"] = completeness

    scores = (
        scores.drop_duplicates(["country_iso3", "year"])
        .sort_values(["country_iso3", "year"])
        .reset_index(drop=True)
    )

    drivers = drivers[
        [
            "country_iso3",
            "year",
            "indicator_code",
            "category",
            "label",
            "z_risk",
            "weighted_contribution",
        ]
    ]

    return scores, drivers


def top_drivers(
    drivers: pd.DataFrame,
    country_iso3: str,
    year: int,
    n: int = 6,
) -> pd.DataFrame:
    subset = drivers[
        drivers["country_iso3"].astype(str).eq(str(country_iso3))
        & pd.to_numeric(drivers["year"], errors="coerce").eq(int(year))
    ].copy()

    if subset.empty:
        return subset

    subset = subset.sort_values(
        "weighted_contribution",
        key=lambda s: s.abs(),
        ascending=False,
    )

    return subset.head(int(n)).reset_index(drop=True)
