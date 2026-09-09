"""
Transparent cross-sectional country-risk scoring engine.

Method:
1. Read indicator metadata/weights from config/indicators.yaml.
2. For every year and indicator, calculate a cross-sectional z-score.
3. Flip the sign where a higher raw value means lower risk.
4. Multiply by configured weights.
5. Renormalize over indicators actually observed for the country-year.
6. Convert the weighted signal to a 0–100 score around a neutral midpoint.

Scoring is pillar-based: each indicator belongs to a pillar (Monetary
Stability, Macro Growth, Fiscal Sustainability, External Vulnerability,
Banking Health). Pillar scores are computed independently, then the
composite score is a weighted average of pillar scores.

This is intentionally deterministic and inspectable.
"""

from __future__ import annotations

from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[2]
CONFIG_PATH = ROOT / "config" / "indicators.yaml"

# Pillar display order
PILLAR_ORDER = [
    "Monetary Stability",
    "Macro Growth",
    "Fiscal Sustainability",
    "External Vulnerability",
    "Banking Health",
]


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
    records = {
        str(r["code"]): r
        for r in _records()
        if r.get("code")
    }
    active_weights = [float(item.get("weight", 0)) for item in records.values() if float(item.get("weight", 0)) > 0]
    if not active_weights or not np.isclose(sum(active_weights), 1.0):
        raise ValueError("Configured positive indicator weights must sum to 1.0.")
    return records


def _pillar_weights(indicator_map: dict[str, dict]) -> dict[str, float]:
    """Compute total weight per pillar from indicator weights."""
    pw: dict[str, float] = defaultdict(float)
    for code, meta in indicator_map.items():
        pillar = meta.get("pillar", "Other")
        pw[pillar] += float(meta.get("weight", 0))
    return dict(pw)


def _zscore(series: pd.Series) -> pd.Series:
    numeric = pd.to_numeric(series, errors="coerce")
    mean = numeric.mean()
    std = numeric.std(ddof=0)

    if pd.isna(std) or std == 0:
        return pd.Series(0.0, index=series.index)

    return (numeric - mean) / std


def _score_from_signal(normalized_signal: float) -> float:
    """Map a normalized signal (typically -1..+1) to a 0-100 score."""
    return float(np.clip(50.0 + 18.0 * normalized_signal, 0.0, 100.0))


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
                    "pillar",
                    "z_risk",
                    "weighted_contribution",
                    "label",
                ]
            ),
            pd.DataFrame(
                columns=[
                    "country_iso3",
                    "year",
                    "pillar",
                    "pillar_score",
                    "pillar_band",
                    "pillar_weight",
                ]
            ),
        )

    cfg = _indicator_map()
    pillar_w = _pillar_weights(cfg)

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
                    "pillar": meta.get("pillar", "Other"),
                    "label": meta.get("label", code),
                    "raw_value": values,
                    "unit": meta.get("unit", ""),
                    "risk_direction": risk_direction,
                    "weight": weight,
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

    # --- Composite score (same logic as before, unchanged) ---
    group_keys = ["country_iso3", "year"]

    available_weight = (
        drivers["_weight"]
        .where(drivers["_available"], 0.0)
        .groupby([drivers[k] for k in group_keys])
        .transform("sum")
    )

    drivers["_available_weight"] = available_weight

    valid = drivers["_available"] & drivers["_available_weight"].gt(0)

    normalized_signal = (
        drivers["weighted_contribution"]
        .where(valid, 0.0)
        .groupby([drivers[k] for k in group_keys])
        .transform("sum")
        / drivers["_available_weight"].replace(0, np.nan)
    )

    normalized_signal = normalized_signal.fillna(0.0)

    score = normalized_signal.apply(_score_from_signal)

    total_weight = sum(float(m.get("weight", 0)) for m in cfg.values())
    completeness = (drivers["_available_weight"] / total_weight).clip(0.0, 1.0)

    scores = working[["country_iso3", "year"]].copy()
    scores["risk_score"] = score
    scores["risk_band"] = scores["risk_score"].map(_band)
    scores["data_completeness"] = completeness

    scores = (
        scores.drop_duplicates(["country_iso3", "year"])
        .sort_values(["country_iso3", "year"])
        .reset_index(drop=True)
    )

    # --- Pillar scores ---
    pillar_scores = _compute_pillar_scores(drivers, cfg, pillar_w, group_keys)

    # Merge pillar scores into the scores DataFrame
    if not pillar_scores.empty:
        for pillar_name in PILLAR_ORDER:
            col = f"pillar_{pillar_name.lower().replace(' ', '_')}_score"
            ps = pillar_scores[pillar_scores["pillar"] == pillar_name][
                ["country_iso3", "year", "pillar_score"]
            ].rename(columns={"pillar_score": col})
            scores = scores.merge(ps, on=["country_iso3", "year"], how="left")

    drivers = drivers[
        [
            "country_iso3",
            "year",
            "indicator_code",
            "category",
            "pillar",
            "label",
            "raw_value",
            "unit",
            "risk_direction",
            "weight",
            "z_risk",
            "weighted_contribution",
        ]
    ]

    return scores, drivers, pillar_scores


def _compute_pillar_scores(
    drivers: pd.DataFrame,
    cfg: dict[str, dict],
    pillar_w: dict[str, float],
    group_keys: list[str],
) -> pd.DataFrame:
    """Compute a 0-100 score for each pillar independently."""
    rows = []

    for pillar_name, pillar_total_weight in pillar_w.items():
        if pillar_total_weight <= 0:
            continue

        pillar_drivers = drivers[drivers["pillar"] == pillar_name].copy()
        if pillar_drivers.empty:
            continue

        # For each country-year, sum weighted contributions within this pillar
        # and normalize by available weight in this pillar
        pillar_drivers["_pw"] = pillar_drivers["_weight"].where(
            pillar_drivers["_available"], 0.0
        )

        pillar_sum = (
            pillar_drivers["weighted_contribution"]
            .where(pillar_drivers["_available"], 0.0)
            .groupby([pillar_drivers[k] for k in group_keys])
            .transform("sum")
        )

        pillar_available = (
            pillar_drivers["_pw"]
            .groupby([pillar_drivers[k] for k in group_keys])
            .transform("sum")
        )

        pillar_signal = (pillar_sum / pillar_available.replace(0, np.nan)).fillna(0.0)
        pillar_score = pillar_signal.apply(_score_from_signal)

        tmp = pillar_drivers[group_keys].copy()
        tmp["pillar"] = pillar_name
        tmp["pillar_score"] = pillar_score.values
        tmp["pillar_band"] = tmp["pillar_score"].map(_band)
        tmp["pillar_weight"] = pillar_total_weight

        rows.append(tmp.drop_duplicates(group_keys))

    if not rows:
        return pd.DataFrame(
            columns=["country_iso3", "year", "pillar", "pillar_score", "pillar_band", "pillar_weight"]
        )

    return pd.concat(rows, ignore_index=True).sort_values(group_keys + ["pillar"])


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
