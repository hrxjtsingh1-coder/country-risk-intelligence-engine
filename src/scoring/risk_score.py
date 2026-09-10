"""
Transparent country-risk scoring engine (Phase 4 methodology).

Method:
1. Read indicator metadata/pillar weights from config/indicators.yaml.
2. For every indicator, compute a risk z-score that blends TWO normalizations:
     - time z : the country's value vs its OWN history over the panel window
                (nominally 10 years)
     - peer z : the country's value vs its PEER GROUP (advanced/emerging,
                from config/countries.yaml) in the SAME year
   A side with too few observations is dropped and the other side carries the
   weight, so we never force garbage into the blend.
3. Flip sign where a higher raw value means lower risk and multiply by the
   indicator's (intra-pillar relative) weight.
4. Score each pillar 0-100 from its own indicator set, renormalized over what
   is actually available for that country-year.
5. Optionally group pillars into sectors (a "super-pillar" layer) and score
   each sector 0-100 as the renormalized weighted average of its pillars.
6. Combine everything into a 0-100 composite: the sector score (when enabled)
   is blended into the pillar-weighted composite by `sector_composite_weight`,
   with both sides' weights renormalized over whatever had data.
7. Emit trend deltas (vs 1 and 2 periods behind, finest cadence the panel
   supports) so a user can say "deteriorating" or "improving", and carry the
   z components on every driver row so a score can always be explained.

This is intentionally deterministic and inspectable.
"""

from __future__ import annotations

import re
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[2]
CONFIG_PATH = ROOT / "config" / "indicators.yaml"
COUNTRIES_PATH = ROOT / "config" / "countries.yaml"

# Pillar display order (Political / Event Risk is reserved at 0 weight).
PILLAR_ORDER = [
    "Macro Growth & Inflation",
    "Fiscal Sustainability",
    "External Vulnerability",
    "Monetary & Currency Stability",
    "Banking & Financial Sector Health",
    "Political / Event Risk",
]

DEFAULT_SCORING: dict[str, float] = {
    "history_z_weight": 0.5,
    "peer_z_weight": 0.5,
    "min_history_obs": 5.0,
    "min_peer_obs": 2.0,
    "trend_threshold_points": 1.5,
    "sector_composite_weight": 0.0,
}


def _pillar_slug(name: str) -> str:
    """'Macro Growth & Inflation' -> 'macro_growth_inflation' (column friendly)."""
    return re.sub(r"[^a-z0-9]+", "_", name.lower()).strip("_")


def _load_config() -> dict:
    with open(CONFIG_PATH, encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def _records() -> list[dict]:
    cfg = _load_config()
    records = cfg.get("indicators", []) if isinstance(cfg, dict) else []
    return [r for r in records if isinstance(r, dict)]


def _load_scoring() -> dict[str, float]:
    cfg = _load_config()
    block = cfg.get("scoring", {}) if isinstance(cfg, dict) else {}
    merged = dict(DEFAULT_SCORING)
    if isinstance(block, dict):
        merged.update({k: float(v) for k, v in block.items() if v is not None})
    return merged


def _load_pillar_weights() -> dict[str, float]:
    cfg = _load_config()
    weights = cfg.get("pillar_weights", {}) if isinstance(cfg, dict) else {}
    if not isinstance(weights, dict) or not weights:
        raise ValueError("config/indicators.yaml must define pillar_weights.")
    pw = {str(k): float(v) for k, v in weights.items()}
    if not np.isclose(sum(pw.values()), 1.0):
        raise ValueError(f"Configured pillar weights must sum to 1.0 (got {sum(pw.values()):.4f}).")
    if not any(w > 0 for w in pw.values()):
        raise ValueError("At least one pillar must carry positive weight.")
    return pw


def _load_peer_groups() -> dict[str, str]:
    """iso3 -> peer-group name, from config/countries.yaml peer_groups."""
    try:
        with open(COUNTRIES_PATH, encoding="utf-8") as f:
            cfg = yaml.safe_load(f) or {}
    except OSError:
        return {}
    groups = cfg.get("peer_groups", {}) if isinstance(cfg, dict) else {}
    mapping: dict[str, str] = {}
    for group_name, members in groups.items():
        if isinstance(members, list):
            for iso3 in members:
                mapping[str(iso3).upper()] = str(group_name)
    return mapping


def _load_sectors() -> tuple[dict[str, float], dict[str, list[str]]]:
    """Return (sector_name -> weight, sector_name -> list of pillar names).

    Sectors are the optional "super-pillar" layer: each sector groups one or
    more pillars, and the sector score is the renormalized weighted average of
    those pillars' scores. The composite blends the pillar-weighted composite
    with the sector_score by `sector_composite_weight`. An empty result means
    sectors are deactivated (pure pillar-driven composite).
    """
    cfg = _load_config()
    if not isinstance(cfg, dict):
        return {}, {}

    weights_cfg = cfg.get("sector_weights", {})
    members_cfg = cfg.get("sectors", {})
    if not isinstance(weights_cfg, dict) and not isinstance(members_cfg, dict):
        return {}, {}

    sector_weights: dict[str, float] = {}
    sectors: dict[str, list[str]] = {}
    for name, members in members_cfg.items():
        sector_name = str(name)
        sectors[sector_name] = [str(m) for m in members] if isinstance(members, list) else []
    for name, weight in weights_cfg.items():
        sector_weights[str(name)] = float(weight)

    if not sectors and not sector_weights:
        return {}, {}

    _validate_sectors(sector_weights, sectors, _load_pillar_weights())
    return sector_weights, sectors


def _validate_sectors(
    sector_weights: dict[str, float],
    sectors: dict[str, list[str]],
    pillar_w: dict[str, float],
) -> None:
    """Fail fast on a misconfigured sectors block."""
    known_pillars = set(PILLAR_ORDER)
    pillar_names = {p for p, w in pillar_w.items() if w > 0}

    if not sector_weights or not sectors:
        raise ValueError("sector_weights and sectors must both be defined to activate sectors.")

    if not np.isclose(sum(sector_weights.values()), 1.0):
        raise ValueError(f"Configured sector weights must sum to 1.0 (got {sum(sector_weights.values()):.4f}).")
    if set(sector_weights) != set(sectors):
        raise ValueError("sector_weights and sectors must define the same sector names.")

    covered: list[str] = []
    for sector_name, members in sectors.items():
        if sector_weights.get(sector_name, 0.0) <= 0:
            raise ValueError(f"Sector '{sector_name}' must carry positive weight.")
        if not members:
            raise ValueError(f"Sector '{sector_name}' must list at least one pillar.")
        unknown = [m for m in members if m not in known_pillars]
        if unknown:
            raise ValueError(f"Sector '{sector_name}' lists unknown pillar(s): {', '.join(unknown)}")
        covered.extend(members)

    missing = sorted(pillar_names - set(covered))
    if missing:
        raise ValueError(
            "Every active pillar must belong to exactly one sector; missing pillar(s): " + ", ".join(missing)
        )
    duplicated = sorted({m for m in covered if covered.count(m) > 1})
    if duplicated:
        raise ValueError(f"Pillar(s) assigned to more than one sector: {', '.join(duplicated)}")


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
    return {str(r["code"]): r for r in _records() if r.get("code")}


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


def _empty_frames():
    scores = pd.DataFrame(
        columns=[
            "country_iso3",
            "year",
            "risk_score",
            "risk_band",
            "data_completeness",
            "trend_1y_delta",
            "trend_2y_delta",
            "trend_direction",
            "sector_score",
        ]
    )
    drivers = pd.DataFrame(
        columns=[
            "country_iso3",
            "year",
            "indicator_code",
            "category",
            "pillar",
            "z_time",
            "z_peer",
            "z_combined",
            "z_risk",
            "weighted_contribution",
            "label",
        ]
    )
    pillar_scores = pd.DataFrame(
        columns=["country_iso3", "year", "pillar", "pillar_score", "pillar_band", "pillar_weight"]
    )
    return scores, drivers, pillar_scores


def score_panel(panel: pd.DataFrame):
    if panel is None or panel.empty:
        return _empty_frames()

    cfg = _indicator_map()
    pillar_w = _load_pillar_weights()
    scoring = _load_scoring()
    peer_map = _load_peer_groups()

    min_history_obs = float(scoring["min_history_obs"])
    min_peer_obs = float(scoring["min_peer_obs"])

    working = panel.copy()

    if "country_iso3" not in working.columns or "year" not in working.columns:
        raise ValueError("Panel must contain country_iso3 and year columns.")

    working["country_iso3"] = working["country_iso3"].astype(str).str.upper()
    working["year"] = pd.to_numeric(working["year"], errors="coerce").astype(int)
    working["_peer_group"] = working["country_iso3"].map(peer_map)

    driver_frames = []

    for code, meta in cfg.items():
        if code not in working.columns:
            continue

        values = pd.to_numeric(working[code], errors="coerce")
        risk_direction = float(meta.get("risk_direction", 1))
        weight = float(meta.get("weight", 0))

        # time z — the country's value against its own full panel history
        n_hist = values.notna().groupby(working["country_iso3"]).transform("sum")
        time_z = values.groupby(working["country_iso3"]).transform(_zscore)
        time_z = time_z.where(n_hist >= min_history_obs)

        # peer z — the country's value against its peer group in the same year
        # (whole-panel cross-section when the country has no peer group)
        gkey = working["_peer_group"].fillna("__all__")
        n_peer = values.notna().groupby([working["year"], gkey]).transform("sum")
        peer_z = values.groupby([working["year"], gkey]).transform(_zscore)
        peer_z = peer_z.where(n_peer >= min_peer_obs)

        # blend whatever sides cleared the minimum-observation bar
        tw = float(scoring["history_z_weight"])
        p_w = float(scoring["peer_z_weight"])
        has_time = time_z.notna()
        has_peer = peer_z.notna()
        denom = has_time.to_numpy(dtype=float) * tw + has_peer.to_numpy(dtype=float) * p_w
        combined: pd.Series = (time_z.fillna(0.0) * tw + peer_z.fillna(0.0) * p_w) / np.where(denom == 0, np.nan, denom)
        combined = combined.fillna(0.0)

        z_risk = combined * risk_direction
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
                    "z_time": time_z,
                    "z_peer": peer_z,
                    "z_combined": combined,
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
    group_keys = ["country_iso3", "year"]

    available_weight = (
        drivers["_weight"].where(drivers["_available"], 0.0).groupby([drivers[k] for k in group_keys]).transform("sum")
    )
    drivers["_available_weight"] = available_weight

    total_weight = round(sum(float(m.get("weight", 0)) for m in cfg.values()), 6)
    completeness = (available_weight.round(6) / total_weight).clip(0.0, 1.0)

    pillar_scores = _compute_pillar_scores(drivers, pillar_w, group_keys)

    pillar_composite = _composite_from_pillars(pillar_scores, pillar_w, working)["composite"]

    sector_weights, sectors = _load_sectors()
    sector_scores_df = pd.DataFrame(
        columns=["country_iso3", "year", "sector", "sector_score", "sector_band", "sector_weight"]
    )
    if sectors:
        sector_scores_df = _compute_sector_scores(pillar_scores, sector_weights, sectors, group_keys)
    aggregate_sector_score = _aggregate_sector_score(sector_scores_df, sector_weights, working)

    sector_beta = float(scoring["sector_composite_weight"])
    if aggregate_sector_score.notna().any():
        composite = (1.0 - sector_beta) * pillar_composite + sector_beta * aggregate_sector_score
    else:
        composite = pillar_composite

    scores = working[["country_iso3", "year"]].copy()
    scores["risk_score"] = composite
    scores["risk_band"] = scores["risk_score"].map(_band)
    scores["data_completeness"] = completeness
    scores["sector_score"] = aggregate_sector_score
    scores = _add_trend_deltas(scores, scoring)

    if not pillar_scores.empty:
        for pillar_name in PILLAR_ORDER:
            col = f"pillar_{_pillar_slug(pillar_name)}_score"
            ps = pillar_scores[pillar_scores["pillar"] == pillar_name][["country_iso3", "year", "pillar_score"]].rename(
                columns={"pillar_score": col}
            )
            if ps.empty:
                continue
            scores = scores.merge(ps, on=["country_iso3", "year"], how="left")

    if not sector_scores_df.empty:
        for sector_name in sectors:
            col = f"sector_{_pillar_slug(sector_name)}_score"
            ss = sector_scores_df[sector_scores_df["sector"] == sector_name][
                ["country_iso3", "year", "sector_score"]
            ].rename(columns={"sector_score": col})
            if ss.empty:
                continue
            scores = scores.merge(ss, on=["country_iso3", "year"], how="left")

    scores = (
        scores.drop_duplicates(["country_iso3", "year"]).sort_values(["country_iso3", "year"]).reset_index(drop=True)
    )

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
            "z_time",
            "z_peer",
            "z_combined",
            "z_risk",
            "weighted_contribution",
        ]
    ]

    return scores, drivers, pillar_scores


def _composite_from_pillars(
    pillar_scores: pd.DataFrame,
    pillar_w: dict[str, float],
    working: pd.DataFrame,
) -> pd.DataFrame:
    """Weighted average of pillar scores, renormalized over pillars with data."""
    keys = working[["country_iso3", "year"]].copy()

    if pillar_scores.empty or pillar_scores["pillar_score"].notna().empty:
        keys["composite"] = np.nan
        return keys

    ps = pillar_scores.dropna(subset=["pillar_score"])[["country_iso3", "year", "pillar", "pillar_score"]].copy()
    if ps.empty:
        keys["composite"] = np.nan
        return keys

    ps["_w"] = ps["pillar"].map(pillar_w).fillna(0.0)
    agg = (
        ps.assign(_contrib=ps["pillar_score"] * ps["_w"])
        .groupby(["country_iso3", "year"], as_index=False)
        .agg(contribution=("_contrib", "sum"), weight=("_w", "sum"))
    )
    agg["composite"] = agg["contribution"] / agg["weight"].replace(0, np.nan)

    return keys.merge(agg[["country_iso3", "year", "composite"]], on=["country_iso3", "year"], how="left")


def _compute_pillar_scores(
    drivers: pd.DataFrame,
    pillar_w: dict[str, float],
    group_keys: list[str],
) -> pd.DataFrame:
    """Compute a 0-100 score for each pillar independently."""
    rows = []

    active_pillars = {p for p, w in pillar_w.items() if w > 0}

    for pillar_name in PILLAR_ORDER:
        if pillar_name not in active_pillars:
            continue

        pillar_drivers = drivers[drivers["pillar"] == pillar_name].copy()
        if pillar_drivers.empty:
            continue

        pillar_drivers["_pw"] = pillar_drivers["_weight"].where(pillar_drivers["_available"], 0.0)

        agg = pillar_drivers.groupby(group_keys, as_index=False).agg(
            contribution=("weighted_contribution", "sum"), available=("_pw", "sum")
        )
        agg = agg[agg["available"] > 0].copy()
        if agg.empty:
            continue

        agg["pillar"] = pillar_name
        agg["pillar_score"] = (agg["contribution"] / agg["available"]).apply(_score_from_signal)
        agg["pillar_band"] = agg["pillar_score"].map(_band)
        agg["pillar_weight"] = pillar_w.get(pillar_name, 0.0)

        rows.append(agg[[*group_keys, "pillar", "pillar_score", "pillar_band", "pillar_weight"]])

    if not rows:
        return pd.DataFrame(columns=["country_iso3", "year", "pillar", "pillar_score", "pillar_band", "pillar_weight"])

    return pd.concat(rows, ignore_index=True).sort_values(group_keys + ["pillar"])


def _compute_sector_scores(
    pillar_scores: pd.DataFrame,
    sector_weights: dict[str, float],
    sectors: dict[str, list[str]],
    group_keys: list[str],
) -> pd.DataFrame:
    """Compute a 0-100 score for each sector (a weighted avg of its pillar scores)."""
    rows = []

    for sector_name, members in sectors.items():
        if sector_weights.get(sector_name, 0.0) <= 0:
            continue

        member = pillar_scores[pillar_scores["pillar"].isin(members)].copy()
        member = member.dropna(subset=["pillar_score"])
        if member.empty:
            continue

        agg = (
            member.assign(_contrib=member["pillar_score"] * member["pillar_weight"])
            .groupby(group_keys, as_index=False)
            .agg(contribution=("_contrib", "sum"), weight=("pillar_weight", "sum"))
        )
        agg = agg[agg["weight"] > 0].copy()
        if agg.empty:
            continue

        agg["sector"] = sector_name
        agg["sector_score"] = agg["contribution"] / agg["weight"]
        agg["sector_band"] = agg["sector_score"].map(_band)
        agg["sector_weight"] = sector_weights.get(sector_name, 0.0)

        rows.append(agg[[*group_keys, "sector", "sector_score", "sector_band", "sector_weight"]])

    if not rows:
        return pd.DataFrame(columns=["country_iso3", "year", "sector", "sector_score", "sector_band", "sector_weight"])

    return pd.concat(rows, ignore_index=True).sort_values(group_keys + ["sector"])


def _aggregate_sector_score(
    sector_scores_df: pd.DataFrame,
    sector_weights: dict[str, float],
    working: pd.DataFrame,
) -> pd.Series:
    """Combine per-sector scores into one 0-100 sector_score per country-year."""
    keys = working[["country_iso3", "year"]].copy()

    if sector_scores_df.empty or not sector_weights:
        keys["sector_score"] = np.nan
        return keys["sector_score"]

    ss = sector_scores_df.dropna(subset=["sector_score"])[["country_iso3", "year", "sector", "sector_score"]].copy()
    if ss.empty:
        keys["sector_score"] = np.nan
        return keys["sector_score"]

    ss["_w"] = ss["sector"].map(sector_weights).fillna(0.0)
    agg = (
        ss.assign(_contrib=ss["sector_score"] * ss["_w"])
        .groupby(["country_iso3", "year"], as_index=False)
        .agg(contribution=("_contrib", "sum"), weight=("_w", "sum"))
    )
    agg["sector_score"] = agg["contribution"] / agg["weight"].replace(0, np.nan)

    merged = keys.merge(agg[["country_iso3", "year", "sector_score"]], on=["country_iso3", "year"], how="left")
    return merged["sector_score"]


def _add_trend_deltas(scores: pd.DataFrame, scoring: dict[str, float]) -> pd.DataFrame:
    """Trend deltas on the finest cadence the panel supports (annual here)."""
    out = scores.copy().sort_values(["country_iso3", "year"])

    for lag in (1, 2):
        col = f"trend_{lag}y_delta"
        out[col] = out["risk_score"] - out.groupby("country_iso3")["risk_score"].shift(lag)

    thr = float(scoring["trend_threshold_points"])
    d1 = out["trend_1y_delta"]
    direction = np.select(
        [d1.isna().to_numpy(), np.abs(d1.to_numpy(dtype=float)) <= thr, d1.to_numpy(dtype=float) > thr],
        ["", "Stable", "Deteriorating"],
        default="Improving",
    )
    out["trend_direction"] = pd.Series(direction, index=out.index).replace("", pd.NA)

    return out


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
