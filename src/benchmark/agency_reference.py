"""Sovereign-rating reference benchmark.

Turns the static `config/agency_ratings.yaml` snapshot into a comparable
0-100 "reference risk" scale (AAA/Aaa = 0, default = 100) and measures how the
engine's ordering agrees with real agency views via rank correlation and band
agreement. Reference-only: used to sanity-check direction, never as an outcome
or an optimization target for scoring.
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Any

import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[2]
REFERENCE_PATH = ROOT / "config" / "agency_ratings.yaml"

# Long-term foreign-currency issuer scale, ordinal 1 (safest) .. 21 (worst).
_SP_FITCH = {
    "AAA": 1,
    "AA+": 2,
    "AA": 3,
    "AA-": 4,
    "A+": 5,
    "A": 6,
    "A-": 7,
    "BBB+": 8,
    "BBB": 9,
    "BBB-": 10,
    "BB+": 11,
    "BB": 12,
    "BB-": 13,
    "B+": 14,
    "B": 15,
    "B-": 16,
    "CCC+": 17,
    "CCC": 18,
    "CCC-": 19,
    "CC": 20,
    "C": 21,
    "D": 21,
}
_MOODYS = {
    "Aaa": 1,
    "Aa1": 2,
    "Aa2": 3,
    "Aa3": 4,
    "A1": 5,
    "A2": 6,
    "A3": 7,
    "Baa1": 8,
    "Baa2": 9,
    "Baa3": 10,
    "Ba1": 11,
    "Ba2": 12,
    "Ba3": 13,
    "B1": 14,
    "B2": 15,
    "B3": 16,
    "Caa1": 17,
    "Caa2": 18,
    "Caa3": 19,
    "Ca": 20,
    "C": 21,
}


def _ordinal(agency: str, letter: Any) -> float | None:
    if letter is None:
        return None
    key = str(letter).strip()
    if not key:
        return None
    if agency == "Moody's":
        value = _MOODYS.get(key)
    else:
        value = _SP_FITCH.get(key)
    return float(value) if value is not None else None


@lru_cache(maxsize=1)
def load_agency_reference() -> dict[str, Any]:
    if not REFERENCE_PATH.exists():
        return {"asof": "", "entries": []}
    raw = yaml.safe_load(REFERENCE_PATH.read_text(encoding="utf-8"))
    return raw if isinstance(raw, dict) else {"asof": "", "entries": []}


def reference_rows(reference: dict[str, Any] | None = None) -> list[dict]:
    """Flatten entries to iso3-level rows with ordinal and 0-100 reference risk."""
    ref = reference if reference is not None else load_agency_reference()
    rows: list[dict] = []
    for entry in ref.get("entries", []) or []:
        if not isinstance(entry, dict) or not entry.get("iso3"):
            continue
        ordinals = [
            o
            for o in (
                _ordinal("S&P", entry.get("S&P")),
                _ordinal("Moody's", entry.get("Moody's")),
                _ordinal("Fitch", entry.get("Fitch")),
            )
            if o is not None
        ]
        if not ordinals:
            continue
        avg_ordinal = sum(ordinals) / len(ordinals)
        reference_risk = ((avg_ordinal - 1.0) / 20.0) * 100.0
        rows.append(
            {
                "iso3": str(entry["iso3"]).upper(),
                "country": str(entry.get("country", "") or entry["iso3"]),
                "S&P": str(entry.get("S&P", "") or ""),
                "Moody's": str(entry.get("Moody's", "") or ""),
                "Fitch": str(entry.get("Fitch", "") or ""),
                "ordinal": round(avg_ordinal, 3),
                "reference_risk": round(reference_risk, 1),
            }
        )
    return rows


def _spearman(a: pd.Series, b: pd.Series) -> float | None:
    joint = pd.DataFrame({"a": a, "b": b}).dropna()
    if len(joint) < 2:
        return None
    if joint["a"].nunique() < 2 or joint["b"].nunique() < 2:
        return None
    rank_a = joint["a"].rank()
    rank_b = joint["b"].rank()
    return float(rank_a.corr(rank_b))


def benchmark_against_scores(scores: pd.DataFrame, reference: dict[str, Any] | None = None) -> dict[str, Any]:
    """Compare the engine's latest per-country score with reference ratings."""
    ref_rows = reference_rows(reference)
    by_iso = {r["iso3"]: r for r in ref_rows}

    latest: dict[str, tuple[float, str]] = {}
    if isinstance(scores, pd.DataFrame) and not scores.empty and "risk_score" in scores.columns:
        working = scores.dropna(subset=["risk_score"]).copy()
        working["year"] = pd.to_numeric(working["year"], errors="coerce")
        for iso3, group in working.groupby(working["country_iso3"].astype(str).str.upper()):
            group = group.dropna(subset=["year"])
            if group.empty:
                continue
            top = group.sort_values("year", ascending=False).iloc[0]
            latest[str(iso3).upper()] = (float(top["risk_score"]), str(top.get("risk_band", "") or ""))

    rows: list[dict] = []
    engine_scores: list[float] = []
    ref_risks: list[float] = []
    band_matches = 0
    band_compared = 0
    for iso3, r in by_iso.items():
        engine = latest.get(iso3)
        if engine is None:
            continue
        row = dict(r)
        row["engine_score"] = round(engine[0], 1)
        row["engine_band"] = engine[1]
        row["bands_match"] = None
        if engine[1]:
            row["bands_match"] = engine[1] == band_for_reference_risk(r["reference_risk"])
            band_compared += 1
            band_matches += 1 if row["bands_match"] else 0
        rows.append(row)
        engine_scores.append(float(engine[0]))
        ref_risks.append(float(r["reference_risk"]))

    rows.sort(key=lambda row: row["reference_risk"], reverse=True)

    spearman = _spearman(pd.Series(engine_scores), pd.Series(ref_risks))
    band_share = (band_matches / band_compared) if band_compared else None
    return {
        "asof": str((reference if reference is not None else load_agency_reference()).get("asof", "") or ""),
        "n": len(rows),
        "spearman": round(spearman, 3) if spearman is not None else None,
        "band_match_share": round(band_share, 3) if band_share is not None else None,
        "rows": rows,
    }


def band_for_reference_risk(reference_risk: float) -> str:
    if reference_risk < 20:
        return "Low"
    if reference_risk < 40:
        return "Moderate"
    if reference_risk < 60:
        return "Elevated"
    if reference_risk < 80:
        return "High"
    return "Severe"
