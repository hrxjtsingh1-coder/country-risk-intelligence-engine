"""Single-year data contract used across the engine.

The public engine intentionally exposes only the current calendar year. Older
observations are not part of the production data contract and must not leak
through loaders, APIs, or the dashboard.
"""
from __future__ import annotations

from datetime import date
from typing import Any

CURRENT_YEAR = date.today().year


def validate_current_year(year: int | str, *, name: str = "year") -> int:
    """Return ``year`` as an int or raise when it is not the current year."""
    try:
        value = int(year)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be an integer year") from exc
    if value != CURRENT_YEAR:
        raise ValueError(f"Only current year {CURRENT_YEAR} is supported; received {value}.")
    return value


def validate_current_year_range(start_year: int | str, end_year: int | str) -> tuple[int, int]:
    """Validate a supported single-year request."""
    start = validate_current_year(start_year, name="start_year")
    end = validate_current_year(end_year, name="end_year")
    if start != end:
        raise ValueError(f"Only a single current-year slice is supported: {CURRENT_YEAR}.")
    return start, end


def filter_current_year(df: Any, year_column: str = "year"):
    """Keep only current-year observations without mutating the input object."""
    if df is None or getattr(df, "empty", False):
        return df
    if year_column not in getattr(df, "columns", []):
        return df.iloc[0:0].copy()
    out = df.copy()
    years = __import__("pandas").to_numeric(out[year_column], errors="coerce")
    return out.loc[years.eq(CURRENT_YEAR)].copy()
