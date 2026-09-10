"""
Tests for src/runtime/live_data.py.

Per the project's own testing rule: never depend on the real World Bank API
for unit tests — everything here mocks requests or the fetch boundary
directly. tests/fixtures/worldbank_success.json provides a realistic
response shape for the HTTP-parsing test.
"""

import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import pandas as pd
import pytest

from src.indicators.build_panel import FetchMetadata, _world_bank_indicator_batch
from src.runtime.live_data import (
    LiveDataUnavailable,
    fetch_live_panel,
    select_latest_common_year,
)

FIXTURE_PATH = Path(__file__).resolve().parent / "fixtures" / "worldbank_success.json"


def _load_fixture_json() -> list:
    # The fixture file has a leading `#`-comment line before the JSON array.
    lines = FIXTURE_PATH.read_text(encoding="utf-8").splitlines()
    json_text = "\n".join(line for line in lines if not line.lstrip().startswith("#"))
    return json.loads(json_text)


def _synthetic_long_panel(thin_latest_year: bool = True) -> pd.DataFrame:
    rows = []
    countries = ["USA", "IND", "DEU", "BRA", "ZAF", "JPN"]
    for year in range(2020, 2026):
        for i, c in enumerate(countries):
            if thin_latest_year and year == 2025 and i >= 2:
                continue  # simulate a newest year that only 2/6 countries have reported yet
            for code, base in [
                ("FP.CPI.TOTL.ZG", 3),
                ("NY.GDP.MKTP.KD.ZG", 2),
                ("GC.DOD.TOTL.GD.ZS", 50),
            ]:
                rows.append(
                    {
                        "country_iso3": c,
                        "indicator_code": code,
                        "year": year,
                        "value": base + i,
                        "source": "World Bank",
                        "flag": "ok",
                    }
                )
    return pd.DataFrame(rows)


INDICATORS_CFG = {
    "indicators": [
        {"code": "FP.CPI.TOTL.ZG", "weight": 0.4},
        {"code": "NY.GDP.MKTP.KD.ZG", "weight": 0.3},
        {"code": "GC.DOD.TOTL.GD.ZS", "weight": 0.3},
    ]
}
COUNTRIES = ["USA", "IND", "DEU", "BRA", "ZAF", "JPN"]


def test_select_latest_common_year_skips_thin_newest_year():
    panel = _synthetic_long_panel(thin_latest_year=True)
    weighted = [("FP.CPI.TOTL.ZG", 0.4), ("NY.GDP.MKTP.KD.ZG", 0.3), ("GC.DOD.TOTL.GD.ZS", 0.3)]
    year = select_latest_common_year(panel, weighted, min_coverage=0.8, min_countries=5)
    assert year == 2024  # 2025 only has 2/6 countries and must be skipped


def test_select_latest_common_year_uses_newest_when_fully_covered():
    panel = _synthetic_long_panel(thin_latest_year=False)
    weighted = [("FP.CPI.TOTL.ZG", 0.4), ("NY.GDP.MKTP.KD.ZG", 0.3), ("GC.DOD.TOTL.GD.ZS", 0.3)]
    year = select_latest_common_year(panel, weighted, min_coverage=0.8, min_countries=5)
    assert year == 2025


def test_fetch_live_panel_success_path():
    panel = _synthetic_long_panel(thin_latest_year=True)
    meta = FetchMetadata()
    with patch("src.runtime.live_data.build_long_panel_batched", return_value=(panel, meta)):
        result = fetch_live_panel(COUNTRIES, INDICATORS_CFG, 2020, 2025)

    assert result.latest_common_year == 2024
    assert result.world_bank_ok is True
    assert result.fred_ok is False  # not in the synthetic fixture -> honestly reported
    assert result.provenance.coverage_pct == 100.0
    assert not result.wide_panel.empty


def test_fetch_live_panel_raises_cleanly_on_total_failure():
    with patch("src.runtime.live_data.build_long_panel_batched", side_effect=ConnectionError("blocked")):
        with pytest.raises(LiveDataUnavailable) as excinfo:
            fetch_live_panel(COUNTRIES, INDICATORS_CFG, 2020, 2025)
    # Message must be end-user-safe: no raw exception text as the primary message.
    assert "could not be retrieved" in str(excinfo.value)
    assert "ConnectionError" in excinfo.value.technical_detail


def test_fetch_live_panel_raises_on_empty_response():
    meta = FetchMetadata()
    with patch("src.runtime.live_data.build_long_panel_batched", return_value=(pd.DataFrame(), meta)):
        with pytest.raises(LiveDataUnavailable):
            fetch_live_panel(COUNTRIES, INDICATORS_CFG, 2020, 2025)


def test_world_bank_batch_parses_realistic_fixture_shape():
    """Exercises the actual HTTP-response-parsing code against the shape the
    real World Bank API returns (tests/fixtures/worldbank_success.json),
    with requests.Session.get mocked — no network involved."""
    payload = _load_fixture_json()
    mock_response = MagicMock()
    mock_response.json.return_value = payload
    mock_response.raise_for_status.return_value = None

    mock_session = MagicMock()
    mock_session.get.return_value = mock_response

    df = _world_bank_indicator_batch(mock_session, ["IND", "USA"], "NY.GDP.MKTP.KD.ZG", 2022, 2022)

    assert set(df["country_iso3"]) == {"IND", "USA"}
    assert df.loc[df.country_iso3 == "IND", "value"].iloc[0] == pytest.approx(6.8)
    assert df.loc[df.country_iso3 == "USA", "value"].iloc[0] == pytest.approx(2.1)
    # One request for both countries, not one per country.
    assert mock_session.get.call_count == 1
