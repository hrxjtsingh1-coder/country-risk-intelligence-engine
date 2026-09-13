"""Shared test fixtures for the Country Risk Intelligence Engine test suite."""

from __future__ import annotations

import pandas as pd
import pytest


@pytest.fixture()
def sample_long_panel() -> pd.DataFrame:
    """Minimal long-format panel with 5 countries, 6 years, 10 indicators."""
    rows = []
    countries = ["USA", "CAN", "DEU", "IND", "BRA"]
    indicators = [
        ("FP.CPI.TOTL.ZG", 3, 1),
        ("NY.GDP.MKTP.KD.ZG", 5, -1),
        ("SL.UEM.TOTL.ZS", 6, 1),
        ("GC.DOD.TOTL.GD.ZS", 60, 1),
        ("BN.CAB.XOKA.GD.ZS", -2, -1),
        ("FI.RES.TOTL.MO", 8, -1),
        ("DT.DOD.DECT.GN.ZS", 30, 1),
        ("FB.AST.NPER.ZS", 3, 1),
        ("FX_YOY_DEPRECIATION_PCT", 2, 1),
        ("POLICY_RATE_YOY_CHANGE_BPS", 0, 1),
    ]
    for year in range(2020, 2026):
        for i, c in enumerate(countries):
            for code, base, _sign in indicators:
                rows.append(
                    {
                        "country_iso3": c,
                        "indicator_code": code,
                        "year": year,
                        "value": base + i + (year - 2020) * _sign,
                        "source": "World Bank",
                        "flag": "ok",
                    }
                )
    return pd.DataFrame(rows)


@pytest.fixture()
def sample_wide_panel(sample_long_panel: pd.DataFrame) -> pd.DataFrame:
    """Wide-format panel pivoted from sample_long_panel."""
    from src.cleaning.clean import to_wide_panel

    return to_wide_panel(sample_long_panel)


@pytest.fixture()
def indicators_cfg() -> dict:
    """Minimal indicator configuration matching sample_long_panel indicators."""
    return {
        "indicators": [
            {"code": "FP.CPI.TOTL.ZG", "weight": 0.16, "risk_direction": 1, "category": "Prices"},
            {"code": "NY.GDP.MKTP.KD.ZG", "weight": 0.16, "risk_direction": -1, "category": "Growth"},
            {"code": "SL.UEM.TOTL.ZS", "weight": 0.10, "risk_direction": 1, "category": "Labor"},
            {"code": "GC.DOD.TOTL.GD.ZS", "weight": 0.18, "risk_direction": 1, "category": "Fiscal"},
            {"code": "BN.CAB.XOKA.GD.ZS", "weight": 0.09, "risk_direction": -1, "category": "External"},
            {"code": "FI.RES.TOTL.MO", "weight": 0.10, "risk_direction": -1, "category": "External"},
            {"code": "DT.DOD.DECT.GN.ZS", "weight": 0.09, "risk_direction": 1, "category": "External"},
            {"code": "FB.AST.NPER.ZS", "weight": 0.06, "risk_direction": 1, "category": "Banking"},
            {"code": "FX_YOY_DEPRECIATION_PCT", "weight": 0.06, "risk_direction": 1, "category": "FX"},
        ]
    }
