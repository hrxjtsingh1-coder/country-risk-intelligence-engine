"""
Offline tests for the commodity-index and BIS credit-gap collectors.

These exercise the parsing/derivation logic with canned payloads and mocked
cache boundaries — no network involved, matching the project's testing rule.
"""

import pandas as pd
import pytest

from src.indicators import build_panel as bp
from src.indicators.build_panel import (
    _bis_credit_gap,
    _commodity_price_annual_yoy,
    _commodity_price_index,
    _parse_bis_credit_gap_payload,
)


def test_commodity_price_annual_yoy():
    levels = pd.DataFrame({"year": [2004, 2005, 2006, 2007], "value": [100.0, 110.0, 121.0, 133.1]})
    yoy = _commodity_price_annual_yoy(levels, start=2005, end=2006)
    assert yoy["year"].tolist() == [2005, 2006]
    assert yoy["value"].tolist() == pytest.approx([10.0, 10.0])
    # 2005 uses the 2004 level as its base; it must not be dropped.
    assert yoy.loc[yoy.year == 2005, "value"].iloc[0] == pytest.approx(10.0)


def test_commodity_price_annual_yoy_empty_on_garbage():
    assert _commodity_price_annual_yoy(pd.DataFrame(), 2005, 2010).empty
    assert _commodity_price_annual_yoy(None, 2005, 2010).empty


def test_commodity_price_index_fans_out_to_all_countries(monkeypatch):
    canned = pd.DataFrame({"year": [2004, 2005, 2006, 2007], "value": [100.0, 110.0, 121.0, 133.1]})
    meta = bp.FetchMetadata()
    monkeypatch.setattr(bp, "_cached_fred_fetch", lambda *a, **k: canned)

    frame = _commodity_price_index(None, ["USA", "DEU"], 2005, 2006, meta, series_id="PALLFNFINDEXM")

    assert not frame.empty
    assert set(frame["country_iso3"]) == {"USA", "DEU"}
    assert frame["indicator_code"].eq("COMMODITY_PRICE_INDEX_PCT").all()
    assert frame["value"].tolist() == pytest.approx([10.0] * 4)
    assert frame["flag"].eq("ok").all()
    assert frame["source"].iloc[0].startswith("FRED PALLFNFINDEXM")


def _bis_payload(periods, values):
    return {
        "meta": {"id": "x"},
        "data": {
            "dataSets": [
                {"series": {"0:0:0:0:0": {"observations": {str(i): [v, 0, 0, None] for i, v in enumerate(values)}}}}
            ],
            "structure": {
                "dimensions": {
                    "series": [
                        {"id": "FREQ", "values": [{"id": "Q"}]},
                        {"id": "BORROWERS_CTY", "values": [{"id": "US"}]},
                        {"id": "TC_BORROWERS", "values": [{"id": "P"}]},
                        {"id": "TC_LENDERS", "values": [{"id": "A"}]},
                        {"id": "CG_DTYPE", "values": [{"id": "C"}]},
                    ],
                    "observation": [{"id": "TIME_PERIOD", "values": [{"id": p} for p in periods]}],
                }
            },
        },
    }


def test_parse_bis_credit_gap_payload_annualizes_to_q4():
    payload = _bis_payload(
        ["2005-Q1", "2005-Q2", "2005-Q3", "2005-Q4", "2006-Q1", "2006-Q2"],
        [7.1409, 7.6848, 7.8697, 8.1691, 8.7706, 9.5533],
    )
    frame = _parse_bis_credit_gap_payload(payload)

    assert frame["indicator_code"].eq("BIS_CREDIT_GAP").all()
    assert frame["flag"].eq("ok").all()
    assert frame.loc[frame.year == 2005, "value"].iloc[0] == pytest.approx(8.1691)
    assert frame.loc[frame.year == 2006, "value"].iloc[0] == pytest.approx(9.5533)
    assert "WS_CREDIT_GAP" in frame["source"].iloc[0]
    assert frame["year"].tolist() == [2005, 2006]


def test_parse_bis_credit_gap_payload_empty_on_garbage():
    assert _parse_bis_credit_gap_payload(None).empty
    assert _parse_bis_credit_gap_payload({}).empty
    assert _parse_bis_credit_gap_payload({"data": {"dataSets": []}}).empty
    assert _parse_bis_credit_gap_payload("not-a-dict").empty


def test_bis_credit_gap_stamps_each_country(monkeypatch):
    canned = pd.DataFrame(
        [
            {"country_iso3": "USA", "year": 2005, "value": 8.1, "source": "BIS", "flag": "ok"},
            {"country_iso3": "USA", "year": 2006, "value": 9.5, "source": "BIS", "flag": "ok"},
            {"country_iso3": "DEU", "year": 2005, "value": -4.3, "source": "BIS", "flag": "ok"},
        ]
    )
    canned["indicator_code"] = "BIS_CREDIT_GAP"

    def fake_fetch(session, url, params, meta, ttl):
        iso2 = str(url).split("/Q.")[1][:2]
        subset = canned[canned["country_iso3"].eq({"US": "USA", "DE": "DEU"}[iso2])]
        return subset

    monkeypatch.setattr(bp, "_cached_bis_fetch", fake_fetch)

    frame = _bis_credit_gap(None, ["USA", "DEU"], 2005, 2006, bp.FetchMetadata())

    assert set(frame["country_iso3"]) == {"USA", "DEU"}
    assert frame["indicator_code"].eq("BIS_CREDIT_GAP").all()
    assert frame.shape[0] == 3


def test_batched_builder_wires_both_new_sources(monkeypatch):
    """The batched builder must route the two new indicator codes to their
    collectors even when every other source returns nothing."""
    bis = pd.DataFrame(
        [
            {
                "country_iso3": "USA",
                "year": 2005,
                "value": 8.1,
                "source": "BIS",
                "flag": "ok",
                "indicator_code": "BIS_CREDIT_GAP",
            }
        ]
    )
    comm = pd.DataFrame(
        [
            {
                "country_iso3": "USA",
                "year": 2005,
                "value": 10.0,
                "source": "FRED",
                "flag": "ok",
                "indicator_code": "COMMODITY_PRICE_INDEX_PCT",
            }
        ]
    )
    monkeypatch.setattr(bp, "_cached_world_bank_batch", lambda *a, **k: pd.DataFrame())
    monkeypatch.setattr(bp, "_world_bank_indicator", lambda *a, **k: pd.DataFrame())
    monkeypatch.setattr(bp, "_fred_annual_mean", lambda *a, **k: pd.DataFrame())
    monkeypatch.setattr(bp, "_cached_fred_fetch", lambda *a, **k: pd.DataFrame())
    monkeypatch.setattr(bp, "fetch_imf_indicator", lambda *a, **k: pd.DataFrame())
    monkeypatch.setattr(bp, "_commodity_price_index", lambda *a, **k: comm)
    monkeypatch.setattr(bp, "_bis_credit_gap", lambda *a, **k: bis)

    panel, meta = bp.build_long_panel_batched(["USA"], 2005, 2005)
    codes = set(panel["indicator_code"])
    assert {"BIS_CREDIT_GAP", "COMMODITY_PRICE_INDEX_PCT"} <= codes
    assert panel.loc[panel.indicator_code == "BIS_CREDIT_GAP", "value"].iloc[0] == pytest.approx(8.1)
    assert panel.loc[panel.indicator_code == "COMMODITY_PRICE_INDEX_PCT", "value"].iloc[0] == pytest.approx(10.0)
