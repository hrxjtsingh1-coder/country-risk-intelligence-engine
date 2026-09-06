import pandas as pd
from src.runtime import live_data


def test_world_bank_parser_preserves_iso3_and_series():
    class Response:
        def raise_for_status(self): pass
        def json(self):
            return [{"total": 1}, [{"countryiso3code":"IND","country":{"value":"India"},"date":"2024","value":7.2}]]
    class Session:
        def get(self, *args, **kwargs): return Response()
    out = live_data._wb_indicator(Session(), "NY.GDP.MKTP.KD.ZG", 2024, 2024)
    assert out.iloc[0]["country_iso3"] == "IND"
    assert out.iloc[0]["indicator_code"] == "NY.GDP.MKTP.KD.ZG"


def test_fx_change_is_calculated_within_country():
    raw = pd.DataFrame([
        {"country_iso3":"USA","country_name":"US","year":2022,"value":1.0},
        {"country_iso3":"USA","country_name":"US","year":2023,"value":1.1},
        {"country_iso3":"IND","country_name":"India","year":2022,"value":80.0},
        {"country_iso3":"IND","country_name":"India","year":2023,"value":88.0},
    ])
    out = live_data._fx_from_raw(raw, 2023, 2023)
    vals = dict(zip(out.country_iso3, out.value))
    assert round(vals["USA"], 6) == 10.0
    assert round(vals["IND"], 6) == 10.0


def test_clean_long_flags_bounds():
    df = pd.DataFrame([{"country_iso3":"USA","country_name":"US","indicator_code":"FP.CPI.TOTL.ZG","year":2024,"value":200,"source":"WB"}])
    out = live_data._clean_long(df)
    assert out.iloc[0]["flag"] == "out_of_range"


def test_wide_panel_excludes_invalid_observation():
    df = pd.DataFrame([
        {"country_iso3":"USA","country_name":"US","indicator_code":"FP.CPI.TOTL.ZG","year":2024,"value":2,"source":"WB","flag":"ok"},
        {"country_iso3":"USA","country_name":"US","indicator_code":"NY.GDP.MKTP.KD.ZG","year":2024,"value":3,"source":"WB","flag":"out_of_range"},
    ])
    wide = live_data.create_wide_panel_from_long(df)
    assert float(wide.loc[0, "FP.CPI.TOTL.ZG"]) == 2
    assert "NY.GDP.MKTP.KD.ZG" not in wide.columns


def test_latest_valid_analysis_year_prefers_latest_well_covered_year():
    codes = [x["code"] for x in live_data.indicator_records() if float(x.get("weight", 0)) > 0]
    rows = [{"country_iso3":f"X{i:02d}","indicator_code":c,"year":y,"value":i+1,"flag":"ok"}
            for y in (2024,2025) for i in range(35) for c in codes]
    assert live_data.latest_valid_analysis_year(pd.DataFrame(rows), codes, .55, 30) == 2025


def test_quality_reports_missing_values():
    df = pd.DataFrame([
        {"country_iso3":"USA","indicator_code":"FP.CPI.TOTL.ZG","year":2024,"value":2,"flag":"ok"},
        {"country_iso3":"IND","indicator_code":"FP.CPI.TOTL.ZG","year":2024,"value":None,"flag":"missing"},
    ])
    q = live_data.data_quality(df, 2024)
    assert q["countries"] == 2
    assert q["missing"] == 1
    assert q["coverage"] == .5
