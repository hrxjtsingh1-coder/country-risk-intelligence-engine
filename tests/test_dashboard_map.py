import pandas as pd

from dashboard import ui
from dashboard.context import Context
from dashboard.sections import map as map_section
from src.scoring.risk_score import score_panel


def _ctx(**overrides):
    panel = pd.DataFrame(
        [
            {
                "country_iso3": c,
                "year": y,
                "FP.CPI.TOTL.ZG": i + y - 2020,
                "NY.GDP.MKTP.KD.ZG": 10 - i,
                "SL.UEM.TOTL.ZS": i + 1,
                "OUTPUT_GAP_PROXY_PCT": i - 3,
                "GC.DOD.TOTL.GD.ZS": 40 + i,
                "GC.NLD.TOTL.GD.ZS": -2 - i,
                "PUBLIC_DEBT_TRAJECTORY_PCT": i * 0.5,
                "BN.CAB.XOKA.GD.ZS": -i,
                "FI.RES.TOTL.MO": 8 - i / 5,
                "DT.DOD.DECT.GN.ZS": 20 + i,
                "NE.RSB.GNFS.ZS": 2 - i,
                "FB.AST.NPER.ZS": 1 + i / 10,
                "FX_YOY_DEPRECIATION_PCT": i,
                "POLICY_RATE_YOY_CHANGE_BPS": i * 10,
            }
            for y in range(2020, 2026)
            for i, c in enumerate(["USA", "CAN", "DEU", "BRA", "ZAF"], 1)
        ]
    )
    scores, drivers, pillar = score_panel(panel)
    kwargs = dict(
        panel=panel,
        scores=scores,
        drivers=drivers,
        pillar_scores=pillar,
        indicators_cfg={},
        country="USA",
        year=2025,
        iso="USA",
        country_label="United States",
        band="Moderate",
        score_value=50.0,
        score_color="#ffffff",
        coverage_value=1.0,
        current_row=pd.Series(dtype=object),
        country_drivers=pd.DataFrame(),
        report="",
        scenario=None,
        scenario_result=None,
        scenario_error=None,
        shock=0,
        run_scenario_btn=False,
        using_demo_data=True,
        live_provenance=None,
        generated_at="2026-09-10 12:00:00",
    )
    kwargs.update(overrides)
    return Context(**kwargs)


def test_map_builds_choropleth_with_year_frames():
    ctx = _ctx()
    ui.set_countries_cfg({"countries": []})
    fig = map_section._build_figure(ctx)
    assert fig.data and fig.data[0].type == "choropleth"
    assert len(fig.frames) == len(ctx.scores["year"].unique())
    first_frame = fig.frames[0].data
    assert all(t["type"] in ("choropleth", "scattergeo") for t in first_frame)


def test_map_five_band_colorscale_and_drill_customdata():
    ctx = _ctx()
    ui.set_countries_cfg({"countries": []})
    fig = map_section._build_figure(ctx)
    colorscale = fig.data[0]["colorscale"]
    assert len(colorscale) == 10  # five bands, two stops each
    hover = fig.data[0]["hovertemplate"]
    assert "Score" in hover and "Band" in hover

    custom = fig.data[0]["customdata"]
    assert custom and all(len(item) == 4 for item in custom)
    assert any(item[3] == "USA" for item in custom)


def test_map_glow_marks_high_risk_countries():
    ctx = _ctx()
    ui.set_countries_cfg({"countries": []})
    fig = map_section._build_figure(ctx)
    geo_traces = [t for t in fig.data if t.type == "scattergeo"]
    assert geo_traces, "expected a glow overlay trace"
