"""Smoke test for the agency benchmark section on the Overview page."""

from pathlib import Path


def _overview_app(monkeypatch):
    from streamlit.testing.v1 import AppTest

    monkeypatch.setenv("COUNTRY_RISK_OFFLINE", "1")
    app = AppTest.from_file(Path(__file__).resolve().parents[1] / "dashboard" / "app.py", default_timeout=120)
    app.run(timeout=120)
    assert not app.exception

    if app.button:
        app.button[0].click()
        app.run(timeout=120)
    assert not app.exception
    return app


def test_agency_benchmark_section_renders(monkeypatch):
    app = _overview_app(monkeypatch)
    body = "\n".join(str(m.value) for m in app.markdown if m.value)
    assert "BENCHMARK VS AGENCY RATINGS" in body
    assert "Reference-only" in body
    assert "COUNTRIES MATCHED" in body
    assert "RANK AGREEMENT" in body


def test_agency_benchmark_table_renders(monkeypatch):
    app = _overview_app(monkeypatch)
    assert not app.exception
    dataframes = [d for d in app.dataframe if getattr(d, "value", None) is not None]
    assert dataframes, "expected at least one rendered dataframe"
