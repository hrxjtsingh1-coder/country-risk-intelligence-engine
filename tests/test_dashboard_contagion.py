"""Smoke tests for the Contagion & Correlations page.

The page is a source-level network of the engine's own scored history: YoY
risk-score deltas correlated pairwise between countries, clusters discovered
from the thresholded co-movement graph. Descriptive only — never presented as
a causal contagion model.
"""

from pathlib import Path


def _switched_app(monkeypatch):
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


def test_contagion_page_is_reachable_from_sidebar_nav(monkeypatch):
    app = _switched_app(monkeypatch)

    assert any("Contagion & Correlations" in str(o) for r in app.radio for o in r.options)
    app.radio[0].set_value("Contagion & Correlations").run(timeout=120)
    assert not app.exception

    body = "\n".join(str(m.value) for m in app.markdown if m.value)
    assert "Contagion &amp; correlation network" in body
    assert "section-title" in body


def test_contagion_page_renders_network_kpis(monkeypatch):
    app = _switched_app(monkeypatch)
    app.radio[0].set_value("Contagion & Correlations").run(timeout=120)
    body = "\n".join(str(m.value) for m in app.markdown if m.value)

    assert "COUNTRIES" in body
    assert "EDGES" in body
    assert "MEAN |R|" in body


def test_contagion_page_labels_demo_data_and_caveat(monkeypatch):
    app = _switched_app(monkeypatch)
    app.radio[0].set_value("Contagion & Correlations").run(timeout=120)

    body = "\n".join(str(m.value) for m in app.markdown if m.value)
    assert "DESCRIPTIVE CO-MOVEMENT / NOT CAUSAL" in body
    errors = "\n".join(str(e.value) for e in app.error)
    assert "SYNTHETIC" in errors
