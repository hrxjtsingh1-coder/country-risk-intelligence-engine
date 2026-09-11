"""Smoke tests for the peer-percentile ranking.

Each country's score display now shows its percentile standing among peers
("ranks #N for relative risk, riskier than ~X% of peers"), derived from the
peer-group configuration as a display ranking over already-computed scores.
"""

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


def test_peer_position_kpi_renders(monkeypatch):
    app = _overview_app(monkeypatch)
    body = "\n".join(str(m.value) for m in app.markdown if m.value)

    # The KPI surfaces the peer group size and the rank semantics.
    assert "Peer position" in body
    assert "peers (1st = riskiest)" in body


def test_interpretation_sentence_names_the_peer_set(monkeypatch):
    app = _overview_app(monkeypatch)

    if app.selectbox:
        app.selectbox[0].select("BRA")  # BRA is in the emerging peer group
        app.run(timeout=120)
        assert not app.exception

    body = "\n".join(str(m.value) for m in app.markdown if m.value)
    assert "ranks" in body
    assert "riskier than ~" in body
