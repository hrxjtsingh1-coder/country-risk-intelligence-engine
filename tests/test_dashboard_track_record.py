"""Smoke tests for the Track Record page.

The Track Record page ("Model Validation & Track Record") is wired into the
sidebar navigation and replays every configured historical stress episode
against the engine's own composite scores — reporting an honest hit-rate and
a false-alarm base rate, not a claim.
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


def test_track_record_is_reachable_from_sidebar_nav(monkeypatch):
    app = _switched_app(monkeypatch)

    assert any("Track Record & Model Validation" in str(o) for r in app.radio for o in r.options)
    app.radio[0].set_value("Track Record & Model Validation").run(timeout=120)
    assert not app.exception

    body = "\n".join(str(m.value) for m in app.markdown if m.value)
    assert "Track record" in body
    assert "Would this model have caught it?" in body


def test_track_record_shows_hit_rate_and_false_alarm_rate(monkeypatch):
    app = _switched_app(monkeypatch)
    app.radio[0].set_value("Track Record & Model Validation").run(timeout=120)
    body = "\n".join(str(m.value) for m in app.markdown if m.value)

    # The honest companion numbers are both present, never the hit rate alone.
    assert "DETECTION RATE" in body
    assert "FALSE-ALARM RATE" in body
    assert "EPISODES" in body
    assert "FLAGGED" in body and "MISSED" in body


def test_track_record_shows_quantitative_metrics(monkeypatch):
    app = _switched_app(monkeypatch)
    app.radio[0].set_value("Track Record & Model Validation").run(timeout=120)
    body = "\n".join(str(m.value) for m in app.markdown if m.value)

    # New confusion-matrix + lead-time KPIs are rendered, not just the hit rate.
    assert "MEDIAN LEAD TIME" in body
    assert "PRECISION" in body
    assert "RECALL" in body
    assert "FALSE-POSITIVE RATE" in body
    assert "FALSE-NEGATIVE RATE" in body
    assert "WARNING FREQUENCY" in body


def test_track_record_page_shows_the_relevant_episode_verdicts(monkeypatch):
    app = _switched_app(monkeypatch)
    app.radio[0].set_value("Track Record & Model Validation").run(timeout=120)
    body = "\n".join(str(m.value) for m in app.markdown if m.value)

    # Every configured episode is replayed; demo data is labeled honestly.
    assert "Turkiye — 2018 currency crisis" in body
    assert "Turkiye — 2021 lira crash" in body
    assert "Brazil — 2015-16 recession" in body
    assert "United States — 2008 global financial crisis" in body
    errors = "\n".join(str(e.value) for e in app.error)
    assert "SYNTHETIC" in errors
