"""
Historical backtest harness ("instant replay").

Does the scoring engine actually flag known real macro-stress episodes, using
the country's OWN scored history — not just "does the UI look right".

This never re-fetches data: it evaluates whatever scores/panel the caller
passes (the live-fetched or demo panel already in memory), so it costs nothing
extra in production and is transparent about running on demo/synthetic data
when that's what's loaded (see `data_is_synthetic`).

A full run produces three layers, all derived from the same numbers:

1. Per-episode replay — baseline vs event score, the peak inside a detection
   window after the event year, the verdict, the dominant pillar/sector that
   actually moved, and a peer-drift diagnostic (what fraction of OTHER panel
   countries also rose the same way — a systemic rise is low-information).
2. A summary with detection rate and the median peak delta.
3. A CLI (`python -m src.analysis.backtest`) that scores a panel CSV and
   prints/JSON-dumps the whole thing.

Episodes live in config/episodes.yaml so new cases are data, not code.
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import pandas as pd
import yaml

from src.scoring.risk_score import score_panel

ROOT = Path(__file__).resolve().parents[2]
EPISODES_PATH = ROOT / "config" / "episodes.yaml"

PASS_THRESHOLD_POINTS = 3.0  # a rise smaller than this is noise, not a signal
DEFAULT_WINDOW_YEARS = 1

# Built-in fallback episodes, used only if config/episodes.yaml is missing.
DEFAULT_EPISODES: list[dict[str, Any]] = [
    {
        "iso3": "TUR",
        "label": "Turkiye — 2018 currency crisis",
        "baseline_year": 2016,
        "event_year": 2018,
        "note": "The lira lost roughly 30% of its value against the US dollar in 2018, "
        "driven by a large current-account deficit, heavy external corporate debt, and "
        "rapid monetary tightening.",
    },
    {
        "iso3": "BRA",
        "label": "Brazil — 2015-16 recession",
        "baseline_year": 2013,
        "event_year": 2016,
        "note": "GDP contracted roughly 7% cumulatively over 2015-16, alongside double-digit "
        "inflation and a widening fiscal deficit during a deep political and fiscal crisis.",
    },
    {
        "iso3": "ZAF",
        "label": "South Africa — fiscal deterioration, 2017-2020",
        "baseline_year": 2017,
        "event_year": 2020,
        "note": "Persistent fiscal slippage and state-utility stress widened the budget deficit "
        "and public debt trajectory well before the COVID-19 shock added a sharp additional "
        "contraction in 2020.",
    },
    {
        "iso3": "GBR",
        "label": "United Kingdom — 2022 gilt market shock",
        "baseline_year": 2019,
        "event_year": 2022,
        "note": "September 2022's unfunded fiscal package triggered a sharp gilt sell-off and "
        "sterling depreciation, on top of the UK's post-2020 inflation and current-account "
        "pressures.",
    },
]


@dataclass
class Episode:
    iso3: str
    label: str
    baseline_year: int
    event_year: int
    note: str = ""
    window_years: int = DEFAULT_WINDOW_YEARS
    pass_threshold_points: float = PASS_THRESHOLD_POINTS


@dataclass
class EpisodeResult:
    iso3: str
    label: str
    baseline_year: int
    event_year: int
    baseline_score: float | None
    event_score: float | None
    peak_score: float | None
    peak_year: int | None
    delta: float | None  # event_year - baseline (at-event snapshot)
    peak_delta: float | None  # peak-in-window - baseline (drives verdict)
    threshold_points: float
    window_years: int
    verdict: str  # "flagged" | "missed" | "inconclusive"
    peer_drift: float | None  # share of other panel countries that also rose >= threshold
    dominant_pillar: str | None  # pillar_<slug>_score column with the largest rise
    dominant_sector: str | None  # sector_<slug>_score column with the largest rise
    note: str


@dataclass
class BacktestSummary:
    episodes_total: int
    flagged: int
    missed: int
    inconclusive: int
    detection_rate: float | None  # flagged / (flagged + missed)
    median_peak_delta: float | None  # median peak_delta across evaluable episodes
    systemic_episode: str | None  # highest peer_drift episode (least informative)
    highest_peer_drift: float | None  # drift fraction of the systemic_episode


def _load_episodes() -> list[Episode]:
    """Load episodes from config/episodes.yaml, falling back to built-ins."""
    cfg: dict = {}
    try:
        with open(EPISODES_PATH, encoding="utf-8") as f:
            loaded = yaml.safe_load(f) or {}
        if isinstance(loaded, dict):
            cfg = loaded
    except OSError:
        pass

    raw_episodes = cfg.get("episodes") if isinstance(cfg, dict) else None
    default_window = cfg.get("detection_window_years", DEFAULT_WINDOW_YEARS)
    default_threshold = cfg.get("pass_threshold_points", PASS_THRESHOLD_POINTS)

    if isinstance(raw_episodes, list) and raw_episodes:
        episodes = []
        for e in raw_episodes:
            if not isinstance(e, dict) or not e.get("iso3"):
                continue
            episodes.append(
                Episode(
                    iso3=str(e["iso3"]).upper(),
                    label=str(e.get("label", e["iso3"])),
                    baseline_year=int(e["baseline_year"]),
                    event_year=int(e["event_year"]),
                    note=str(e.get("note", "")),
                    window_years=int(e.get("window_years", default_window)),
                    pass_threshold_points=float(e.get("pass_threshold_points", default_threshold)),
                )
            )
        if episodes:
            return episodes

    return [
        Episode(
            iso3=str(e["iso3"]).upper(),
            label=str(e["label"]),
            baseline_year=int(e["baseline_year"]),
            event_year=int(e["event_year"]),
            note=str(e["note"]),
            window_years=DEFAULT_WINDOW_YEARS,
            pass_threshold_points=PASS_THRESHOLD_POINTS,
        )
        for e in DEFAULT_EPISODES
    ]


_EPISODES = _load_episodes()

# Backward-compatible module constant: same episodes, as plain dicts.
EPISODES: list[dict[str, Any]] = [asdict(e) for e in _EPISODES]


def _prep(scores: pd.DataFrame) -> pd.DataFrame:
    s = scores.copy()
    s["country_iso3"] = s["country_iso3"].astype(str).str.upper()
    s["year"] = pd.to_numeric(s["year"], errors="coerce")
    s["risk_score"] = pd.to_numeric(s["risk_score"], errors="coerce")
    return s


def _score_for(scores: pd.DataFrame, iso3: str, year: int) -> float | None:
    row = scores[
        scores["country_iso3"].eq(str(iso3).upper()) & pd.to_numeric(scores["year"], errors="coerce").eq(int(year))
    ]
    if row.empty or pd.isna(row["risk_score"].iloc[0]):
        return None
    return float(row["risk_score"].iloc[0])


def _peak_in_window(
    scores: pd.DataFrame,
    iso3: str,
    start_year: int,
    window_years: int,
) -> tuple[float | None, int | None]:
    """(max score, year of that max) across [start_year, start_year + window]."""
    years = list(range(int(start_year), int(start_year) + int(window_years) + 1))
    points = [(int(y), _score_for(scores, iso3, y)) for y in years]
    available = [(y, v) for y, v in points if v is not None]
    if not available:
        return None, None
    peak_year, peak = max(available, key=lambda t: t[1])
    return peak, peak_year


def _peer_drift(
    scores: pd.DataFrame,
    iso3: str,
    baseline_year: int,
    event_year: int,
    window_years: int,
    threshold: float,
) -> float | None:
    """Share of OTHER countries that also rose >= threshold over the window."""
    others = [c for c in scores["country_iso3"].dropna().unique() if str(c).upper() != str(iso3).upper()]
    flagged = 0
    compared = 0
    for other in others:
        other = str(other)
        base = _score_for(scores, other, baseline_year)
        if base is None:
            continue
        peak, _ = _peak_in_window(scores, other, event_year, window_years)
        if peak is None:
            continue
        compared += 1
        if peak - base >= threshold:
            flagged += 1
    if compared == 0:
        return None
    return round(flagged / compared, 3)


def _largest_rising_component(
    scores: pd.DataFrame,
    iso3: str,
    baseline_year: int,
    target_year: int | None,
    prefix: str,
) -> str | None:
    if target_year is None:
        return None
    sub = scores[scores["country_iso3"].eq(str(iso3).upper())]
    base = sub[pd.to_numeric(sub["year"], errors="coerce").eq(int(baseline_year))]
    target = sub[pd.to_numeric(sub["year"], errors="coerce").eq(int(target_year))]
    if base.empty or target.empty:
        return None
    best: tuple[str, float] | None = None
    for col in scores.columns:
        if not col.startswith(prefix):
            continue
        b = pd.to_numeric(base[col], errors="coerce").iloc[0]
        t = pd.to_numeric(target[col], errors="coerce").iloc[0]
        if pd.isna(b) or pd.isna(t):
            continue
        rise = float(t) - float(b)
        if best is None or rise > best[1]:
            best = (col, rise)
    return best[0] if best is not None else None


def _inconclusive_result(ep: Episode) -> EpisodeResult:
    return EpisodeResult(
        iso3=ep.iso3,
        label=ep.label,
        baseline_year=ep.baseline_year,
        event_year=ep.event_year,
        baseline_score=None,
        event_score=None,
        peak_score=None,
        peak_year=None,
        delta=None,
        peak_delta=None,
        threshold_points=ep.pass_threshold_points,
        window_years=ep.window_years,
        verdict="inconclusive",
        peer_drift=None,
        dominant_pillar=None,
        dominant_sector=None,
        note=ep.note,
    )


def run_backtest(
    scores: pd.DataFrame,
    data_is_synthetic: bool = False,
    window_years: int | None = None,
    threshold_points: float | None = None,
) -> list[EpisodeResult]:
    """
    Evaluate every configured episode against the given scores DataFrame.

    The DataFrame needs at least country_iso3, year, risk_score (plus the
    pillar_<slug>_score / sector_<slug>_score columns optionally emitted by
    src.scoring.risk_score.score_panel for the component drill-down).

    Returns one EpisodeResult per episode in config order.
    """
    if not isinstance(scores, pd.DataFrame) or scores.empty:
        return [_inconclusive_result(ep) for ep in _EPISODES]

    s = _prep(scores)
    results: list[EpisodeResult] = []

    for ep in _EPISODES:
        window = int(ep.window_years if window_years is None else window_years)
        threshold = float(ep.pass_threshold_points if threshold_points is None else threshold_points)

        baseline_score = _score_for(s, ep.iso3, ep.baseline_year)
        event_score = _score_for(s, ep.iso3, ep.event_year)
        peak_score, peak_year = _peak_in_window(s, ep.iso3, ep.event_year, window)

        if baseline_score is None or peak_score is None:
            results.append(_inconclusive_result(ep))
            continue

        delta = None
        if event_score is not None:
            delta = round(event_score - baseline_score, 1)
        peak_delta = round(peak_score - baseline_score, 1)
        verdict = "flagged" if peak_delta >= threshold else "missed"

        results.append(
            EpisodeResult(
                iso3=ep.iso3,
                label=ep.label,
                baseline_year=ep.baseline_year,
                event_year=ep.event_year,
                baseline_score=baseline_score,
                event_score=event_score,
                peak_score=peak_score,
                peak_year=peak_year,
                delta=delta,
                peak_delta=peak_delta,
                threshold_points=threshold,
                window_years=window,
                verdict=verdict,
                peer_drift=_peer_drift(s, ep.iso3, ep.baseline_year, ep.event_year, window, threshold),
                dominant_pillar=_largest_rising_component(s, ep.iso3, ep.baseline_year, peak_year, "pillar_"),
                dominant_sector=_largest_rising_component(s, ep.iso3, ep.baseline_year, peak_year, "sector_"),
                note=ep.note,
            )
        )

    return results


def backtest_summary(results: list[EpisodeResult]) -> BacktestSummary:
    """Aggregate detection-rate + diagnostics across all episodes."""
    evaluated = [r for r in results if r.verdict != "inconclusive"]
    flagged = sum(1 for r in results if r.verdict == "flagged")
    missed = sum(1 for r in results if r.verdict == "missed")
    inconclusive = sum(1 for r in results if r.verdict == "inconclusive")

    detection_rate: float | None = None
    if evaluated:
        detection_rate = round(flagged / len(evaluated), 3)

    deltas = [r.peak_delta for r in evaluated if r.peak_delta is not None]
    median_peak_delta = round(float(pd.Series(deltas).median()), 1) if deltas else None

    drift_entries = [(r.label, r.peer_drift) for r in results if r.peer_drift is not None]
    top_drift = max(drift_entries, key=lambda t: t[1], default=(None, None))

    return BacktestSummary(
        episodes_total=len(results),
        flagged=flagged,
        missed=missed,
        inconclusive=inconclusive,
        detection_rate=detection_rate,
        median_peak_delta=median_peak_delta,
        systemic_episode=top_drift[0],
        highest_peer_drift=top_drift[1],
    )


def false_alarm_rate(
    scores: pd.DataFrame,
    window_years: int = DEFAULT_WINDOW_YEARS,
    threshold_points: float = PASS_THRESHOLD_POINTS,
    exclude_episodes: bool = True,
) -> float | None:
    """
    False-alarm base rate: the share of ordinary country-year windows where the
    model's own "rise >= threshold over the detection window" rule would trip.

    This is the honest companion to `detection_rate`: a high detection rate is
    only meaningful if the rule does not also fire on plenty of times that no
    known crisis happened. For every (country, baseline_year) in the panel we
    measure peak-in-window minus baseline, exactly as episodes are judged, and
    count how many clear the threshold. The return value is that share (0-1).

    Episode windows are excluded by default so the known crises do not inflate
    the base rate against themselves.
    """
    if not isinstance(scores, pd.DataFrame) or scores.empty:
        return None

    s = _prep(scores)
    excluded_windows: dict[str, set[int]] = {}
    if exclude_episodes:
        for ep in _EPISODES:
            iso3 = str(ep.iso3).upper()
            excluded_windows.setdefault(iso3, set()).add(int(ep.baseline_year))
            for year in range(int(ep.event_year) - int(ep.window_years), int(ep.event_year) + int(ep.window_years) + 1):
                excluded_windows[iso3].add(int(year))

    years = sorted(pd.to_numeric(s["year"], errors="coerce").dropna().astype(int).unique().tolist())
    alarmed = 0
    compared = 0

    for iso3 in sorted(s["country_iso3"].dropna().unique()):
        iso3 = str(iso3).upper()
        country_years = excluded_windows.get(iso3, set())
        for baseline_year in years:
            if baseline_year in country_years:
                continue
            base = _score_for(s, iso3, baseline_year)
            if base is None:
                continue
            peak, _ = _peak_in_window(s, iso3, baseline_year, window_years)
            if peak is None:
                continue
            compared += 1
            if peak - base >= threshold_points:
                alarmed += 1

    if compared == 0:
        return None
    return round(alarmed / compared, 3)


def evaluate_panel(
    panel: pd.DataFrame,
    data_is_synthetic: bool = False,
) -> tuple[list[EpisodeResult], BacktestSummary, pd.DataFrame]:
    """Score a wide panel and replay the episodes against the freshly-computed scores."""
    scores, _drivers, _pillar_scores = score_panel(panel)
    results = run_backtest(scores, data_is_synthetic=data_is_synthetic)
    return results, backtest_summary(results), scores


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

DEFAULT_PANEL = ROOT / "data" / "demo" / "panel_wide.csv"


def _format_table(results: list[EpisodeResult]) -> str:
    lines = [
        "",
        "HISTORICAL BACKTEST (instant replay)",
        "=" * 78,
        f"{'EPISODE':<42}{'BASE':>6}{'EVENT':>7}{'PEAK':>7}{'DELTA':>7}  VERDICT",
        "-" * 78,
    ]
    for r in results:
        base = f"{r.baseline_score:.1f}" if r.baseline_score is not None else "-"
        event = f"{r.event_score:.1f}" if r.event_score is not None else "-"
        peak = f"{r.peak_score:.1f}" if r.peak_score is not None else "-"
        delta = f"{r.peak_delta:+.1f}" if r.peak_delta is not None else "-"
        lines.append(f"{r.label[:42]:<42}{base:>6}{event:>7}{peak:>7}{delta:>7}  {r.verdict.upper()}")
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run the historical backtest harness on a wide panel CSV.")
    parser.add_argument("--panel", default=str(DEFAULT_PANEL), help="Wide-format panel CSV path.")
    parser.add_argument(
        "--mode",
        choices=["demo", "live"],
        default="demo",
        help="demo = synthetic (results are NOT a real validation); live = real data.",
    )
    parser.add_argument("--json", default=None, help="Optional path to write the JSON report.")
    args = parser.parse_args(argv)

    panel_path = Path(args.panel)
    if not panel_path.exists():
        print(f"Panel file not found: {panel_path}", file=sys.stderr)
        return 2

    panel = pd.read_csv(panel_path)
    if "country_iso3" not in panel.columns or "year" not in panel.columns:
        print(f"Panel at {panel_path} is missing country_iso3/year columns.", file=sys.stderr)
        return 2

    results, summary, scores = evaluate_panel(panel, data_is_synthetic=(args.mode == "demo"))

    print(_format_table(results))
    rate_txt = (
        f"Detection rate: {summary.detection_rate:.0%} ({summary.flagged} flagged / "
        f"{summary.flagged + summary.missed} evaluable) · {summary.inconclusive} inconclusive"
        f" · median peak delta {summary.median_peak_delta} pts"
        if summary.detection_rate is not None
        else f"All {summary.inconclusive} episodes inconclusive — nothing scored "
        f"(check that the panel covers the episode years)."
    )
    print("\n" + rate_txt)
    if summary.systemic_episode:
        print(f"Least informative: {summary.systemic_episode} (peer drift {summary.highest_peer_drift})")

    if args.json:
        payload = {
            "mode": args.mode,
            "synthetic": args.mode == "demo",
            "summary": asdict(summary),
            "episodes": [asdict(r) for r in results],
        }
        out = Path(args.json)
        out.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        print(f"\nWrote JSON report -> {out}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
