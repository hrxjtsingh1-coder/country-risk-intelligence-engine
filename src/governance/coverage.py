"""Coverage-repair protocol for the risk model.

"Coverage" here means the whole vulnerability surface the methodology claims
to score: the configured pillars, the sectors that group them, every
positive-weight indicator, and the historical episodes used for validation.
If any configured component is not actually represented in the panel being
scored — a new indicator added to config but never generated, a pillar weight
that no longer sums to 1.0, an episode whose years fall outside the data —
the numbers silently drift and traceability breaks.

This module verifies that surface against a loaded panel and FAILS loudly
(exit code 1 / error-severity issues) instead of letting stale or
shape-broken artifacts be treated as current. Issues come with concrete
repair hints (the exact command to regenerate the stale artifact), so the
protocol is "verify, then repair with the printed command", never "hope".

Everything here is offline and deterministic: no network, no random state.
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict, dataclass
from pathlib import Path

import pandas as pd
import yaml

from src.indicators.build_panel import _derive_indicators
from src.scoring import risk_score as rs

ROOT = Path(__file__).resolve().parents[2]
CONFIG_PATH = ROOT / "config" / "indicators.yaml"
COUNTRIES_PATH = ROOT / "config" / "countries.yaml"
EPISODES_PATH = ROOT / "config" / "episodes.yaml"
DEMO_PANEL_PATH = ROOT / "data" / "demo" / "panel_wide.csv"

# Repair hints: the exact command that regenerates the stale artifact.
REPAIR_CMDS: dict[str, str] = {
    "demo_panel": "python scripts/create_demo_data.py --output data/demo/panel_wide.csv",
    "live_panel": "python -m src.pipeline.run_all",
}

# Derived indicators re-derivable from base series via _derive_indicators.
DERIVABLE: dict[str, list[str]] = {
    "OUTPUT_GAP_PROXY_PCT": ["NY.GDP.MKTP.KD.ZG"],
    "PUBLIC_DEBT_TRAJECTORY_PCT": ["GC.DOD.TOTL.GD.ZS"],
}


@dataclass
class CoverageIssue:
    severity: str  # "error" | "warning"
    ref: str
    message: str
    repair: str | None = None


@dataclass
class CoverageReport:
    panel_source: str
    countries: int
    year_min: int | None
    year_max: int | None
    panel_columns: list[str]
    indicator_total: int
    indicator_covered: int
    checks: dict[str, bool]
    issues: list[CoverageIssue]

    @property
    def status(self) -> str:
        return "ok" if not any(i.severity == "error" for i in self.issues) else "issues"


def _load_countries_iso3() -> set[str]:
    try:
        with open(COUNTRIES_PATH, encoding="utf-8") as f:
            cfg = yaml.safe_load(f) or {}
    except OSError:
        return set()
    return {str(c["iso3"]).upper() for c in cfg.get("countries", []) if isinstance(c, dict) and c.get("iso3")}


def _load_episodes() -> list[dict]:
    try:
        with open(EPISODES_PATH, encoding="utf-8") as f:
            cfg = yaml.safe_load(f) or {}
    except OSError:
        cfg = {}
    return [e for e in cfg.get("episodes", []) if isinstance(e, dict)] if isinstance(cfg, dict) else []


def _verify_derivable(panel: pd.DataFrame, issues: list[CoverageIssue]) -> bool:
    """Re-derive derived indicators from base series and compare to the panel.

    Guards the "wrong shape" failure mode: a column can exist in the panel yet
    no longer match its documented derivation — coverage would otherwise look
    fine while the signal silently changed meaning.
    """
    long_rows: list[dict] = []
    for code in {c for cols in DERIVABLE.values() for c in cols}:
        if code not in panel.columns:
            continue
        for iso3, year, value in panel[["country_iso3", "year", code]].itertuples(index=False):
            if pd.isna(value):
                continue
            long_rows.append({"country_iso3": iso3, "year": year, "value": value, "indicator_code": code})
    base = pd.DataFrame(long_rows)
    if base.empty:
        return True

    derived = _derive_indicators(base)
    if derived.empty:
        return True

    ok = True
    for code in DERIVABLE:
        if code not in panel.columns or code not in derived["indicator_code"].unique():
            if code in DERIVABLE and code not in panel.columns:
                continue  # missing column is reported by the indicator-coverage check
            continue
        expected = derived[derived["indicator_code"].eq(code)][["country_iso3", "year", "value"]]
        merged = expected.merge(
            panel[["country_iso3", "year", code]].rename(columns={code: "actual"}),
            on=["country_iso3", "year"],
        )
        drift = (
            pd.to_numeric(merged["value"], errors="coerce") - pd.to_numeric(merged["actual"], errors="coerce")
        ).abs()
        if drift.max(skipna=True) > 1e-6:
            issues.append(
                CoverageIssue(
                    severity="error",
                    ref=f"indicator/{code}",
                    message=(
                        f"Derived indicator {code} in the panel no longer matches its documented derivation "
                        f"(max |re-derived - panel| = {drift.max():.6f} over {merged.shape[0]} rows). Regenerate the panel."
                    ),
                    repair=REPAIR_CMDS["live_panel"],
                )
            )
            ok = False
    return ok


def verify_model_coverage(panel: pd.DataFrame, panel_source: str = "demo") -> CoverageReport:
    """Verify the full vulnerability surface against a wide panel and report issues."""
    issues: list[CoverageIssue] = []
    checks: dict[str, bool] = {}

    indicators = rs._records()
    positive_weight = [i for i in indicators if float(i.get("weight", 0.0)) > 0]
    panel_columns = list(panel.columns)
    panel_countries = set(panel["country_iso3"].astype(str).str.upper()) if not panel.empty else set()
    years = pd.to_numeric(panel["year"], errors="coerce") if not panel.empty else pd.Series(dtype=float)
    year_min = int(years.min()) if not panel.empty and not years.isna().all() else None
    year_max = int(years.max()) if not panel.empty and not years.isna().all() else None

    # --- Configuration validity (surfaces scoring-time failures as issues) ---
    try:
        rs._load_pillar_weights()
        checks["pillar_weights"] = True
    except ValueError as exc:
        checks["pillar_weights"] = False
        issues.append(CoverageIssue("error", "config/pillar_weights", f"Pillar weights invalid: {exc}"))

    try:
        _sector_weights, sector_map = rs._load_sectors()
        checks["sector_config"] = True
    except ValueError as exc:
        sector_map = {}
        checks["sector_config"] = False
        issues.append(CoverageIssue("error", "config/sectors", f"Sector configuration invalid: {exc}"))

    if checks["sector_config"]:
        scoring = rs._load_scoring()
        active_sectors = float(scoring.get("sector_composite_weight", 0.0)) > 0.0
        if active_sectors:
            occupied = {i["pillar"] for i in positive_weight}
            listed = {p for members in sector_map.values() for p in members}
            not_partitioned = sorted(occupied - listed) or sorted(set(sector_map) and (listed - occupied))
            checks["sector_partition"] = not not_partitioned
            if not_partitioned:
                issues.append(
                    CoverageIssue(
                        "error",
                        "config/sectors",
                        f"Sectors do not partition the occupied pillars cleanly: {not_partitioned}.",
                    )
                )
        else:
            checks["sector_partition"] = True

    # --- Indicator coverage ---
    missing_columns = [i["code"] for i in positive_weight if i["code"] not in panel_columns]
    checks["indicator_columns"] = not missing_columns
    if missing_columns:
        for code in missing_columns:
            rec = next((i for i in positive_weight if i["code"] == code), {})
            repair = None
            if str(rec.get("source", "")) == "derived":
                repair = REPAIR_CMDS["live_panel"]
            issues.append(
                CoverageIssue(
                    severity="error",
                    ref=f"panel/{code}",
                    message=(
                        f"Positive-weight indicator {code} has no column in the panel. "
                        f"The configured surface is not covered; the score will ship with a hole in it."
                    ),
                    repair=repair,
                )
            )

    # --- Pillar coverage: every positively-weighted pillar has >=1 indicator ---
    pillars = rs.PILLAR_ORDER
    pillar_of = {i["code"]: i["pillar"] for i in positive_weight}
    try:
        weighted_pillars = {p for p, w in rs._load_pillar_weights().items() if w > 0}
    except ValueError:
        weighted_pillars = set(pillars)
    empty_pillars = sorted(p for p in weighted_pillars if p not in set(pillar_of.values()))
    checks["pillar_coverage"] = not empty_pillars
    if empty_pillars:
        issues.append(
            CoverageIssue(
                "error",
                "config/pillars",
                f"Positive-weight pillars with no positive-weight indicator: {empty_pillars}.",
            )
        )

    # --- Derivation integrity ---
    checks["derivations"] = _verify_derivable(panel, issues)

    # --- Episode coverage ---
    configured = _load_countries_iso3()
    configured |= {str(country.upper()) for country in panel_countries}
    for ep in _load_episodes():
        ep_iso3 = str(ep.get("iso3", "")).upper()
        if ep_iso3 not in configured:
            issues.append(
                CoverageIssue(
                    severity="warning",
                    ref=f"episodes/{ep.get('label', ep_iso3)}",
                    message=f"Episode country {ep_iso3} is not in config/countries.yaml nor the loaded panel.",
                )
            )
        elif year_min is not None and year_max is not None:
            baseline = int(ep.get("baseline_year", 0))
            event = int(ep.get("event_year", 0))
            outside = [y for y in (baseline, event) if y < year_min or y > year_max]
            if outside:
                issues.append(
                    CoverageIssue(
                        severity="warning",
                        ref=f"episodes/{ep.get('label', ep_iso3)}",
                        message=(
                            f"Episode years {outside} fall outside the panel year range "
                            f"({year_min}-{year_max}); the episode cannot be evaluated until the window is expanded."
                        ),
                    )
                )
    checks["episode_windows_in_range"] = not any(
        i.ref.startswith("episodes/") and i.severity == "warning" for i in issues
    )

    return CoverageReport(
        panel_source=panel_source,
        countries=len(panel_countries),
        year_min=year_min,
        year_max=year_max,
        panel_columns=panel_columns,
        indicator_total=len(positive_weight),
        indicator_covered=sum(1 for i in positive_weight if i["code"] in panel_columns),
        checks=checks,
        issues=issues,
    )


def report_to_text(report: CoverageReport) -> str:
    lines = [
        "MODEL COVERAGE REPORT",
        "=" * 72,
        f"Panel: {report.panel_source} · {report.countries} countries",
        f"Years: {report.year_min}–{report.year_max}" if report.year_min is not None else "Years: <empty panel>",
        f"Indicators (positive weight): {report.indicator_covered}/{report.indicator_total} columns present",
        "-" * 72,
    ]
    for name, passed in report.checks.items():
        lines.append(f"  [{'OK ' if passed else 'FAIL'}] {name}")
    errors = [i for i in report.issues if i.severity == "error"]
    warnings = [i for i in report.issues if i.severity == "warning"]
    lines.append("-" * 72)
    lines.append(f"Issues: {len(errors)} errors, {len(warnings)} warnings")
    for issue in report.issues:
        lines.append(f"  [{issue.severity.upper()}] {issue.ref}: {issue.message}")
        if issue.repair:
            lines.append(f"        repair: {issue.repair}")
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Verify model coverage of a panel and repair stale artifacts.")
    parser.add_argument("--panel", default=str(DEMO_PANEL_PATH), help="Wide-format panel CSV path.")
    parser.add_argument("--format", choices=["text", "json"], default="text")
    args = parser.parse_args(argv)

    panel_path = Path(args.panel)
    if not panel_path.exists():
        print(f"Panel file not found: {panel_path}", file=sys.stderr)
        return 2
    panel = pd.read_csv(panel_path)
    if "country_iso3" not in panel.columns or "year" not in panel.columns:
        print(f"Panel at {panel_path} is missing country_iso3/year columns.", file=sys.stderr)
        return 2

    report = verify_model_coverage(panel, panel_source=f"{panel_path}")

    if args.format == "json":
        print(
            json.dumps(
                {
                    "status": report.status,
                    "panel_source": report.panel_source,
                    "countries": report.countries,
                    "year_min": report.year_min,
                    "year_max": report.year_max,
                    "indicator_covered": report.indicator_covered,
                    "indicator_total": report.indicator_total,
                    "checks": report.checks,
                    "issues": [asdict(i) for i in report.issues],
                },
                indent=2,
            )
        )
    else:
        print(report_to_text(report))

    return 1 if report.status == "issues" else 0


if __name__ == "__main__":
    raise SystemExit(main())
