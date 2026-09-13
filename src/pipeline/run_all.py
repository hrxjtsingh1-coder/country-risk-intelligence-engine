"""
End-to-end Country Risk Intelligence Engine pipeline.

The production pipeline is intentionally single-year: every invocation is
restricted to the current calendar year so stale historical observations
cannot be regenerated into the committed panel.

Usage:
    python -m src.pipeline.run_all
    python -m src.pipeline.run_all --countries USA,IND,DEU
    python -m src.pipeline.run_all --shock POLICY_RATE_YOY_CHANGE_BPS:100
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import UTC, datetime
from pathlib import Path

import yaml

from src.cleaning.clean import coverage_report, to_wide_panel
from src.commentary.generate_commentary import generate_report
from src.db import db_utils
from src.governance.manifest import MANIFEST_PATH, METHODOLOGY_VERSION, build_manifest
from src.indicators.build_panel import build_long_panel_batched
from src.runtime.year_policy import CURRENT_YEAR, validate_current_year_range
from src.scenario.scenario_engine import run_shock_scenario
from src.scoring.risk_score import score_panel

ROOT = Path(__file__).resolve().parents[2]


def _load_countries_cfg() -> dict:
    with open(ROOT / "config" / "countries.yaml", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def parse_args():
    parser = argparse.ArgumentParser(description="Run the current-year country risk intelligence pipeline.")
    parser.add_argument(
        "--countries",
        type=str,
        default=None,
        help="Comma-separated ISO3 codes; default = all configured countries",
    )
    parser.add_argument("--start", type=int, default=CURRENT_YEAR, help=f"Current calendar year only ({CURRENT_YEAR})")
    parser.add_argument("--end", type=int, default=CURRENT_YEAR, help=f"Current calendar year only ({CURRENT_YEAR})")
    parser.add_argument("--shock", type=str, default="POLICY_RATE_YOY_CHANGE_BPS:100", help="driver_code:amount")
    parser.add_argument("--skip-db", action="store_true", help="Skip writing to the SQLite warehouse")
    return parser.parse_args()


def main():
    args = parse_args()
    try:
        start_year, end_year = validate_current_year_range(args.start, args.end)
    except ValueError as exc:
        print(f"Current-year data contract violation: {exc}", file=sys.stderr)
        sys.exit(2)

    cfg = _load_countries_cfg()
    all_iso3 = [str(c["iso3"]) for c in cfg.get("countries", [])]
    iso3_codes = [c.strip().upper() for c in args.countries.split(",") if c.strip()] if args.countries else all_iso3
    name_lookup = {str(c["iso3"]): c["name"] for c in cfg.get("countries", [])}

    print(f"[1/6] Collecting + cleaning current-year ({CURRENT_YEAR}) indicators for {len(iso3_codes)} countries...")
    started_at = datetime.now(UTC)
    long_panel, fetch_meta = build_long_panel_batched(iso3_codes, start_year, end_year)
    del fetch_meta

    if long_panel.empty:
        print("No current-year data returned. Check official provider availability and retry.")
        sys.exit(1)

    indicator_codes = sorted(long_panel["indicator_code"].unique())
    cov = coverage_report(long_panel, iso3_codes, indicator_codes, [CURRENT_YEAR])
    print(cov.to_string(index=False))

    wide_panel = to_wide_panel(long_panel)
    if wide_panel.empty or not wide_panel["year"].eq(CURRENT_YEAR).all():
        print(f"Current-year validation failed: panel is not restricted to {CURRENT_YEAR}.", file=sys.stderr)
        sys.exit(1)

    out_dir = ROOT / "data" / "processed"
    out_dir.mkdir(parents=True, exist_ok=True)
    wide_panel.to_csv(out_dir / "panel_wide.csv", index=False)
    configured_codes = [str(item.get("code")) for item in yaml.safe_load((ROOT / "config" / "indicators.yaml").read_text()).get("indicators", [])]
    metadata = {
        "run_id": started_at.strftime("live-%Y%m%dT%H%M%SZ"),
        "mode": "LIVE",
        "started_at": started_at.isoformat(),
        "completed_at": datetime.now(UTC).isoformat(),
        "retrieved_at": datetime.now(UTC).isoformat(),
        "requested_period": str(CURRENT_YEAR),
        "latest_available_observation": CURRENT_YEAR,
        "country_count": int(wide_panel["country_iso3"].nunique()),
        "indicator_count": int(len([c for c in configured_codes if c in wide_panel.columns])),
        "observations_received": int(long_panel.shape[0]),
        "observations_missing": int(max(0, len(iso3_codes) * len(configured_codes) - long_panel.shape[0])),
        "sources": ["World Bank", "FRED (US-only enrichment)"],
        "source_urls": ["https://api.worldbank.org/v2/", "https://fred.stlouisfed.org/graph/fredgraph.csv"],
        "methodology_version": METHODOLOGY_VERSION,
        "config_version": "config/indicators.yaml",
    }
    (out_dir / "data_metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    print("[2/6] Scoring...")
    scores, drivers, pillar_scores = score_panel(wide_panel)
    del pillar_scores

    print("[3/6] Preparing scenario...")
    driver_code, amount_text = args.shock.split(":", 1)
    amount = float(amount_text)
    latest_year = CURRENT_YEAR
    scenario_targets = ["FX_YOY_DEPRECIATION_PCT", "NY.GDP.MKTP.KD.ZG", "GC.DOD.TOTL.GD.ZS"]

    if not args.skip_db:
        print("[4/6] Writing SQLite warehouse...")
        conn = db_utils.get_connection()
        try:
            db_utils.init_schema(conn)
            db_utils.clear_run_data(conn)
            db_utils.load_countries(conn)
            db_utils.load_indicator_values(conn, long_panel)
            db_utils.load_scores(conn, scores, drivers)
            print(db_utils.top_risk_countries(conn, latest_year))
        finally:
            conn.close()
    else:
        print("[4/6] Skipping DB (--skip-db)...")

    print("[5/6] Generating analyst commentary...")
    commentary_dir = out_dir / "commentary"
    commentary_dir.mkdir(parents=True, exist_ok=True)
    peer_groups = cfg.get("peer_groups", {})

    for iso3 in iso3_codes:
        rows_for_country = wide_panel[wide_panel["country_iso3"].astype(str).eq(iso3)]
        if rows_for_country.empty:
            continue
        year = CURRENT_YEAR
        peer_group = next((members for members in peer_groups.values() if iso3 in members), None)
        peer_group = [c for c in (peer_group or []) if c != iso3]
        scenario_result = None
        try:
            scenario_result = run_shock_scenario(wide_panel, iso3, year, driver_code, amount, scenario_targets)
        except Exception as exc:
            print(f"  (scenario skipped for {iso3}: {exc})")
        report = generate_report(
            country_name=name_lookup.get(iso3, iso3),
            country_iso3=iso3,
            year=year,
            scores=scores,
            drivers=drivers,
            scenario_result=scenario_result,
            peer_group=peer_group,
        )
        (commentary_dir / f"{iso3}_{year}.md").write_text(report, encoding="utf-8")

    print("[6/6] Done. Current-year panel, DB, and commentary files written under data/processed/.")
    print("Launch the dashboard with: streamlit run dashboard/app.py")
    print("[7/7] Building reproducibility manifest + coverage gate...")
    manifest = build_manifest(panel=wide_panel, scores=scores, dataset_mode="live", signed_by=os.environ.get("ANALYST_SIGNED_BY", "pipeline-bot"))
    coverage_status = manifest["model"]["coverage"]["status"]
    detection_rate = manifest["model"]["backtest"]["detection_rate"]
    print(f"Manifest written to {MANIFEST_PATH} (coverage={coverage_status} · backtest detection_rate={detection_rate})")
    if coverage_status == "issues":
        print("WARNING: model coverage has error-severity issues. Run `python -m src.governance.coverage` for details and repair hints.", file=sys.stderr)


if __name__ == "__main__":
    main()
