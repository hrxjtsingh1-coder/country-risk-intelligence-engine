"""
End-to-end pipeline: Country -> Data collection -> Cleaning -> Indicators ->
Risk scoring -> Scenario analysis -> Dashboard-ready DB -> Analyst commentary.

Usage:
    python -m src.pipeline.run_all
    python -m src.pipeline.run_all --countries USA,IND,DEU --start 2015 --end 2025
    python -m src.pipeline.run_all --shock POLICY_RATE_YOY_CHANGE_BPS:100

This hits the real World Bank / FRED / ECB / BIS APIs over the network —
run it somewhere with normal internet access. Output lands in:
    data/processed/country_risk.db          (SQLite warehouse)
    data/processed/panel_wide.csv           (the indicator panel, for the dashboard/notebooks)
    data/processed/commentary/<ISO3>_<YEAR>.md   (one file per country, latest year)
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[2]
sys.path.append(str(ROOT))

from src.commentary.generate_commentary import generate_report
from src.db import db_utils
from src.indicators.build_panel import build_long_panel
from src.cleaning.clean import to_wide_panel, coverage_report
from src.scenario.scenario_engine import run_shock_scenario
from src.scoring.risk_score import score_panel


def _load_countries_cfg() -> dict:
    with open(ROOT / "config" / "countries.yaml") as f:
        return yaml.safe_load(f)


def parse_args():
    p = argparse.ArgumentParser(description="Run the country risk intelligence pipeline end to end.")
    p.add_argument("--countries", type=str, default=None, help="Comma-separated ISO3 codes; default = everyone in config/countries.yaml")
    p.add_argument("--start", type=int, default=2012)
    p.add_argument("--end", type=int, default=2025)
    p.add_argument("--shock", type=str, default="POLICY_RATE_YOY_CHANGE_BPS:100", help="driver_code:amount, e.g. POLICY_RATE_YOY_CHANGE_BPS:100")
    p.add_argument("--skip-db", action="store_true", help="Skip writing to the SQLite warehouse")
    return p.parse_args()


def main():
    args = parse_args()
    countries_cfg = _load_countries_cfg()
    all_iso3 = [c["iso3"] for c in countries_cfg["countries"]]
    iso3_codes = args.countries.split(",") if args.countries else all_iso3
    name_lookup = {c["iso3"]: c["name"] for c in countries_cfg["countries"]}

    print(f"[1/6] Collecting + cleaning indicators for {len(iso3_codes)} countries, {args.start}-{args.end}...")
    long_panel = build_long_panel(iso3_codes, args.start, args.end)
    if long_panel.empty:
        print("No data returned. Check network access to the source APIs (see README) and retry.")
        sys.exit(1)

    cov = coverage_report(long_panel, iso3_codes, sorted(long_panel["indicator_code"].unique()), list(range(args.start, args.end + 1)))
    print(cov.to_string(index=False))

    wide_panel = to_wide_panel(long_panel)
    out_dir = ROOT / "data" / "processed"
    out_dir.mkdir(parents=True, exist_ok=True)
    wide_panel.to_csv(out_dir / "panel_wide.csv", index=False)

    print("[2/6] Scoring...")
    scores, drivers = score_panel(wide_panel)

    print("[3/6] Running scenario shock...")
    driver_code, amount = args.shock.split(":")
    amount = float(amount)
    latest_year = int(wide_panel["year"].max())
    scenario_targets = ["FX_YOY_DEPRECIATION_PCT", "NY.GDP.MKTP.KD.ZG", "GC.DOD.TOTL.GD.ZS"]

    print("[4/6] Writing SQLite warehouse..." if not args.skip_db else "[4/6] Skipping DB (--skip-db)...")
    if not args.skip_db:
        conn = db_utils.get_connection()
        db_utils.init_schema(conn)
        db_utils.load_countries(conn)
        db_utils.load_indicator_values(conn, long_panel)
        db_utils.load_scores(conn, scores, drivers)
        print(db_utils.top_risk_countries(conn, latest_year))
        conn.close()

    print("[5/6] Generating analyst commentary...")
    commentary_dir = out_dir / "commentary"
    commentary_dir.mkdir(parents=True, exist_ok=True)
    peer_groups = countries_cfg.get("peer_groups", {})

    for iso3 in iso3_codes:
        rows_for_country = wide_panel[wide_panel["country_iso3"] == iso3]
        if rows_for_country.empty:
            continue
        year = int(rows_for_country["year"].max())
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
        (commentary_dir / f"{iso3}_{year}.md").write_text(report)

    print(f"[6/6] Done. Panel, DB, and {len(iso3_codes)} commentary files written under data/processed/.")
    print("Launch the dashboard with: streamlit run dashboard/app.py")


if __name__ == "__main__":
    main()
