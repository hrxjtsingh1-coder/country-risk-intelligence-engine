"""
End-to-end Country Risk Intelligence Engine pipeline.

Usage:
    python -m src.pipeline.run_all
    python -m src.pipeline.run_all --countries USA,IND,DEU --start 2015 --end 2025
    python -m src.pipeline.run_all --shock POLICY_RATE_YOY_CHANGE_BPS:100
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from datetime import datetime, timezone

import yaml

ROOT = Path(__file__).resolve().parents[2]
sys.path.append(str(ROOT))

from src.cleaning.clean import coverage_report, to_wide_panel
from src.commentary.generate_commentary import generate_report
from src.db import db_utils
from src.indicators.build_panel import build_long_panel
from src.scenario.scenario_engine import run_shock_scenario
from src.scoring.risk_score import score_panel


def _load_countries_cfg() -> dict:
    with open(ROOT / "config" / "countries.yaml", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run the country risk intelligence pipeline end to end."
    )
    parser.add_argument(
        "--countries",
        type=str,
        default=None,
        help="Comma-separated ISO3 codes; default = all configured countries",
    )
    parser.add_argument("--start", type=int, default=2012)
    parser.add_argument("--end", type=int, default=2025)
    parser.add_argument(
        "--shock",
        type=str,
        default="POLICY_RATE_YOY_CHANGE_BPS:100",
        help="driver_code:amount",
    )
    parser.add_argument(
        "--skip-db",
        action="store_true",
        help="Skip writing to the SQLite warehouse",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    cfg = _load_countries_cfg()

    all_iso3 = [str(c["iso3"]) for c in cfg.get("countries", [])]
    iso3_codes = (
        [c.strip().upper() for c in args.countries.split(",") if c.strip()]
        if args.countries
        else all_iso3
    )

    name_lookup = {
        str(c["iso3"]): c["name"]
        for c in cfg.get("countries", [])
    }

    print(
        f"[1/6] Collecting + cleaning indicators for "
        f"{len(iso3_codes)} countries, {args.start}-{args.end}..."
    )

    started_at = datetime.now(timezone.utc)
    long_panel = build_long_panel(iso3_codes, args.start, args.end)

    if long_panel.empty:
        print(
            "No data returned. Check network access to the World Bank/FRED "
            "endpoints and retry."
        )
        sys.exit(1)

    indicator_codes = sorted(long_panel["indicator_code"].unique())
    cov = coverage_report(
        long_panel,
        iso3_codes,
        indicator_codes,
        list(range(args.start, args.end + 1)),
    )
    print(cov.to_string(index=False))

    wide_panel = to_wide_panel(long_panel)

    out_dir = ROOT / "data" / "processed"
    out_dir.mkdir(parents=True, exist_ok=True)
    wide_panel.to_csv(out_dir / "panel_wide.csv", index=False)
    configured_codes = [str(item.get("code")) for item in yaml.safe_load((ROOT / "config" / "indicators.yaml").read_text()).get("indicators", [])]
    metadata = {
        "run_id": started_at.strftime("live-%Y%m%dT%H%M%SZ"),
        "mode": "LIVE",
        "started_at": started_at.isoformat(),
        "completed_at": datetime.now(timezone.utc).isoformat(),
        "retrieved_at": datetime.now(timezone.utc).isoformat(),
        "requested_period": f"{args.start}–{args.end}",
        "latest_available_observation": int(wide_panel["year"].max()),
        "country_count": int(wide_panel["country_iso3"].nunique()),
        "indicator_count": int(len([c for c in configured_codes if c in wide_panel.columns])),
        "observations_received": int(long_panel.shape[0]),
        "observations_missing": int(max(0, len(iso3_codes) * len(configured_codes) * (args.end - args.start + 1) - long_panel.shape[0])),
        "sources": ["World Bank", "FRED (US-only enrichment)"],
        "source_urls": ["https://api.worldbank.org/v2/", "https://fred.stlouisfed.org/graph/fredgraph.csv"],
        "methodology_version": "1.1.0",
        "config_version": "config/indicators.yaml",
    }
    (out_dir / "data_metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    print("[2/6] Scoring...")
    scores, drivers = score_panel(wide_panel)

    print("[3/6] Preparing scenario...")
    driver_code, amount_text = args.shock.split(":", 1)
    amount = float(amount_text)

    latest_year = int(wide_panel["year"].max())

    scenario_targets = [
        "FX_YOY_DEPRECIATION_PCT",
        "NY.GDP.MKTP.KD.ZG",
        "GC.DOD.TOTL.GD.ZS",
    ]

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
        rows_for_country = wide_panel[
            wide_panel["country_iso3"].astype(str).eq(iso3)
        ]

        if rows_for_country.empty:
            continue

        year = int(rows_for_country["year"].max())

        peer_group = next(
            (members for members in peer_groups.values() if iso3 in members),
            None,
        )
        peer_group = [c for c in (peer_group or []) if c != iso3]

        scenario_result = None

        try:
            scenario_result = run_shock_scenario(
                wide_panel,
                iso3,
                year,
                driver_code,
                amount,
                scenario_targets,
            )
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

        (
            commentary_dir / f"{iso3}_{year}.md"
        ).write_text(report, encoding="utf-8")

    print(
        f"[6/6] Done. Panel, DB, and commentary files written "
        f"under data/processed/."
    )
    print("Launch the dashboard with: streamlit run dashboard/app.py")


if __name__ == "__main__":
    main()
