"""Create a deterministic offline panel for dashboard demonstrations.

The generated values are synthetic. They are useful for showcasing the
interface, testing the scoring contract, and developing without network access;
they must not be presented as real economic observations.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[1]
OUTPUT_PATH = ROOT / "data" / "processed" / "panel_wide.csv"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--countries", default=None, help="Comma-separated ISO3 codes")
    parser.add_argument("--start", type=int, default=2015)
    parser.add_argument("--end", type=int, default=2025)
    parser.add_argument(
        "--output",
        default=str(OUTPUT_PATH),
        help="Output CSV path (default: data/processed/panel_wide.csv)",
    )
    return parser.parse_args()


def configured_countries() -> list[dict]:
    with open(ROOT / "config" / "countries.yaml", encoding="utf-8") as handle:
        config = yaml.safe_load(handle) or {}
    return [country for country in config.get("countries", []) if isinstance(country, dict)]


def create_panel(countries: list[dict], start: int, end: int) -> pd.DataFrame:
    years = list(range(start, end + 1))
    rows: list[dict] = []

    for country_index, country in enumerate(countries):
        iso3 = str(country["iso3"]).upper()
        profile = (country_index % 7) - 3

        for year in years:
            elapsed = year - start
            cycle = ((elapsed + country_index) % 5) - 2
            rows.append(
                {
                    "country_iso3": iso3,
                    "year": year,
                    "FP.CPI.TOTL.ZG": round(3.0 + profile * 0.35 + cycle * 0.18, 3),
                    "NY.GDP.MKTP.KD.ZG": round(4.0 - profile * 0.28 - cycle * 0.12, 3),
                    "SL.UEM.TOTL.ZS": round(6.0 + profile * 0.7 + cycle * 0.2, 3),
                    "GC.DOD.TOTL.GD.ZS": round(52.0 + profile * 9.0 + elapsed * 0.45, 3),
                    "BN.CAB.XOKA.GD.ZS": round(1.5 - profile * 0.6 + cycle * 0.15, 3),
                    "FI.RES.TOTL.MO": round(4.5 - profile * 0.22 + cycle * 0.08, 3),
                    "DT.DOD.DECT.GN.ZS": round(38.0 + profile * 7.0 + elapsed * 0.25, 3),
                    "FB.AST.NPER.ZS": round(2.8 + profile * 0.35 + cycle * 0.1, 3),
                    "FX_YOY_DEPRECIATION_PCT": round(2.0 + profile * 1.1 + cycle * 0.25, 3),
                    "POLICY_RATE_YOY_CHANGE_BPS": round(35.0 + profile * 7.0 + cycle * 4.0, 3),
                }
            )

    return pd.DataFrame(rows).sort_values(["country_iso3", "year"]).reset_index(drop=True)


def main() -> None:
    args = parse_args()
    if args.start > args.end:
        raise SystemExit("--start must be less than or equal to --end")

    configured = configured_countries()
    requested = (
        [item.strip().upper() for item in args.countries.split(",") if item.strip()]
        if args.countries
        else [str(country["iso3"]).upper() for country in configured]
    )
    by_iso3 = {str(country["iso3"]).upper(): country for country in configured}
    unknown = [iso3 for iso3 in requested if iso3 not in by_iso3]
    if unknown:
        raise SystemExit(f"Unknown configured country code(s): {', '.join(unknown)}")

    panel = create_panel([by_iso3[iso3] for iso3 in requested], args.start, args.end)
    output_path = Path(args.output)
    if not output_path.is_absolute():
        output_path = ROOT / output_path
    output_path.parent.mkdir(parents=True, exist_ok=True)
    panel.to_csv(output_path, index=False)
    print(
        f"Wrote synthetic demo panel: {len(panel):,} rows, "
        f"{len(requested)} countries, {args.start}-{args.end} -> {output_path}"
    )


if __name__ == "__main__":
    main()