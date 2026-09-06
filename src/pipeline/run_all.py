"""Run the live public-data pipeline and persist a reproducible snapshot."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from src.runtime.live_data import create_wide_panel_from_long, fetch_live_data


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--countries", default=None, help="Optional comma-separated ISO3 filter after collection")
    p.add_argument("--start", type=int, default=2012)
    p.add_argument("--end", type=int, default=None)
    args = p.parse_args()

    long, meta = fetch_live_data(args.start, args.end)
    if args.countries:
        wanted = {x.strip().upper() for x in args.countries.split(",") if x.strip()}
        long = long[long["country_iso3"].isin(wanted)].copy()

    out = Path("data/processed")
    out.mkdir(parents=True, exist_ok=True)
    create_wide_panel_from_long(long).to_csv(out / "panel_wide.csv", index=False)
    (out / "data_metadata.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    print(f"Live observations: {len(long):,}; latest valid analysis year: {meta.get('latest_valid_analysis_year')}")


if __name__ == "__main__":
    main()
