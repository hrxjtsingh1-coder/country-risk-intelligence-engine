"""Reproducibility manifest + signoff pack.

Any number shipped by this system is only meaningful if it can be re-derived
from committed inputs, and if anyone can tell at a glance whether the shipped
artifacts still match those inputs. That's what this module is for:

    python -m src.governance.manifest --build --signed-by "Analyst Name"
    python -m src.governance.manifest --verify

`--build` fingerprints every committed input (config yamls) and fixture
(data/demo/panel_wide.csv), plus any processed outputs present, records the
methodology version, the environment's python + dependency versions, and the
state of the model at that moment (coverage verdicts + backtest detection
rate), then attaches a metadata-only signoff record (`signed_by`,
`signed_at`, `notes`). `--verify` recomputes the hashes and reports per-file
ok/drifted/missing with an overall verdict — so a stale artifact is detected,
not silently trusted.

Signoff is a structured record, deliberately not cryptographic: the project
has no key infrastructure, and the audit value here is the deterministic
input->artifact binding plus an explicit human sign-off line.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from importlib import metadata
from pathlib import Path
from typing import Any

import pandas as pd

from src.analysis.backtest import backtest_summary, run_backtest
from src.governance.coverage import verify_model_coverage
from src.scoring import risk_score as rs

ROOT = Path(__file__).resolve().parents[2]

METHODOLOGY_VERSION = "2.0.0"  # sector (super-pillar) layer + full backtest harness

MANIFEST_PATH = ROOT / "data" / "processed" / "manifest.json"
FORMAT_ID = "country-risk-manifest"
SCHEMA_VERSION = 1

DEFAULT_INPUTS = [
    ROOT / "config" / "indicators.yaml",
    ROOT / "config" / "countries.yaml",
    ROOT / "config" / "episodes.yaml",
]
DEFAULT_FIXTURES = [ROOT / "data" / "demo" / "panel_wide.csv"]
DEFAULT_OUTPUTS = [
    ROOT / "data" / "processed" / "panel_wide.csv",
    ROOT / "data" / "processed" / "data_metadata.json",
]

ENV_PACKAGES = ["streamlit", "pandas", "numpy", "plotly", "PyYAML", "requests"]


@dataclass
class FileStatus:
    path: str
    present: bool
    hash_matches: bool | None
    status: str  # "ok" | "drifted" | "missing" | "added"


@dataclass
class Verification:
    overall: str  # "consistent" | "drifted" | "inconsistent"
    files: list[FileStatus]
    built_at: str
    methodology_version: str


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _file_entry(path: Path) -> dict[str, str | int]:
    return {"sha256": _sha256(path), "size_bytes": path.stat().st_size}


def _collect_files(root: Path, paths: Sequence[Path]) -> dict[str, dict[str, str | int]]:
    entries: dict[str, dict[str, str | int]] = {}
    for path in paths:
        if not path.exists():
            continue
        resolved = path.resolve()
        try:
            rel = str(resolved.relative_to(root))
        except ValueError:
            rel = str(resolved)  # outside the repo root; keep the absolute path
        entries[rel] = _file_entry(path)
    return dict(sorted(entries.items()))


def _env_fingerprint() -> dict[str, str | int]:
    env: dict[str, str | int] = {
        "python": sys.version.split()[0],
        "python_version_tuple": ".".join(str(v) for v in sys.version_info[:3]),
    }
    for package in ENV_PACKAGES:
        try:
            env[package] = metadata.version(package)
        except metadata.PackageNotFoundError:
            env[package] = "not-installed"
    return env


def _load_scoring_weight() -> float:
    return float(rs._load_scoring().get("sector_composite_weight", 0.0))


def _model_state(panel: pd.DataFrame, dataset_mode: str, scores: pd.DataFrame | None = None) -> dict[str, Any]:
    coverage = verify_model_coverage(panel, panel_source=dataset_mode)
    if scores is None:
        scores, _drivers, _pillars = rs.score_panel(panel)
    bt_results = run_backtest(scores, data_is_synthetic=(dataset_mode == "demo"))
    bt_summary = backtest_summary(bt_results)
    return {
        "methodology_version": METHODOLOGY_VERSION,
        "sector_composite_weight": _load_scoring_weight(),
        "countries_in_panel": coverage.countries,
        "coverage": {
            "status": coverage.status,
            "checks": coverage.checks,
            "indicator_covered": coverage.indicator_covered,
            "indicator_total": coverage.indicator_total,
            "error_count": sum(1 for i in coverage.issues if i.severity == "error"),
            "warning_count": sum(1 for i in coverage.issues if i.severity == "warning"),
        },
        "backtest": {
            "episodes_checked": bt_summary.episodes_total,
            "flagged": bt_summary.flagged,
            "missed": bt_summary.missed,
            "inconclusive": bt_summary.inconclusive,
            "detection_rate": bt_summary.detection_rate,
            "median_peak_delta": bt_summary.median_peak_delta,
        },
    }


def build_manifest(
    *,
    inputs: Sequence[Path] = DEFAULT_INPUTS,
    outputs: Sequence[Path] = DEFAULT_OUTPUTS,
    panel: pd.DataFrame | None = None,
    scores: pd.DataFrame | None = None,
    dataset_mode: str = "demo",
    signed_by: str = "",
    notes: str = "",
    manifest_path: Path = MANIFEST_PATH,
) -> dict[str, Any]:
    """Fingerprint committed inputs/fixtures, attach model + env + signoff, write JSON."""
    fixture = DEFAULT_FIXTURES if dataset_mode == "demo" else []
    used_panel = panel if panel is not None else pd.read_csv(DEFAULT_FIXTURES[0])

    manifest = {
        "format": FORMAT_ID,
        "schema_version": SCHEMA_VERSION,
        "methodology_version": METHODOLOGY_VERSION,
        "dataset_mode": dataset_mode,
        "built_at": datetime.now(UTC).isoformat(),
        "files": _collect_files(ROOT, [*inputs, *fixture, *outputs]),
        "env": _env_fingerprint(),
        "model": _model_state(used_panel, dataset_mode, scores),
        "signoff": {"signed_by": signed_by, "signed_at": datetime.now(UTC).isoformat(), "notes": notes},
    }
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return manifest


def verify_manifest(manifest_path: Path = MANIFEST_PATH) -> Verification:
    """Recompute hashes of the manifest's recorded files and diff them."""
    if not manifest_path.exists():
        raise FileNotFoundError(f"Manifest not found: {manifest_path}")

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    recorded = manifest.get("files", {})

    statuses: list[FileStatus] = []
    for rel, entry in recorded.items():
        path = ROOT / rel
        present = path.exists()
        matches: bool | None = None
        if present:
            matches = _sha256(path) == entry["sha256"]
        status = "ok" if present and matches else ("drifted" if present else "missing")
        statuses.append(FileStatus(path=str(rel), present=present, hash_matches=matches, status=status))

    missing = sorted(p for p in recorded if not (ROOT / p).exists())
    drifted = sorted(p for p, s in zip(recorded, statuses, strict=False) if s.status == "drifted")
    overall = "consistent" if not missing and not drifted else ("drifted" if drifted else "inconsistent")

    return Verification(
        overall=overall,
        files=statuses,
        built_at=str(manifest.get("built_at", "")),
        methodology_version=str(manifest.get("methodology_version", "")),
    )


def _text_verdict(verification: Verification) -> str:
    lines = [
        "REPRODUCIBILITY VERIFICATION",
        "=" * 72,
        f"Manifest: {verification.methodology_version} · built {verification.built_at}",
        f"Overall: {verification.overall.upper()}",
        "-" * 72,
        f"{'FILE':<58}{'STATUS':<10}",
        "-" * 72,
    ]
    for file_status in verification.files:
        lines.append(f"{file_status.path:<58}{file_status.status:<10}")
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Reproducibility manifest + signoff pack.")
    sub = parser.add_subparsers(dest="command", required=True)

    build = sub.add_parser("build", help="Build the manifest for the current model/data state.")
    build.add_argument("--signed-by", default="", help="Analyst / responsible party who signs this run.")
    build.add_argument("--notes", default="", help="Free-text notes attached to the signoff.")
    build.add_argument("--mode", choices=["demo", "live"], default="demo", help="Dataset mode.")
    build.add_argument("--panel", default=None, help="Wide panel CSV path (default: demo fixture).")
    build.add_argument("--output", default=str(MANIFEST_PATH))

    verify = sub.add_parser("verify", help="Verify a built manifest against current files.")
    verify.add_argument("--manifest", default=str(MANIFEST_PATH))

    args = parser.parse_args(argv)

    if args.command == "build":
        panel = None
        if args.panel:
            panel = pd.read_csv(args.panel)
        manifest = build_manifest(
            panel=panel,
            dataset_mode=args.mode,
            signed_by=args.signed_by,
            notes=args.notes,
            manifest_path=Path(args.output),
        )
        print(
            f"Built manifest: {args.output}\n"
            f"  methodology {manifest['methodology_version']} · mode {manifest['dataset_mode']}\n"
            f"  {len(manifest['files'])} files fingerprinted\n"
            f"  coverage={manifest['model']['coverage']['status']} · "
            f"backtest detection_rate={manifest['model']['backtest']['detection_rate']}\n"
            f"  signed_by: {manifest['signoff']['signed_by'] or '(unsigned)'} at {manifest['signoff']['signed_at']}"
        )
        return 0

    if args.command == "verify":
        try:
            verification = verify_manifest(Path(args.manifest))
        except FileNotFoundError as exc:
            print(str(exc), file=sys.stderr)
            return 2
        print(_text_verdict(verification))
        return 0 if verification.overall == "consistent" else 1

    return 2


if __name__ == "__main__":
    raise SystemExit(main())
