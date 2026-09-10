"""
Transparent what-if scenario engine.

The scenario estimates pooled-panel linear relationships between a selected
shock driver and configured target indicators. It then recomputes the score
after applying estimated target deltas to the selected country-year.

This is a sensitivity analysis, not a causal structural model.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.scoring.risk_score import score_panel

SHOCK_LIBRARY: dict[str, dict] = {
    "rate_hike": {
        "name": "Rate shock (+100bps)",
        "driver_code": "POLICY_RATE_YOY_CHANGE_BPS",
        "default_shock_amount": 100.0,
        "scenario_targets": ["FX_YOY_DEPRECIATION_PCT", "NY.GDP.MKTP.KD.ZG", "GC.DOD.TOTL.GD.ZS"],
        "display_unit": "bps",
        "description": (
            "Monetary tightening: +100 bps policy-rate YoY change. The panel is "
            "annual, so an N-quarter hiking campaign is expressed as its "
            "annualized year-over-year policy-rate change."
        ),
    },
    "growth_slowdown": {
        "name": "Growth shock (−2pp GDP)",
        "driver_code": "NY.GDP.MKTP.KD.ZG",
        "default_shock_amount": -2.0,
        "scenario_targets": ["SL.UEM.TOTL.ZS", "GC.NLD.TOTL.GD.ZS", "BN.CAB.XOKA.GD.ZS"],
        "display_unit": "pp",
        "description": (
            "Demand shock: real GDP growth falls 2 percentage points from the "
            "baseline, transmitting into labour markets, the fiscal balance and "
            "the external position."
        ),
    },
    "fx_devaluation": {
        "name": "FX shock (20% devaluation)",
        "driver_code": "FX_YOY_DEPRECIATION_PCT",
        "default_shock_amount": 20.0,
        "scenario_targets": ["FP.CPI.TOTL.ZG", "FI.RES.TOTL.MO", "DT.DOD.DECT.GN.ZS", "NE.RSB.GNFS.ZS"],
        "display_unit": "pp",
        "description": (
            "Currency shock: a 20pp year-over-year depreciation, transmitting "
            "into import prices, reserves coverage and external-debt service."
        ),
    },
    "commodity_collapse": {
        "name": "Commodity shock (exporters)",
        "driver_code": "COMMODITY_PRICE_INDEX_PCT",
        "default_shock_amount": -20.0,
        "scenario_targets": ["BN.CAB.XOKA.GD.ZS", "NE.RSB.GNFS.ZS", "FX_YOY_DEPRECIATION_PCT"],
        "display_unit": "%",
        "description": (
            "Commodity shock: a 20% fall in export commodity prices, hitting "
            "commodity exporters' external and fiscal positions. The driver is a "
            "commodity-price index series; no such series is collected in "
            "config/indicators.yaml yet, so this preset runs only once that "
            "series exists in the panel."
        ),
    },
    "banking_stress": {
        "name": "Banking stress (credit gap)",
        "driver_code": "BIS_CREDIT_GAP",
        "default_shock_amount": 10.0,
        "scenario_targets": ["FB.AST.NPER.ZS", "NY.GDP.MKTP.KD.ZG", "FX_YOY_DEPRECIATION_PCT"],
        "display_unit": "pp",
        "description": (
            "Banking stress: the credit-to-GDP gap widens 10pp (a BIS-calculated "
            "early-warning signal), transmitting into loan quality, growth and "
            "financing conditions. Requires the BIS_CREDIT_GAP series in the panel."
        ),
    },
}


def available_shock_presets() -> list[str]:
    """Names of the built-in named shock presets (SHOCK_LIBRARY keys)."""
    return list(SHOCK_LIBRARY)


def shock_preset(name: str) -> dict:
    """Return a copy of a named preset, raising a clear error if unknown."""
    if name not in SHOCK_LIBRARY:
        raise ValueError(f"Unknown scenario preset '{name}'. Known presets: {', '.join(available_shock_presets())}.")
    return dict(SHOCK_LIBRARY[name])


def _ols(x: pd.Series, y: pd.Series) -> tuple[float, float, int, float, float]:
    data = pd.DataFrame({"x": x, "y": y}).replace([np.inf, -np.inf], np.nan).dropna()

    n = len(data)
    if n < 5:
        return float("nan"), float("nan"), n, float("nan"), float("nan")

    xv = data["x"].to_numpy(dtype=float)
    yv = data["y"].to_numpy(dtype=float)

    x_mean = xv.mean()
    y_mean = yv.mean()

    denom = np.sum((xv - x_mean) ** 2)
    if denom == 0:
        return float("nan"), float("nan"), n, float(xv.min()), float(xv.max())

    beta = np.sum((xv - x_mean) * (yv - y_mean)) / denom
    alpha = y_mean - beta * x_mean

    fitted = alpha + beta * xv
    ss_res = np.sum((yv - fitted) ** 2)
    ss_tot = np.sum((yv - y_mean) ** 2)

    r2 = 1.0 - ss_res / ss_tot if ss_tot else float("nan")

    return float(beta), float(r2), n, float(xv.min()), float(xv.max())


def _information_assessment(models: list[dict]) -> str:
    """Transparent information-quality label, not statistical confidence."""
    usable = [m for m in models if not np.isnan(m["estimated_delta"])]
    if not usable:
        return "INSUFFICIENT DATA"
    median_r2 = float(np.nanmedian([m["r_squared"] for m in usable]))
    min_n = min(m["n_obs"] for m in usable)
    if min_n >= 80 and median_r2 >= 0.35:
        return "HIGH INFORMATION"
    if min_n >= 30 and median_r2 >= 0.10:
        return "MODERATE INFORMATION"
    return "LOW INFORMATION"


def run_shock_scenario(
    panel: pd.DataFrame,
    country_iso3: str,
    year: int,
    driver_code: str | None = None,
    shock_amount: float = 0.0,
    scenario_targets: list[str] | None = None,
    preset: str | None = None,
) -> dict:
    if panel is None or panel.empty:
        raise ValueError("Panel is empty.")

    preset_name = None

    if preset is not None:
        resolved_preset = shock_preset(preset)
        preset_name = str(resolved_preset["name"])
        driver_code = str(resolved_preset["driver_code"])
        shock_amount = float(resolved_preset["default_shock_amount"])
        scenario_targets = list(resolved_preset["scenario_targets"])

    if not driver_code:
        raise ValueError("A shock driver_code or a named preset must be supplied.")

    if driver_code not in panel.columns:
        raise ValueError(f"Scenario driver {driver_code} is not present in the panel.")

    scenario_targets = list(scenario_targets or [])

    selected = panel[
        panel["country_iso3"].astype(str).eq(str(country_iso3))
        & pd.to_numeric(panel["year"], errors="coerce").eq(int(year))
    ].copy()

    if selected.empty:
        raise ValueError("Selected country-year is not present in the panel.")

    baseline_panel = panel.copy()

    target_deltas = []

    for target in scenario_targets:
        if target not in baseline_panel.columns:
            continue

        beta, r2, n_obs, observed_min, observed_max = _ols(
            baseline_panel[driver_code],
            baseline_panel[target],
        )

        if np.isnan(beta):
            estimated_delta = float("nan")
        else:
            estimated_delta = beta * float(shock_amount)

        baseline_value = pd.to_numeric(
            selected.iloc[0][target],
            errors="coerce",
        )

        target_deltas.append(
            {
                "indicator_code": target,
                "baseline_value": (float(baseline_value) if not pd.isna(baseline_value) else float("nan")),
                "estimated_delta": estimated_delta,
                "r_squared": r2,
                "n_obs": n_obs,
                "observed_driver_min": observed_min,
                "observed_driver_max": observed_max,
                "shocked_driver_value": float(pd.to_numeric(selected.iloc[0][driver_code], errors="coerce"))
                + float(shock_amount),
            }
        )

    baseline_scores, _, _ = score_panel(baseline_panel)

    base_row = baseline_scores[
        baseline_scores["country_iso3"].astype(str).eq(str(country_iso3))
        & pd.to_numeric(baseline_scores["year"], errors="coerce").eq(int(year))
    ]

    if base_row.empty or pd.isna(base_row.iloc[0]["risk_score"]):
        raise ValueError("No baseline score is available for the selected slice.")

    baseline_score = float(base_row.iloc[0]["risk_score"])
    baseline_band = str(base_row.iloc[0]["risk_band"])

    scenario_panel = baseline_panel.copy()

    for target_delta in target_deltas:
        code = str(target_delta.get("indicator_code", ""))
        raw_delta = target_delta.get("estimated_delta")
        if not code or raw_delta is None:
            continue
        if not isinstance(raw_delta, (int, float, np.number)):
            continue
        if not np.isfinite(float(raw_delta)):
            continue
        delta = float(raw_delta)

        mask = scenario_panel["country_iso3"].astype(str).eq(str(country_iso3)) & pd.to_numeric(
            scenario_panel["year"], errors="coerce"
        ).eq(int(year))

        if code in scenario_panel.columns:
            scenario_panel.loc[mask, code] = pd.to_numeric(scenario_panel.loc[mask, code], errors="coerce") + delta

    scenario_scores, _, _ = score_panel(scenario_panel)

    scenario_row = scenario_scores[
        scenario_scores["country_iso3"].astype(str).eq(str(country_iso3))
        & pd.to_numeric(scenario_scores["year"], errors="coerce").eq(int(year))
    ]

    scenario_score = float(scenario_row.iloc[0]["risk_score"]) if not scenario_row.empty else float("nan")

    def band(score):
        if np.isnan(score):
            return "Unavailable"
        if score < 20:
            return "Low"
        if score < 40:
            return "Moderate"
        if score < 60:
            return "Elevated"
        if score < 80:
            return "High"
        return "Severe"

    scenario_band = band(scenario_score)

    baseline_driver = float(pd.to_numeric(selected.iloc[0][driver_code], errors="coerce"))
    shocked_driver = baseline_driver + float(shock_amount)
    observed_driver = (
        pd.to_numeric(baseline_panel[driver_code], errors="coerce").replace([np.inf, -np.inf], np.nan).dropna()
    )
    out_of_sample = bool(
        not observed_driver.empty and (shocked_driver < observed_driver.min() or shocked_driver > observed_driver.max())
    )
    return {
        "preset": preset_name,
        "driver_code": driver_code,
        "shock_amount": float(shock_amount),
        "baseline_score": baseline_score,
        "baseline_band": baseline_band,
        "scenario_score": scenario_score,
        "scenario_band": scenario_band,
        "delta": scenario_score - baseline_score,
        "indicator_deltas": target_deltas,
        "baseline_driver_value": baseline_driver,
        "shocked_driver_value": shocked_driver,
        "estimation_window": f"{int(pd.to_numeric(panel['year'], errors='coerce').min())}–{int(pd.to_numeric(panel['year'], errors='coerce').max())}",
        "model_specification": "Pooled-panel bivariate OLS: target = alpha + beta × shock driver; no causal controls or lags.",
        "information_assessment": _information_assessment(target_deltas),
        "out_of_sample_shock": out_of_sample,
    }
