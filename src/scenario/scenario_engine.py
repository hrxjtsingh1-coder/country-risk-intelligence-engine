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


def _ols(x: pd.Series, y: pd.Series) -> tuple[float, float, int]:
    data = pd.DataFrame({"x": x, "y": y}).replace([np.inf, -np.inf], np.nan).dropna()

    n = len(data)
    if n < 5:
        return float("nan"), float("nan"), n

    xv = data["x"].to_numpy(dtype=float)
    yv = data["y"].to_numpy(dtype=float)

    x_mean = xv.mean()
    y_mean = yv.mean()

    denom = np.sum((xv - x_mean) ** 2)
    if denom == 0:
        return float("nan"), float("nan"), n

    beta = np.sum((xv - x_mean) * (yv - y_mean)) / denom
    alpha = y_mean - beta * x_mean

    fitted = alpha + beta * xv
    ss_res = np.sum((yv - fitted) ** 2)
    ss_tot = np.sum((yv - y_mean) ** 2)

    r2 = 1.0 - ss_res / ss_tot if ss_tot else float("nan")

    return float(beta), float(r2), n


def run_shock_scenario(
    panel: pd.DataFrame,
    country_iso3: str,
    year: int,
    driver_code: str,
    shock_amount: float,
    scenario_targets: list[str],
) -> dict:
    if panel is None or panel.empty:
        raise ValueError("Panel is empty.")

    if driver_code not in panel.columns:
        raise ValueError(
            f"Scenario driver {driver_code} is not present in the panel."
        )

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

        beta, r2, n_obs = _ols(
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
                "baseline_value": (
                    float(baseline_value)
                    if not pd.isna(baseline_value)
                    else float("nan")
                ),
                "estimated_delta": estimated_delta,
                "r_squared": r2,
                "n_obs": n_obs,
            }
        )

    baseline_scores, _ = score_panel(baseline_panel)

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
        code = target_delta["indicator_code"]
        delta = target_delta["estimated_delta"]

        if pd.isna(delta):
            continue

        mask = (
            scenario_panel["country_iso3"].astype(str).eq(str(country_iso3))
            & pd.to_numeric(scenario_panel["year"], errors="coerce").eq(int(year))
        )

        if code in scenario_panel.columns:
            scenario_panel.loc[mask, code] = (
                pd.to_numeric(scenario_panel.loc[mask, code], errors="coerce")
                + float(delta)
            )

    scenario_scores, _ = score_panel(scenario_panel)

    scenario_row = scenario_scores[
        scenario_scores["country_iso3"].astype(str).eq(str(country_iso3))
        & pd.to_numeric(scenario_scores["year"], errors="coerce").eq(int(year))
    ]

    scenario_score = (
        float(scenario_row.iloc[0]["risk_score"])
        if not scenario_row.empty
        else float("nan")
    )

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

    return {
        "driver_code": driver_code,
        "shock_amount": float(shock_amount),
        "baseline_score": baseline_score,
        "baseline_band": baseline_band,
        "scenario_score": scenario_score,
        "scenario_band": scenario_band,
        "delta": scenario_score - baseline_score,
        "indicator_deltas": target_deltas,
    }
