"""Deterministic commentary whose claims can be traced to model outputs."""
from __future__ import annotations
import pandas as pd


def generate_report(country_name,country_iso3,year,scores,drivers,scenario_result=None,peer_group=None,n_drivers=3):
    row=scores[(scores["country_iso3"]==country_iso3)&(scores["year"]==year)]
    if row.empty or pd.isna(row.iloc[0]["risk_score"]):
        return f"No sufficient data to score {country_name} for {year}."
    r=row.iloc[0]
    d=drivers[(drivers["country_iso3"]==country_iso3)&(drivers["year"]==year)].copy()
    d=d.reindex(d["weighted_contribution"].abs().sort_values(ascending=False).index).head(n_drivers)
    bullets="\n".join(f"- {x['label']}: {x['weighted_contribution']:+.2f} contribution" for _,x in d.iterrows()) or "- Insufficient driver coverage."
    lines=[f"**{country_name} — {r['risk_score']:.1f}/100 ({r['risk_band']})**","","**Main drivers:**",bullets,"","**Analyst view:**",f"The model places {country_name} in the {str(r['risk_band']).lower()} relative-risk band for {year}.","","**Limitation:**","This is a relative-positioning research signal, not a credit rating, forecast, probability of default, or investment recommendation."]
    if scenario_result and pd.notna(scenario_result.get("delta")):
        lines += ["","**Scenario:**",f"Historical sensitivity: {scenario_result['baseline_score']:.1f} → {scenario_result['scenario_score']:.1f} ({scenario_result['delta']:+.1f} pts)."]
    return "\n".join(lines)
