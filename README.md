# 🌌 Country Risk Intelligence Engine

A production-style, deterministic country-risk intelligence platform built around a
country × year macroeconomic panel.

## What it does

```text
Country configuration
        ↓
Public-data collection
        ↓
Cleaning / validation
        ↓
Indicator panel
        ↓
Cross-sectional risk scoring
        ↓
Driver decomposition
        ↓
Scenario sensitivity
        ↓
SQLite warehouse + CSV
        ↓
Analyst commentary
        ↓
Streamlit intelligence cockpit
```

## Dashboard

The interface includes:

- 🌌 animated dark fintech / terminal presentation
- 📊 0–100 risk score gauge
- 📈 historical risk trajectory
- 🧩 driver decomposition
- 🌍 peer comparison
- 🧪 scenario laboratory
- 📡 indicator signal board
- 🧠 deterministic analyst intelligence
- 📋 metadata and coverage
- 📥 CSV export
- 🔄 refresh controls
- 📱 responsive layout

The dashboard is intentionally a presentation layer over the analytical engine:
`score_panel`, `top_drivers`, `run_shock_scenario`, and `generate_report` remain
the analytical source of truth.

## Run

Install dependencies:

```bash
pip install -r requirements.txt
```

Build the panel:

```bash
python -m src.pipeline.run_all
```

Or test a smaller universe first:

```bash
python -m src.pipeline.run_all --countries USA,IND,DEU --start 2015 --end 2025
```

Launch:

```bash
streamlit run dashboard/app.py
```

The pipeline writes:

```text
data/processed/panel_wide.csv
data/processed/country_risk.db
data/processed/commentary/<ISO3>_<YEAR>.md
```

## Data sources

The core collector uses the World Bank Indicators API for annual macroeconomic
series. The US policy-rate proxy uses the Federal Funds Effective Rate from FRED.
Optional external-source slots are deliberately fault-tolerant: unavailable
series remain missing rather than being fabricated.

## Model transparency

Risk scoring is deterministic and configured in `config/indicators.yaml`.

The score uses cross-sectional z-scores within each year, direction-adjusted
risk signals, configured weights, and available-weight renormalization.

Scenario analysis is a pooled-panel sensitivity calculation. It should be read
as a what-if stress test, not as a causal forecast or credit rating.

## Repository structure

```text
country-risk-intelligence-engine/
├── .streamlit/
│   └── config.toml
├── dashboard/
│   └── app.py
├── src/
│   ├── cleaning/
│   │   └── clean.py
│   ├── commentary/
│   │   └── generate_commentary.py
│   ├── db/
│   │   ├── db_utils.py
│   │   └── schema.sql
│   ├── indicators/
│   │   └── build_panel.py
│   ├── pipeline/
│   │   └── run_all.py
│   ├── scenario/
│   │   └── scenario_engine.py
│   └── scoring/
│       └── risk_score.py
├── config/
│   ├── countries.yaml
│   └── indicators.yaml
├── data/
│   └── processed/
├── requirements.txt
└── README.md
```
