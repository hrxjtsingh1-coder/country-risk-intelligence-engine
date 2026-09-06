# Country Risk Intelligence Engine

A public-facing macroeconomic risk research application that turns official public data into transparent **relative country-risk positioning**, driver attribution, global comparison, and exploratory historical sensitivity analysis.

> **Responsible use:** this is a research/education prototype. It is not a credit rating, probability-of-default model, investment recommendation, or professional financial advice.

## Live application

The Streamlit entrypoint automatically attempts to retrieve official public data at runtime. It does not require a pre-generated local `data/processed/panel_wide.csv` to show live analysis.

Live data is cached for six hours. Each successful live run exposes the retrieval timestamp, latest valid analysis year, observation coverage, and source status.

If a source cannot be verified, the application does **not** silently substitute synthetic observations. It presents Retry and an explicit Open Demo Dataset action.

## Product experience

The interface is built around an analyst workflow rather than a generic dashboard:

- **Global Risk Pulse:** an animated world map across recent model years.
- **Executive view:** score, risk band, year-over-year movement, peer position, and score completeness in one place.
- **Why is the score here?:** largest contributions translated from technical indicator codes into human-readable signals.
- **Peer context:** global, advanced-economy, and emerging-market comparison views.
- **Scenario Laboratory:** hypothetical shocks expressed as historical sensitivity, with model-quality and out-of-sample diagnostics.
- **Evidence & Methodology:** data quality, indicator dictionary, model card, and downloadable outputs.
- **Live / Demo separation:** synthetic data is explicit and never silently mixed with live public data.
- **Mobile-aware design:** charts and analytical sections stack cleanly on narrow screens.
- **Controlled charts:** hover and animation are retained, while zoom and pan are intentionally disabled.

## Data sources

- [World Bank Indicators API](https://api.worldbank.org/v2/) for the global annual macroeconomic panel.
- [FRED DFF](https://fred.stlouisfed.org/series/DFF) as a **US-only enrichment** for changes in the annual-average Effective Federal Funds Rate. It is excluded from the composite score and is not presented as a global policy-rate series.

BIS and ECB are not advertised because the collectors are not implemented in this version.

## Methodology

### Relative risk score

For each indicator and year, the engine computes a cross-sectional z-score:

`z(i,t) = (value(i,t) - mean(value(*,t))) / std(value(*,t))`

Configured directionality puts indicators on a common risk orientation. Weights are applied and renormalized when observations are missing. The resulting signal is mapped around a 50-point midpoint and bounded to 0–100.

The score is therefore **relative positioning within the available country panel**. A score of 70 is not a 70% probability of default.

### Scenario analysis

The Scenario Laboratory uses pooled-panel bivariate OLS as a **historical sensitivity approximation**:

`target = alpha + beta × shock driver`

Results report baseline and shocked scores, the score delta, estimated target deltas, R², observations, estimation window, information assessment, and whether the shock is outside the observed historical range. The output is not a causal estimate or forecast.

## Live data architecture

```text
Official public APIs
        ↓
Runtime collection with retries
        ↓
Validation + bounds + deduplication
        ↓
Canonical country-year panel
        ↓
Deterministic scoring / drivers / scenarios
        ↓
Public Streamlit research interface
```

World Bank observations are fetched in global indicator batches rather than issuing one request per country/indicator. Optional FRED enrichment is isolated from the core World Bank path.

## Demo mode

`data/demo/panel_wide.csv` is a tracked deterministic synthetic fixture for interface and methodology demonstration. It is deliberately labelled and uses the same analytical functions as live mode.

## Repository layout

```text
config/                         model and country metadata
data/demo/                      synthetic showcase fixture
dashboard/public_app.py        public Streamlit research interface
dashboard/app.py               legacy/alternate dashboard retained for comparison
src/runtime/live_data.py       runtime World Bank/FRED collector and validation
src/scoring/                   deterministic relative-risk scoring
src/scenario/                  historical-sensitivity scenarios
src/commentary/                traceable commentary
src/indicators/build_panel.py  local compatibility wrapper
src/pipeline/run_all.py        local live batch pipeline
tests/                         deterministic analytical/runtime tests
streamlit_app.py               Streamlit Cloud entrypoint
```

## Run locally

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
streamlit run streamlit_app.py
```

For an offline interface preview, choose **Open Demo Dataset** in the application. For the local batch collector:

```bash
python -m src.pipeline.run_all --start 2012 --end 2025
```

## Testing

```bash
pytest -q
python -m compileall -q dashboard src scripts streamlit_app.py
```

GitHub Actions runs the automated test suite on pushes and pull requests.

## Limitations

- Public annual indicators have publication lags and can be revised.
- Country coverage differs by indicator.
- Cross-sectional scores depend on the comparison panel and can move when panel composition changes.
- Missing indicators change effective weights and may conceal unmeasured vulnerabilities.
- The scenario model is intentionally simple and exploratory.
- The FRED policy-rate enrichment is US-only.

## Portfolio intent

The project is designed to demonstrate the intersection of **economics + risk methodology + data engineering + software engineering + analytical communication**. The point is not to imitate institutional production infrastructure; it is to make the data path, model assumptions, diagnostics, limitations, and user experience inspectable.
