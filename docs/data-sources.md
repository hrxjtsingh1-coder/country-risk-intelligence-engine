# Official data sources

The ingestion layer now has direct adapters for five official providers:

- World Bank Indicators API
- IMF DataMapper API v2
- BIS Statistics SDMX API
- Eurostat Statistics API
- OECD SDMX REST API

No API keys are configured for the public endpoints used by the smoke tests.

Run the provider checks with:

```bash
python -m src.ingestion.smoke_test
```

The adapters are deliberately separate from `src/indicators`, scoring, and the dashboard. This first pass proves provider connectivity and keeps source identifiers explicit. Integration into the existing panel/risk calculations is a separate step after the live checks pass.
