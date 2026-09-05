-- Country Risk Intelligence Engine — warehouse schema.
-- Written as plain SQLite-compatible SQL (no vendor-specific extensions),
-- so it also runs unmodified on Postgres if you swap the connection string
-- in db_utils.py for a SQLAlchemy Postgres URL.

CREATE TABLE IF NOT EXISTS countries (
    country_iso3   TEXT PRIMARY KEY,
    name           TEXT NOT NULL,
    region         TEXT,
    income_group   TEXT,
    monetary_union TEXT
);

-- Long-format raw/cleaned indicator values. This is the single source of
-- truth everything else derives from — wide panels, scores, and drivers are
-- all reproducible from this table plus config/indicators.yaml.
CREATE TABLE IF NOT EXISTS indicator_values (
    country_iso3   TEXT NOT NULL REFERENCES countries(country_iso3),
    indicator_code TEXT NOT NULL,
    year           INTEGER NOT NULL,
    value          REAL,
    source         TEXT NOT NULL,
    flag           TEXT,               -- 'ok' | 'out_of_range' | 'unbounded', from cleaning/clean.py
    loaded_at      TEXT DEFAULT (datetime('now')),
    PRIMARY KEY (country_iso3, indicator_code, year)
);

CREATE TABLE IF NOT EXISTS risk_scores (
    country_iso3      TEXT NOT NULL REFERENCES countries(country_iso3),
    year              INTEGER NOT NULL,
    risk_score        REAL,
    risk_band         TEXT,
    data_completeness REAL,            -- fraction of indicator weight populated for this row
    computed_at       TEXT DEFAULT (datetime('now')),
    PRIMARY KEY (country_iso3, year)
);

-- One row per (country, year, indicator) driver contribution — this is what
-- "Main drivers" in the commentary is generated from, kept so a score can
-- always be explained after the fact without recomputation.
CREATE TABLE IF NOT EXISTS score_drivers (
    country_iso3          TEXT NOT NULL,
    year                  INTEGER NOT NULL,
    indicator_code        TEXT NOT NULL,
    category              TEXT,
    z_risk                REAL,
    weighted_contribution REAL,
    FOREIGN KEY (country_iso3, year) REFERENCES risk_scores(country_iso3, year)
);

CREATE INDEX IF NOT EXISTS idx_indicator_values_lookup ON indicator_values (indicator_code, year);
CREATE INDEX IF NOT EXISTS idx_score_drivers_lookup ON score_drivers (country_iso3, year);

-- ---------------------------------------------------------------------------
-- A few example analytical queries — the kind you'd actually run, not just
-- SELECT * demos.
-- ---------------------------------------------------------------------------

-- 1. Current riskiest countries in the latest year available.
-- SELECT c.name, s.risk_score, s.risk_band
-- FROM risk_scores s
-- JOIN countries c ON c.country_iso3 = s.country_iso3
-- WHERE s.year = (SELECT MAX(year) FROM risk_scores)
-- ORDER BY s.risk_score DESC
-- LIMIT 5;

-- 2. Biggest year-over-year score deteriorations (largest positive delta).
-- SELECT curr.country_iso3, curr.year,
--        curr.risk_score - prev.risk_score AS score_delta
-- FROM risk_scores curr
-- JOIN risk_scores prev
--   ON curr.country_iso3 = prev.country_iso3 AND curr.year = prev.year + 1
-- ORDER BY score_delta DESC
-- LIMIT 5;

-- 3. Most common top driver category across the panel in a given year —
--    "what's actually moving global risk this year".
-- SELECT category, COUNT(*) AS times_top_driver
-- FROM (
--     SELECT country_iso3, year, category,
--            ROW_NUMBER() OVER (
--                PARTITION BY country_iso3, year
--                ORDER BY ABS(weighted_contribution) DESC
--            ) AS rnk
--     FROM score_drivers
--     WHERE year = 2025
-- )
-- WHERE rnk = 1
-- GROUP BY category
-- ORDER BY times_top_driver DESC;
