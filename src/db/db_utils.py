"""
Thin SQLite wrapper. Deliberately not an ORM — this project is small enough
that raw SQL (schema.sql) stays readable, and readable SQL is the point of
having the SQL layer at all for a portfolio project.

Swap to Postgres later by changing DB_PATH usage to a SQLAlchemy engine and
pointing `run_query` at that instead — schema.sql has no SQLite-only syntax.
"""
from __future__ import annotations

import sqlite3
from pathlib import Path

import pandas as pd
import yaml

DB_PATH = Path(__file__).resolve().parents[2] / "data" / "processed" / "country_risk.db"
SCHEMA_PATH = Path(__file__).resolve().parent / "schema.sql"
CONFIG_DIR = Path(__file__).resolve().parents[2] / "config"


def get_connection() -> sqlite3.Connection:
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    return sqlite3.connect(DB_PATH)


def init_schema(conn: sqlite3.Connection) -> None:
    with open(SCHEMA_PATH) as f:
        # Strip the trailing commented-out example queries block; sqlite3
        # handles `--` comments fine, but keep executescript input minimal.
        conn.executescript(f.read())


def load_countries(conn: sqlite3.Connection) -> None:
    with open(CONFIG_DIR / "countries.yaml") as f:
        cfg = yaml.safe_load(f)
    rows = [
        (c["iso3"], c["name"], c["region"], c["income"], c.get("monetary_union"))
        for c in cfg["countries"]
    ]
    conn.executemany(
        "INSERT OR REPLACE INTO countries (country_iso3, name, region, income_group, monetary_union) VALUES (?,?,?,?,?)",
        rows,
    )
    conn.commit()


def load_indicator_values(conn: sqlite3.Connection, df_long: pd.DataFrame) -> None:
    cols = ["country_iso3", "indicator_code", "year", "value", "source", "flag"]
    df = df_long[[c for c in cols if c in df_long.columns]].copy()
    for c in cols:
        if c not in df.columns:
            df[c] = None
    df[cols].to_sql("indicator_values", conn, if_exists="append", index=False, method="multi", chunksize=500)
    conn.commit()


def load_scores(conn: sqlite3.Connection, scores: pd.DataFrame, drivers: pd.DataFrame) -> None:
    scores[["country_iso3", "year", "risk_score", "risk_band", "data_completeness"]].to_sql(
        "risk_scores", conn, if_exists="append", index=False, method="multi", chunksize=500
    )
    drivers[["country_iso3", "year", "indicator_code", "category", "z_risk", "weighted_contribution"]].to_sql(
        "score_drivers", conn, if_exists="append", index=False, method="multi", chunksize=500
    )
    conn.commit()


def run_query(conn: sqlite3.Connection, sql: str) -> pd.DataFrame:
    return pd.read_sql_query(sql, conn)


def top_risk_countries(conn: sqlite3.Connection, year: int, n: int = 5) -> pd.DataFrame:
    return run_query(
        conn,
        f"""
        SELECT c.name, c.country_iso3, s.risk_score, s.risk_band
        FROM risk_scores s
        JOIN countries c ON c.country_iso3 = s.country_iso3
        WHERE s.year = {int(year)}
        ORDER BY s.risk_score DESC
        LIMIT {int(n)}
        """,
    )


def clear_run_data(conn: sqlite3.Connection) -> None:
    """Clear derived rows before a fresh pipeline load.

    This keeps the primary-key tables reproducible across repeated pipeline
    executions without changing the schema or analytical formulas.
    """
    conn.execute("DELETE FROM score_drivers")
    conn.execute("DELETE FROM risk_scores")
    conn.execute("DELETE FROM indicator_values")
    conn.commit()
