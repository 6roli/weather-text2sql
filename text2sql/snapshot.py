"""A frozen copy of the warehouse's gold layer, so training data and evaluation are reproducible.

The live bucket changes every hour (the forecast is replaced, today and yesterday are rebuilt), so a question
like "how much will it rain tomorrow?" has a different answer each hour. Everything here runs against
data/snapshot/*.parquet instead, downloaded once with:

    python -m text2sql.snapshot
"""
from __future__ import annotations

import argparse
from datetime import date, timedelta
from pathlib import Path

import duckdb

BUCKET = "https://huggingface.co/buckets/groli/weather-warehouse/resolve"
SNAPSHOT_DIR = Path(__file__).resolve().parent.parent / "data" / "snapshot"
TABLES = ("latest_forecast", "location_station", "daily_summary", "forecast_vs_observed")
PARTITIONED = ("daily_summary", "forecast_vs_observed")


def connect(snapshot_dir: Path = SNAPSHOT_DIR) -> duckdb.DuckDBPyConnection:
    """In-memory DuckDB with one view per gold table, as the model sees them."""
    connection = duckdb.connect()
    for table in TABLES:
        connection.execute(f"CREATE VIEW {table} AS SELECT * FROM read_parquet('{(snapshot_dir / table).as_posix()}.parquet')")
    return connection


def download(first: date, last: date, snapshot_dir: Path = SNAPSHOT_DIR) -> None:
    snapshot_dir.mkdir(parents=True, exist_ok=True)
    connection = duckdb.connect()
    for table in TABLES:
        if table in PARTITIONED:
            days = [first + timedelta(days=i) for i in range((last - first).days + 1)]
            source = "[" + ", ".join(f"'{BUCKET}/gold/{table}/date={d.isoformat()}/data.parquet'" for d in days) + "]"
        else:
            source = f"'{BUCKET}/gold/{table}/data.parquet'"
        # forecast_vs_observed has a `date` column, daily_summary too: union_by_name keeps a single one
        output = (snapshot_dir / f"{table}.parquet").as_posix()
        connection.execute(
            f"COPY (SELECT * FROM read_parquet({source}, union_by_name = true)) TO '{output}' (FORMAT parquet, COMPRESSION zstd)"
        )
        rows = connection.execute(f"SELECT count(*) FROM '{output}'").fetchone()[0]
        print(f"{table}: {rows} rows")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--first", type=date.fromisoformat, default=date(2026, 10, 18))
    parser.add_argument("--last", type=date.fromisoformat, default=date(2026, 10, 25))
    args = parser.parse_args()
    download(args.first, args.last)


if __name__ == "__main__":
    main()
