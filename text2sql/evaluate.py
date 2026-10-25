"""Execution accuracy: a predicted query is correct when it returns the same data as the reference query.

Comparing results instead of SQL text accepts any correct way of writing the query. The rules:
- row order is ignored, and floats are compared rounded to 2 decimals;
- column names and order are ignored, and extra columns are allowed, as long as every reference column
  appears in the prediction with the same values row by row.

Score a predictions file (JSONL with `question`, `today`, `sql` and `prediction`) with:

    python -m text2sql.evaluate predictions.jsonl
"""
from __future__ import annotations

import argparse
import json
import threading
from collections import Counter
from dataclasses import dataclass
from datetime import date, datetime
from decimal import Decimal
from pathlib import Path

import duckdb

from .schema import CAPITALS
from .snapshot import connect

MAX_ROWS = 5000
TIMEOUT_S = 10.0
# a capital may come back as its id or its name depending on the table; both mean the same answer
NAME_TO_ID = {names[0]: location_id for location_id, names in CAPITALS.items()}


@dataclass
class Result:
    columns: list[str]
    rows: list[tuple]


def run(connection: duckdb.DuckDBPyConnection, sql: str) -> Result:
    """Runs a query with a time limit; raises duckdb.Error on invalid SQL, TimeoutError when it is too slow."""
    timer = threading.Timer(TIMEOUT_S, connection.interrupt)
    timer.start()
    try:
        cursor = connection.execute(sql)
        rows = cursor.fetchmany(MAX_ROWS + 1)
        columns = [d[0] for d in cursor.description or []]
    except duckdb.InterruptException as error:
        raise TimeoutError(f"query took more than {TIMEOUT_S} s") from error
    finally:
        timer.cancel()
    return Result(columns, rows)


def _normalise(value):
    if isinstance(value, bool) or value is None:
        return value
    if isinstance(value, (float, Decimal)):
        value = round(float(value), 2)
        return 0.0 if value == 0 else value  # -0.0 == 0.0
    if isinstance(value, int):
        return float(value)
    if isinstance(value, datetime):
        return value.isoformat()
    if isinstance(value, date):  # a day may come back as DATE or as a midnight TIMESTAMP
        return datetime(value.year, value.month, value.day).isoformat()
    return NAME_TO_ID.get(str(value), str(value))


def _column(rows: list[tuple], index: int) -> list:
    return [_normalise(row[index]) for row in rows]


def same_result(reference: Result, prediction: Result) -> bool:
    if len(reference.rows) != len(prediction.rows):
        return False
    if not reference.rows:
        return not prediction.rows
    gold = [_column(reference.rows, i) for i in range(len(reference.columns))]
    pred = [_column(prediction.rows, j) for j in range(len(prediction.columns))]
    # match every reference column to a distinct prediction column holding the same values
    mapping: list[int] = []
    for values in gold:
        wanted = Counter(values)
        match = next((j for j, col in enumerate(pred) if j not in mapping and Counter(col) == wanted), None)
        if match is None:
            return False
        mapping.append(match)
    gold_rows = Counter(zip(*gold))
    pred_rows = Counter(zip(*(pred[j] for j in mapping)))
    return gold_rows == pred_rows


@dataclass
class Verdict:
    correct: bool
    valid_sql: bool
    error: str | None = None


def judge(connection: duckdb.DuckDBPyConnection, reference_sql: str, predicted_sql: str) -> Verdict:
    reference = run(connection, reference_sql)
    try:
        prediction = run(connection, predicted_sql)
    except (duckdb.Error, TimeoutError) as error:
        return Verdict(correct=False, valid_sql=False, error=str(error).splitlines()[0][:200])
    return Verdict(correct=same_result(reference, prediction), valid_sql=True)


def clean_prediction(text: str) -> str:
    """Models sometimes wrap the query in ```sql fences or add a trailing explanation; keep the query."""
    text = text.strip()
    if "```" in text:
        parts = text.split("```")
        text = parts[1] if len(parts) > 1 else parts[0]
        if text.lower().startswith("sql"):
            text = text[3:]
    return text.strip().split(";")[0].strip()


def score(examples: list[dict], connection: duckdb.DuckDBPyConnection | None = None) -> dict:
    """Adds `correct`, `valid_sql` and `error` to each example and returns the summary."""
    connection = connection or connect()
    for example in examples:
        verdict = judge(connection, example["sql"], clean_prediction(example["prediction"]))
        example.update(correct=verdict.correct, valid_sql=verdict.valid_sql, error=verdict.error)
    total = len(examples)
    return {
        "examples": total,
        "execution_accuracy": round(sum(e["correct"] for e in examples) / total, 4) if total else 0.0,
        "valid_sql": round(sum(e["valid_sql"] for e in examples) / total, 4) if total else 0.0,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Execution accuracy of a predictions JSONL file.")
    parser.add_argument("predictions", type=Path)
    parser.add_argument("--show-errors", type=int, default=10, help="print this many wrong answers")
    args = parser.parse_args()
    examples = [json.loads(line) for line in args.predictions.read_text(encoding="utf-8").splitlines() if line]
    summary = score(examples)
    for example in [e for e in examples if not e["correct"]][: args.show_errors]:
        print(f"\nQ: {example['question']}\n  reference:  {example['sql']}\n  prediction: {clean_prediction(example['prediction'])}")
        if example["error"]:
            print(f"  error: {example['error']}")
    print("\n" + json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
