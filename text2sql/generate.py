"""Builds the train / validation / test sets from the question families, validated on the snapshot.

    python -m text2sql.generate

Each candidate example is run against data/snapshot; it is kept only when the query works and returns at
least one non-empty row. Examples are split by their SQL (the same query never appears in two splits, even
with different wording). Test examples keep the snapshot's real dates, so they can be scored directly;
most training examples have every date shifted by a random number of days (see templates.py).

Writes data/train.jsonl, data/val.jsonl and data/test.jsonl, and checks data/handwritten_test.jsonl.
"""
from __future__ import annotations

import hashlib
import json
import random
import re
from collections import Counter
from datetime import date, timedelta
from pathlib import Path

import duckdb

from .evaluate import run
from .snapshot import connect
from .templates import FAMILIES, Ctx, real_today_for

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
SEED = 2026
ENGLISH_SHARE = 0.2
SHIFTED_SHARE = 0.7  # training examples whose dates are moved
SHIFT_RANGE = (-250, 400)  # days
DATE_LITERAL = re.compile(r"DATE '(\d{4}-\d{2}-\d{2})'")


def make(family: str, seed: int, shift: int) -> dict:
    kind, build, _ = FAMILIES[family]
    rng = random.Random(seed)
    lang = "en" if rng.random() < ENGLISH_SHARE else "es"
    ctx = Ctx(rng=rng, real_today=real_today_for(kind, rng), shift=shift, lang=lang)
    question, sql = build(ctx)
    return {"family": family, "lang": lang, "today": ctx.today.isoformat(), "question": question, "sql": sql}


def shift_dates(sql: str, days: int) -> str:
    return DATE_LITERAL.sub(lambda m: f"DATE '{(date.fromisoformat(m.group(1)) + timedelta(days=days)).isoformat()}'", sql)


def has_data(connection: duckdb.DuckDBPyConnection, sql: str) -> bool:
    try:
        result = run(connection, sql)
    except (duckdb.Error, TimeoutError) as error:
        raise ValueError(f"template produced invalid SQL: {sql}\n{error}") from error
    return any(any(value is not None for value in row) for row in result.rows)


def split_of(sql: str) -> str:
    bucket = int(hashlib.sha256(sql.encode()).hexdigest(), 16) % 100
    return "train" if bucket < 85 else "val" if bucket < 90 else "test"


def generate(connection: duckdb.DuckDBPyConnection) -> dict[str, list[dict]]:
    splits: dict[str, list[dict]] = {"train": [], "val": [], "test": []}
    seen: set[str] = set()
    rejected: Counter = Counter()
    for family, (_, _, tries) in FAMILIES.items():
        for i in range(tries):
            seed = int(hashlib.sha256(f"{SEED}:{family}:{i}".encode()).hexdigest()[:12], 16)
            example = make(family, seed, 0)
            if example["question"] in seen:
                rejected["duplicate"] += 1
                continue
            seen.add(example["question"])
            if not has_data(connection, example["sql"]):
                rejected["empty result"] += 1
                continue
            split = split_of(example["sql"])
            rng = random.Random(seed)
            if split == "train" and rng.random() < SHIFTED_SHARE:
                shift = rng.randint(*SHIFT_RANGE)
                shifted = make(family, seed, shift)
                if shift_dates(shifted["sql"], -shift) == example["sql"]:  # same example, only dates moved
                    example = shifted
                else:
                    rejected["shift mismatch (kept unshifted)"] += 1
            example["id"] = f"{family}-{i}"
            splits[split].append(example)
    for reason, count in rejected.items():
        print(f"skipped {count:5d}  {reason}")
    return splits


def check_handwritten(connection: duckdb.DuckDBPyConnection, path: Path) -> int:
    examples = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    for example in examples:
        if not has_data(connection, example["sql"]):
            raise ValueError(f"hand-written example returns no data: {example['question']}")
    return len(examples)


def write_jsonl(path: Path, rows: list[dict]) -> None:
    fields = ("id", "family", "lang", "today", "question", "sql")
    path.write_text("".join(json.dumps({k: row[k] for k in fields}, ensure_ascii=False) + "\n" for row in rows), encoding="utf-8")


def main() -> None:
    connection = connect()
    splits = generate(connection)
    for name, rows in splits.items():
        random.Random(SEED).shuffle(rows)
        write_jsonl(DATA_DIR / f"{name}.jsonl", rows)
        langs = Counter(r["lang"] for r in rows)
        print(f"{name:5s} {len(rows):5d} examples  ({langs['es']} es, {langs['en']} en)")
    handwritten = DATA_DIR / "handwritten_test.jsonl"
    if handwritten.exists():
        print(f"handwritten {check_handwritten(connection, handwritten)} examples OK")


if __name__ == "__main__":
    main()
