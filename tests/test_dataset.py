import json
from pathlib import Path

from text2sql.generate import make, shift_dates, split_of
from text2sql.schema import CAPITALS, messages
from text2sql.templates import FAMILIES

DATA = Path(__file__).resolve().parent.parent / "data"


def load(name):
    return [json.loads(line) for line in (DATA / name).read_text(encoding="utf-8").splitlines() if line]


def test_shifting_only_moves_dates():
    for family in FAMILIES:
        for seed in range(20):
            base = make(family, seed, 0)
            moved = make(family, seed, 123)
            if shift_dates(moved["sql"], -123) == base["sql"]:
                continue
            # the only allowed difference is a phrase like "el día 28" falling back near a month change
            assert family.startswith(("ds_", "fvo_")), (family, seed, base["sql"], moved["sql"])


def test_no_query_appears_in_two_splits():
    train = {r["sql"] for r in load("train.jsonl")}
    for name in ("val", "test"):
        rows = load(f"{name}.jsonl")
        assert all(split_of(r["sql"]) == name for r in rows)  # held-out queries keep their real dates
        assert not train & {r["sql"] for r in rows}


def test_test_sets_use_the_snapshot_dates():
    for row in load("test.jsonl") + load("handwritten_test.jsonl"):
        assert row["today"] <= "2026-10-25"


def test_prompt_lists_every_capital_and_today():
    chat = messages("¿Lloverá mañana en Soria?", "2026-11-03", "SELECT 1")
    assert [m["role"] for m in chat] == ["system", "user", "assistant"]
    assert all(location_id in chat[1]["content"] for location_id in CAPITALS)
    assert "Today (UTC): 2026-11-03" in chat[1]["content"]
