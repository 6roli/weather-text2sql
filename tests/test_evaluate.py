import duckdb

from text2sql.evaluate import Result, clean_prediction, judge, same_result


def test_order_and_column_names_do_not_matter():
    gold = Result(["a", "b"], [(1, "x"), (2, "y")])
    pred = Result(["b2", "a2"], [("y", 2), ("x", 1)])
    assert same_result(gold, pred)


def test_extra_columns_are_allowed_but_missing_ones_are_not():
    gold = Result(["temperature"], [(20.0,)])
    assert same_result(gold, Result(["valid_at", "temperature"], [("2026-10-26 12:00", 20.004)]))
    assert not same_result(Result(["a", "b"], [(1, 2)]), Result(["a"], [(1,)]))


def test_rows_must_line_up_not_only_columns():
    gold = Result(["city", "t"], [("madrid", 10.0), ("bilbao", 20.0)])
    swapped = Result(["city", "t"], [("madrid", 20.0), ("bilbao", 10.0)])
    assert not same_result(gold, swapped)


def test_capital_name_and_id_are_the_same_answer():
    assert same_result(Result(["location_id"], [("a-coruna",)]), Result(["location_name"], [("A Coruña",)]))


def test_date_and_midnight_timestamp_are_the_same_day():
    connection = duckdb.connect()
    verdict = judge(connection, "SELECT DATE '2026-10-26' AS d", "SELECT DATE '2026-10-25' + INTERVAL 1 DAY")
    assert verdict.correct


def test_invalid_sql_is_reported_not_raised():
    verdict = judge(duckdb.connect(), "SELECT 1", "SELEC 1")
    assert not verdict.valid_sql and not verdict.correct and verdict.error


def test_clean_prediction_strips_fences_and_trailing_text():
    assert clean_prediction("```sql\nSELECT 1;\n```\nThis returns one.") == "SELECT 1"
    assert clean_prediction("SELECT 2; -- done") == "SELECT 2"
