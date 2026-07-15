from browsecomp250.report.aggregate import aggregate_records, wilson_interval


def test_aggregate_records() -> None:
    records = [
        {
            "item_id": "a",
            "attempt": 1,
            "correct": True,
            "status": "completed",
            "confidence": 90,
            "extracted_answer": "x",
            "metrics": {"duration_seconds": 2, "search_calls": 3, "page_opens": 4},
        },
        {
            "item_id": "b",
            "attempt": 1,
            "correct": False,
            "status": "completed",
            "confidence": 20,
            "extracted_answer": "y",
            "metrics": {"duration_seconds": 4, "search_calls": 5, "page_opens": 6},
        },
    ]
    summary = aggregate_records(records, bootstrap_samples=200)
    assert summary["accuracy"] == 0.5
    assert summary["n_items"] == 2
    assert summary["duration_seconds"]["median"] == 3
    assert 0 <= summary["calibration"]["brier"] <= 1


def test_wilson_interval_bounds() -> None:
    low, high = wilson_interval(5, 10)
    assert 0 < low < 0.5 < high < 1


def test_errors_count_as_incorrect() -> None:
    records = [
        {
            "item_id": "a",
            "attempt": 1,
            "correct": True,
            "status": "completed",
            "confidence": 80,
            "extracted_answer": "x",
            "metrics": {},
        },
        {
            "item_id": "b",
            "attempt": 1,
            "correct": None,
            "status": "timeout",
            "confidence": None,
            "extracted_answer": None,
            "metrics": {},
        },
    ]
    summary = aggregate_records(records, bootstrap_samples=200)
    assert summary["accuracy"] == 0.5
    assert summary["n_scored"] == 2
    assert summary["n_graded"] == 1
    assert summary["n_ungraded"] == 1
