"""Synthetic grid tests. TEST_FIXTURE_NOT_RESEARCH_EVIDENCE.

Expected Arm A dates were read from utilsforecast 0.2.16 ``fill_gaps`` in the
existing preprocess environment. They are not prior-audit aggregate counts.
"""

from __future__ import annotations

import json
import subprocess
from datetime import date
from pathlib import Path

import pytest

from p12_bank_liquidity.candidate_a.grid_arms import (
    EVIDENCE_LABEL,
    DuplicateEntityQuarter,
    NonQuarterEndSourceDate,
    SourceRow,
    build_arm_b,
    is_natural_quarter_end,
)

PREPROCESS_PYTHON = Path(
    r"C:\Users\hanqi\Desktop\projects\_p12_audit_scratch\preprocess_venv\Scripts\python.exe"
)
UPSTREAM = Path(
    r"C:\Users\hanqi\Desktop\projects\_p12_audit_scratch\upstream_src\robust_preprocessing.py"
)
UPSTREAM_SHA256 = "52503024b7f4a5717bdb4d783b43645f51efad1d96c15429deb3fdbce745ef3c"


def _row(entity: str, day: str, value: float | None, row_id: str) -> SourceRow:
    year, month, day_number = (int(part) for part in day.split("-"))
    return SourceRow(entity, date(year, month, day_number), value, row_id)


def _fill_gaps(dates: list[str]) -> list[dict[str, object]]:
    if not PREPROCESS_PYTHON.exists():
        pytest.skip("preprocess environment is absent; Arm A was not re-run")
    script = r"""
import json
from datetime import datetime
import polars as pl
from utilsforecast.preprocessing import fill_gaps
dates = json.loads(input())
rows = []
for index, item in enumerate(dates):
    year, month, day = (int(part) for part in item.split("-"))
    rows.append({
        "unique_id": "TEST_FIXTURE_NOT_RESEARCH_EVIDENCE",
        "ds": datetime(year, month, day),
        "y": float(index + 1),
    })
frame = pl.DataFrame(rows).with_columns(pl.col("ds").cast(pl.Datetime("ns")))
filled = fill_gaps(frame, freq="3mo", start="per_serie", end="per_serie").sort("ds")
payload = []
for row in filled.iter_rows(named=True):
    value = row["y"]
    payload.append({"ds": str(row["ds"])[:10], "y": None if value is None else value})
print(json.dumps(payload))
"""
    completed = subprocess.run(
        [str(PREPROCESS_PYTHON), "-X", "utf8", "-c", script],
        input=json.dumps(dates),
        text=True,
        capture_output=True,
        check=False,
    )
    assert completed.returncode == 0, completed.stderr
    return json.loads(completed.stdout)


def test_fixture_banner_is_present() -> None:
    assert EVIDENCE_LABEL == "TEST_FIXTURE_NOT_RESEARCH_EVIDENCE"


def test_arm_a_june_start_shifts_december_and_march() -> None:
    payload = _fill_gaps(["1976-06-30", "1976-09-30", "1976-12-31", "1977-03-31"])
    assert [item["ds"] for item in payload] == [
        "1976-06-30",
        "1976-09-30",
        "1976-12-30",
        "1977-03-30",
    ]
    assert payload[2]["y"] is None
    assert payload[3]["y"] is None


def test_arm_a_september_start_shifts_internal_quarter_ends() -> None:
    payload = _fill_gaps(["1976-09-30", "1976-12-31", "1977-03-31", "1977-06-30"])
    assert [item["ds"] for item in payload] == [
        "1976-09-30",
        "1976-12-30",
        "1977-03-30",
        "1977-06-30",
    ]
    assert payload[1]["y"] is None
    assert payload[2]["y"] is None
    assert payload[3]["y"] == 4.0


def test_arm_a_march_and_december_starts_stay_on_quarter_ends() -> None:
    march = _fill_gaps(["1976-03-31", "1976-06-30", "1976-09-30", "1976-12-31"])
    december = _fill_gaps(["1976-12-31", "1977-03-31", "1977-06-30", "1977-09-30"])
    assert [item["ds"] for item in march] == [
        "1976-03-31",
        "1976-06-30",
        "1976-09-30",
        "1976-12-31",
    ]
    assert [item["ds"] for item in december] == [
        "1976-12-31",
        "1977-03-31",
        "1977-06-30",
        "1977-09-30",
    ]


def test_arm_b_preserves_natural_quarter_ends_including_leap_year() -> None:
    rows = build_arm_b(
        [
            _row("E", "2019-12-31", 0.2, "a"),
            _row("E", "2020-03-31", 0.3, "b"),
            _row("E", "2020-06-30", 0.4, "c"),
        ]
    )
    stamps = [row.grid_timestamp.isoformat() for row in rows]
    assert stamps == ["2019-12-31", "2020-03-31", "2020-06-30"]
    assert date(2020, 2, 29) not in [row.grid_timestamp for row in rows]
    assert all(is_natural_quarter_end(row.grid_timestamp) for row in rows)
    assert rows[1].source_reporting_date == date(2020, 3, 31)
    assert rows[1].grid_timestamp == date(2020, 3, 31)
    assert rows[1].source_row_id == "b"
    assert rows[1].evidence_label == "TEST_FIXTURE_NOT_RESEARCH_EVIDENCE"


def test_arm_b_missing_quarter_is_synthetic_and_target_is_missing() -> None:
    rows = build_arm_b(
        [
            _row("E", "2020-03-31", 0.1, "a"),
            _row("E", "2020-09-30", 0.2, "b"),
        ]
    )
    missing = rows[1]
    assert missing.grid_timestamp == date(2020, 6, 30)
    assert missing.is_synthetic_grid_row is True
    assert missing.observed_target is False
    assert missing.target_value is None
    assert missing.source_reporting_date is None


def test_arm_b_explicit_end_can_leave_a_synthetic_final_row() -> None:
    rows = build_arm_b(
        [_row("E", "2020-06-30", 0.5, "a")],
        grid_end=date(2020, 12, 31),
    )
    assert rows[-1].grid_timestamp == date(2020, 12, 31)
    assert rows[-1].is_synthetic_grid_row is True
    assert rows[-1].observed_target is False


def test_duplicate_entity_quarter_raises() -> None:
    with pytest.raises(DuplicateEntityQuarter):
        build_arm_b(
            [
                _row("E", "2020-03-31", 0.1, "a"),
                _row("E", "2020-03-31", 0.2, "b"),
            ]
        )


def test_non_quarter_end_source_date_is_not_moved() -> None:
    with pytest.raises(NonQuarterEndSourceDate):
        build_arm_b([_row("E", "2020-03-30", 0.1, "a")])


def test_upstream_preprocessing_bytes_are_unchanged() -> None:
    if not UPSTREAM.exists():
        pytest.skip("saved upstream preprocessing file is absent")
    import hashlib

    digest = hashlib.sha256(UPSTREAM.read_bytes()).hexdigest()
    assert digest == UPSTREAM_SHA256
    text = UPSTREAM.read_text(encoding="utf-8")
    assert '"QE": "3mo"' in text
    assert "variance_threshold" in text
