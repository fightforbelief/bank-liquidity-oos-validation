"""Synthetic quarter-grid checks. TEST_FIXTURE_NOT_RESEARCH_EVIDENCE.

These tests call the saved upstream functions in the isolated preprocess
environment. They use invented dates and values, not bank filings.
"""

import json
import subprocess
from functools import lru_cache
from pathlib import Path

import pytest

FIXTURE_BANNER = "TEST_FIXTURE_NOT_RESEARCH_EVIDENCE"
PREPROCESS_PYTHON = Path(
    r"C:\Users\hanqi\Desktop\projects\_p12_audit_scratch\preprocess_venv\Scripts\python.exe"
)
UPSTREAM_SRC = Path(r"C:\Users\hanqi\Desktop\projects\_p12_audit_scratch\upstream_src")

SCRIPT = r"""
import contextlib, io, json, sys
from pathlib import Path
import numpy as np
import pandas as pd
import polars as pl
sys.path.insert(0, r"C:\Users\hanqi\Desktop\projects\_p12_audit_scratch\upstream_src")
from robust_preprocessing import (
    build_canonical_grid,
    filter_series_by_quality,
    get_data_requirements,
    normalize_month_end_dates,
    split_train_test_aligned,
)
BANNER = "TEST_FIXTURE_NOT_RESEARCH_EVIDENCE"

def walk(start, count):
    stamp = pd.Timestamp(start)
    year, quarter = stamp.year, (stamp.month - 1) // 3 + 1
    month_day = ((3, 31), (6, 30), (9, 30), (12, 31))
    dates = []
    for _ in range(count):
        month, day = month_day[quarter - 1]
        dates.append(pd.Timestamp(year=year, month=month, day=day))
        quarter += 1
        if quarter == 5:
            quarter, year = 1, year + 1
    return dates

def is_qe(text):
    return text[5:] in {"03-31", "06-30", "09-30", "12-31"}

def run_grid(dates, values):
    frame = pl.DataFrame({
        "unique_id": [BANNER] * len(dates),
        "ds": dates,
        "y": values,
    }).with_columns(pl.col("ds").cast(pl.Datetime("us")), pl.col("y").cast(pl.Float32))
    normalized = normalize_month_end_dates(frame, "QE")
    buffer = io.StringIO()
    with contextlib.redirect_stdout(buffer):
        filled = build_canonical_grid(normalized, "QE").sort("ds")
    generated = [str(pd.Timestamp(value).date()) for value in filled["ds"].to_list()]
    paired = list(zip(generated, filled["y"].to_list()))
    direct = pl.select(pl.datetime_ranges(
        pl.datetime(pd.Timestamp(dates[0]).year, pd.Timestamp(dates[0]).month, pd.Timestamp(dates[0]).day, time_unit="us"),
        pl.datetime(pd.Timestamp(dates[-1]).year, pd.Timestamp(dates[-1]).month, pd.Timestamp(dates[-1]).day, time_unit="us"),
        interval="3mo",
    ).alias("ds")).explode("ds")
    direct_dates = [str(pd.Timestamp(value).date()) for value in direct["ds"].to_list()]
    return {
        "normalize_equals_input": bool(normalized.equals(frame)),
        "input": [str(pd.Timestamp(value).date()) for value in dates],
        "generated": generated,
        "direct": direct_dates,
        "finite_on_non_quarter_end": sum(1 for date, value in paired if value is not None and not is_qe(date)),
        "missing": sorted(set(str(pd.Timestamp(value).date()) for value in dates) - set(generated)),
    }

def quarter_end_grid(dates, values):
    start, end = pd.Timestamp(dates[0]), pd.Timestamp(dates[-1])
    q0 = start.year * 4 + (start.month - 1) // 3
    q1 = end.year * 4 + (end.month - 1) // 3
    rows = []
    for q in range(q0, q1 + 1):
        year, rem = divmod(q, 4)
        month, day = ((3, 31), (6, 30), (9, 30), (12, 31))[rem]
        rows.append(pd.Timestamp(year=year, month=month, day=day))
    observed = {pd.Timestamp(date): value for date, value in zip(dates, values)}
    return pl.DataFrame({
        "unique_id": [BANNER] * len(rows),
        "ds": rows,
        "y": [observed.get(row, None) for row in rows],
    }).with_columns(pl.col("ds").cast(pl.Datetime("us")), pl.col("y").cast(pl.Float32))

def passes(grid, requirements):
    buffer = io.StringIO()
    try:
        with contextlib.redirect_stdout(buffer):
            train_df, test_df = split_train_test_aligned(grid, 1)
            _, _, valid = filter_series_by_quality(train_df, test_df, requirements)
    except ValueError:
        return False
    return BANNER in [str(value) for value in valid]

requirements = get_data_requirements("QE", 1, 4)
anchors = {}
for start in ("1976-03-31", "1976-06-30", "1976-09-30", "1976-12-31"):
    anchors[start] = run_grid(walk(start, 5), [0.1, 0.2, 0.3, 0.4, 0.5])
off = run_grid(walk("1976-06-30", 4), [0.1, 0.2, 0.3, 0.4])
y = np.linspace(0.05, 0.55, 27)
june = walk("1976-06-30", 27)
march = walk("1976-03-31", 27)
buffer = io.StringIO()
with contextlib.redirect_stdout(buffer):
    june_frame = pl.DataFrame({"unique_id": [BANNER]*27, "ds": june, "y": y}).with_columns(
        pl.col("ds").cast(pl.Datetime("us")), pl.col("y").cast(pl.Float32)
    )
    march_frame = pl.DataFrame({"unique_id": [BANNER]*27, "ds": march, "y": y}).with_columns(
        pl.col("ds").cast(pl.Datetime("us")), pl.col("y").cast(pl.Float32)
    )
    june_3 = build_canonical_grid(june_frame, "QE")
    march_3 = build_canonical_grid(march_frame, "QE")
june_q = quarter_end_grid(june, y)
march_q = quarter_end_grid(march, y)
result = {
    "banner": BANNER,
    "requirements": requirements,
    "anchors": anchors,
    "june_ending_march": off,
    "june_dates_differ": sorted(set(str(pd.Timestamp(v).date()) for v in june_3["ds"].to_list())
                                .symmetric_difference(set(str(pd.Timestamp(v).date()) for v in june_q["ds"].to_list()))),
    "march_dates_equal": [str(pd.Timestamp(v).date()) for v in march_3.sort("ds")["ds"].to_list()]
        == [str(pd.Timestamp(v).date()) for v in march_q.sort("ds")["ds"].to_list()],
    "june_upstream_passes": passes(june_3, requirements),
    "june_quarter_end_passes": passes(june_q, requirements),
    "march_upstream_passes": passes(march_3, requirements),
    "march_quarter_end_passes": passes(march_q, requirements),
}
print(json.dumps(result))
"""


@pytest.fixture(scope="module")
def fixture_result():
    if not PREPROCESS_PYTHON.exists() or not UPSTREAM_SRC.exists():
        pytest.skip("isolated preprocess environment is not on this machine")
    completed = subprocess.run(
        [str(PREPROCESS_PYTHON), "-X", "utf8", "-"],
        input=SCRIPT,
        text=True,
        capture_output=True,
        check=False,
    )
    assert completed.returncode == 0, completed.stderr[-2000:]
    payload = json.loads(completed.stdout)
    assert payload["banner"] == FIXTURE_BANNER
    return payload


def test_banner_marks_fixture(fixture_result):
    assert FIXTURE_BANNER in fixture_result["banner"]


def test_requirements_stay_at_the_table15_thresholds(fixture_result):
    assert fixture_result["requirements"] == {
        "min_total_obs": 27,
        "min_train_obs": 18,
        "min_test_obs": 1,
        "variance_threshold": 0.01,
        "max_gap_ratio": 0.6,
        "min_test_completion_ratio": 0.75,
    }


def test_march_and_december_anchors_stay_on_quarter_ends(fixture_result):
    for start in ("1976-03-31", "1976-12-31"):
        anchor = fixture_result["anchors"][start]
        assert anchor["normalize_equals_input"] is True
        assert anchor["generated"] == anchor["input"]
        assert anchor["direct"] == anchor["generated"]
        assert anchor["missing"] == []
        assert anchor["finite_on_non_quarter_end"] == 0
        assert all(date[5:] in {"03-31", "06-30", "09-30", "12-31"} for date in anchor["generated"])


def test_june_anchor_creates_non_quarter_end_dates_and_drops_originals(fixture_result):
    anchor = fixture_result["anchors"]["1976-06-30"]
    assert anchor["generated"] == [
        "1976-06-30",
        "1976-09-30",
        "1976-12-30",
        "1977-03-30",
        "1977-06-30",
    ]
    assert anchor["direct"] == anchor["generated"]
    assert anchor["missing"] == ["1976-12-31", "1977-03-31"]
    assert "1976-12-30" in anchor["generated"]
    assert "1977-03-30" in anchor["generated"]
    assert anchor["finite_on_non_quarter_end"] == 0


def test_september_anchor_drops_december_and_march_filings(fixture_result):
    anchor = fixture_result["anchors"]["1976-09-30"]
    assert anchor["generated"] == [
        "1976-09-30",
        "1976-12-30",
        "1977-03-30",
        "1977-06-30",
        "1977-09-30",
    ]
    assert anchor["missing"] == ["1976-12-31", "1977-03-31"]
    assert anchor["finite_on_non_quarter_end"] == 0
    assert anchor["direct"] == anchor["generated"]


def test_series_ending_on_march_31_does_not_keep_that_report_as_the_grid_end(fixture_result):
    off = fixture_result["june_ending_march"]
    assert off["input"][-1] == "1977-03-31"
    assert off["generated"][-1] == "1977-03-30"
    assert "1977-03-31" in off["missing"]
    assert off["finite_on_non_quarter_end"] == 0


def test_strict_quarter_end_grid_differs_only_for_the_drifting_anchor(fixture_result):
    assert fixture_result["march_dates_equal"] is True
    drifted = fixture_result["june_dates_differ"]
    assert "1976-12-30" in drifted or any(date.endswith("-12-30") or date.endswith("-03-30") for date in drifted)
    assert any(date.endswith("-12-31") or date.endswith("-03-31") for date in drifted)


def test_same_thresholds_keep_different_entities_on_the_two_grids(fixture_result):
    assert fixture_result["june_quarter_end_passes"] is True
    assert fixture_result["june_upstream_passes"] is False
    assert fixture_result["march_upstream_passes"] is True
    assert fixture_result["march_quarter_end_passes"] is True


def test_partition_counts_do_not_treat_a_dropped_filing_as_a_missing_quarter():
    """Counting rule only. This does not call the upstream grid."""
    original = {"1976-06-30", "1976-09-30", "1976-12-31", "1977-03-31"}
    grid = {"1976-06-30", "1976-09-30", "1976-12-30", "1977-03-30"}
    matched = original & grid
    unmatched = original - grid
    synthetic_dates = {date for date in grid - original if date[5:] not in {"03-31", "06-30", "09-30", "12-31"}}
    true_missing_quarter_ends = {date for date in grid - original if date[5:] in {"03-31", "06-30", "09-30", "12-31"}}
    assert matched | unmatched == original
    assert len(unmatched) == 2
    assert synthetic_dates == {"1976-12-30", "1977-03-30"}
    assert true_missing_quarter_ends == set()
