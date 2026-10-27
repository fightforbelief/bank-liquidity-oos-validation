"""Independent audit of constructed Bank Cash Liquidity columns.

Expected ratios are cash / assets, written in this file. The project function
is imported only afterwards, and only to obtain actual_y. Synthetic helpers
are marked TEST_FIXTURE_NOT_RESEARCH_EVIDENCE and are not bank filings.

This script does not download data, fit a model, or change gate status.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd

FIXTURE_BANNER = "TEST_FIXTURE_NOT_RESEARCH_EVIDENCE"
MIN_ASSETS = 1e-10
ZIP_SHA256 = "83abc553c7c55e04d4b0ea1cc2a46566ba330b0eb992a25cdc451fb68e0bae25"
DTA_SHA256 = "40692c2ec9af404b3148ffbd7ad970a4ce191d9ac8f781b415e8a10909533ad0"
ZIP_BYTES = 455791391
DTA_BYTES = 2910552178

ROOT = Path(r"C:\Users\hanqi\Desktop\projects\p12-bank-liquidity-oos-validation")
SCRATCH = Path(r"C:\Users\hanqi\Desktop\projects\_p12_audit_scratch")
ZIP_PATH = SCRATCH / "callreports_1976_2020_WRDS.dta.zip"
DTA_PATH = SCRATCH / "extract" / "callreports_1976_2020_WRDS.dta"
REPORTS = ROOT / "reports"

# Published table text, used only when writing the discrepancy note.
# These constants are not filters and are not used to choose rows.
PUBLISHED_TABLE4_ENTITIES = 23862
PUBLISHED_TABLE4_MIN = 1
PUBLISHED_TABLE4_MEDIAN = 66
PUBLISHED_TABLE4_MAX = 177
PUBLISHED_TABLE15_BEFORE = 23862
PUBLISHED_TABLE15_AFTER = 17383
PUBLISHED_TABLE15_RETENTION = 72.8
PUBLISHED_TABLE15_MEDIAN_AFTER = 82


def independent_ratio(cash, assets):
    """Explicit constructed target. Does not call project code."""
    if cash is None or assets is None:
        return None
    cash_value = float(cash)
    asset_value = float(assets)
    if not math.isfinite(cash_value) or not math.isfinite(asset_value):
        return None
    if not (asset_value > MIN_ASSETS and cash_value >= 0):
        return None
    ratio = cash_value / asset_value
    if not math.isfinite(ratio):
        return None
    return ratio


def qe_quality_requirements(test_size=1, seasonality=4):
    """robust_preprocessing.get_data_requirements for frequency QE.

    The arithmetic is copied from that function's QE branch. test_size is the
    forecast horizon from get_test_size_from_frequency, which is 1 for QE.
    """
    base_min_total = 27
    base_min_train = 18
    base_min_test = 9
    train_ratio = 1.0
    test_ratio = 0.5
    completion_ratio = 0.75
    min_train_seasons = 2
    seasonal_floor = seasonality * min_train_seasons if seasonality > 1 else base_min_train
    min_train = max(base_min_train, math.ceil(test_size * train_ratio), seasonal_floor)
    adaptive_test_floor = max(1, math.ceil(test_size * test_ratio))
    min_test = max(
        min(test_size, max(base_min_test, adaptive_test_floor)),
        math.ceil(test_size * completion_ratio),
    )
    min_total = max(base_min_total, min_train + min_test, test_size + min_train)
    return {
        "min_total_obs": min_total,
        "min_train_obs": min_train,
        "min_test_obs": min_test,
        "variance_threshold": 0.01,
        "max_gap_ratio": 0.6,
    }


def forecasting_length_threshold(horizon, seasonality, min_buffer=6):
    """filter_series_for_forecasting minimum length, including its floor of 18."""
    if seasonality <= 1:
        base_min = max(horizon + min_buffer, 18)
    else:
        base_min = max(seasonality + 1, horizon + min_buffer)
    return max(base_min, 18)


def cv_requirement_thresholds(test_size=1, seasonality=4, multiplier=3.0):
    """get_minimum_requirements_by_frequency for QE. Not a forecast entry point."""
    base_min_train = max(test_size * 2, seasonality * 2 if seasonality > 1 else 12)
    min_train = int(base_min_train * multiplier)
    min_test = max(int(test_size * 0.8), 3)
    return {
        "min_total_obs": min_train + min_test,
        "min_train_obs": min_train,
        "min_test_obs": min_test,
        "min_variance": 0.001,
    }


def select_three(dates, rssdid, rows, eligible):
    """Choose three absolute indexes using rules fixed before any ratio sort.

    A is the eligible row with the smallest date, then the smallest rssdid,
    then the smallest file row number.
    B is the earliest remaining eligible row. If another rssdid exists, B is
    restricted to those other banks. Same tie break.
    C is the latest remaining eligible row. If a bank other than A and B
    exists, C is restricted to those banks. Same tie break.
    """
    dates = np.asarray(dates)
    rssdid = np.asarray(rssdid)
    rows = np.asarray(rows)
    eligible = np.asarray(eligible, dtype=bool)
    indexes = np.flatnonzero(eligible)
    if len(indexes) < 3:
        raise RuntimeError(f"fewer than three eligible rows: {len(indexes)}")
    order = np.lexsort((rows[indexes], rssdid[indexes], dates[indexes]))
    ordered = indexes[order]
    choice_a = int(ordered[0])
    id_a = rssdid[choice_a]
    rest = ordered[1:]
    other = rest[rssdid[rest] != id_a]
    if len(other):
        choice_b = int(other[0])
        b_distinct = True
        b_reason = "earliest eligible row whose rssdid differs from A"
    else:
        choice_b = int(rest[0])
        b_distinct = False
        b_reason = "no eligible row has an rssdid different from A"
    id_b = rssdid[choice_b]
    remaining_mask = eligible.copy()
    remaining_mask[choice_a] = False
    remaining_mask[choice_b] = False
    remaining = np.flatnonzero(remaining_mask)
    if len(remaining) == 0:
        raise RuntimeError("no eligible row remains for C")
    other_c = remaining[(rssdid[remaining] != id_a) & (rssdid[remaining] != id_b)]
    if len(other_c):
        pool = other_c
        c_distinct = True
        c_reason = "latest eligible row whose rssdid differs from A and B"
    else:
        pool = remaining
        c_distinct = False
        c_reason = "no eligible row has an rssdid different from both A and B"
    pool_order = np.lexsort((rows[pool], rssdid[pool], dates[pool]))
    pool_sorted = pool[pool_order]
    last_date = dates[pool_sorted[-1]]
    start = int(np.searchsorted(dates[pool_sorted], last_date, side="left"))
    choice_c = int(pool_sorted[start])
    return {
        "A": choice_a,
        "B": choice_b,
        "C": choice_c,
        "B_distinct_bank": b_distinct,
        "C_distinct_bank": c_distinct,
        "B_reason": b_reason,
        "C_reason": c_reason,
    }


def synthetic_self_check():
    """Arithmetic and ordering checks. TEST_FIXTURE_NOT_RESEARCH_EVIDENCE."""
    ratio = independent_ratio(20, 100)
    dropped = independent_ratio(-1, 100)
    dates = np.array([19760331, 19760331, 19760630, 20200331, 20200331])
    rssdid = np.array([2, 1, 1, 9, 1])
    rows = np.array([1, 2, 3, 4, 5])
    eligible = np.array([True, True, True, True, True])
    chosen = select_three(dates, rssdid, rows, eligible)
    return {
        "banner": FIXTURE_BANNER,
        "ratio_20_over_100": ratio,
        "negative_cash": dropped,
        "chosen_rows": {
            "A": int(rows[chosen["A"]]),
            "B": int(rows[chosen["B"]]),
            "C": int(rows[chosen["C"]]),
        },
        "requirements": qe_quality_requirements(),
        "length_threshold": forecasting_length_threshold(1, 4),
        "cv_requirements": cv_requirement_thresholds(),
    }


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _require_archive():
    if not ZIP_PATH.is_file() or not DTA_PATH.is_file():
        raise SystemExit("STOP archive file missing")
    zip_size = ZIP_PATH.stat().st_size
    dta_size = DTA_PATH.stat().st_size
    if zip_size != ZIP_BYTES or dta_size != DTA_BYTES:
        raise SystemExit(f"STOP size mismatch zip={zip_size} dta={dta_size}")
    zip_hash = _sha256(ZIP_PATH)
    dta_hash = _sha256(DTA_PATH)
    if zip_hash != ZIP_SHA256 or dta_hash != DTA_SHA256:
        raise SystemExit("STOP sha256 mismatch")
    print("archive_ok", flush=True)
    return zip_hash, dta_hash


def _load_columns():
    frames = []
    seen = 0
    reader = pd.read_stata(
        DTA_PATH,
        columns=["rssdid", "date", "dateq", "cash", "assets"],
        chunksize=20000,
        convert_categoricals=False,
        convert_dates=False,
    )
    for chunk in reader:
        frames.append(chunk)
        seen += len(chunk)
        if seen % 200000 == 0:
            print(f"rows_read {seen}", flush=True)
    frame = pd.concat(frames, ignore_index=True)
    print(f"rows_loaded {len(frame)}", flush=True)
    return frame


def _middle(lengths: np.ndarray):
    ordered = np.sort(np.asarray(lengths))
    count = len(ordered)
    if count == 0:
        return {"n": 0, "min": None, "max": None, "low": None, "high": None, "mean_of_two": None, "int_truncated": None}
    if count % 2:
        mid = float(ordered[count // 2])
        low = high = mid
    else:
        low = float(ordered[count // 2 - 1])
        high = float(ordered[count // 2])
    mean_of_two = (low + high) / 2.0
    return {
        "n": int(count),
        "min": int(ordered[0]),
        "max": int(ordered[-1]),
        "low": low,
        "high": high,
        "mean_of_two": mean_of_two,
        "int_truncated": int(mean_of_two),
    }


def _json_ready(value):
    if isinstance(value, dict):
        return {str(key): _json_ready(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_ready(item) for item in value]
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        if not np.isfinite(value):
            return None
        return float(value)
    if isinstance(value, np.bool_):
        return bool(value)
    return value


def _project_ratio(cash, assets):
    src = ROOT / "src"
    if str(src) not in sys.path:
        sys.path.insert(0, str(src))
    from p12_bank_liquidity.target.bank_cash_liquidity import cash_liquidity_ratio

    return cash_liquidity_ratio(cash, assets)


def run_audit():
    zip_hash, dta_hash = _require_archive()
    frame = _load_columns()
    row_number = np.arange(1, len(frame) + 1, dtype=np.int64)
    rss_float = frame["rssdid"].to_numpy(dtype=np.float64)
    date_float = frame["date"].to_numpy(dtype=np.float64)
    dateq = frame["dateq"].to_numpy(dtype=np.int64)
    cash = frame["cash"].to_numpy(dtype=np.float64)
    assets = frame["assets"].to_numpy(dtype=np.float64)

    date_ok = np.isfinite(date_float) & (date_float == np.floor(date_float))
    date_int = np.zeros(len(frame), dtype=np.int64)
    date_int[date_ok] = date_float[date_ok].astype(np.int64)
    invalid_dates = int((~date_ok).sum())
    if invalid_dates:
        raise SystemExit(f"STOP non-integer date values: {invalid_dates}")
    parsed = pd.to_datetime(date_int.astype(str), format="%Y%m%d", errors="coerce")
    if pd.isna(parsed).any():
        raise SystemExit(f"STOP dates that are not YYYYMMDD: {int(pd.isna(parsed).sum())}")
    sas_days = (
        np.asarray(parsed, dtype="datetime64[D]") - np.datetime64("1960-01-01", "D")
    ).astype(np.int64)
    dateq_mismatch = int((sas_days != dateq).sum())
    if dateq_mismatch:
        raise SystemExit(f"STOP dateq does not match SAS day number: {dateq_mismatch}")
    print("date_encoding_ok YYYYMMDD and dateq", flush=True)

    month_day = date_int % 10000
    quarter_end = np.isin(month_day, np.array([331, 630, 930, 1231]))
    non_quarter_end = int((~quarter_end).sum())
    year = date_int // 10000
    month = (date_int // 100) % 100
    quarter_index = year * 4 + (month - 1) // 3

    finite_id = np.isfinite(rss_float) & (rss_float == np.floor(rss_float))
    fractional_id = int((np.isfinite(rss_float) & ~finite_id).sum())
    missing_id = int((~np.isfinite(rss_float)).sum())
    rss = np.zeros(len(frame), dtype=np.int64)
    rss[finite_id] = rss_float[finite_id].astype(np.int64)

    finite_pair = np.isfinite(cash) & np.isfinite(assets)
    eligible = finite_id & finite_pair & (assets > MIN_ASSETS) & (cash >= 0)
    ratio = np.full(len(frame), np.nan, dtype=np.float64)
    ratio[eligible] = cash[eligible] / assets[eligible]
    eligible = eligible & np.isfinite(ratio)

    chosen = select_three(date_int, rss, row_number, eligible)
    records = []
    for label in ("A", "B", "C"):
        index = chosen[label]
        expected = independent_ratio(cash[index], assets[index])
        actual = _project_ratio(cash[index], assets[index])
        if expected is None or actual is None:
            difference = None
            status = "FAIL"
        else:
            difference = abs(expected - actual)
            status = "PASS" if expected == actual else "FAIL"
        records.append(
            {
                "category": label,
                "source_file": DTA_PATH.name,
                "row_locator": int(row_number[index]),
                "reporting_date": str(date_int[index]),
                "dateq": int(dateq[index]),
                "rssdid": int(rss[index]),
                "cash": cash[index],
                "assets": assets[index],
                "expected_y": expected,
                "actual_y": actual,
                "absolute_difference": difference,
                "assets_gt_min": bool(assets[index] > MIN_ASSETS),
                "cash_nonnegative": bool(cash[index] >= 0),
                "both_finite": bool(finite_pair[index]),
                "ratio_finite": expected is not None,
                "selection_note": chosen[f"{label}_reason"] if label != "A" else "earliest eligible by date, rssdid, row",
                "distinct_bank": True if label == "A" else bool(chosen[f"{label}_distinct_bank"]),
                "status": status,
            }
        )
    print(
        "three_rows",
        [(item["category"], item["status"], item["reporting_date"], item["distinct_bank"]) for item in records],
        flush=True,
    )

    raw_rows = int(len(frame))
    raw_entities = int(pd.unique(rss[finite_id]).size)
    finite_keys = pd.DataFrame({"id": rss[finite_id], "date": date_int[finite_id]})
    duplicate_keys = int(finite_keys.duplicated().sum())

    kept_positions = np.flatnonzero(eligible)
    kept_frame = pd.DataFrame(
        {
            "id": rss[kept_positions],
            "y": ratio[kept_positions],
            "q": quarter_index[kept_positions],
            "row": row_number[kept_positions],
        }
    )
    std = kept_frame.groupby("id")["y"].std(ddof=1)
    constant_ids = std.index[std == 0]
    constant_id_count = int(len(constant_ids))
    one_row_entities = int((kept_frame.groupby("id").size() == 1).sum())
    constructed = kept_frame.loc[~kept_frame["id"].isin(constant_ids)].copy()
    constructed_lengths = constructed.groupby("id").size().to_numpy()
    table4 = _middle(constructed_lengths)
    constructed_rows = int(len(constructed))
    constructed_entities = int(constructed["id"].nunique())

    y32 = constructed["y"].to_numpy(dtype=np.float32).astype(np.float64)
    float32_nonfinite = int((~np.isfinite(y32)).sum())
    constructed["y"] = y32
    duplicate_quarters = int(constructed.duplicated(["id", "q"]).sum())
    work = constructed.sort_values(["id", "q", "row"], kind="mergesort")

    work = work.sort_values(["id", "q", "row"], kind="mergesort")
    max_q = work.groupby("id")["q"].transform("max")
    train = work.loc[work["q"] != max_q]
    train_stats = train.groupby("id")["y"].agg(train_n="count", train_std="std")
    span = work.groupby("id")["q"].agg(min_q="min", max_q="max")
    panel = span.join(train_stats, how="left")
    panel["train_n"] = panel["train_n"].fillna(0).astype(np.int64)
    panel["grid_len"] = (panel["max_q"] - panel["min_q"] + 1).astype(np.int64)
    train_len = panel["grid_len"] - 1
    gap_ratio = np.where(train_len.to_numpy() > 0, 1.0 - panel["train_n"].to_numpy() / train_len.to_numpy(), 1.0)
    requirements = qe_quality_requirements()
    std_ok = np.where(
        panel["train_n"].to_numpy() > 1,
        panel["train_std"].to_numpy() >= requirements["variance_threshold"],
        True,
    )
    fail_total = panel["grid_len"].to_numpy() < requirements["min_total_obs"]
    fail_train = ~fail_total & (panel["train_n"].to_numpy() < requirements["min_train_obs"])
    fail_std = ~fail_total & ~fail_train & ~std_ok
    fail_gap = ~fail_total & ~fail_train & std_ok & (gap_ratio > requirements["max_gap_ratio"])
    passed = ~fail_total & ~fail_train & std_ok & ~fail_gap
    # The last grid point is an observed finite value because the grid ends
    # at the series maximum. min_test_obs for this horizon is 1.
    if requirements["min_test_obs"] > 1:
        raise SystemExit("STOP test rule is not the one-row tail this script implements")

    pass_index = panel.index[passed]
    grid_before = int(panel["grid_len"].sum())
    grid_after = int(panel.loc[pass_index, "grid_len"].sum())
    entities_after = int(passed.sum())
    observed_after_lengths = constructed.groupby("id").size()
    observed_after = observed_after_lengths.loc[observed_after_lengths.index.isin(pass_index)].to_numpy()
    median_after = _middle(panel.loc[pass_index, "grid_len"].to_numpy())
    median_after_observed = _middle(observed_after)
    retention = round((entities_after / constructed_entities) * 100, 1) if constructed_entities else None

    min_train_std = float(np.nanmin(panel.loc[pass_index, "train_std"].to_numpy())) if entities_after else None
    passer_y = work.loc[work["id"].isin(pass_index), "y"].to_numpy()
    max_abs_y = float(np.nanmax(np.abs(passer_y))) if len(passer_y) else None
    max_grid = int(panel["grid_len"].max()) if len(panel) else 0
    imputation_removes_zero = bool(
        min_train_std is not None and min_train_std >= 0.01 and max_abs_y is not None and max_abs_y < 1e10 and max_grid <= 177
    )

    length_at_least_18 = int((constructed.groupby("id").size() >= 18).sum())
    summary = {
        "zip_sha256": zip_hash,
        "dta_sha256": dta_hash,
        "rows": raw_rows,
        "raw_entities": raw_entities,
        "fractional_rssdid": fractional_id,
        "missing_rssdid": missing_id,
        "duplicate_rssdid_date": duplicate_keys,
        "duplicate_quarters_after_construction": duplicate_quarters,
        "non_quarter_end_dates": non_quarter_end,
        "date_min": int(date_int.min()),
        "date_max": int(date_int.max()),
        "distinct_dates": int(pd.unique(date_int).size),
        "finite_pair_rows": int(finite_pair.sum()),
        "eligible_rows": int(eligible.sum()),
        "eligible_entities": int(pd.unique(rss[eligible]).size),
        "one_row_entities_after_row_filter": one_row_entities,
        "constant_entities_removed": constant_id_count,
        "constructed_rows": constructed_rows,
        "constructed_entities": constructed_entities,
        "table4_lengths": table4,
        "float32_nonfinite_rows": float32_nonfinite,
        "requirements": requirements,
        "fail_total": int(fail_total.sum()),
        "fail_train": int(fail_train.sum()),
        "fail_std": int(fail_std.sum()),
        "fail_gap": int(fail_gap.sum()),
        "entities_after_quality": entities_after,
        "grid_rows_before": grid_before,
        "grid_rows_after": grid_after,
        "median_grid_after": median_after,
        "median_observed_after": median_after_observed,
        "retention_rounded_1": retention,
        "min_train_std_passed": min_train_std,
        "max_abs_y": max_abs_y,
        "max_grid": max_grid,
        "imputation_removes_zero_by_bound": imputation_removes_zero,
        "observed_entities_length_at_least_18": length_at_least_18,
        "forecast_length_threshold": forecasting_length_threshold(1, 4),
        "cv_requirements": cv_requirement_thresholds(),
        "records": records,
        "pandas": pd.__version__,
        "numpy": np.__version__,
    }
    out_json = SCRATCH / "stage3_summary.json"
    out_json.write_text(json.dumps(_json_ready(summary), indent=2), encoding="utf-8")
    _write_row_csv(records)
    _write_panel_csv(summary)
    print("summary_written", out_json, flush=True)
    return summary


def _fmt(value):
    if value is None:
        return ""
    if isinstance(value, float):
        return format(value, ".16g")
    return str(value)


def _write_row_csv(records):
    path = REPORTS / "constructed_target_audit.csv"
    fields = [
        "category",
        "source_file",
        "row_locator",
        "reporting_date",
        "dateq",
        "rssdid",
        "cash",
        "assets",
        "expected_y",
        "actual_y",
        "absolute_difference",
        "assets_gt_min",
        "cash_nonnegative",
        "both_finite",
        "ratio_finite",
        "selection_note",
        "distinct_bank",
        "status",
    ]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for record in records:
            writer.writerow({key: _fmt(record[key]) if key in {"cash", "assets", "expected_y", "actual_y", "absolute_difference"} else record[key] for key in fields})


def _step(name, source, rows_before, rows_after, entities_before, entities_after, rule, status, notes):
    return {
        "step_name": name,
        "source_function": source,
        "rows_before": rows_before,
        "rows_after": rows_after,
        "entities_before": entities_before,
        "entities_after": entities_after,
        "rule": rule,
        "evidence_status": status,
        "discrepancy_notes": notes,
    }


def _write_panel_csv(summary):
    table4 = summary["table4_lengths"]
    after = summary["median_grid_after"]
    notes_before = (
        f"exact median {table4['mean_of_two']}; int(median) as in create_dataset_statistics.py is {table4['int_truncated']}; "
        f"published Table 4 text is entities {PUBLISHED_TABLE4_ENTITIES}, min {PUBLISHED_TABLE4_MIN}, "
        f"median {PUBLISHED_TABLE4_MEDIAN}, max {PUBLISHED_TABLE4_MAX}"
    )
    notes_after = (
        f"grid-length median {after['mean_of_two']}; int(median) as in create_filtered_dataset_statistics.py is {after['int_truncated']}; "
        f"published Table 15 text is {PUBLISHED_TABLE15_BEFORE} to {PUBLISHED_TABLE15_AFTER}, "
        f"retention {PUBLISHED_TABLE15_RETENTION}, median after {PUBLISHED_TABLE15_MEDIAN_AFTER}; "
        f"this retention rounded to 1 decimal is {summary['retention_rounded_1']}"
    )
    steps = [
        _step(
            "raw_stata_rows",
            "callreports_1976_2020_WRDS.dta",
            summary["rows"],
            summary["rows"],
            summary["raw_entities"],
            summary["raw_entities"],
            "no filter; count rows and finite integer rssdid",
            "VERIFIED",
            f"duplicate rssdid-date rows {summary['duplicate_rssdid_date']}; non-quarter-end dates {summary['non_quarter_end_dates']}; date {summary['date_min']} to {summary['date_max']}",
        ),
        _step(
            "row_eligibility",
            "src/nyu_call_report/create_ftsfr_datasets.py cash block",
            summary["rows"],
            summary["eligible_rows"],
            summary["raw_entities"],
            summary["eligible_entities"],
            "finite rssdid, finite cash and assets, assets > 1e-10, cash >= 0, finite cash/assets",
            "VERIFIED",
            "raw MDRM items are not in this file; this is the constructed-column filter only",
        ),
        _step(
            "drop_zero_sample_std",
            "create_ftsfr_datasets.py groupby rssdid std()==0",
            summary["eligible_rows"],
            summary["constructed_rows"],
            summary["eligible_entities"],
            summary["constructed_entities"],
            "drop rssdid whose pandas/numpy sample std (ddof=1) equals 0; one-row series have NaN std and are kept",
            "VERIFIED",
            f"constant entities removed {summary['constant_entities_removed']}; one-row entities after the row filter {summary['one_row_entities_after_row_filter']}. {notes_before}",
        ),
        _step(
            "float32_cast_before_forecast_loader",
            "forecast_stats.py and create_filtered_dataset_statistics.py",
            summary["constructed_rows"],
            summary["constructed_rows"] - summary["float32_nonfinite_rows"],
            summary["constructed_entities"],
            summary["constructed_entities"],
            "cast y to float32 and null non-finite values; rows with a finite cast stay",
            "VERIFIED",
            f"non-finite after cast {summary['float32_nonfinite_rows']}",
        ),
        _step(
            "quarter_grid",
            "robust_preprocessing.build_canonical_grid QE fill_gaps freq=3mo start=per_serie end=per_serie",
            summary["constructed_rows"],
            summary["grid_rows_before"],
            summary["constructed_entities"],
            summary["constructed_entities"],
            "insert missing quarter ends between each series first and last observed date; do not pad outside that span",
            "VERIFIED"
            if summary["non_quarter_end_dates"] == 0
            and summary["duplicate_rssdid_date"] == 0
            and summary["duplicate_quarters_after_construction"] == 0
            else "BLOCKED",
            "reimplemented from the source rule; utilsforecast was not executed",
        ),
        _step(
            "robust_quality_filter",
            "robust_preprocess_pipeline -> filter_series_by_quality(get_data_requirements QE)",
            summary["grid_rows_before"],
            summary["grid_rows_after"],
            summary["constructed_entities"],
            summary["entities_after_quality"],
            (
                f"total>={summary['requirements']['min_total_obs']}, "
                f"train finite>={summary['requirements']['min_train_obs']}, "
                f"train sample std>={summary['requirements']['variance_threshold']}, "
                f"train gap ratio<={summary['requirements']['max_gap_ratio']}, "
                f"test finite>={summary['requirements']['min_test_obs']}"
            ),
            "VERIFIED",
            (
                f"first-failure entities total {summary['fail_total']}, train {summary['fail_train']}, "
                f"std {summary['fail_std']}, gap {summary['fail_gap']}. {notes_after}"
            ),
        ),
        _step(
            "train_forward_fill_validation",
            "light_train_imputation forward_fill then validate_series_after_imputation",
            summary["grid_rows_after"],
            summary["grid_rows_after"] if summary["imputation_removes_zero_by_bound"] else "",
            summary["entities_after_quality"],
            summary["entities_after_quality"] if summary["imputation_removes_zero_by_bound"] else "",
            "remove a series only for non-finite values, abs percentile beyond 1e10, or finite sample std < 1e-15",
            "VERIFIED" if summary["imputation_removes_zero_by_bound"] else "NOT_CHECKED",
            (
                f"min train std among passers {summary['min_train_std_passed']}; "
                f"max abs y {summary['max_abs_y']}; max grid {summary['max_grid']}. "
                "Imputed arrays were not materialized. The bound shows the 1e-15 rule cannot remove a passer."
            ),
        ),
        _step(
            "forecast_stats_extra_validation",
            "forecast_stats.py validate_statistical_series",
            summary["grid_rows_after"],
            summary["grid_rows_after"] if summary["imputation_removes_zero_by_bound"] else "",
            summary["entities_after_quality"],
            summary["entities_after_quality"] if summary["imputation_removes_zero_by_bound"] else "",
            "after the shared pipeline, drop >50% null, fewer than 10 finite points, or sample std < 1e-12",
            "VERIFIED" if summary["imputation_removes_zero_by_bound"] else "NOT_CHECKED",
            "Same bound. This function is not part of create_filtered_dataset_statistics.py, so Table 15 does not include it. It removes no additional entity under the bound.",
        ),
        _step(
            "filter_series_for_forecasting",
            "forecast_utils.load_and_preprocess_data; example ex_statsforecast.py",
            "",
            "",
            "",
            "",
            f"minimum length {summary['forecast_length_threshold']} on the training slice after fill_gaps(..., end=global)",
            "NOT_CHECKED",
            "forecast_stats.py, forecast_neural.py, and forecast_neural_auto.py do not call this function. A full-series length count would be a different rule, so it is not reported as this function's result. "
            f"Post-constructor entities with observed length >= 18: {summary['observed_entities_length_at_least_18']} (illustration only).",
        ),
        _step(
            "get_minimum_requirements_by_frequency",
            "forecast_utils.filter_series_by_cv_requirements",
            "",
            "",
            "",
            "",
            (
                f"QE multiplier 3: min_total {summary['cv_requirements']['min_total_obs']}, "
                f"min_train {summary['cv_requirements']['min_train_obs']}, "
                f"min_test {summary['cv_requirements']['min_test_obs']}, "
                f"min_variance {summary['cv_requirements']['min_variance']}"
            ),
            "NOT_CHECKED",
            "No call from forecast_stats.py, forecast_neural.py, or forecast_neural_auto.py. Entity count was not produced.",
        ),
    ]
    path = REPORTS / "panel_filter_reconciliation.csv"
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(steps[0].keys()))
        writer.writeheader()
        writer.writerows(steps)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--command", choices=("audit", "synthetic"), default="audit")
    args = parser.parse_args()
    if args.command == "synthetic":
        print(json.dumps(synthetic_self_check(), indent=2))
        return
    run_audit()


if __name__ == "__main__":
    main()
