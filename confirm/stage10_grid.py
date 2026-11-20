"""Grid and screen routines used by the historical Stage 10 preparation script."""
from __future__ import annotations

import contextlib
import io
import os
import sys
from pathlib import Path

import polars as pl

import common as C

upstream = Path(os.environ.get("P12_UPSTREAM", C.SCRATCH / "upstream_src"))
sys.path.insert(0, str(upstream))
from robust_preprocessing import (  # noqa: E402
    build_canonical_grid,
    filter_series_by_quality,
    get_data_requirements,
    light_train_imputation,
    split_train_test_aligned,
)


def quarter_label(year: int, month: int) -> str:
    return f"{int(year)}Q{(int(month) - 1) // 3 + 1}"


def requirements_or_stop():
    requirements = get_data_requirements("QE", 1, 4)
    expected = {
        "min_total_obs": 27,
        "min_train_obs": 18,
        "min_test_obs": 1,
        "variance_threshold": 0.01,
        "max_gap_ratio": 0.6,
    }
    for key, value in expected.items():
        if requirements[key] != value:
            raise SystemExit(f"STOP threshold {key} is {requirements[key]}")
    return requirements


def quarter_grid(frame: pl.DataFrame) -> pl.DataFrame:
    bounds = frame.group_by("unique_id").agg(
        pl.col("ds").min().alias("start"),
        pl.col("ds").max().alias("end"),
    )
    bounds = bounds.with_columns(
        (pl.col("start").dt.year() * 4 + (pl.col("start").dt.month() - 1) // 3).alias("start_i"),
        (pl.col("end").dt.year() * 4 + (pl.col("end").dt.month() - 1) // 3).alias("end_i"),
    )
    long = bounds.select(
        "unique_id",
        pl.int_ranges("start_i", pl.col("end_i") + 1).alias("qi"),
    ).explode("qi")
    long = long.with_columns(
        (pl.col("qi") // 4).alias("year"),
        (pl.col("qi") % 4).alias("q0"),
    )
    long = long.with_columns((((pl.col("q0") + 1) * 3)).alias("month"))
    long = long.with_columns(
        pl.when(pl.col("month").is_in([3, 12])).then(31).otherwise(30).alias("day")
    )
    long = long.with_columns(
        pl.date("year", "month", "day").cast(pl.Datetime("ns")).alias("ds")
    ).select(["unique_id", "ds"])
    observed = frame.select(
        ["unique_id", pl.col("ds").cast(pl.Datetime("ns")), pl.col("y").alias("y_obs")]
    )
    grid = long.join(observed, on=["unique_id", "ds"], how="left")
    grid = grid.with_columns(pl.col("y_obs").alias("y"))
    return grid.select(["unique_id", "ds", "y"])


def first_failures(train: pl.DataFrame, test: pl.DataFrame, requirements: dict) -> dict:
    train_stats = train.group_by("unique_id").agg(
        pl.len().alias("train_len"),
        pl.col("y").is_finite().sum().alias("train_finite"),
        pl.col("y").filter(pl.col("y").is_finite()).std(ddof=1).alias("train_std"),
    )
    test_stats = test.group_by("unique_id").agg(
        pl.len().alias("test_len"),
        pl.col("y").is_finite().sum().alias("test_finite"),
    )
    stats = train_stats.join(test_stats, on="unique_id", how="full", coalesce=True).with_columns(
        pl.col("train_len").fill_null(0),
        pl.col("test_len").fill_null(0),
        pl.col("train_finite").fill_null(0),
        pl.col("test_finite").fill_null(0),
    )
    stats = stats.with_columns(
        (pl.col("train_len") + pl.col("test_len")).alias("total"),
        pl.when(pl.col("train_len") > 0)
        .then(1 - pl.col("train_finite") / pl.col("train_len"))
        .otherwise(1.0)
        .alias("gap"),
    )
    total_fail = pl.col("total") < requirements["min_total_obs"]
    train_fail = pl.col("train_finite") < requirements["min_train_obs"]
    std_fail = pl.col("train_std").is_null() | (pl.col("train_std") < requirements["variance_threshold"])
    # one finite training value has no sample std in the source only when train_non_null > 1
    std_fail = (pl.col("train_finite") > 1) & std_fail
    gap_fail = pl.col("gap") > requirements["max_gap_ratio"]
    test_fail = pl.col("test_finite") < requirements["min_test_obs"]
    reason = (
        pl.when(total_fail)
        .then(pl.lit("total"))
        .when(train_fail)
        .then(pl.lit("train"))
        .when(std_fail)
        .then(pl.lit("std"))
        .when(gap_fail)
        .then(pl.lit("gap"))
        .when(test_fail)
        .then(pl.lit("test"))
        .otherwise(pl.lit("pass"))
    )
    labeled = stats.with_columns(reason.alias("reason"))
    counts = {row["reason"]: int(row["len"]) for row in labeled.group_by("reason").len().iter_rows(named=True)}
    return counts


def run_arm(frame: pl.DataFrame, arm: str, requirements: dict) -> dict:
    print(f"PHASE arm {arm} grid", flush=True)
    sink = io.StringIO()
    with contextlib.redirect_stdout(sink):
        if arm == "A":
            grid = build_canonical_grid(frame, "QE")
        else:
            grid = quarter_grid(frame)
        train, test = split_train_test_aligned(grid, 1)
    grid = grid.with_columns(pl.col("ds").cast(pl.Datetime("ns")))
    original = frame.select(
        ["unique_id", pl.col("ds").cast(pl.Datetime("ns")).alias("ds"), pl.col("y").alias("source_y")]
    )
    joined = grid.join(original, on=["unique_id", "ds"], how="left")
    observed_rows = int(joined.filter(pl.col("y").is_finite() & pl.col("source_y").is_finite()).height)
    synthetic_rows = int(joined.filter(pl.col("source_y").is_null()).height)
    finite_without_source = int(joined.filter(pl.col("y").is_finite() & pl.col("source_y").is_null()).height)
    if finite_without_source:
        raise SystemExit(f"STOP {arm} finite grid value without a source date")
    source_missing = int(
        original.join(grid.select(["unique_id", "ds"]), on=["unique_id", "ds"], how="anti").height
    )
    failures = first_failures(train, test, requirements)
    with contextlib.redirect_stdout(sink):
        train_f, test_f, valid = filter_series_by_quality(train, test, requirements)
    if int(failures.get("pass", 0)) != len(valid):
        raise SystemExit(
            f"STOP {arm} filter count mismatch vector {failures.get('pass', 0)} official {len(valid)}"
        )
    before_impute = train_f["unique_id"].n_unique()
    with contextlib.redirect_stdout(sink):
        if train_f["y"].null_count() > 0:
            train_i = light_train_imputation(train_f, 4, "forward_fill")
        else:
            train_i = train_f.with_columns(pl.col("y").alias("y_imputed"))
    after_ids = set(train_i["unique_id"].unique().to_list())
    test_i = test_f.filter(pl.col("unique_id").is_in(list(after_ids)))
    removed_impute = int(before_impute - len(after_ids))
    if "y_imputed" in train_i.columns:
        train_cv = train_i.select(["unique_id", "ds", pl.col("y_imputed").alias("y")])
    else:
        train_cv = train_i.select(["unique_id", "ds", "y"])
    test_cv = test_i.select(["unique_id", "ds", "y"])
    cv = pl.concat([train_cv, test_cv], how="vertical")
    cv_stats = cv.group_by("unique_id").agg(
        pl.len().alias("total"),
        pl.col("y").is_not_null().sum().alias("non_null"),
        pl.col("y").filter(pl.col("y").is_finite()).std(ddof=1).alias("std"),
        pl.col("y").filter(pl.col("y").is_finite()).len().alias("finite_n"),
    )
    cv_stats = cv_stats.with_columns(
        pl.when(pl.col("non_null") < pl.col("total") * 0.5)
        .then(pl.lit("half_null"))
        .when(pl.col("non_null") < 10)
        .then(pl.lit("under_10"))
        .when((pl.col("finite_n") > 1) & pl.col("std").is_not_null() & (pl.col("std") < 1e-12))
        .then(pl.lit("near_zero_std"))
        .otherwise(pl.lit("pass"))
        .alias("reason")
    )
    classical = {
        row["reason"]: int(row["len"]) for row in cv_stats.group_by("reason").len().iter_rows(named=True)
    }
    keep = cv_stats.filter(pl.col("reason") == "pass")["unique_id"]
    final_cv = cv.filter(pl.col("unique_id").is_in(keep))
    lengths = final_cv.group_by("unique_id").len()
    min_length = int(lengths["len"].min()) if lengths.height else 0
    horizon = 1
    n_windows = 1 if min_length <= 0 else max(1, min(6, max(1, min_length // horizon)))
    if min_length <= 0:
        n_windows = 0
    observed = joined.select(
        ["unique_id", "ds", pl.col("source_y").is_finite().alias("observed")]
    )
    final_rows = final_cv.join(observed, on=["unique_id", "ds"], how="left")
    final_rows = final_rows.with_columns(pl.col("observed").fill_null(False))
    final_rows = final_rows.sort(["unique_id", "ds"]).with_columns(
        pl.col("ds").rank(method="ordinal").over("unique_id").alias("pos"),
        pl.len().over("unique_id").alias("n"),
    )
    targets = final_rows.filter(pl.col("pos") > pl.col("n") - n_windows)
    targets = targets.with_columns(
        pl.col("ds").shift(1).over("unique_id").alias("origin_ds")
    )
    valid_targets = targets.filter(pl.col("observed") & pl.col("origin_ds").is_not_null())
    no_origin_entities = int(
        final_rows["unique_id"].n_unique()
        - valid_targets["unique_id"].n_unique()
    )
    keys = valid_targets.select(
        [
            "unique_id",
            pl.col("origin_ds").dt.year().alias("oy"),
            pl.col("origin_ds").dt.month().alias("om"),
            pl.col("ds").dt.year().alias("ty"),
            pl.col("ds").dt.month().alias("tm"),
            pl.col("ds").dt.day().alias("td"),
            pl.col("origin_ds").dt.day().alias("od"),
        ]
    )
    key_rows = [
        (
            row["unique_id"],
            quarter_label(row["oy"], row["om"]),
            quarter_label(row["ty"], row["tm"]),
            int(row["od"]),
            int(row["td"]),
        )
        for row in keys.iter_rows(named=True)
    ]
    target_month_days = {}
    for row in targets.group_by(pl.col("ds").dt.strftime("%m-%d").alias("md")).len().iter_rows(named=True):
        target_month_days[row["md"]] = int(row["len"])
    print(
        f"PHASE arm {arm} pass {failures.get('pass', 0)} final {final_cv['unique_id'].n_unique()} windows {n_windows}",
        flush=True,
    )
    return {
        "grid_rows": int(grid.height),
        "grid_entities": int(grid["unique_id"].n_unique()),
        "observed_exact_date_rows": observed_rows,
        "synthetic_or_unmatched_grid_rows": synthetic_rows,
        "source_rows_absent_from_grid": source_missing,
        "finite_value_on_non_source_date": finite_without_source,
        "first_failure": failures,
        "entities_after_quality_filter": int(before_impute),
        "entities_removed_by_post_imputation_validation": removed_impute,
        "entities_after_imputation": int(len(after_ids)),
        "classical_filter": classical,
        "final_eligible_entities": int(final_cv["unique_id"].n_unique()),
        "final_rows": int(final_cv.height),
        "observed_finite_rows_in_final_frame": int(final_rows.filter(pl.col("observed")).height),
        "synthetic_rows_in_final_frame": int(final_rows.filter(~pl.col("observed")).height),
        "panel_min_length": min_length,
        "panel_n_windows": int(n_windows),
        "target_rows_in_window": int(targets.height),
        "valid_observed_targets_in_window": int(valid_targets.height),
        "entities_with_a_valid_target": int(valid_targets["unique_id"].n_unique()) if valid_targets.height else 0,
        "entities_with_no_valid_evaluation_origin": no_origin_entities,
        "target_month_day_counts": target_month_days,
        "_keys": key_rows,
    }
