"""Grids, quality screen, schedule, and prepared series for one target, both arms.

Same logic as the Stage 18 prepare script for Bank Cash Liquidity, parameterized:
Arm A = upstream build_canonical_grid(frame, "QE") (fill_gaps "3mo", per-series anchors),
Arm B = strict calendar quarter-end grid; frozen QE thresholds; forward-fill training
imputation; last six grid positions as one-step targets. Usage:
    python confirm/prepare.py bank_leverage
"""
from __future__ import annotations

import contextlib
import io
import json
import sys
import time

import polars as pl

import common as C
from p12_bank_liquidity.candidate_a.pilot_guard import entity_rank


def stamp(quarter: str, day: int) -> str:
    return f"{int(quarter[:4]):04d}-{int(quarter[5]) * 3:02d}-{int(day):02d}"


def finite_or_none(value):
    if value is None:
        return None
    number = float(value)
    return None if number != number or number in (float("inf"), float("-inf")) else number


def arm_targets(keys):
    found, dup = {}, 0
    for entity, oq, tq, od, td in keys:
        slot = found.setdefault(entity, {})
        target = stamp(tq, td)
        dup += target in slot
        slot[target] = (stamp(oq, od), oq, tq)
    return found, dup


def main() -> int:
    started = time.perf_counter()
    target = sys.argv[1]
    run = C.run_dir(target)
    s10 = C.stage10()
    frame = pl.read_parquet(run / "panel.parquet").select(
        pl.col("unique_id").cast(pl.Utf8), pl.col("ds").cast(pl.Datetime("ns")), pl.col("y").cast(pl.Float64)
    )
    req = s10.requirements_or_stop()
    arm_a = s10.run_arm(frame, "A", req)
    arm_b = s10.run_arm(frame, "B", req)
    left, dup_a = arm_targets(arm_a["_keys"])
    right, dup_b = arm_targets(arm_b["_keys"])
    if dup_a or dup_b:
        raise SystemExit("STOP duplicate keys")

    pairs, only_a, only_b = [], 0, 0
    for entity in set(left) | set(right):
        ta, tb = left.get(entity, {}), right.get(entity, {})
        for t in set(ta) | set(tb):
            if t in ta and t in tb:
                (oa, qa, tqa), (ob, qb, tqb) = ta[t], tb[t]
                pairs.append({
                    "entity_hash": entity_rank(entity), "origin_timestamp_arm_a": oa, "origin_timestamp_arm_b": ob,
                    "target_timestamp": t, "origin_timestamp_equal": oa == ob, "target_quarter_equal": tqa == tqb,
                })
            elif t in ta:
                only_a += 1
            else:
                only_b += 1
    pairs = pl.DataFrame(pairs)

    schedule = pl.DataFrame([
        {"entity_hash": entity_rank(e), "arm_id": arm, "origin_timestamp": o, "target_timestamp": t,
         "origin_quarter": oq, "target_quarter": tq, "label": C.LABEL}
        for arm, mapping in (("A", left), ("B", right))
        for e, ts in mapping.items()
        for t, (o, oq, tq) in ts.items()
    ])

    chosen = frame.filter(pl.col("unique_id").is_in(list(set(left) | set(right))))
    rows = []
    sink = io.StringIO()
    for arm in ("A", "B"):
        with contextlib.redirect_stdout(sink):
            grid = s10.build_canonical_grid(chosen, "QE") if arm == "A" else s10.quarter_grid(chosen)
            train, test = s10.split_train_test_aligned(grid, 1)
            train_f, test_f, _ = s10.filter_series_by_quality(train, test, req)
            train_i = (s10.light_train_imputation(train_f, 4, "forward_fill") if train_f["y"].null_count() > 0
                       else train_f.with_columns(pl.col("y").alias("y_imputed")))
        source = chosen.select(["unique_id", pl.col("ds").cast(pl.Datetime("ns")), pl.col("y").alias("source_y")])
        tr = train_i.select(["unique_id", pl.col("ds").cast(pl.Datetime("ns")), "y_imputed"]).join(
            source, on=["unique_id", "ds"], how="left")
        te = (test_f.select(["unique_id", pl.col("ds").cast(pl.Datetime("ns"))])
              .with_columns(pl.lit(None).cast(pl.Float64).alias("y_imputed"))
              .join(source, on=["unique_id", "ds"], how="left"))
        for r in pl.concat([tr, te], how="diagonal").iter_rows(named=True):
            v = finite_or_none(r["source_y"])
            rows.append({"entity_hash": entity_rank(r["unique_id"]), "arm_id": arm, "timestamp": str(r["ds"])[:10],
                         "value": v, "y_imputed": finite_or_none(r["y_imputed"]), "synthetic_grid_row": v is None})
    series = pl.DataFrame(rows)

    # every scheduled target and origin must be on that arm's prepared grid with a finite target
    present = series.select(["entity_hash", "arm_id", "timestamp", "value"]).with_columns(pl.lit(True).alias("on"))
    tcheck = schedule.join(present, left_on=["entity_hash", "arm_id", "target_timestamp"],
                           right_on=["entity_hash", "arm_id", "timestamp"], how="left")
    ocheck = schedule.join(present, left_on=["entity_hash", "arm_id", "origin_timestamp"],
                           right_on=["entity_hash", "arm_id", "timestamp"], how="left")
    bad = int(tcheck["on"].is_null().sum() + tcheck["value"].is_null().sum() + ocheck["on"].is_null().sum())
    if bad:
        raise SystemExit(f"STOP prepared grid does not contain the schedule ({bad})")

    # first stored filing per entity (model-independent covariate fixed by the protocol)
    first = (frame.group_by("unique_id").agg(pl.col("ds").min().alias("first"), pl.col("ds").max().alias("last"))
             .select(pl.col("unique_id").map_elements(entity_rank, return_dtype=pl.Utf8).alias("entity_hash"),
                     pl.col("first").dt.strftime("%Y-%m-%d"), pl.col("last").dt.strftime("%Y-%m-%d")))

    schedule.write_parquet(run / "schedule.parquet")
    series.write_parquet(run / "prepared_series.parquet")
    pairs.write_parquet(run / "primary_pairs.parquet")
    first.write_parquet(run / "entity_dates.parquet")
    summary = {
        "label": C.LABEL, "target": target,
        "arm_a": s10.public_arm(arm_a), "arm_b": s10.public_arm(arm_b),
        "counts": {"arm_a_eligible_rows": len(arm_a["_keys"]), "arm_b_eligible_rows": len(arm_b["_keys"]),
                   "primary_common_keys": pairs.height, "arm_a_only_keys": only_a, "arm_b_only_keys": only_b,
                   "different_origin_keys": int((~pairs["origin_timestamp_equal"]).sum()) if pairs.height else 0},
        "seconds": round(time.perf_counter() - started, 1),
    }
    (run / "prepare_summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print("PREPARED", target, json.dumps(summary["counts"]), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
