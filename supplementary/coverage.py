"""Model-independent grid and coverage summaries from the saved schedule and grids.

Per-target eligibility uses the frozen thresholds in the frozen first-failure
order: total length, finite training count, training standard deviation, then
gap fraction. No call-report panel is read.
"""
from __future__ import annotations

import math
from collections import Counter

import numpy as np
import pandas as pd

QUARTER_ENDS = {"03-31", "06-30", "09-30", "12-31"}
MIN_TOTAL = 27
MIN_TRAIN = 18
MIN_STD = 0.01
MAX_GAP = 0.6


def _monthday(timestamp: str) -> str:
    return timestamp[5:10]


def _quarter(timestamp: str) -> tuple[int, int]:
    year = int(timestamp[:4])
    month = int(timestamp[5:7])
    return year, (month - 1) // 3


def _finite_array(values: list[object]) -> np.ndarray:
    out = np.empty(len(values), dtype=bool)
    for index, value in enumerate(values):
        if value is None:
            out[index] = False
            continue
        try:
            number = float(value)
        except (TypeError, ValueError):
            out[index] = False
            continue
        out[index] = math.isfinite(number)
    return out


def grid_anchor(prepared: pd.DataFrame) -> dict:
    arm = prepared.loc[prepared["arm_id"] == "A", ["entity_hash", "timestamp"]].copy()
    arm["monthday"] = arm["timestamp"].str.slice(5, 10)
    stamp_counts = {key: int(value) for key, value in arm["monthday"].value_counts().sort_index().items()}
    n_stamps = int(len(arm))
    off_counts = {key: value for key, value in stamp_counts.items() if key not in QUARTER_ENDS}
    first = arm.sort_values(["entity_hash", "timestamp"], kind="mergesort").groupby("entity_hash", sort=False).head(1)
    off_entities = set(arm.loc[~arm["monthday"].isin(QUARTER_ENDS), "entity_hash"])
    by_anchor: dict[str, dict[str, int]] = {}
    for anchor, entity in zip(first["monthday"], first["entity_hash"]):
        slot = by_anchor.setdefault(str(anchor), {"entities": 0, "entities_with_off_stamp": 0})
        slot["entities"] += 1
        if entity in off_entities:
            slot["entities_with_off_stamp"] += 1
    anchors_with_off = {key for key, value in by_anchor.items() if value["entities_with_off_stamp"] > 0}
    if not anchors_with_off:
        drift = "no off-quarter-end stamp"
    elif all(int(key[3:]) == 30 for key in anchors_with_off):
        drift = "only from day-30 anchors"
    else:
        drift = "from any anchor"
    n_entities = int(first["entity_hash"].nunique())
    n_with_off = int(len(off_entities))
    return {
        "n_stamps": n_stamps,
        "n_entities": n_entities,
        "n_entities_with_off_stamp": n_with_off,
        "share_entities_with_off_stamp": float(n_with_off / n_entities) if n_entities else float("nan"),
        "stamp_monthday_counts": stamp_counts,
        "off_stamp_monthday_counts": off_counts,
        "by_first_stamp": dict(sorted(by_anchor.items())),
        "drift": drift,
    }


def _eligibility(timestamps: list[str], finite_mask: np.ndarray, values: np.ndarray, target: str) -> str:
    total = len(timestamps)
    target_at = timestamps.index(target)
    train_finite_mask = finite_mask[:target_at]
    train_len = int(target_at)
    train_finite = int(train_finite_mask.sum())
    if total < MIN_TOTAL:
        return "total_length"
    if train_finite < MIN_TRAIN:
        return "min_18_finite_train"
    if train_finite > 1:
        train_values = values[:target_at][train_finite_mask]
        center = float(train_values.mean())
        variance = float(np.sum((train_values - center) ** 2) / (train_finite - 1))
        if not math.isfinite(variance) or math.sqrt(variance) < MIN_STD:
            return "std"
    gap = 1.0 if train_len == 0 else 1.0 - train_finite / train_len
    if gap > MAX_GAP:
        return "gap_fraction"
    return "pass"


def _classify_side(
    only_keys: set[tuple[str, str]],
    other_entities: set[str],
    other_grids: dict[str, dict],
    missing_label: str,
    not_on_grid_label: str,
    missing_value_label: str,
    failed_label: str,
) -> dict:
    rules = [
        missing_label,
        not_on_grid_label,
        missing_value_label,
        failed_label,
        "other",
    ]
    counts = {rule: 0 for rule in rules}
    subs: dict[str, Counter] = {rule: Counter() for rule in rules}
    for entity, target in only_keys:
        if entity not in other_entities:
            counts[missing_label] += 1
            subs[missing_label]["no per-entity screen log in the saved run"] += 1
            continue
        grid = other_grids.get(entity)
        if grid is None:
            raise RuntimeError("scheduled entity is missing from the saved grid")
        pos = grid["pos"]
        if target not in pos:
            counts[not_on_grid_label] += 1
            quarter = _quarter(target)
            same = [_monthday(timestamp) for timestamp in grid["ts"] if _quarter(timestamp) == quarter]
            label = "+".join(same) if same else "none"
            subs[not_on_grid_label][label] += 1
            continue
        if not bool(grid["finite"][pos[target]]):
            counts[missing_value_label] += 1
            subs[missing_value_label]["source value not finite"] += 1
            continue
        reason = _eligibility(grid["ts"], grid["finite"], grid["values"], target)
        if reason == "pass":
            counts["other"] += 1
            subs["other"]["numeric criteria pass and the target is absent from the saved schedule"] += 1
            continue
        counts[failed_label] += 1
        subs[failed_label][reason] += 1
    if sum(counts.values()) != len(only_keys):
        raise RuntimeError("classification does not sum to the key count")
    payload = []
    for rule in rules:
        payload.append(
            {
                "rule": rule,
                "count": int(counts[rule]),
                "subs": [{"label": label, "count": int(count)} for label, count in sorted(subs[rule].items())],
            }
        )
    return {"n": int(len(only_keys)), "rules": payload}


def _grids_by_arm(prepared: pd.DataFrame) -> dict[str, dict[str, dict]]:
    out: dict[str, dict[str, dict]] = {"A": {}, "B": {}}
    for keys, group in prepared.groupby(["entity_hash", "arm_id"], sort=False):
        group = group.sort_values("timestamp", kind="mergesort")
        timestamps = group["timestamp"].tolist()
        finite_mask = _finite_array(group["value"].tolist())
        values = np.array([float(value) if flag else np.nan for value, flag in zip(group["value"], finite_mask)], dtype=float)
        out[str(keys[1])][keys[0]] = {
            "ts": timestamps,
            "pos": {timestamp: index for index, timestamp in enumerate(timestamps)},
            "finite": finite_mask,
            "values": values,
        }
    return out


def classify_exclusive(schedule: pd.DataFrame, prepared: pd.DataFrame) -> dict:
    grids = _grids_by_arm(prepared)
    entities = {"A": set(), "B": set()}
    keys = {"A": set(), "B": set()}
    for row in schedule.itertuples(index=False):
        entities[row.arm_id].add(row.entity_hash)
        keys[row.arm_id].add((row.entity_hash, row.target_timestamp))
    b_only = keys["B"] - keys["A"]
    a_only = keys["A"] - keys["B"]
    if len(b_only) != 12_594 or len(a_only) != 18:
        raise RuntimeError(f"exclusive key counts B {len(b_only)} A {len(a_only)}")
    return {
        "arm_b_only": _classify_side(
            b_only,
            entities["A"],
            grids["A"],
            "entity_not_in_A",
            "target_not_on_A_grid",
            "target_on_A_grid_missing",
            "failed_A_eligibility",
        ),
        "arm_a_only": _classify_side(
            a_only,
            entities["B"],
            grids["B"],
            "entity_not_in_B",
            "target_not_on_B_grid",
            "target_on_B_grid_missing",
            "failed_B_eligibility",
        ),
    }


def origin_monthdays(pairs: pd.DataFrame) -> list[dict[str, object]]:
    differ = pairs.loc[~pairs["origin_timestamp_equal"].astype(bool)]
    if len(differ) != 1_190:
        raise RuntimeError("different-origin pair count is not 1190")
    left = differ["origin_timestamp_arm_a"].str.slice(5, 10)
    right = differ["origin_timestamp_arm_b"].str.slice(5, 10)
    counted = Counter(zip(left.tolist(), right.tolist()))
    rows = [
        {"arm_a_monthday": key[0], "arm_b_monthday": key[1], "count": int(value)}
        for key, value in sorted(counted.items(), key=lambda item: (-item[1], item[0]))
    ]
    if sum(row["count"] for row in rows) != 1_190:
        raise RuntimeError("month-day tabulation does not sum to 1190")
    return rows
