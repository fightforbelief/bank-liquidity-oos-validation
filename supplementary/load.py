"""Map the frozen Stage 18 files onto the canonical long schema.

Nothing here refits a model, regenerates a forecast, or rewrites a frozen file.
The MASE denominator is not a column in the forecast batches. It is derived from
the saved imputed training grid with the same retained-position rule that
matched every stored MASE in the Stage 19 audit.
"""
from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pandas as pd

from p12_supplementary import MODELS

RUN = Path(r"C:\Users\hanqi\Desktop\projects\_p12_stage18_full_run\run_20261028")
N_KEYS = 85_205
N_ENTITIES = 17_377
N_SAME_ORIGIN = 84_015
N_DIFF_ORIGIN = 1_190
N_ARM_A = 85_223
N_ARM_B = 97_799

LAST_IDENTITY: dict[str, float | int] = {}
_CACHE: dict[str, object] = {}


def finite(value: object) -> bool:
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return False
    try:
        number = float(value)
    except (TypeError, ValueError):
        return False
    return math.isfinite(number)


def stamp(value: object) -> str:
    return str(value)[:10]


def as_bool(value: object) -> bool:
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "t", "yes"}
    if not finite(value) and not isinstance(value, (bool, np.bool_)):
        return False
    return bool(value)


def training_scale(
    timestamps: list[str], imputed: list[object], origin: str, target: str
) -> tuple[float | None, str, int]:
    """Retained-position lag-4 scale. Leading non-finite values are dropped.

    A later non-finite imputed value makes the scale undefined. The denominator
    is the mean absolute lag-4 difference of the retained tail.
    """
    selected: list[object] = []
    for timestamp, value in zip(timestamps, imputed):
        if timestamp > origin or timestamp >= target:
            continue
        selected.append(value)
    if not any(finite(value) for value in selected):
        return None, "empty_training", 0
    first = next(index for index, value in enumerate(selected) if finite(value))
    tail = selected[first:]
    if any(not finite(value) for value in tail):
        return None, "interior_gap", len(tail)
    values = [float(value) for value in tail]
    pairs = [abs(values[index] - values[index - 4]) for index in range(4, len(values))]
    if not pairs:
        return None, "short_history", len(values)
    scale = sum(pairs) / len(pairs)
    if scale == 0.0:
        return None, "zero_denominator", len(values)
    return scale, "defined", len(values)


def _window_record(grid: dict, origin: str, target: str) -> dict[str, object] | str:
    timestamps: list[str] = grid["ts"]
    values: list[object] = grid["value"]
    imputed: list[object] = grid["imputed"]
    synthetic: list[bool] = grid["synthetic"]
    pos: dict[str, int] = grid["pos"]
    if origin not in pos or target not in pos:
        return "stamp_missing"
    origin_index = pos[origin]
    if not (origin < target):
        return "origin_not_before_target"
    scale, reason, n_train = training_scale(timestamps, imputed, origin, target)
    if scale is None:
        return reason
    selected_val: list[object] = []
    selected_imp: list[object] = []
    for timestamp, value, imp in zip(timestamps, values, imputed):
        if timestamp > origin or timestamp >= target:
            continue
        selected_val.append(value)
        selected_imp.append(imp)
    first = next(index for index, value in enumerate(selected_imp) if finite(value))
    tail_val = selected_val[first:]
    tail_imp = [float(value) for value in selected_imp[first:]]
    pairs = [abs(tail_imp[index] - tail_imp[index - 4]) for index in range(4, len(tail_imp))]
    if len(tail_imp) != n_train or not pairs:
        return "tail_mismatch"
    steps = 0
    cursor = origin_index
    while cursor >= 0 and not (finite(values[cursor]) and not synthetic[cursor]):
        steps += 1
        cursor -= 1
    actual = values[pos[target]]
    if not finite(actual):
        return "nonfinite_target"
    return {
        "y_true": float(actual),
        "mase_den": float(scale),
        "n_train_obs": int(n_train),
        "n_imputed_train": int(sum(1 for value in tail_val if not finite(value))),
        "zero_seasonal_share": float(sum(pair == 0.0 for pair in pairs) / len(pairs)),
        "origin_is_synthetic": bool(synthetic[origin_index] or not finite(values[origin_index])),
        "staleness_q": int(steps),
    }


def read_inputs() -> dict[str, pd.DataFrame]:
    if "inputs" in _CACHE:
        return _CACHE["inputs"]  # type: ignore[return-value]
    print("LOAD pairs schedule prepared", flush=True)
    pairs = pd.read_parquet(RUN / "primary_pairs.parquet")
    schedule = pd.read_parquet(RUN / "schedule.parquet")
    prepared = pd.read_parquet(
        RUN / "prepared_series.parquet",
        columns=["entity_hash", "arm_id", "timestamp", "value", "y_imputed", "synthetic_grid_row"],
    )
    print("LOAD forecasts", flush=True)
    batches = sorted((RUN / "batches").glob("batch_*.parquet"))
    forecasts = pd.concat(
        [
            pd.read_parquet(
                path,
                columns=[
                    "entity_hash",
                    "arm_id",
                    "model_id",
                    "origin_timestamp",
                    "target_timestamp",
                    "prediction",
                    "mase_value",
                    "status",
                    "mase_status",
                ],
            )
            for path in batches
        ],
        ignore_index=True,
    )
    for frame, columns in (
        (pairs, ["origin_timestamp_arm_a", "origin_timestamp_arm_b", "target_timestamp"]),
        (schedule, ["origin_timestamp", "target_timestamp"]),
        (prepared, ["timestamp"]),
        (forecasts, ["origin_timestamp", "target_timestamp"]),
    ):
        for column in columns:
            frame[column] = frame[column].map(stamp)
    _CACHE["inputs"] = {"pairs": pairs, "schedule": schedule, "prepared": prepared, "forecasts": forecasts}
    _CACHE["batch_names"] = [path.name for path in batches]
    return _CACHE["inputs"]  # type: ignore[return-value]


def build_grids(prepared: pd.DataFrame) -> dict[tuple[str, str], dict]:
    if "grids" in _CACHE:
        return _CACHE["grids"]  # type: ignore[return-value]
    print("BUILD grids", flush=True)
    duplicate = int(prepared.duplicated(["entity_hash", "arm_id", "timestamp"]).sum())
    if duplicate:
        raise RuntimeError(f"prepared grid has {duplicate} duplicate stamps")
    grids: dict[tuple[str, str], dict] = {}
    for keys, group in prepared.groupby(["entity_hash", "arm_id"], sort=False):
        group = group.sort_values("timestamp", kind="mergesort")
        timestamps = group["timestamp"].tolist()
        grids[(keys[0], keys[1])] = {
            "ts": timestamps,
            "value": group["value"].tolist(),
            "imputed": group["y_imputed"].tolist(),
            "synthetic": [as_bool(value) for value in group["synthetic_grid_row"].tolist()],
            "pos": {timestamp: index for index, timestamp in enumerate(timestamps)},
        }
    _CACHE["grids"] = grids
    return grids


def load_long() -> pd.DataFrame:
    """Primary common keys, pair-file order within each model and arm."""
    inputs = read_inputs()
    pairs = inputs["pairs"]
    forecasts = inputs["forecasts"]
    grids = build_grids(inputs["prepared"])
    if len(pairs) != N_KEYS:
        raise RuntimeError(f"pair rows {len(pairs)} != {N_KEYS}")
    if int(pairs["origin_timestamp_equal"].sum()) != N_SAME_ORIGIN:
        raise RuntimeError("same-origin count mismatch")
    if int((~pairs["origin_timestamp_equal"]).sum()) != N_DIFF_ORIGIN:
        raise RuntimeError("different-origin count mismatch")
    schedule = inputs["schedule"]
    arm_counts = schedule["arm_id"].value_counts().to_dict()
    if int(arm_counts.get("A", 0)) != N_ARM_A or int(arm_counts.get("B", 0)) != N_ARM_B:
        raise RuntimeError(f"schedule arm counts {arm_counts}")
    if set(forecasts["status"]) != {"RECORDED_FORECAST"} or set(forecasts["mase_status"]) != {"DEFINED"}:
        raise RuntimeError("forecast status is not entirely recorded and defined")
    if set(forecasts["model_id"]) != set(MODELS):
        raise RuntimeError("forecast model ids differ from the frozen list")

    print("INDEX forecasts", flush=True)
    prediction: dict[tuple[str, str, str, str], float] = {}
    stored_mase: dict[tuple[str, str, str, str], float] = {}
    forecast_origin: dict[tuple[str, str, str, str], str] = {}
    for row in forecasts.itertuples(index=False):
        key = (row.entity_hash, row.model_id, row.arm_id, row.target_timestamp)
        if key in prediction:
            raise RuntimeError("duplicate forecast key")
        prediction[key] = float(row.prediction)
        stored_mase[key] = float(row.mase_value)
        forecast_origin[key] = row.origin_timestamp
    del forecasts

    print("WINDOW metadata", flush=True)
    meta: dict[tuple[str, str, str, str], dict[str, object]] = {}
    failures: dict[str, int] = {}
    for arm, origin_column in (("A", "origin_timestamp_arm_a"), ("B", "origin_timestamp_arm_b")):
        for row in pairs.itertuples(index=False):
            key = (row.entity_hash, arm, getattr(row, origin_column), row.target_timestamp)
            if key in meta:
                continue
            grid = grids.get((row.entity_hash, arm))
            if grid is None:
                failures["grid_missing"] = failures.get("grid_missing", 0) + 1
                continue
            record = _window_record(grid, getattr(row, origin_column), row.target_timestamp)
            if isinstance(record, str):
                failures[record] = failures.get(record, 0) + 1
                continue
            meta[key] = record
    if failures:
        raise RuntimeError("window metadata failed " + ",".join(f"{k}:{v}" for k, v in sorted(failures.items())))

    print("ASSEMBLE long frame", flush=True)
    columns: dict[str, list] = {
        name: []
        for name in (
            "entity",
            "target_ts",
            "arm",
            "model",
            "origin_ts",
            "y_true",
            "y_pred",
            "mase_den",
            "mase",
            "n_train_obs",
            "n_imputed_train",
            "zero_seasonal_share",
            "origin_is_synthetic",
            "staleness_q",
            "pair_ord",
        )
    }
    missing_forecast = 0
    origin_mismatch = 0
    for pair_ord, row in enumerate(pairs.itertuples(index=False)):
        for model in MODELS:
            for arm, origin_column in (("A", "origin_timestamp_arm_a"), ("B", "origin_timestamp_arm_b")):
                origin = getattr(row, origin_column)
                fkey = (row.entity_hash, model, arm, row.target_timestamp)
                if fkey not in prediction:
                    missing_forecast += 1
                    continue
                if forecast_origin[fkey] != origin:
                    origin_mismatch += 1
                    continue
                record = meta[(row.entity_hash, arm, origin, row.target_timestamp)]
                columns["entity"].append(row.entity_hash)
                columns["target_ts"].append(row.target_timestamp)
                columns["arm"].append(arm)
                columns["model"].append(model)
                columns["origin_ts"].append(origin)
                columns["y_true"].append(record["y_true"])
                columns["y_pred"].append(prediction[fkey])
                columns["mase_den"].append(record["mase_den"])
                columns["mase"].append(stored_mase[fkey])
                columns["n_train_obs"].append(record["n_train_obs"])
                columns["n_imputed_train"].append(record["n_imputed_train"])
                columns["zero_seasonal_share"].append(record["zero_seasonal_share"])
                columns["origin_is_synthetic"].append(record["origin_is_synthetic"])
                columns["staleness_q"].append(record["staleness_q"])
                columns["pair_ord"].append(pair_ord)
    if missing_forecast or origin_mismatch:
        raise RuntimeError(f"forecast join missing={missing_forecast} origin_mismatch={origin_mismatch}")

    frame = pd.DataFrame(columns)
    frame["target_ts"] = pd.to_datetime(frame["target_ts"])
    frame["origin_ts"] = pd.to_datetime(frame["origin_ts"])
    frame["model"] = pd.Categorical(frame["model"], MODELS, ordered=True)
    frame = frame.sort_values(["model", "arm", "pair_ord"], kind="mergesort").reset_index(drop=True)
    frame["model"] = frame["model"].astype(str)
    _assert_long(frame, pairs)
    return frame.drop(columns=["pair_ord"])


def _assert_long(frame: pd.DataFrame, pairs: pd.DataFrame) -> None:
    counts = frame.groupby(["model", "arm"], sort=False).size()
    if not bool((counts == N_KEYS).all()) or len(counts) != len(MODELS) * 2:
        raise RuntimeError("key count per model and arm is not 85205")
    if int(frame["entity"].nunique()) != N_ENTITIES:
        raise RuntimeError("entity count mismatch")
    if not np.isfinite(frame["mase"].to_numpy(dtype=float)).all():
        raise RuntimeError("non-finite stored MASE")
    y_true = frame["y_true"].to_numpy(dtype=float)
    y_pred = frame["y_pred"].to_numpy(dtype=float)
    den = frame["mase_den"].to_numpy(dtype=float)
    mase = frame["mase"].to_numpy(dtype=float)
    recomputed = np.abs(y_true - y_pred) / den
    abs_diff = np.abs(recomputed - mase)
    denom = np.maximum(np.abs(mase), 1e-300)
    max_rel = float(np.max(abs_diff / denom))
    max_abs = float(np.max(abs_diff))
    LAST_IDENTITY.clear()
    LAST_IDENTITY.update({"max_relative_diff": max_rel, "max_absolute_diff": max_abs, "rows": int(len(frame))})
    if max_rel > 1e-12:
        raise RuntimeError(f"MASE identity failed max_relative={max_rel} max_absolute={max_abs}")
    arm_a = frame[(frame["model"] == MODELS[0]) & (frame["arm"] == "A")]
    pair_targets = pairs["target_timestamp"].tolist()
    got_targets = arm_a["target_ts"].dt.strftime("%Y-%m-%d").tolist()
    if arm_a["entity"].tolist() != pairs["entity_hash"].tolist() or got_targets != pair_targets:
        raise RuntimeError("Arm A row order does not follow the pair file")
    origin_a = arm_a["origin_ts"].to_numpy()
    origin_b = frame[(frame["model"] == MODELS[0]) & (frame["arm"] == "B")]["origin_ts"].to_numpy()
    n_same = int(np.sum(origin_a == origin_b))
    if n_same != N_SAME_ORIGIN or (N_KEYS - n_same) != N_DIFF_ORIGIN:
        raise RuntimeError(f"paired origin split {n_same}")
