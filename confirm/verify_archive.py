"""Cross-check archived coverage and accuracy contrasts against saved forecast batches."""
from __future__ import annotations

import json
from math import isclose
from pathlib import Path

import pandas as pd

import common as C


def verify(target: str, run: Path) -> None:
    archive = json.loads((C.REPO / "confirm" / "out" / f"{target}_analysis.json").read_text())
    keys = ["entity_hash", "arm_id", "target_timestamp"]
    schedule = pd.read_parquet(run / "schedule.parquet", columns=keys)
    schedule["target_timestamp"] = schedule.target_timestamp.astype(str).str[:10]
    if schedule.duplicated(keys).any():
        raise ValueError(f"{target}: duplicate scheduled target")
    frames = []
    for path in sorted((run / "batches").glob("batch_*.parquet")):
        frames.append(pd.read_parquet(path, columns=keys + ["model_id", "mase_value", "mase_status", "status"]))
    rows = pd.concat(frames, ignore_index=True)
    rows["target_timestamp"] = rows.target_timestamp.astype(str).str[:10]
    if set(rows.model_id) != set(C.MODELS) or rows.duplicated(keys + ["model_id"]).any():
        raise ValueError(f"{target}: unexpected model set or duplicate forecast")
    counts = rows.groupby(keys).size().reindex(pd.MultiIndex.from_frame(schedule))
    if not counts.eq(len(C.MODELS)).all() or len(rows) != len(schedule) * len(C.MODELS):
        raise ValueError(f"{target}: forecast batches do not cover the schedule exactly")

    if target == "discovery":
        series = pd.read_parquet(run / "prepared_series.parquet", columns=["entity_hash", "arm_id", "timestamp", "value"])
        finite = series.dropna(subset=["value"])
        first = finite.groupby(["entity_hash", "arm_id"]).timestamp.min().unstack()
        dates = first.B.combine_first(first.A)
    else:
        dates = pd.read_parquet(run / "entity_dates.parquet").set_index("entity_hash")["first"]
    dates = dates.astype(str).str[:10]
    sets = {arm: set(group.entity_hash) for arm, group in schedule.groupby("arm_id")}
    if (sets["A"] | sets["B"]) - set(dates.index):
        raise ValueError(f"{target}: missing first filing")
    by_arm = {arm: set(zip(group.entity_hash, group.target_timestamp)) for arm, group in schedule.groupby("arm_id")}
    lost = by_arm["B"] - by_arm["A"]
    day30 = set(dates.index[dates.str.endswith(("06-30", "09-30"))])
    if len(lost) != archive["H1"]["arm_b_only_keys"] or sum(e in day30 for e, _ in lost) != archive["H1"]["b_only_keys_day30_first_filing"]:
        raise ValueError(f"{target}: H1 coverage mismatch")
    entrants = set(dates.index[dates > C.PANEL_START])
    for arm, entities in sets.items():
        share = len(entities & entrants) / len(entities)
        if not isclose(share, archive["H2"][f"entrant_share_{arm}"], abs_tol=1e-12):
            raise ValueError(f"{target}: H2 composition mismatch")
    ok = rows.arm_id.eq("B") & rows.status.eq("RECORDED_FORECAST") & rows.mase_status.eq("DEFINED")
    wide = rows.loc[ok].pivot(index=["entity_hash", "target_timestamp"], columns="model_id", values="mase_value").dropna()
    gaps = (wide.Naive - wide.AutoTheta).groupby(level="entity_hash").mean()
    entrant = gaps.index.isin(entrants)
    contrast = gaps[entrant].mean() - gaps[~entrant].mean()
    if not isclose(contrast, archive["H3"]["naive_minus_theta"]["contrast"]["est"], abs_tol=1e-12):
        raise ValueError(f"{target}: H3 contrast mismatch")
    print(f"{target}: {len(rows):,} forecasts, B-only targets {len(lost):,}, H3 {contrast:+.6f}", flush=True)


if __name__ == "__main__":
    verify("discovery", C.DISCOVERY_RUN)
    for target in C.TARGETS:
        verify(target, C.RUNS / target)
