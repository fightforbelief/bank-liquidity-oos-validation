"""Exploratory (discovery dataset only), last mechanism check: level shifts.

For each Arm B forecast key, a "recent level shift" is an absolute quarter-on-quarter
change in the stored training values, within the last 8 quarters before the target,
larger than 5 times the median absolute change over the preceding 24 quarters
(robust scale). Reports the shift rate by group and the entrant/incumbent contrast
in (Naive - AutoTheta) separately for keys with and without a recent shift.
Writes supplementary/out/entrant_shifts.json.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

import load

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "supplementary" / "out" / "entrant_shifts.json"
rng = np.random.default_rng(20261118)


def shift_flag(ts, vals, target) -> float:
    i = int(np.searchsorted(np.asarray(ts), target))
    v = vals[max(0, i - 32):i]
    d = np.diff(v)
    d = d[np.isfinite(d)]
    if d.size < 16:
        return np.nan
    scale = np.median(np.abs(d[:-8])) if d.size > 8 else np.nan
    if not np.isfinite(scale) or scale == 0:
        return np.nan
    return float(np.any(np.abs(d[-8:]) > 5 * scale))


def boot(x):
    reps = [x[rng.integers(0, x.size, x.size)].mean() for _ in range(1000)]
    return [float(x.mean()), *np.quantile(reps, [0.025, 0.975], method="linear").tolist(), int(x.size)]


def main() -> int:
    inputs = load.read_inputs()
    prep, fc = inputs["prepared"], inputs["forecasts"]
    b = prep.loc[prep["arm_id"] == "B"].sort_values("timestamp")
    grids, first = {}, {}
    for e, g in b.groupby("entity_hash", sort=False):
        ts, vals = g["timestamp"].tolist(), g["value"].to_numpy(dtype=float)
        grids[e] = (ts, vals)
        first[e] = next(t for t, v in zip(ts, vals) if np.isfinite(v))
    w = fc.loc[fc["arm_id"] == "B"].pivot_table(index=["entity_hash", "target_timestamp"], columns="model_id",
                                                values="mase_value").reset_index()
    w["shift"] = [shift_flag(*grids[e], t) for e, t in zip(w["entity_hash"], w["target_timestamp"])]
    w["entrant"] = w["entity_hash"].map(first) != "1976-03-31"
    w["d"] = w["Naive"] - w["AutoTheta"]
    out = {"shift_rate": {}, "contrast": {}}
    for grp, name in ((True, "entrants"), (False, "incumbents")):
        part = w.loc[(w["entrant"] == grp) & w["shift"].notna()]
        out["shift_rate"][name] = float(part["shift"].mean())
    for flag, label in ((1.0, "recent_shift"), (0.0, "no_recent_shift")):
        row = {}
        for grp, name in ((True, "entrants"), (False, "incumbents")):
            x = w.loc[(w["entrant"] == grp) & (w["shift"] == flag)].groupby("entity_hash")["d"].mean().to_numpy()
            row[name] = boot(x)
        out["contrast"][label] = row
    OUT.write_text(json.dumps(out, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(out, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
