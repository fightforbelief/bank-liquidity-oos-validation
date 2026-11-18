"""Exploratory (discovery dataset only): why does Naive do relatively better for entrants?

Arm B saved forecasts and grid. For each forecast key, from the training values the
model saw (stored values before the target, last 24 quarters):
  phi    : OLS AR(1) slope of y_t on y_{t-1}  (persistence; Naive is optimal near 1)
  trend  : R^2 of a linear time trend
  season : share of variance explained by quarter-of-year means
and bank age at the target (years since first filing). Reports how the entrant
advantage (Naive - AutoTheta) varies with age and with phi.
Writes supplementary/out/entrant_mechanism.json.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

import load

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "supplementary" / "out" / "entrant_mechanism.json"
WINDOW = 24
SEED = 20261118
rng = np.random.default_rng(SEED)


def feats(ts: list[str], vals: np.ndarray, target: str) -> tuple[float, float, float]:
    i = np.searchsorted(np.asarray(ts), target)
    v = vals[max(0, i - WINDOW):i]
    q = np.array([int(t[5:7]) for t in ts[max(0, i - WINDOW):i]])
    ok = np.isfinite(v)
    v, q = v[ok], q[ok]
    if v.size < 8 or np.var(v) == 0:
        return np.nan, np.nan, np.nan
    x, y = v[:-1], v[1:]
    phi = np.cov(x, y, ddof=0)[0, 1] / np.var(x) if np.var(x) > 0 else np.nan
    t = np.arange(v.size)
    trend = np.corrcoef(t, v)[0, 1] ** 2
    qm = pd.Series(v).groupby(q).transform("mean").to_numpy()
    season = np.var(qm) / np.var(v)
    return float(phi), float(trend), float(season)


def boot(x: np.ndarray) -> list[float]:
    reps = [x[rng.integers(0, x.size, x.size)].mean() for _ in range(1000)]
    return [float(x.mean()), *np.quantile(reps, [0.025, 0.975], method="linear").tolist()]


def main() -> int:
    inputs = load.read_inputs()
    prep, fc = inputs["prepared"], inputs["forecasts"]
    b = prep.loc[prep["arm_id"] == "B"].sort_values("timestamp")
    grids, first = {}, {}
    for e, g in b.groupby("entity_hash", sort=False):
        ts = g["timestamp"].tolist()
        vals = g["value"].to_numpy(dtype=float)
        grids[e] = (ts, vals)
        first[e] = next(t for t, v in zip(ts, vals) if np.isfinite(v))
    w = fc.loc[fc["arm_id"] == "B"].pivot_table(index=["entity_hash", "target_timestamp"], columns="model_id",
                                                values="mase_value").reset_index()
    rows = [feats(*grids[e], t) for e, t in zip(w["entity_hash"], w["target_timestamp"])]
    w[["phi", "trend", "season"]] = rows
    w["first"] = w["entity_hash"].map(first)
    w["entrant"] = w["first"] != "1976-03-31"
    w["age"] = (pd.to_datetime(w["target_timestamp"]) - pd.to_datetime(w["first"])).dt.days / 365.25
    w["d"] = w["Naive"] - w["AutoTheta"]

    out: dict = {"feature_medians": {}, "by_age_entrants": {}, "by_phi": {}}
    for grp, part in w.groupby("entrant"):
        key = "entrants" if grp else "incumbents"
        out["feature_medians"][key] = {c: float(part[c].median()) for c in ("phi", "trend", "season", "age")}
    for lo, hi in ((0, 7), (7, 10), (10, 15), (15, 20), (20, 50)):
        part = w.loc[w["entrant"] & w["age"].between(lo, hi, inclusive="left")]
        ent = part.groupby("entity_hash")["d"].mean().to_numpy()
        if ent.size >= 30:
            out["by_age_entrants"][f"{lo}-{hi}y"] = {"naive_minus_theta": boot(ent), "n_entities": int(ent.size),
                                                     "phi_median": float(part["phi"].median())}
    qs = w["phi"].quantile([0, .2, .4, .6, .8, 1]).to_numpy()
    for i in range(5):
        part = w.loc[w["phi"].between(qs[i], qs[i + 1])]
        row = {"phi_range": [float(qs[i]), float(qs[i + 1])]}
        for grp, name in ((True, "entrants"), (False, "incumbents")):
            ent = part.loc[part["entrant"] == grp].groupby("entity_hash")["d"].mean().to_numpy()
            if ent.size >= 30:
                row[name] = {"naive_minus_theta": boot(ent), "n_entities": int(ent.size)}
        out["by_phi"][f"q{i + 1}"] = row
    OUT.write_text(json.dumps(out, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(out, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
