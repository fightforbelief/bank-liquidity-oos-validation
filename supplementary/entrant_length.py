"""Exploratory: is the entrant/incumbent ranking reversal explained by training length?

Arm B forecasts only. For each forecast, training length = number of finite stored
values on the Arm B grid strictly before the target (the history the model saw).
Within training-length bins, compare entrants and 1976 incumbents on the key-level
MASE differences Naive - AutoTheta and Naive - AutoARIMA, with entity-cluster
bootstrap intervals (entities resampled within bin and group).
Writes supplementary/out/entrant_length.json.
"""
from __future__ import annotations

import json
from bisect import bisect_left
from pathlib import Path

import numpy as np
import pandas as pd

import coverage
import load

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "supplementary" / "out" / "entrant_length.json"
BINS = [(18, 30), (31, 45), (46, 60), (61, 80), (81, 200)]
SEED = 20261118
N_BOOT = 1000


def boot_mean(df: pd.DataFrame, col: str, rng: np.random.Generator) -> dict:
    g = df.groupby("entity")[col].agg(["sum", "count"])
    s, c = g["sum"].to_numpy(float), g["count"].to_numpy(float)
    n = len(g)
    est = s.sum() / c.sum()
    reps = np.empty(N_BOOT)
    for b in range(N_BOOT):
        i = rng.integers(0, n, n)
        reps[b] = s[i].sum() / c[i].sum()
    lo, hi = np.quantile(reps, [0.025, 0.975], method="linear")
    return {"est": float(est), "lo": float(lo), "hi": float(hi), "n_entities": int(n), "n_keys": int(c.sum())}


def main() -> int:
    inputs = load.read_inputs()
    prepared, fc = inputs["prepared"], inputs["forecasts"]
    grids = coverage._grids_by_arm(prepared)["B"]
    b = fc.loc[fc["arm_id"] == "B", ["entity_hash", "model_id", "target_timestamp", "mase_value"]]
    w = b.pivot_table(index=["entity_hash", "target_timestamp"], columns="model_id", values="mase_value").reset_index()

    first, ntrain = {}, []
    cum = {}
    for e, g in grids.items():
        f = np.asarray(g["finite"], bool)
        cum[e] = (g["ts"], np.concatenate([[0], np.cumsum(f)]))
        first[e] = next(t for t, ok in zip(g["ts"], f) if ok)[:10]
    for e, t in zip(w["entity_hash"], w["target_timestamp"]):
        ts, c = cum[e]
        ntrain.append(int(c[bisect_left(ts, t)]))
    w["n_train"] = ntrain
    w["entity"] = w["entity_hash"]
    w["group"] = np.where(w["entity_hash"].map(first) == "1976-03-31", "incumbent", "entrant")
    w["naive_minus_theta"] = w["Naive"] - w["AutoTheta"]
    w["naive_minus_arima"] = w["Naive"] - w["AutoARIMA"]

    rng = np.random.default_rng(SEED)
    out = {"bins": {}}
    for lo, hi in BINS:
        part = w.loc[(w["n_train"] >= lo) & (w["n_train"] <= hi)]
        key = f"{lo}-{hi}"
        out["bins"][key] = {}
        for grp in ("entrant", "incumbent"):
            p = part.loc[part["group"] == grp]
            if p["entity"].nunique() < 30:
                continue
            out["bins"][key][grp] = {
                "naive_minus_theta": boot_mean(p, "naive_minus_theta", rng),
                "naive_minus_arima": boot_mean(p, "naive_minus_arima", rng),
            }
    # same calendar period check: targets 2000-2020 only, all lengths
    late = w.loc[w["target_timestamp"] >= "2000-01-01"]
    out["targets_2000_2020"] = {
        grp: {
            "naive_minus_theta": boot_mean(late.loc[late["group"] == grp], "naive_minus_theta", rng),
            "naive_minus_arima": boot_mean(late.loc[late["group"] == grp], "naive_minus_arima", rng),
            "n_train_median": float(late.loc[late["group"] == grp, "n_train"].median()),
        }
        for grp in ("entrant", "incumbent")
    }
    out["notes"] = "Exploratory; Arm B saved forecasts; key-level means pooled within entity, entity-cluster bootstrap."
    OUT.write_text(json.dumps(out, indent=2) + "\n", encoding="utf-8")
    for k, v in out["bins"].items():
        for grp, r in v.items():
            a, t = r["naive_minus_arima"], r["naive_minus_theta"]
            print(f"n_train {k:7s} {grp:9s} ents {t['n_entities']:6d}  N-Theta {t['est']:+.4f} [{t['lo']:+.4f},{t['hi']:+.4f}]  N-ARIMA {a['est']:+.4f} [{a['lo']:+.4f},{a['hi']:+.4f}]")
    for grp, r in out["targets_2000_2020"].items():
        print("2000-2020", grp, "median n_train", r["n_train_median"], {k: [round(r[k][x], 4) for x in ("est", "lo", "hi")] for k in ("naive_minus_theta", "naive_minus_arima")})
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
