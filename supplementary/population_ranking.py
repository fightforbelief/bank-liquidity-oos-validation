"""Exploratory: does the benchmark's model ranking depend on which banks the grid admits?

Full-population comparison, each arm on its own eligible set (the way a benchmark
reports results), using saved forecasts only:
  * per arm: mean entity-level MASE per model, aggregate order, and every pairwise
    model difference with an entity-cluster bootstrap interval;
  * entrant-matched check: banks whose first filing is after the 1976-03-31 panel
    start, split into retained-on-A vs excluded-from-A, Arm B forecasts only.
Writes supplementary/out/population_ranking.json.
"""
from __future__ import annotations

import itertools
import json
from pathlib import Path

import numpy as np
import pandas as pd

import coverage
import load

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "supplementary" / "out" / "population_ranking.json"
MODELS = ["HistoricAverage", "Naive", "SeasonalNaive", "AutoARIMA", "SimpleExponentialSmoothing", "AutoTheta"]
SEED = 20261118
N_BOOT = 2000


def boot_pairs(m: pd.DataFrame, rng: np.random.Generator) -> dict:
    """Entity-cluster bootstrap of mean entity MASE for all models jointly."""
    x = m[MODELS].to_numpy(float)
    n = x.shape[0]
    est = x.mean(axis=0)
    reps = np.empty((N_BOOT, len(MODELS)))
    for b in range(N_BOOT):
        reps[b] = x[rng.integers(0, n, n)].mean(axis=0)
    out = {"n_entities": int(n), "mean": dict(zip(MODELS, est.tolist())),
           "order": [MODELS[i] for i in np.argsort(est, kind="stable")], "pairs": {}}
    for i, j in itertools.combinations(range(len(MODELS)), 2):
        d = reps[:, i] - reps[:, j]
        lo, hi = np.quantile(d, [0.025, 0.975], method="linear")
        out["pairs"][f"{MODELS[i]} - {MODELS[j]}"] = {
            "est": float(est[i] - est[j]), "lo": float(lo), "hi": float(hi),
            "share_reps_i_better": float((d < 0).mean()),
        }
    # probability each model is ranked first across replicates
    first = np.bincount(reps.argmin(axis=1), minlength=len(MODELS)) / N_BOOT
    out["p_first"] = dict(zip(MODELS, first.tolist()))
    return out


def main() -> int:
    inputs = load.read_inputs()
    schedule, prepared, fc = inputs["schedule"], inputs["prepared"], inputs["forecasts"]
    grids = coverage._grids_by_arm(prepared)
    rng = np.random.default_rng(SEED)

    ent = fc.groupby(["arm_id", "entity_hash", "model_id"], sort=False)["mase_value"].mean().unstack("model_id")
    payload: dict = {"full_population": {}}
    for arm in ("A", "B"):
        payload["full_population"][arm] = boot_pairs(ent.loc[arm], rng)

    # entrant-matched comparison on Arm B
    in_a = set(schedule.loc[schedule["arm_id"] == "A", "entity_hash"])
    rows = []
    for e in ent.loc["B"].index:
        g = grids["B"][e]
        first = next(t for t, f in zip(g["ts"], g["finite"]) if f)
        rows.append((e, first[:10], e in in_a))
    meta = pd.DataFrame(rows, columns=["entity", "first", "on_A"]).set_index("entity")
    entrants = meta.loc[meta["first"] != "1976-03-31"]
    mb = ent.loc["B"]
    payload["entrants_arm_b"] = {
        "retained_on_A": boot_pairs(mb.loc[entrants.index[entrants["on_A"]]], rng),
        "excluded_from_A": boot_pairs(mb.loc[entrants.index[~entrants["on_A"]]], rng),
        "incumbents_1976": boot_pairs(mb.loc[meta.index[meta["first"] == "1976-03-31"]], rng),
        "n_entrants": int(len(entrants)),
        "share_entrants_excluded": float((~entrants["on_A"]).mean()),
    }
    payload["notes"] = "Exploratory, specified after post-hoc results were seen; saved forecasts only; seed 20261118."
    OUT.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    for arm in ("A", "B"):
        p = payload["full_population"][arm]
        print(arm, p["n_entities"], p["order"], {k: round(v, 4) for k, v in p["mean"].items()})
        print("  p_first", {k: round(v, 3) for k, v in p["p_first"].items()})
        for k in ("Naive - AutoARIMA", "AutoARIMA - AutoTheta", "Naive - AutoTheta"):
            q = p["pairs"][k]
            print("  ", k, round(q["est"], 4), [round(q["lo"], 4), round(q["hi"], 4)])
    for k, p in payload["entrants_arm_b"].items():
        if isinstance(p, dict):
            print(k, p["n_entities"], p["order"], {m: round(v, 3) for m, v in p["mean"].items()})
            q = p["pairs"]["Naive - AutoARIMA"]
            print("   Naive-AutoARIMA", round(q["est"], 4), [round(q["lo"], 4), round(q["hi"], 4)])
            q = p["pairs"]["Naive - AutoTheta"]
            print("   Naive-AutoTheta", round(q["est"], 4), [round(q["lo"], 4), round(q["hi"], 4)])
    print("entrants", payload["entrants_arm_b"]["n_entrants"], "share excluded", round(payload["entrants_arm_b"]["share_entrants_excluded"], 3))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
