"""Exploratory: who does the 3mo grid exclude, and would the benchmark's conclusions differ on them?

Uses only saved Arm B forecasts and saved grids (no refit). Groups banks into
  excluded  : Arm B schedule rows, no Arm A schedule rows (all have a 30 Jun / 30 Sep first filing)
  drifted   : on both schedules, Arm A grid has an off-quarter-end stamp
  unaffected: on both schedules, Arm A grid has only calendar quarter-ends
and compares series characteristics and Arm B forecast accuracy / rankings.
Writes supplementary/out/excluded_banks.json.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

import coverage
import load

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "supplementary" / "out" / "excluded_banks.json"
MODELS = ["HistoricAverage", "Naive", "SeasonalNaive", "AutoARIMA", "SimpleExponentialSmoothing", "AutoTheta"]
QE = {"03-31", "06-30", "09-30", "12-31"}
SEED = 20261118
N_BOOT = 2000


def group_of(schedule: pd.DataFrame, grids: dict) -> dict[str, str]:
    in_a = set(schedule.loc[schedule["arm_id"] == "A", "entity_hash"])
    in_b = set(schedule.loc[schedule["arm_id"] == "B", "entity_hash"])
    out = {}
    for e in in_b:
        if e not in in_a:
            out[e] = "excluded"
        else:
            ts = grids["A"][e]["ts"]
            out[e] = "drifted" if any(coverage._monthday(t) not in QE for t in ts) else "unaffected"
    return out


def series_stats(grid_b: dict) -> dict:
    vals = grid_b["values"]
    fin = np.isfinite(vals)
    v = vals[fin]
    first = next(t for t, f in zip(grid_b["ts"], grid_b["finite"]) if f)
    d = np.diff(vals)
    d = d[np.isfinite(d)]
    return {
        "first_year": int(first[:4]),
        "first_is_panel_start": first[:10] == "1976-03-31",
        "n_obs": int(fin.sum()),
        "level_mean": float(v.mean()),
        "abs_diff_mean": float(np.abs(d).mean()) if d.size else np.nan,
    }


def boot_diff(x: np.ndarray, y: np.ndarray, rng: np.random.Generator) -> tuple[float, float, float]:
    """mean(x) - mean(y), independent entity resampling within each group, type-7 percentile CI."""
    est = x.mean() - y.mean()
    reps = np.empty(N_BOOT)
    for b in range(N_BOOT):
        reps[b] = x[rng.integers(0, x.size, x.size)].mean() - y[rng.integers(0, y.size, y.size)].mean()
    lo, hi = np.quantile(reps, [0.025, 0.975], method="linear")
    return float(est), float(lo), float(hi)


def main() -> int:
    inputs = load.read_inputs()
    schedule, prepared, fc = inputs["schedule"], inputs["prepared"], inputs["forecasts"]
    grids = coverage._grids_by_arm(prepared)
    groups = group_of(schedule, grids)

    # --- series characteristics (Arm B grid, model-independent)
    rows = []
    for e, g in groups.items():
        s = series_stats(grids["B"][e])
        s["entity"], s["group"] = e, g
        rows.append(s)
    st = pd.DataFrame(rows)
    chars = {}
    for g, part in st.groupby("group"):
        entrants = part.loc[~part["first_is_panel_start"]]
        chars[g] = {
            "n_entities": int(len(part)),
            "share_first_at_panel_start": float(part["first_is_panel_start"].mean()),
            "first_year_median": float(part["first_year"].median()),
            "first_year_median_entrants": float(entrants["first_year"].median()) if len(entrants) else None,
            "n_obs_median": float(part["n_obs"].median()),
            "level_mean_median": float(part["level_mean"].median()),
            "abs_diff_mean_median": float(part["abs_diff_mean"].median()),
        }
    # entrants only (first filing after the panel start), retained vs excluded
    ent = st.loc[~st["first_is_panel_start"]]
    chars["entrants_first_year_by_group"] = {
        g: {str(k): int(v) for k, v in p["first_year"].value_counts().sort_index().items()}
        for g, p in ent.groupby("group")
    }

    # --- Arm B forecast accuracy by group
    b = fc.loc[fc["arm_id"] == "B", ["entity_hash", "model_id", "target_timestamp", "mase_value"]].copy()
    b["group"] = b["entity_hash"].map(groups)
    ent_mase = b.groupby(["group", "entity_hash", "model_id"], sort=False)["mase_value"].mean().unstack("model_id")[MODELS]
    rng = np.random.default_rng(SEED)
    acc = {}
    for g in ("excluded", "drifted", "unaffected"):
        m = ent_mase.loc[g]
        ranks = m.rank(axis=1, method="average")
        best = m.idxmin(axis=1).value_counts(normalize=True)
        means = m.mean()
        acc[g] = {
            "n_entities": int(len(m)),
            "n_keys": int((b["group"] == g).sum() // len(MODELS)),
            "mean_entity_mase": {k: float(v) for k, v in means.items()},
            "median_entity_mase": {k: float(v) for k, v in m.median().items()},
            "mean_rank": {k: float(v) for k, v in ranks.mean().items()},
            "best_share": {k: float(best.get(k, 0.0)) for k in MODELS},
            "aggregate_order": list(means.sort_values().index),
        }
    # excluded minus retained-unaffected, per model, and the excluded-vs-unaffected rank gap
    diffs = {}
    for model in MODELS:
        x = ent_mase.loc["excluded", model].to_numpy(float)
        y = ent_mase.loc["unaffected", model].to_numpy(float)
        diffs[model] = dict(zip(("est", "lo", "hi"), boot_diff(x, y, rng)))
    # relative performance: each model's MASE relative to the bank's own best model (scale-free across groups)
    rel = {}
    for g in ("excluded", "unaffected"):
        m = ent_mase.loc[g]
        r = m.div(m["Naive"], axis=0)
        rel[g] = {k: float(v) for k, v in r.median().items()}

    payload = {
        "characteristics": chars,
        "arm_b_accuracy": acc,
        "excluded_minus_unaffected_mean_entity_mase": diffs,
        "median_mase_ratio_to_naive": rel,
        "notes": "Exploratory, specified after the post-hoc results were seen. Arm B forecasts only; no refit.",
    }
    OUT.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(payload, indent=1)[:6000], flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
