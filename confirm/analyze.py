"""Pre-registered analysis (confirm/PROTOCOL.md) for one run directory.

Usage:
    python confirm/analyze.py bank_leverage          # confirmatory run
    python confirm/analyze.py discovery              # Bank Cash Liquidity, the discovery run
Writes <run>/analysis.json (discovery: <repo>/confirm/out/discovery_analysis.json).
"""
from __future__ import annotations

import json
import sys
from bisect import bisect_left
from pathlib import Path

import numpy as np
import pandas as pd

import common as C
from p12_bank_liquidity.candidate_a.stage18_inference import cluster_bootstrap

DAY30 = {"06-30", "09-30"}


def load(run: Path):
    sched = pd.read_parquet(run / "schedule.parquet", columns=["entity_hash", "arm_id", "target_timestamp"])
    prep = pd.read_parquet(run / "prepared_series.parquet",
                           columns=["entity_hash", "arm_id", "timestamp", "value"])
    fc = pd.concat([pd.read_parquet(p, columns=["entity_hash", "arm_id", "model_id", "target_timestamp",
                                                "status", "mase_status", "mase_value"])
                    for p in sorted((run / "batches").glob("batch_*.parquet"))], ignore_index=True)
    for f, cols in ((sched, ["target_timestamp"]), (prep, ["timestamp"]), (fc, ["target_timestamp"])):
        for c in cols:
            f[c] = f[c].astype(str).str[:10]
    if (run / "entity_dates.parquet").exists():
        first = pd.read_parquet(run / "entity_dates.parquet").set_index("entity_hash")["first"].astype(str).str[:10]
    else:  # discovery run: first finite stored value on the B grid (all filings join it), else A
        fin = prep.loc[prep["value"].notna()].sort_values("timestamp")
        fb = fin.loc[fin["arm_id"] == "B"].groupby("entity_hash")["timestamp"].first()
        fa = fin.loc[fin["arm_id"] == "A"].groupby("entity_hash")["timestamp"].first()
        first = fb.combine_first(fa)
    return sched, prep, fc, first


def entity_table(fc: pd.DataFrame, arm: str) -> tuple[pd.DataFrame, dict]:
    """Entity-mean MASE per model over keys where all six models have a defined MASE."""
    f = fc.loc[fc["arm_id"] == arm]
    ok = (f["status"] == "RECORDED_FORECAST") & (f["mase_status"] == "DEFINED")
    w = f.assign(v=np.where(ok, f["mase_value"], np.nan)).pivot_table(
        index=["entity_hash", "target_timestamp"], columns="model_id", values="v", aggfunc="first")
    w = w.reindex(columns=list(C.MODELS))
    full = w.notna().all(axis=1)
    info = {"keys": int(len(w)), "keys_all_models_defined": int(full.sum()),
            "excluded_or_undefined_by_model": {m: int((~ok & (f["model_id"] == m)).sum()) for m in C.MODELS}}
    return w.loc[full], info


def boot_group_diff(x: np.ndarray, y: np.ndarray, rng) -> dict:
    est = float(x.mean() - y.mean())
    reps = np.array([x[rng.integers(0, x.size, x.size)].mean() - y[rng.integers(0, y.size, y.size)].mean()
                     for _ in range(C.N_BOOT)])
    lo, hi = np.quantile(reps, [0.025, 0.975], method="linear")
    return {"est": est, "lo": float(lo), "hi": float(hi), "n_x": int(x.size), "n_y": int(y.size)}


def boot_mean(x: np.ndarray, rng) -> dict:
    reps = np.array([x[rng.integers(0, x.size, x.size)].mean() for _ in range(C.N_BOOT)])
    lo, hi = np.quantile(reps, [0.025, 0.975], method="linear")
    return {"est": float(x.mean()), "lo": float(lo), "hi": float(hi), "n": int(x.size)}


def main() -> int:
    name = sys.argv[1]
    if name == "discovery":
        run, out = C.DISCOVERY_RUN, C.REPO / "confirm" / "out" / "discovery_analysis.json"
        out.parent.mkdir(parents=True, exist_ok=True)
    else:
        run = C.run_dir(name)
        out = run / "analysis.json"
    rng = np.random.default_rng(C.SEED)
    sched, prep, fc, first = load(run)
    md = first.str[5:10]
    entrant = first != C.PANEL_START

    # ---- H1: every Arm B-only target belongs to an entity whose first filing is 30 Jun / 30 Sep
    ka = set(map(tuple, sched.loc[sched["arm_id"] == "A", ["entity_hash", "target_timestamp"]].to_numpy()))
    kb = set(map(tuple, sched.loc[sched["arm_id"] == "B", ["entity_hash", "target_timestamp"]].to_numpy()))
    b_only = kb - ka
    b_only_day30 = sum(md.get(e, "") in DAY30 for e, _ in b_only)
    ents = {arm: set(sched.loc[sched["arm_id"] == arm, "entity_hash"]) for arm in ("A", "B")}
    h1 = {"arm_b_only_keys": len(b_only), "arm_a_only_keys": len(ka - kb),
          "b_only_keys_day30_first_filing": int(b_only_day30),
          "share": (b_only_day30 / len(b_only)) if b_only else None,
          "entities": {arm: len(s) for arm, s in ents.items()},
          "entities_b_not_a": len(ents["B"] - ents["A"]),
          "entities_b_not_a_day30": int(sum(md.get(e, "") in DAY30 for e in ents["B"] - ents["A"]))}
    h1["confirmed"] = bool(h1["arm_b_only_keys"] > 0 and h1["share"] == 1.0)

    # ---- H2: composition
    def entrant_share(s):
        return float(np.mean([bool(entrant.get(e, False)) for e in s]))
    all_entrants = {e for e in ents["A"] | ents["B"] if entrant.get(e, False)}
    h2 = {"entrant_share_A": entrant_share(ents["A"]), "entrant_share_B": entrant_share(ents["B"]),
          "entrants_total": len(all_entrants),
          "share_of_entrants_absent_from_A": float(np.mean([e not in ents["A"] for e in all_entrants]))}
    h2["confirmed"] = h2["entrant_share_A"] < h2["entrant_share_B"]

    # ---- H3: entrant vs incumbent contrast in (Naive - AutoTheta), Arm B entity means
    wb, info_b = entity_table(fc, "B")
    eb = wb.groupby(level="entity_hash").mean()
    is_ent = np.array([bool(entrant.get(e, False)) for e in eb.index], dtype=bool)
    h3 = {"forecast_info_B": info_b, "n_entrants": int(is_ent.sum()), "n_incumbents": int((~is_ent).sum())}
    for label, other in (("naive_minus_theta", "AutoTheta"), ("naive_minus_arima", "AutoARIMA")):
        d = (eb["Naive"] - eb[other]).to_numpy()
        h3[label] = {"entrants": boot_mean(d[is_ent], rng), "incumbents": boot_mean(d[~is_ent], rng),
                     "contrast": boot_group_diff(d[is_ent], d[~is_ent], rng)}
    h3["confirmed"] = h3["naive_minus_theta"]["contrast"]["hi"] < 0

    # ---- H3 secondary: within training-length bins (key level, entity-pooled)
    fin = prep.loc[(prep["arm_id"] == "B")].sort_values("timestamp")
    cum = {}
    for e, g in fin.groupby("entity_hash", sort=False):
        ts = g["timestamp"].tolist()
        cum[e] = (ts, np.concatenate([[0], np.cumsum(g["value"].notna().to_numpy())]))
    kw = wb.reset_index()
    kw["n_train"] = [int(cum[e][1][bisect_left(cum[e][0], t)]) for e, t in zip(kw["entity_hash"], kw["target_timestamp"])]
    kw["entrant"] = kw["entity_hash"].map(lambda e: bool(entrant.get(e, False)))
    kw["d"] = kw["Naive"] - kw["AutoTheta"]
    bins = {}
    for lo, hi in C.LENGTH_BINS:
        part = kw.loc[kw["n_train"].between(lo, hi)]
        row = {}
        for grp, mask in (("entrants", part["entrant"]), ("incumbents", ~part["entrant"])):
            g = part.loc[mask].groupby("entity_hash")["d"].mean().to_numpy()
            if g.size >= 30:
                row[grp] = boot_mean(g, rng)
        bins[f"{lo}-{hi}"] = row
    h3["by_training_length"] = bins

    # ---- descriptive: pre-specified-style paired estimate on common keys, and full-population orders
    wa, info_a = entity_table(fc, "A")
    common = wa.index.intersection(wb.index)
    paired = {}
    for m in C.MODELS:
        ca = wa.loc[common, m].groupby(level="entity_hash")
        cb = wb.loc[common, m].groupby(level="entity_hash")
        sa, na = ca.sum(), ca.size()
        sb = cb.sum()
        delta = (sb / na - sa / na).to_numpy()
        res = cluster_bootstrap(delta)
        level_a = float((sa / na).mean())
        paired[m] = {"estimate": res["estimate"], "lo": res["low"], "hi": res["high"],
                     "mean_entity_mase_A": level_a, "rel_change": res["estimate"] / level_a,
                     "share_entities_delta_zero": float(np.mean(delta == 0))}
    orders = {}
    for arm, w in (("A", wa), ("B", wb)):
        means = w.groupby(level="entity_hash").mean().mean()
        orders[arm] = {"order": list(means.sort_values().index), "mean": {k: float(v) for k, v in means.items()}}
    payload = {"run": name, "H1": h1, "H2": h2, "H3": h3,
               "descriptive": {"common_keys_all_models": int(len(common)), "paired": paired,
                               "full_population_orders": orders, "forecast_info_A": info_a}}
    out.write_text(json.dumps(payload, indent=2, default=float) + "\n", encoding="utf-8")
    c = h3["naive_minus_theta"]["contrast"]
    print(name, "H1", h1["confirmed"], h1["share"], "H2", h2["confirmed"],
          round(h2["entrant_share_A"], 3), round(h2["entrant_share_B"], 3),
          "H3", h3["confirmed"], round(c["est"], 4), [round(c["lo"], 4), round(c["hi"], 4)], flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
