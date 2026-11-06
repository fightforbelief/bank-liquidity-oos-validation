"""
P12 supplementary analyses: reference implementation.

Every analysis here is computed from ALREADY-SAVED forecasts and MASE
components. Nothing refits a model or regenerates a forecast.

All of these analyses are POST-HOC / EXPLORATORY with respect to the frozen
protocol, except `reproduce_primary`, which must reproduce Table 2 / Table 4
exactly before anything else is trusted.

Canonical input: a long DataFrame with one row per (arm, model, entity, target_ts):

    entity      : rssdid
    target_ts   : pd.Timestamp, exact target timestamp (the pairing key with entity)
    arm         : "A" (3mo grid) or "B" (natural quarter-end grid)
    model       : one of MODELS
    origin_ts   : pd.Timestamp, arm-specific origin
    y_true      : float, observed target
    y_pred      : float, one-step forecast
    mase_den    : float, seasonal (lag-4) mean absolute difference of the training values
    mase        : float, stored MASE (= |y_true - y_pred| / mase_den)

Optional per-row training metadata (needed for the mechanism tables only):
    n_train_obs          : number of retained training positions
    n_imputed_train      : number of training positions filled by forward fill
    zero_seasonal_share  : share of lag-4 training differences that are exactly 0
    origin_is_synthetic  : bool, the origin grid position had no observed filing
    staleness_q          : quarters between origin and the last truly observed value

The project's own column names will differ; adapt them in one loader
function rather than editing these functions.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np
import pandas as pd

MODELS = [
    "HistoricAverage",
    "Naive",
    "SeasonalNaive",
    "AutoARIMA",
    "SimpleExponentialSmoothing",
    "AutoTheta",
]
KEY = ["entity", "target_ts"]
PRIMARY_SEED = 20261027
POSTHOC_SEED = 20261106  # new, documented seed for every post-hoc bootstrap
N_BOOT = 2000


# --------------------------------------------------------------------------
# Pairing
# --------------------------------------------------------------------------
def pair_wide(df: pd.DataFrame, model: str) -> pd.DataFrame:
    """One row per common (entity, target_ts) key for one model, A_/B_ columns.

    Row order follows first appearance in `df`, which should be the saved
    pair-file order (the paper's bootstrap indexes entities in that order).
    """
    d = df[df["model"] == model]
    cols = [c for c in d.columns if c not in ("arm", "model") + tuple(KEY)]
    a = d[d["arm"] == "A"].set_index(KEY)[cols].add_prefix("A_")
    b = d[d["arm"] == "B"].set_index(KEY)[cols].add_prefix("B_")
    w = a.join(b, how="inner").reset_index()
    w["abs_err_A"] = (w["A_y_true"] - w["A_y_pred"]).abs()
    w["abs_err_B"] = (w["B_y_true"] - w["B_y_pred"]).abs()
    w["same_origin"] = w["A_origin_ts"] == w["B_origin_ts"]
    return w


def entity_order(w: pd.DataFrame) -> np.ndarray:
    """Entities in order of first appearance (matches the paper's bootstrap)."""
    return pd.unique(w["entity"])


# --------------------------------------------------------------------------
# Cluster bootstrap on an entity-level table
# --------------------------------------------------------------------------
def default_draw(rng: np.random.Generator, n: int) -> np.ndarray:
    # REPLACE with the exact draw used by the frozen project code if it differs
    # (e.g. rng.choice(n, size=n, replace=True)); otherwise Table 2 will not
    # reproduce bit-for-bit.
    return rng.integers(0, n, size=n)


@dataclass
class BootResult:
    estimate: np.ndarray
    lower: np.ndarray
    upper: np.ndarray
    n_valid: int
    n_failed: int


def cluster_bootstrap(
    ent: pd.DataFrame,
    stat: Callable[[pd.DataFrame], np.ndarray | float],
    seed: int,
    n_boot: int = N_BOOT,
    draw: Callable[[np.random.Generator, int], np.ndarray] = default_draw,
) -> BootResult:
    """Resample rows of an entity-level table (one row per entity) with replacement.

    `stat` maps an entity table to a scalar or vector. Percentile interval with
    numpy method='linear' (Hyndman-Fan type 7), as in the paper. Non-finite
    replicates are counted as failed, never replaced.
    """
    est = np.atleast_1d(np.asarray(stat(ent), dtype=float))
    rng = np.random.Generator(np.random.PCG64(seed))
    n = len(ent)
    reps = np.empty((n_boot, est.size))
    for b in range(n_boot):
        idx = draw(rng, n)
        reps[b] = np.atleast_1d(np.asarray(stat(ent.iloc[idx]), dtype=float))
    ok = np.all(np.isfinite(reps), axis=1)
    lo = np.quantile(reps[ok], 0.025, axis=0, method="linear")
    hi = np.quantile(reps[ok], 0.975, axis=0, method="linear")
    return BootResult(est, lo, hi, int(ok.sum()), int((~ok).sum()))


# --------------------------------------------------------------------------
# 0. Reproduction gate
# --------------------------------------------------------------------------
def entity_table(w: pd.DataFrame) -> pd.DataFrame:
    """Entity-level means used by most analyses."""
    w = w.copy()
    w["d_mase"] = w["B_mase"] - w["A_mase"]
    g = w.groupby("entity", sort=False)
    ent = pd.DataFrame(
        {
            "mase_A": g["A_mase"].mean(),
            "mase_B": g["B_mase"].mean(),
            "delta": g["d_mase"].mean(),
            "n_keys": g.size(),
        }
    )
    return ent.loc[entity_order(w)]


def reproduce_primary(w: pd.DataFrame, seed: int = PRIMARY_SEED, draw=default_draw) -> BootResult:
    ent = entity_table(w)
    return cluster_bootstrap(ent, lambda e: e["delta"].mean(), seed, draw=draw)


# --------------------------------------------------------------------------
# 1. Baseline MASE levels and relative change
# --------------------------------------------------------------------------
def levels(w: pd.DataFrame, seed: int = POSTHOC_SEED, draw=default_draw) -> dict:
    ent = entity_table(w)

    def stat(e):
        ma, mb = e["mase_A"].mean(), e["mase_B"].mean()
        return [ma, mb, (mb - ma) / ma]

    r = cluster_bootstrap(ent, stat, seed, draw=draw)
    return {
        "mean_entity_mase_A": r.estimate[0],
        "mean_entity_mase_B": r.estimate[1],
        "median_entity_mase_A": ent["mase_A"].median(),
        "median_entity_mase_B": ent["mase_B"].median(),
        "median_key_mase_A": w["A_mase"].median(),
        "median_key_mase_B": w["B_mase"].median(),
        "rel_change": r.estimate[2],
        "rel_change_lo": r.lower[2],
        "rel_change_hi": r.upper[2],
    }


# --------------------------------------------------------------------------
# 2. Distribution / robustness of delta_i
# --------------------------------------------------------------------------
def trimmed_mean(x: np.ndarray, prop: float = 0.05) -> float:
    x = np.sort(np.asarray(x, dtype=float))
    k = int(np.floor(prop * x.size))
    return float(x[k : x.size - k].mean())


def distribution(w: pd.DataFrame, seed: int = POSTHOC_SEED, draw=default_draw) -> dict:
    ent = entity_table(w)
    d = ent["delta"].to_numpy()
    r = cluster_bootstrap(
        ent,
        lambda e: [np.median(e["delta"]), trimmed_mean(e["delta"].to_numpy())],
        seed,
        draw=draw,
    )
    q = np.quantile(d, [0.05, 0.25, 0.5, 0.75, 0.95], method="linear")
    return {
        "q05": q[0], "q25": q[1], "q50": q[2], "q75": q[3], "q95": q[4],
        "share_negative": float((d < 0).mean()),
        "share_zero": float((d == 0).mean()),
        "share_positive": float((d > 0).mean()),
        "median": r.estimate[0], "median_lo": r.lower[0], "median_hi": r.upper[0],
        "trim5": r.estimate[1], "trim5_lo": r.lower[1], "trim5_hi": r.upper[1],
    }


# --------------------------------------------------------------------------
# 3. Numerator / denominator decomposition
# --------------------------------------------------------------------------
def shapley_terms(eA, eB, sA, sB):
    """Exact symmetric split of eB/sB - eA/sA.

    num = (eB - eA) * (1/sA + 1/sB) / 2      -> change in absolute error, at an average scale
    den = (eA + eB) / 2 * (1/sB - 1/sA)      -> change in the MASE scale, at an average error
    num + den == eB/sB - eA/sA identically.
    """
    eA, eB, sA, sB = map(np.asarray, (eA, eB, sA, sB))
    num = (eB - eA) * (1.0 / sA + 1.0 / sB) / 2.0
    den = (eA + eB) / 2.0 * (1.0 / sB - 1.0 / sA)
    return num, den


def decomposition(w: pd.DataFrame, seed: int = POSTHOC_SEED, draw=default_draw) -> dict:
    w = w.copy()
    w["num"], w["den"] = shapley_terms(w["abs_err_A"], w["abs_err_B"], w["A_mase_den"], w["B_mase_den"])
    w["d_abs"] = w["abs_err_B"] - w["abs_err_A"]
    w["log_s_ratio"] = np.log(w["B_mase_den"] / w["A_mase_den"])
    g = w.groupby("entity", sort=False)
    ent = pd.DataFrame(
        {
            "num": g["num"].mean(),
            "den": g["den"].mean(),
            "d_abs": g["d_abs"].mean(),
            "abs_A": g["abs_err_A"].mean(),
        }
    ).loc[entity_order(w)]
    r = cluster_bootstrap(
        ent,
        lambda e: [e["num"].mean(), e["den"].mean(), e["d_abs"].mean(), e["d_abs"].mean() / e["abs_A"].mean()],
        seed,
        draw=draw,
    )
    return {
        "num_term": r.estimate[0], "num_lo": r.lower[0], "num_hi": r.upper[0],
        "den_term": r.estimate[1], "den_lo": r.lower[1], "den_hi": r.upper[1],
        "d_abs_err": r.estimate[2], "d_abs_lo": r.lower[2], "d_abs_hi": r.upper[2],
        "d_abs_rel": r.estimate[3], "d_abs_rel_lo": r.lower[3], "d_abs_rel_hi": r.upper[3],
        "share_keys_sB_gt_sA": float((w["B_mase_den"] > w["A_mase_den"]).mean()),
        "share_keys_sB_eq_sA": float((w["B_mase_den"] == w["A_mase_den"]).mean()),
        "median_log_s_ratio": float(w["log_s_ratio"].median()),
        "share_keys_errB_lt_errA": float((w["abs_err_B"] < w["abs_err_A"]).mean()),
        "share_keys_err_equal": float((w["abs_err_B"] == w["abs_err_A"]).mean()),
    }


# --------------------------------------------------------------------------
# 4. Contribution of the different-origin keys (exact split of the estimate)
# --------------------------------------------------------------------------
def origin_contribution(w: pd.DataFrame) -> dict:
    """delta_i = w_i * delta_i^diff + (1 - w_i) * delta_i^same, so

    mean_i delta_i = mean_i[w_i delta_i^diff] + mean_i[(1-w_i) delta_i^same]  (exact).
    """
    w = w.copy()
    w["d_mase"] = w["B_mase"] - w["A_mase"]
    n_i = w.groupby("entity", sort=False).size()
    contrib = w.assign(c=w["d_mase"] / w["entity"].map(n_i)).groupby(["entity", "same_origin"], sort=False)["c"].sum()
    contrib = contrib.unstack(fill_value=0.0)
    n_ent = len(n_i)
    diff_term = float(contrib.get(False, pd.Series(0.0)).sum() / n_ent)
    same_term = float(contrib.get(True, pd.Series(0.0)).sum() / n_ent)
    diff_keys = w[~w["same_origin"]]
    return {
        "n_diff_keys": int((~w["same_origin"]).sum()),
        "n_entities_with_diff_keys": int(diff_keys["entity"].nunique()),
        "term_diff_origin": diff_term,
        "term_same_origin": same_term,
        "total": diff_term + same_term,
        "share_from_diff_origin": diff_term / (diff_term + same_term) if (diff_term + same_term) != 0 else np.nan,
        "key_mean_d_mase_diff": float(diff_keys["d_mase"].mean()) if len(diff_keys) else np.nan,
        "key_mean_d_mase_same": float(w.loc[w["same_origin"], "d_mase"].mean()),
    }


# --------------------------------------------------------------------------
# 5. Ranking stability (the paper's original question)
# --------------------------------------------------------------------------
def _avg_rank(m: np.ndarray) -> np.ndarray:
    """Average ranks along axis 1 (1 = lowest MASE), ties averaged."""
    return pd.DataFrame(m).rank(axis=1, method="average").to_numpy()


def kendall_tau_b_rows(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Row-wise Kendall tau-b between x[i] and y[i] (vectorised over rows)."""
    k = x.shape[1]
    iu, ju = np.triu_indices(k, 1)
    dx = np.sign(x[:, iu] - x[:, ju])
    dy = np.sign(y[:, iu] - y[:, ju])
    s = (dx * dy).sum(axis=1)
    n0 = len(iu)
    tx = (dx == 0).sum(axis=1)
    ty = (dy == 0).sum(axis=1)
    denom = np.sqrt((n0 - tx) * (n0 - ty))
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.where(denom > 0, s / denom, np.nan)


def ranking(df: pd.DataFrame, seed: int = POSTHOC_SEED, draw=default_draw) -> dict:
    """Per-entity model ranks by entity-mean MASE in each arm, on common keys
    valid for all six models (in the paper all 85,205 keys qualify)."""
    mats = {}
    for arm in ("A", "B"):
        cols = []
        for m in MODELS:
            w = pair_wide(df, m)
            s = w.groupby("entity", sort=False)[f"{arm}_mase"].mean()
            cols.append(s.rename(m))
        mats[arm] = pd.concat(cols, axis=1)
    order = mats["A"].index
    MA = mats["A"].loc[order, MODELS].to_numpy()
    MB = mats["B"].loc[order, MODELS].to_numpy()
    RA, RB = _avg_rank(MA), _avg_rank(MB)
    tau = kendall_tau_b_rows(RA, RB)
    bestA = np.argmin(MA, axis=1)
    bestB = np.argmin(MB, axis=1)
    ent = pd.DataFrame(
        {"tau": tau, "same_best": (bestA == bestB).astype(float), "identical": np.all(RA == RB, axis=1).astype(float)},
        index=order,
    )
    for j, m in enumerate(MODELS):
        ent[f"rA_{m}"] = RA[:, j]
        ent[f"rB_{m}"] = RB[:, j]
        ent[f"mA_{m}"] = MA[:, j]
        ent[f"mB_{m}"] = MB[:, j]

    def stat(e):
        out = [np.nanmean(e["tau"]), e["same_best"].mean(), e["identical"].mean()]
        out += [e[f"rA_{m}"].mean() for m in MODELS] + [e[f"rB_{m}"].mean() for m in MODELS]
        return out

    r = cluster_bootstrap(ent, stat, seed, draw=draw)
    agg_A = {m: ent[f"mA_{m}"].mean() for m in MODELS}
    agg_B = {m: ent[f"mB_{m}"].mean() for m in MODELS}
    order_A = sorted(MODELS, key=agg_A.get)
    order_B = sorted(MODELS, key=agg_B.get)
    return {
        "n_entities": len(ent),
        "mean_tau": r.estimate[0], "mean_tau_lo": r.lower[0], "mean_tau_hi": r.upper[0],
        "share_same_best": r.estimate[1], "same_best_lo": r.lower[1], "same_best_hi": r.upper[1],
        "share_identical_rank": r.estimate[2], "identical_lo": r.lower[2], "identical_hi": r.upper[2],
        "mean_rank_A": dict(zip(MODELS, r.estimate[3:9])),
        "mean_rank_B": dict(zip(MODELS, r.estimate[9:15])),
        "aggregate_order_A": order_A,
        "aggregate_order_B": order_B,
        "aggregate_order_identical": order_A == order_B,
        "best_model_share_A": {m: float((bestA == j).mean()) for j, m in enumerate(MODELS)},
        "best_model_share_B": {m: float((bestB == j).mean()) for j, m in enumerate(MODELS)},
    }


# --------------------------------------------------------------------------
# 6. Mechanism helper: a 3-month DateOffset grid drifts off quarter-ends
# --------------------------------------------------------------------------
def offset_grid(start: pd.Timestamp, end: pd.Timestamp) -> pd.DatetimeIndex:
    """Illustration only, NOT the project's generator.

    pandas applies DateOffset(months=3) cumulatively, so it drifts from ANY
    anchor (03-31 -> 06-30 -> 09-30 -> 12-30). An anchored generator such as
    polars' "3mo" interval (start + k*3 months, day clipped to month length)
    drifts only from day-30 anchors (06-30, 09-30), which is what the paper's
    Introduction describes. Which behaviour Arm A actually has must be read off
    the saved Arm A grids (see CURSOR_PROMPT.md, task T6); do not assume it."""
    return pd.date_range(start, end, freq=pd.DateOffset(months=3))


def quarter_end_grid(start: pd.Timestamp, end: pd.Timestamp) -> pd.DatetimeIndex:
    return pd.date_range(start, end, freq="QE-DEC")
