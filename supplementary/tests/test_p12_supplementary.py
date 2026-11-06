"""Synthetic-data tests for analysis/p12_supplementary.py. Run: python -m pytest -q tests"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from scipy.stats import kendalltau

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import p12_supplementary as S  # noqa: E402


def make_synthetic(n_ent=300, seed=0, identical_arms=False):
    rng = np.random.default_rng(seed)
    rows = []
    for e in range(n_ent):
        n_keys = rng.integers(1, 12)
        targets = pd.date_range("2000-03-31", periods=n_keys, freq="QE-DEC")
        for t in targets:
            y = rng.uniform(0.01, 0.2)
            diff_origin = rng.random() < 0.05
            for m in S.MODELS:
                sA = rng.uniform(0.002, 0.03)
                eA = rng.normal(0, 0.01)
                if identical_arms:
                    sB, eB = sA, eA
                else:
                    sB = sA * rng.uniform(0.9, 1.4)
                    eB = eA * rng.uniform(0.7, 1.1)
                oA = t - pd.offsets.QuarterEnd(1)
                oB = oA if not diff_origin else oA - pd.Timedelta(days=1)
                for arm, s, err, o in (("A", sA, eA, oA), ("B", sB, eB, oB)):
                    rows.append(
                        dict(entity=e, target_ts=t, arm=arm, model=m, origin_ts=o,
                             y_true=y, y_pred=y - err, mase_den=s, mase=abs(err) / s)
                    )
    return pd.DataFrame(rows)


@pytest.fixture(scope="module")
def df():
    return make_synthetic()


def test_primary_matches_direct_computation(df):
    w = S.pair_wide(df, "Naive")
    direct = (
        w.assign(d=w["B_mase"] - w["A_mase"]).groupby("entity")["d"].mean().mean()
    )
    r = S.reproduce_primary(w)
    assert np.isclose(r.estimate[0], direct)
    assert r.n_valid == S.N_BOOT and r.n_failed == 0
    assert r.lower[0] <= r.estimate[0] <= r.upper[0]


def test_bootstrap_is_deterministic(df):
    w = S.pair_wide(df, "AutoTheta")
    r1, r2 = S.reproduce_primary(w), S.reproduce_primary(w)
    assert np.array_equal(r1.lower, r2.lower) and np.array_equal(r1.upper, r2.upper)


def test_shapley_identity_keywise():
    rng = np.random.default_rng(1)
    eA, eB = rng.uniform(0, 1, 1000), rng.uniform(0, 1, 1000)
    sA, sB = rng.uniform(0.1, 2, 1000), rng.uniform(0.1, 2, 1000)
    num, den = S.shapley_terms(eA, eB, sA, sB)
    assert np.allclose(num + den, eB / sB - eA / sA)


def test_decomposition_sums_to_estimate(df):
    w = S.pair_wide(df, "SeasonalNaive")
    d = S.decomposition(w)
    est = S.entity_table(w)["delta"].mean()
    assert np.isclose(d["num_term"] + d["den_term"], est)


def test_origin_contribution_sums_to_estimate(df):
    w = S.pair_wide(df, "Naive")
    c = S.origin_contribution(w)
    est = S.entity_table(w)["delta"].mean()
    assert np.isclose(c["total"], est)
    assert c["n_diff_keys"] == int((~w["same_origin"]).sum())


def test_tau_matches_scipy():
    rng = np.random.default_rng(2)
    x = rng.integers(1, 4, size=(200, 6)).astype(float)  # includes ties
    y = rng.integers(1, 4, size=(200, 6)).astype(float)
    ours = S.kendall_tau_b_rows(x, y)
    ref = np.array([kendalltau(a, b).statistic for a, b in zip(x, y)])
    assert np.allclose(ours, ref, equal_nan=True)


def test_ranking_identical_arms_gives_tau_one():
    d = make_synthetic(n_ent=80, seed=3, identical_arms=True)
    r = S.ranking(d)
    assert np.isclose(r["mean_tau"], 1.0)
    assert np.isclose(r["share_same_best"], 1.0)
    assert r["aggregate_order_identical"]


def test_ranking_runs_on_different_arms(df):
    r = S.ranking(df)
    assert -1 <= r["mean_tau"] <= 1
    assert set(r["aggregate_order_A"]) == set(S.MODELS)


def test_levels_and_distribution(df):
    w = S.pair_wide(df, "HistoricAverage")
    lv = S.levels(w)
    ent = S.entity_table(w)
    assert np.isclose(lv["rel_change"], ent["delta"].mean() / ent["mase_A"].mean())
    ds = S.distribution(w)
    assert np.isclose(ds["share_negative"] + ds["share_zero"] + ds["share_positive"], 1.0)


def test_offset_grid_drifts_off_quarter_end():
    g = S.offset_grid(pd.Timestamp("1980-06-30"), pd.Timestamp("1981-06-30"))
    assert list(g.strftime("%m-%d")) == ["06-30", "09-30", "12-30", "03-30", "06-30"]
    qe = S.quarter_end_grid(pd.Timestamp("1980-06-30"), pd.Timestamp("1981-06-30"))
    assert not set(g).issuperset(set(qe))  # 1980-12-31 and 1981-03-31 are missing
