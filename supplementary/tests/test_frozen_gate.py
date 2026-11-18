"""The gate must reproduce the frozen aggregation bit-for-bit, not up to an ulp."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve()
sys.path.insert(0, str(HERE.parents[1]))
sys.path.insert(0, str(HERE.parents[2] / "src"))

import run_posthoc as R  # noqa: E402
from p12_bank_liquidity.candidate_a.aggregation import PairedMaseRecord, aggregate_paired_mase  # noqa: E402
from p12_bank_liquidity.candidate_a.metrics import EVIDENCE_LABEL  # noqa: E402
from p12_bank_liquidity.candidate_a import stage18_inference  # noqa: E402


def _frame(seed: int = 7) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    rows = []
    for e in rng.permutation(400):
        for t in range(int(rng.integers(1, 9))):
            rows.append((f"e{e}", f"t{t}", float(rng.lognormal()), float(rng.lognormal())))
    return pd.DataFrame(rows, columns=["entity", "target", "A_mase", "B_mase"])


def test_entity_deltas_match_frozen_aggregation_bitwise():
    frame = _frame()
    records = [
        PairedMaseRecord(e, t, a, b, True, True, True, EVIDENCE_LABEL)
        for e, t, a, b in frame.itertuples(index=False)
    ]
    frozen = aggregate_paired_mase(records)
    got = R.frozen_entity_deltas(frame)
    order = list(dict.fromkeys(frame["entity"]))  # first appearance, as the Stage 18 bootstrap indexes
    # Same per-entity floats, bit for bit; the frozen aggregator only stores them sorted by id.
    assert [repr(float(x)) for x in got] == [repr(frozen.delta_by_entity[e]) for e in order]


def test_gate_bootstrap_is_the_frozen_bootstrap():
    frame = _frame(11)
    got = R.frozen_gate_bootstrap(frame)
    ref = stage18_inference.cluster_bootstrap(R.frozen_entity_deltas(frame))
    assert repr(float(got.estimate[0])) == repr(ref["estimate"])
    assert repr(float(got.lower[0])) == repr(ref["low"])
    assert repr(float(got.upper[0])) == repr(ref["high"])
    assert (got.n_valid, got.n_failed) == (2000, 0)
