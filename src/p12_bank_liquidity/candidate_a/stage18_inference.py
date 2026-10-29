"""Frozen entity-cluster bootstrap. No research identifiers."""

from __future__ import annotations

import numpy as np

FROZEN_SEED = 20261027
FROZEN_REPLICATES = 2000
FROZEN_QUANTILE_METHOD = "linear"


def unweighted_mean(deltas: np.ndarray) -> float:
    if deltas.size == 0 or not np.isfinite(deltas).all():
        raise ValueError("entity deltas are not finite")
    return float(deltas.mean())


def cluster_bootstrap(
    deltas: np.ndarray,
    *,
    seed: int = FROZEN_SEED,
    replicates: int = FROZEN_REPLICATES,
) -> dict[str, object]:
    """Resample entities with replacement and keep each entity's delta.

    A replicate that is not finite is counted as failed. It is not replaced
    with zero. The interval is the frozen 95 percent type-7 percentile only
    when every requested replicate is valid.
    """

    attempted = int(replicates)
    if deltas.ndim != 1 or deltas.size == 0 or not np.isfinite(deltas).all():
        return {
            "attempted": attempted,
            "valid": 0,
            "failed": attempted,
            "estimate": None,
            "low": None,
            "high": None,
            "interval": "unavailable",
            "reason": "no_finite_entity_deltas",
        }
    rng = np.random.Generator(np.random.PCG64(seed))
    draws: list[float] = []
    failed = 0
    size = int(deltas.size)
    for _ in range(attempted):
        sample = deltas[rng.choice(size, size=size, replace=True)]
        if sample.size == 0 or not np.isfinite(sample).all():
            failed += 1
            continue
        draws.append(float(sample.mean()))
    estimate = unweighted_mean(deltas)
    if len(draws) != attempted:
        return {
            "attempted": attempted,
            "valid": len(draws),
            "failed": failed,
            "estimate": estimate,
            "low": None,
            "high": None,
            "interval": "unavailable",
            "reason": "valid_replicates_differ_from_requested",
        }
    low, high = np.quantile(np.asarray(draws, dtype=float), [0.025, 0.975], method=FROZEN_QUANTILE_METHOD)
    return {
        "attempted": attempted,
        "valid": attempted,
        "failed": 0,
        "estimate": estimate,
        "low": float(low),
        "high": float(high),
        "interval": "available",
        "reason": None,
    }
