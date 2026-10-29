"""Frozen entity-cluster bootstrap settings. No research errors are resampled."""

from __future__ import annotations

import math
import random
from dataclasses import dataclass

METHOD = "ENTITY_CLUSTER_BOOTSTRAP"
REPLICATES = 2000
SEED = 20261027
CONFIDENCE_LEVEL = 0.95
INTERVAL = "PERCENTILE"
PRACTICAL_EQUIVALENCE_MARGIN = "NOT_SET"
FORMAL_EQUIVALENCE_CLAIM = "NOT_AUTHORIZED"


@dataclass(frozen=True)
class BootstrapResult:
    estimates: tuple[float, ...]
    failed_replicate_indices: tuple[int, ...]
    lower: float | None
    upper: float | None
    replicates: int
    seed: int
    confidence_level: float
    interval: str


def percentile_type7(sorted_values: list[float], probability: float) -> float:
    """Hyndman-Fan type 7 percentile. This is the planned interval rule."""

    count = len(sorted_values)
    if count == 0:
        raise ValueError("percentile requires at least one value")
    if count == 1:
        return sorted_values[0]
    position = (count - 1) * probability
    low = math.floor(position)
    high = math.ceil(position)
    if low == high:
        return sorted_values[low]
    weight = position - low
    return sorted_values[low] * (1.0 - weight) + sorted_values[high] * weight


def entity_cluster_bootstrap(
    deltas: dict[str, float | None],
    *,
    replicates: int = REPLICATES,
    seed: int = SEED,
    confidence_level: float = CONFIDENCE_LEVEL,
) -> BootstrapResult:
    """Resample entity ids with replacement and recompute the unweighted mean.

    A replicate that draws an undefined entity delta is counted as failed.
    It is not filled with zero or an epsilon.
    """

    if replicates < 1:
        raise ValueError("replicates must be positive")
    if not 0.0 < confidence_level < 1.0:
        raise ValueError("confidence level must lie strictly between 0 and 1")
    entity_ids = sorted(deltas)
    if not entity_ids:
        return BootstrapResult(
            (),
            tuple(range(replicates)),
            None,
            None,
            replicates,
            seed,
            confidence_level,
            INTERVAL,
        )
    generator = random.Random(seed)
    estimates: list[float] = []
    failed: list[int] = []
    for replicate in range(replicates):
        draw = [generator.choice(entity_ids) for _ in entity_ids]
        values = [deltas[entity_id] for entity_id in draw]
        if any(value is None or not math.isfinite(value) for value in values):
            failed.append(replicate)
            continue
        estimates.append(sum(values) / len(values))
    if not estimates:
        lower = None
        upper = None
    else:
        ordered = sorted(estimates)
        tail = (1.0 - confidence_level) / 2.0
        lower = percentile_type7(ordered, tail)
        upper = percentile_type7(ordered, 1.0 - tail)
    return BootstrapResult(
        tuple(estimates),
        tuple(failed),
        lower,
        upper,
        replicates,
        seed,
        confidence_level,
        INTERVAL,
    )
