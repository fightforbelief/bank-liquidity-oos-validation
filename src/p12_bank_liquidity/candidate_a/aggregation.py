"""Entity-level paired MASE aggregation. No forecast is fit or stored."""

from __future__ import annotations

from dataclasses import dataclass

from p12_bank_liquidity.candidate_a.alignment import DuplicateEntityTarget
from p12_bank_liquidity.candidate_a.metrics import EVIDENCE_LABEL


class ResearchRowRejected(ValueError):
    """Stage 11 aggregation does not accept unlabeled research rows."""


@dataclass(frozen=True)
class PairedMaseRecord:
    entity_id: str
    target_timestamp: str
    mase_arm_a: float | None
    mase_arm_b: float | None
    target_observed_both: bool
    forecast_finite_both: bool
    denominator_defined_both: bool
    evidence_label: str


@dataclass(frozen=True)
class Exclusion:
    entity_id: str
    target_timestamp: str
    reason: str
    evidence_label: str


@dataclass(frozen=True)
class AggregationResult:
    delta_by_entity: dict[str, float]
    estimate: float | None
    exclusions: tuple[Exclusion, ...]
    evidence_label: str


def _finite(value: float | None) -> bool:
    return value is not None and value == value and value not in (float("inf"), float("-inf"))


def aggregate_paired_mase(records: list[PairedMaseRecord]) -> AggregationResult:
    """Unweighted mean of entity-level Arm B minus Arm A MASE.

    Each entity mean uses only the paired target quarters that are observed,
    finite, and defined on both arms. Excluded keys stay in the result.
    """

    seen: set[tuple[str, str]] = set()
    valid: dict[str, list[tuple[float, float]]] = {}
    exclusions: list[Exclusion] = []
    for record in records:
        if record.evidence_label != EVIDENCE_LABEL:
            raise ResearchRowRejected(record.evidence_label)
        key = (record.entity_id, record.target_timestamp)
        if key in seen:
            raise DuplicateEntityTarget(f"{record.entity_id} {record.target_timestamp}")
        seen.add(key)
        reasons: list[str] = []
        if not record.target_observed_both:
            reasons.append("target_not_observed_in_both_arms")
        if not record.forecast_finite_both:
            reasons.append("forecast_not_finite_in_both_arms")
        if not record.denominator_defined_both:
            reasons.append("mase_denominator_undefined")
        if not reasons and not (_finite(record.mase_arm_a) and _finite(record.mase_arm_b)):
            reasons.append("forecast_not_finite_in_both_arms")
        if reasons:
            exclusions.extend(
                Exclusion(record.entity_id, record.target_timestamp, reason, EVIDENCE_LABEL)
                for reason in reasons
            )
            continue
        assert record.mase_arm_a is not None and record.mase_arm_b is not None
        valid.setdefault(record.entity_id, []).append((record.mase_arm_a, record.mase_arm_b))
    deltas = {}
    for entity_id in sorted(valid):
        pairs = valid[entity_id]
        mean_a = sum(item[0] for item in pairs) / len(pairs)
        mean_b = sum(item[1] for item in pairs) / len(pairs)
        deltas[entity_id] = mean_b - mean_a
    estimate = None if not deltas else sum(deltas.values()) / len(deltas)
    return AggregationResult(deltas, estimate, tuple(exclusions), EVIDENCE_LABEL)
