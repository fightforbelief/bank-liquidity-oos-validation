"""Primary and secondary target alignment for synthetic records.

The helpers reject any record that is not labelled
``TEST_FIXTURE_NOT_RESEARCH_EVIDENCE``. They do not read research rows.
"""

from __future__ import annotations

from dataclasses import dataclass

from p12_bank_liquidity.candidate_a.metrics import EVIDENCE_LABEL


class DuplicateEntityTarget(ValueError):
    """One arm has two observed records for the same entity and target."""


class ResearchRowRejected(ValueError):
    """Stage 11 alignment does not accept unlabeled research rows."""


@dataclass(frozen=True)
class TargetRecord:
    entity_id: str
    origin_timestamp: str
    target_timestamp: str
    observed: bool
    synthetic_grid_date: bool
    evidence_label: str


@dataclass(frozen=True)
class AlignedPair:
    entity_id: str
    origin_timestamp_arm_a: str
    origin_timestamp_arm_b: str
    target_timestamp: str
    target_quarter: str
    origin_timestamp_equal: bool
    origin_quarter_equal: bool
    target_is_observed_arm_a: bool
    target_is_observed_arm_b: bool
    evidence_label: str


@dataclass(frozen=True)
class AlignmentResult:
    primary_pairs: tuple[AlignedPair, ...]
    secondary_pairs: tuple[AlignedPair, ...]
    unmatched_arm_a: tuple[tuple[str, str], ...]
    unmatched_arm_b: tuple[tuple[str, str], ...]
    evidence_label: str


def calendar_quarter(timestamp: str) -> str:
    year_text, month_text, _day_text = timestamp.split("-")
    quarter = (int(month_text) - 1) // 3 + 1
    return f"{int(year_text):04d}Q{quarter}"


def _require_fixture(record: TargetRecord) -> None:
    if record.evidence_label != EVIDENCE_LABEL:
        raise ResearchRowRejected(record.evidence_label)


def _observed_index(records: list[TargetRecord]) -> dict[tuple[str, str], TargetRecord]:
    index: dict[tuple[str, str], TargetRecord] = {}
    for record in records:
        _require_fixture(record)
        if record.synthetic_grid_date or not record.observed:
            continue
        key = (record.entity_id, record.target_timestamp)
        if key in index:
            raise DuplicateEntityTarget(f"{record.entity_id} {record.target_timestamp}")
        index[key] = record
    return index


def align_observed_targets(
    arm_a: list[TargetRecord],
    arm_b: list[TargetRecord],
) -> AlignmentResult:
    """Pair on entity and exact target timestamp. Keep each arm's origin.

    Secondary pairs are the subset whose origin timestamps are also equal.
    Observed means a finite source target. A synthetic grid date never qualifies.
    """

    index_a = _observed_index(arm_a)
    index_b = _observed_index(arm_b)
    primary: list[AlignedPair] = []
    for key in sorted(set(index_a) & set(index_b)):
        left = index_a[key]
        right = index_b[key]
        origin_a = left.origin_timestamp
        origin_b = right.origin_timestamp
        primary.append(
            AlignedPair(
                entity_id=left.entity_id,
                origin_timestamp_arm_a=origin_a,
                origin_timestamp_arm_b=origin_b,
                target_timestamp=left.target_timestamp,
                target_quarter=calendar_quarter(left.target_timestamp),
                origin_timestamp_equal=origin_a == origin_b,
                origin_quarter_equal=calendar_quarter(origin_a) == calendar_quarter(origin_b),
                target_is_observed_arm_a=True,
                target_is_observed_arm_b=True,
                evidence_label=EVIDENCE_LABEL,
            )
        )
    identity = [(pair.entity_id, pair.target_timestamp) for pair in primary]
    if len(identity) != len(set(identity)):
        raise DuplicateEntityTarget("primary keys are not unique")
    secondary = tuple(pair for pair in primary if pair.origin_timestamp_equal)
    return AlignmentResult(
        primary_pairs=tuple(primary),
        secondary_pairs=secondary,
        unmatched_arm_a=tuple(sorted(set(index_a) - set(index_b))),
        unmatched_arm_b=tuple(sorted(set(index_b) - set(index_a))),
        evidence_label=EVIDENCE_LABEL,
    )
