"""Aggregate uniqueness of eligible entity-target keys.

Fixtures must carry ``TEST_FIXTURE_NOT_RESEARCH_EVIDENCE``. The functions
return counts only. They do not drop, average, or choose among duplicate keys.
"""

from __future__ import annotations

from dataclasses import dataclass

from p12_bank_liquidity.candidate_a.metrics import EVIDENCE_LABEL


class ResearchRowRejected(ValueError):
    """Uniqueness helpers do not accept unlabeled research rows."""


@dataclass(frozen=True)
class EligibleTarget:
    entity_id: str
    target_timestamp: str
    origin_timestamp: str
    observed: bool
    synthetic_grid_row: bool
    evidence_label: str


@dataclass(frozen=True)
class ArmKeySummary:
    eligible_rows: int
    distinct_keys: int
    duplicated_keys: int
    max_multiplicity: int


@dataclass(frozen=True)
class UniquenessDecision:
    status: str
    arm_a: ArmKeySummary
    arm_b: ArmKeySummary
    common_unique_keys: int
    common_entities: int
    inherited_common_keys: int
    difference_from_inherited_common_keys: int
    common_keys_with_different_origins: int
    evidence_label: str


def _require_fixture(row: EligibleTarget) -> None:
    if row.evidence_label != EVIDENCE_LABEL:
        raise ResearchRowRejected(row.evidence_label)


def summarize_arm(rows: list[EligibleTarget]) -> ArmKeySummary:
    counts: dict[tuple[str, str], int] = {}
    for row in rows:
        _require_fixture(row)
        if row.synthetic_grid_row or not row.observed:
            continue
        key = (row.entity_id, row.target_timestamp)
        counts[key] = counts.get(key, 0) + 1
    duplicated = sum(1 for count in counts.values() if count > 1)
    maximum = max(counts.values()) if counts else 0
    return ArmKeySummary(
        eligible_rows=sum(counts.values()),
        distinct_keys=len(counts),
        duplicated_keys=duplicated,
        max_multiplicity=maximum,
    )


def _eligible_index(rows: list[EligibleTarget]) -> dict[tuple[str, str], list[str]]:
    index: dict[tuple[str, str], list[str]] = {}
    for row in rows:
        _require_fixture(row)
        if row.synthetic_grid_row or not row.observed:
            continue
        index.setdefault((row.entity_id, row.target_timestamp), []).append(row.origin_timestamp)
    return index


def decide_uniqueness(
    arm_a: list[EligibleTarget],
    arm_b: list[EligibleTarget],
    *,
    inherited_common_keys: int,
) -> UniquenessDecision:
    """Classify the audited keys without changing the inherited count."""

    summary_a = summarize_arm(arm_a)
    summary_b = summarize_arm(arm_b)
    index_a = _eligible_index(arm_a)
    index_b = _eligible_index(arm_b)
    common = set(index_a) & set(index_b)
    different_origins = 0
    for key in common:
        origins_a = set(index_a[key])
        origins_b = set(index_b[key])
        if len(origins_a) == 1 and len(origins_b) == 1 and origins_a != origins_b:
            different_origins += 1
    if summary_a.duplicated_keys or summary_b.duplicated_keys:
        status = "BLOCKED_DUPLICATE_ENTITY_TARGET_KEYS"
    elif len(common) != inherited_common_keys:
        status = "BLOCKED_INHERITED_COUNT_MISMATCH"
    else:
        status = "VERIFIED_FOR_AUDITED_ELIGIBLE_KEYS"
    return UniquenessDecision(
        status=status,
        arm_a=summary_a,
        arm_b=summary_b,
        common_unique_keys=len(common),
        common_entities=len({entity for entity, _target in common}),
        inherited_common_keys=inherited_common_keys,
        difference_from_inherited_common_keys=len(common) - inherited_common_keys,
        common_keys_with_different_origins=different_origins,
        evidence_label=EVIDENCE_LABEL,
    )
