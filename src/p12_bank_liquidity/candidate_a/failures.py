"""Failure records stay in the log. Nothing is dropped silently."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class FailureRecord:
    entity_id: str
    model_id: str
    reason: str
    evidence_label: str


def record_failure(
    log: list[FailureRecord],
    *,
    entity_id: str,
    model_id: str,
    reason: str,
    evidence_label: str,
) -> list[FailureRecord]:
    log.append(
        FailureRecord(
            entity_id=entity_id,
            model_id=model_id,
            reason=reason,
            evidence_label=evidence_label,
        )
    )
    return log
