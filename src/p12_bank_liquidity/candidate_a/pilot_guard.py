"""Hard limits for the Stage 17 pilot. No research identifiers."""

from __future__ import annotations

import hashlib

PILOT_ENTITY_CAP = 20
PILOT_CANDIDATE_POOL_CAP = 100
PILOT_MINIMUM_COMMON_ENTITIES = 10
PILOT_LABEL = "PILOT_ONLY_NOT_PRIMARY_RESEARCH_RESULTS"


def entity_rank(entity_id: str) -> str:
    """SHA-256 of the canonical UTF-8 text of an entity id."""

    return hashlib.sha256(entity_id.encode("utf-8")).hexdigest()


def candidate_pool(entity_ids: list[str]) -> list[str]:
    ordered = sorted(set(entity_ids), key=entity_rank)
    return ordered[:PILOT_CANDIDATE_POOL_CAP]


def select_fitted_entities(eligible_in_pool_order: list[str]) -> list[str]:
    """Take the hash-ordered eligible entities, never more than the fixed cap."""

    if len(eligible_in_pool_order) < PILOT_MINIMUM_COMMON_ENTITIES:
        raise ValueError("insufficient pilot coverage")
    chosen = eligible_in_pool_order[:PILOT_ENTITY_CAP]
    if len(chosen) > PILOT_ENTITY_CAP:
        raise RuntimeError("pilot cap exceeded")
    return chosen
