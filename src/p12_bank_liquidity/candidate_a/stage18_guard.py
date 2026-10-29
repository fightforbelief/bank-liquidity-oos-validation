"""Stage 18 authorization guard. It does not change the Stage 17 pilot cap."""

from __future__ import annotations

from p12_bank_liquidity.candidate_a.pilot_guard import PILOT_ENTITY_CAP

STAGE18_LABEL = "STAGE18_FULL_SAMPLE_PRIMARY_RESEARCH_RUN"
PILOT_LABEL = "PILOT_ONLY_NOT_PRIMARY_RESEARCH_RESULTS"
FULL_SAMPLE_DECISION = "PASS_FOR_FULL_SAMPLE_NONCOMMERCIAL_RESEARCH"


def assert_pilot_cap_unchanged() -> None:
    if PILOT_ENTITY_CAP != 20:
        raise RuntimeError("the stage 17 pilot cap is no longer 20")


def assert_full_sample_label(label: str) -> None:
    assert_pilot_cap_unchanged()
    if label == PILOT_LABEL or label != STAGE18_LABEL:
        raise RuntimeError("the full-sample runner cannot use the pilot path")
