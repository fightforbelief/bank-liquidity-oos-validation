"""Uncertainty choices stay unset. An i.i.d. row bootstrap is refused."""

from __future__ import annotations

from p12_bank_liquidity.gates import GateBlocked

CANDIDATE_METHODS = (
    "resample_bank_trajectories",
    "time_block_resampling",
    "clustered_two_way",
)


def refuse_iid_row_bootstrap() -> None:
    raise GateBlocked(
        "refusing an i.i.d. row bootstrap; bank-quarter rows are not independent"
    )


def assert_uncertainty_not_selected(method: str | None) -> None:
    if method not in (None, "UNSET"):
        raise GateBlocked(
            f"uncertainty method {method!r} cannot be selected before protocol approval"
        )
