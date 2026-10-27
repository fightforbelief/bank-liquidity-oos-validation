"""Source model roster. Fitting is refused until the compute gate is approved."""

from __future__ import annotations

from p12_bank_liquidity.gates import GateBlocked

SOURCE_MODELS = (
    "Historical Average",
    "ARIMA",
    "SES",
    "Theta",
    "DLinear",
    "DeepAR",
    "KAN",
    "N-BEATS",
    "N-HiTS",
    "NLinear",
    "TiDE",
    "Vanilla Transformer",
)


def assert_model_list_unchanged(names: tuple[str, ...] | list[str]) -> None:
    if tuple(names) != SOURCE_MODELS:
        raise GateBlocked(
            "model list differs from the OFR roster; silent substitution is not allowed"
        )


def refuse_model_fit(authorized: bool, model_name: str) -> None:
    assert_model_list_unchanged(SOURCE_MODELS)
    if model_name not in SOURCE_MODELS:
        raise GateBlocked(f"{model_name} is not in the frozen source roster")
    if not authorized:
        raise GateBlocked(
            f"model fit refused for {model_name}: full_model_run_authorized is false"
        )
