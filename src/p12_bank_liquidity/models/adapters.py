"""Adapters do not call neural or statistical estimators in the audit stage."""

from p12_bank_liquidity.models.registry import refuse_model_fit


def fit_model(model_name: str, authorized: bool = False):
    refuse_model_fit(authorized, model_name)
    raise RuntimeError("fit_model has no estimator implementation in the audit stage")
