"""Call one installed model and return the adapter's scalar.

Importing this module does not import StatsForecast. The caller supplies an
already constructed model object.
"""

from __future__ import annotations

from p12_bank_liquidity.candidate_a.model_output import extract_one_step_scalar


class StatsForecastOneStep:
    """One-step bridge. ``fallback_model`` is not used and has no object here."""

    def __init__(self, model: object, model_id: str) -> None:
        self._model = model
        self.model_id = model_id

    def predict_one(self, y_train: list[float]) -> float:
        import numpy as np

        forecast = getattr(self._model, "forecast")
        output = forecast(np.asarray(y_train, dtype=float), h=1)
        return extract_one_step_scalar(output)
