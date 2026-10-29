"""Scalar extraction for the StatsForecast 2.1.1 one-step return.

Observed on the Stage 15 synthetic fixture: ``forecast(y, h=1)`` returns a
dict whose ``mean`` value is a one-element float64 array. This module does not
import StatsForecast.
"""

from __future__ import annotations

OBSERVED_ONE_STEP_BINDING = "dict_mean_ndarray_shape_1"


class UnexpectedModelOutput(ValueError):
    """The model return does not match the observed one-step binding."""


def extract_one_step_scalar(output: object) -> float:
    """Return the single ``mean`` value. Reject any other shape."""

    if not isinstance(output, dict) or "mean" not in output:
        raise UnexpectedModelOutput("missing mean")
    mean = output["mean"]
    shape = getattr(mean, "shape", None)
    if tuple(shape or ()) != (1,):
        raise UnexpectedModelOutput(f"shape {shape}")
    try:
        value = float(mean[0])
    except (TypeError, ValueError, IndexError) as exc:
        raise UnexpectedModelOutput("unreadable mean") from exc
    return value
