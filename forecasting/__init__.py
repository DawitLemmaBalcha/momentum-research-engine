# forecasting/__init__.py

"""Optional TimesFM forecasting layer.

The trading engine works without this package's heavy dependencies. Enable it
by setting ``TIMESFM_ENABLED = True`` in the active config and installing
``requirements-timesfm.txt``.
"""

from forecasting.gate import ForecastGate, GateDecision
from forecasting.timesfm_backend import (
    ForecastBackend,
    Forecaster,
    TimesFMForecaster,
    TimesFMUnavailable,
)
from forecasting.types import Forecast

__all__ = [
    "Forecast",
    "ForecastBackend",
    "ForecastGate",
    "Forecaster",
    "GateDecision",
    "TimesFMForecaster",
    "TimesFMUnavailable",
]
