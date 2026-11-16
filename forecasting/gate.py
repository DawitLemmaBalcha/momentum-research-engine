# forecasting/gate.py

"""Entry veto driven by a TimesFM forecast.

The gate is deliberately small: it asks for a point path over a short horizon
and vetoes an entry when the forecast disagrees with the direction the anchor
is about to trade. It never sizes positions and never places orders.

No look-ahead: callers pass the close history that ends at the decision bar.
The gate trims to the configured context and caches the forecast for a few
bars because a foundation-model forward pass is far more expensive than the
rest of the entry check.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

from forecasting.timesfm_backend import Forecaster
from forecasting.types import Forecast, normalize_side


@dataclass(frozen=True)
class GateDecision:
    allowed: bool
    reason: str
    side: str
    expected_return_pct: float | None = None
    agreement: float | None = None
    cached: bool = False


class ForecastGate:
    """Veto entries whose direction the forecast does not support."""

    def __init__(
        self,
        forecaster: Forecaster,
        horizon: int = 5,
        min_context: int = 64,
        min_agreement: float = 0.6,
        min_expected_return_pct: float = 0.0,
        cache_bars: int = 5,
        bar_seconds: float = 60.0,
        fail_open: bool = False,
    ) -> None:
        if horizon < 1:
            raise ValueError("horizon must be positive")
        if not 0.0 <= min_agreement <= 1.0:
            raise ValueError("min_agreement must be between 0 and 1")
        self.forecaster = forecaster
        self.horizon = int(horizon)
        self.min_context = int(min_context)
        self.min_agreement = float(min_agreement)
        self.min_expected_return_pct = float(min_expected_return_pct)
        self.cache_bars = max(1, int(cache_bars))
        self.bar_seconds = float(bar_seconds)
        self.fail_open = bool(fail_open)
        self.last_error = ""
        self._cached_forecast: Forecast | None = None
        self._cached_at: float | None = None
        self._cached_last: float | None = None

    def evaluate(self, closes: Sequence[float], side: str, timestamp: float) -> GateDecision:
        normalized = normalize_side(side)
        series = list(closes)
        if len(series) < self.min_context:
            return GateDecision(
                allowed=True,
                reason=f"insufficient history ({len(series)}/{self.min_context})",
                side=normalized,
            )

        forecast, reused = self._forecast_for(series, timestamp)
        if forecast is None:
            return GateDecision(
                allowed=self.fail_open,
                reason="timesfm unavailable; failing open" if self.fail_open else "timesfm unavailable; veto",
                side=normalized,
            )

        agreement = forecast.agreement(normalized)
        expected_return = forecast.expected_return_pct
        if normalized == "long":
            direction_ok = expected_return >= self.min_expected_return_pct
        else:
            direction_ok = expected_return <= -self.min_expected_return_pct

        allowed = agreement >= self.min_agreement and direction_ok
        if allowed:
            reason = "forecast agrees"
        elif not direction_ok:
            reason = "forecast points against the entry"
        else:
            reason = f"forecast agreement {agreement:.2f} < {self.min_agreement:.2f}"

        return GateDecision(
            allowed=allowed,
            reason=reason,
            side=normalized,
            expected_return_pct=expected_return,
            agreement=agreement,
            cached=reused,
        )

    def _forecast_for(self, series: list[float], timestamp: float) -> tuple[Forecast | None, bool]:
        last_value = float(series[-1])
        reuse = (
            self._cached_forecast is not None
            and self._cached_last == last_value
            and self._cached_at is not None
            and 0.0 <= float(timestamp) - self._cached_at < self.cache_bars * self.bar_seconds
        )
        if reuse:
            return self._cached_forecast, True

        try:
            forecast = self.forecaster.forecast(series, self.horizon)
        except Exception as exc:  # noqa: BLE001 - any backend failure must fall back to the configured fail mode
            self._cached_forecast = None
            self._cached_at = None
            self._cached_last = None
            self.last_error = str(exc)
            return None, False

        self._cached_forecast = forecast
        self._cached_at = float(timestamp)
        self._cached_last = last_value
        self.last_error = ""
        return forecast, False
