# tests/test_forecast_gate.py

import numpy as np
import pytest

from forecasting import ForecastGate, TimesFMUnavailable
from forecasting.timesfm_backend import TimesFMForecaster


class FakeBackend:
    def __init__(self, slope: float = 0.01, quantiles: bool = False, fail: bool = False):
        self.slope = slope
        self.quantiles = quantiles
        self.fail = fail
        self.calls = 0

    def forecast(self, closes, horizon):
        self.calls += 1
        if self.fail:
            raise TimesFMUnavailable("no model")
        last = float(closes[-1])
        values = np.array([last * (1.0 + self.slope * (i + 1)) for i in range(horizon)])
        quantiles = None
        if self.quantiles:
            quantiles = np.tile(values[:, None], (1, 10))
        return values, quantiles


def series(n=100, start=100.0):
    return [start + i * 0.1 for i in range(n)]


def gate(backend, **kwargs):
    defaults = {"horizon": 5, "min_context": 64, "min_agreement": 0.6, "min_expected_return_pct": 0.0, "cache_bars": 5}
    defaults.update(kwargs)
    return ForecastGate(TimesFMForecaster(backend=backend), **defaults)


def test_rising_forecast_allows_long_and_vetoes_short():
    g = gate(FakeBackend(slope=0.01))
    long_decision = g.evaluate(series(), "buy", timestamp=0)
    short_decision = g.evaluate(series(), "sell", timestamp=0)
    assert long_decision.allowed is True
    assert long_decision.agreement == pytest.approx(1.0)
    assert short_decision.allowed is False
    assert short_decision.agreement == pytest.approx(0.0)


def test_flat_forecast_is_not_agreement():
    g = gate(FakeBackend(slope=0.0))
    decision = g.evaluate(series(), "long", timestamp=0)
    assert decision.allowed is False
    assert decision.agreement == pytest.approx(0.0)


def test_min_expected_return_filters_tiny_moves():
    g = gate(FakeBackend(slope=0.00001), min_expected_return_pct=1.0)
    decision = g.evaluate(series(), "long", timestamp=0)
    assert decision.allowed is False
    assert decision.reason == "forecast points against the entry"


def test_insufficient_history_passes_through():
    g = gate(FakeBackend(), min_context=64)
    decision = g.evaluate(series(10), "long", timestamp=0)
    assert decision.allowed is True
    assert "insufficient history" in decision.reason


def test_forecast_is_cached_within_window():
    backend = FakeBackend()
    g = gate(backend, cache_bars=5)
    first = g.evaluate(series(), "long", timestamp=0)
    second = g.evaluate(series(), "long", timestamp=120)
    assert backend.calls == 1
    assert second.cached is True
    assert first.cached is False
    g.evaluate(series(), "long", timestamp=600)
    assert backend.calls == 2


def test_new_close_invalidates_cache():
    backend = FakeBackend()
    g = gate(backend)
    g.evaluate(series(), "long", timestamp=0)
    g.evaluate(series(101), "long", timestamp=60)
    assert backend.calls == 2


def test_unavailable_backend_fails_closed_by_default():
    g = gate(FakeBackend(fail=True))
    decision = g.evaluate(series(), "long", timestamp=0)
    assert decision.allowed is False
    assert "unavailable" in decision.reason
    assert "no model" in g.last_error


def test_unavailable_backend_can_fail_open():
    g = gate(FakeBackend(fail=True), fail_open=True)
    decision = g.evaluate(series(), "long", timestamp=0)
    assert decision.allowed is True


def test_quantiles_are_optional_and_ignored_by_gate():
    g = gate(FakeBackend(slope=0.01, quantiles=True))
    decision = g.evaluate(series(), "long", timestamp=0)
    assert decision.allowed is True


def test_side_names_are_normalized():
    g = gate(FakeBackend(slope=0.01))
    assert g.evaluate(series(), "LONG", timestamp=0).allowed is True
    assert g.evaluate(series(), "sell", timestamp=0).side == "short"
    with pytest.raises(ValueError):
        g.evaluate(series(), "sideways", timestamp=0)


def test_forecaster_rejects_short_series():
    forecaster = TimesFMForecaster(backend=FakeBackend())
    with pytest.raises(ValueError):
        forecaster.forecast(series(10), horizon=5)


def test_forecaster_trims_to_context_and_returns_forecast():
    forecaster = TimesFMForecaster(backend=FakeBackend(slope=0.01), max_context=64)
    forecast = forecaster.forecast(series(200), horizon=5)
    assert forecast.context_length == 64
    assert forecast.horizon == 5
    assert forecast.values.shape == (5,)
    assert forecast.last_value == pytest.approx(series(200)[-1])


def test_forecaster_wraps_load_failures(monkeypatch):
    forecaster = TimesFMForecaster()
    monkeypatch.setattr(
        forecaster,
        "_load_timesfm",
        lambda: (_ for _ in ()).throw(TimesFMUnavailable("missing torch")),
    )
    with pytest.raises(TimesFMUnavailable):
        forecaster.forecast(series(), horizon=5)
