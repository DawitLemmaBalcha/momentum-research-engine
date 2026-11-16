# tests/test_bot_gate.py

from types import SimpleNamespace

import numpy as np

from core.core_types import Order, SimulationClock
from core.trading_bot_base import MomentumAnchorBotBase
from forecasting import ForecastGate
from forecasting.timesfm_backend import TimesFMForecaster


class FakeBackend:
    def __init__(self, slope: float):
        self.slope = slope
        self.calls = 0

    def forecast(self, closes, horizon):
        self.calls += 1
        last = float(closes[-1])
        values = np.array([last * (1.0 + self.slope * (i + 1)) for i in range(horizon)])
        return values, None


class StubConnector:
    def __init__(self):
        self.orders = []

    def place_order(self, **kwargs):
        self.orders.append(kwargs)
        return {"success": True, "entry_price": kwargs["price"], "trade_id": "t-1", "qty": kwargs["qty"]}

    def close_position(self, position):
        return {"success": True, "pnl": 0.0, "close_price": 1.0}


class GateTestBot(MomentumAnchorBotBase):
    def calculate_position_size(self, price, return_details=False):
        return (1.0, "test") if return_details else 1.0

    def close_position(self, position, current_price, reason):
        return None


def make_config(**overrides):
    values = {
        "SYMBOL": "BTC/USDT",
        "KELLY_FRACTION": 1.0,
        "MAX_POSITION_SIZE_PERCENT": 0.5,
        "MAX_POSITIONS": 10,
        "PYRAMID_SCALING_FACTOR": 0.5,
        "ATR_INITIAL_STOP_MULTIPLIER": 0.5,
        "TIMESFM_CONTEXT": 128,
        "TIMESFM_ENABLED": False,
    }
    values.update(overrides)
    return SimpleNamespace(**values)


def make_bot(slope=None, min_context=64, context=128):
    clock = SimulationClock(1_000_000)
    bot = GateTestBot(100_000.0, clock, make_config(TIMESFM_CONTEXT=context), StubConnector(), silent=True)
    if slope is not None:
        bot.forecast_gate = ForecastGate(TimesFMForecaster(backend=FakeBackend(slope)), horizon=5, min_context=min_context)
    return bot, clock


def feed(bot, count=80, start_ts=1_000_000):
    for i in range(count):
        bot.observe_close(start_ts + i * 60, 100.0 + i * 0.1)


def test_history_is_ignored_when_gate_disabled():
    bot, _ = make_bot()
    feed(bot)
    assert len(bot._close_history) == 0


def test_history_is_bounded_and_deduplicated():
    bot, _ = make_bot(slope=0.01, context=128)
    feed(bot, 200)
    assert len(bot._close_history) == 128
    assert bot._close_history[-1] == 100.0 + 199 * 0.1

    before = len(bot._close_history)
    bot.observe_close(1_000_000, 999.0)  # older than the last recorded bar
    assert len(bot._close_history) == before


def test_falling_forecast_vetoes_long_entry():
    bot, clock = make_bot(slope=-0.01)
    feed(bot)
    clock.current_time = 1_000_000 + 80 * 60
    order = Order(price=110.0, order_type="buy")

    placed = bot.execute_pyramid_entry(order, 1.0, {}, 0.5)

    assert placed is False
    assert bot.connector.orders == []
    assert bot.open_positions == []


def test_rising_forecast_allows_long_entry():
    bot, clock = make_bot(slope=0.01)
    feed(bot)
    clock.current_time = 1_000_000 + 80 * 60
    order = Order(price=110.0, order_type="buy")

    placed = bot.execute_pyramid_entry(order, 1.0, {}, 0.5)

    assert placed is True
    assert len(bot.connector.orders) == 1
    assert bot.connector.orders[0]["side"] == "buy"
    assert len(bot.open_positions) == 1


def test_entry_before_enough_history_is_not_vetoed():
    bot, clock = make_bot(slope=-0.01, min_context=64)
    feed(bot, count=10)
    clock.current_time = 1_000_000 + 10 * 60
    order = Order(price=110.0, order_type="buy")

    assert bot.execute_pyramid_entry(order, 1.0, {}, 0.5) is True


def test_unavailable_model_fails_closed_for_entries():
    class Broken:
        def forecast(self, closes, horizon):
            raise RuntimeError("not loaded")

    bot, clock = make_bot(slope=0.01)
    bot.forecast_gate.forecaster = Broken()
    feed(bot)
    clock.current_time = 1_000_000 + 80 * 60

    assert bot.execute_pyramid_entry(Order(110.0, "buy"), 1.0, {}, 0.5) is False
