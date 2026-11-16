# forecasting/types.py

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class Forecast:
    """A point forecast path plus optional quantiles.

    ``values`` has shape ``(horizon,)`` and is expressed in the same units as
    the input series (price, not returns). ``quantiles``, when present, has
    shape ``(horizon, n_quantiles)`` ordered from the lowest to the highest
    quantile.
    """

    values: np.ndarray
    quantiles: np.ndarray | None
    horizon: int
    checkpoint: str
    context_length: int
    last_value: float

    @property
    def expected_return_pct(self) -> float:
        """Return from the last observed value to the end of the forecast path."""
        if self.last_value <= 0 or self.values.size == 0:
            return 0.0
        return (float(self.values[-1]) / float(self.last_value) - 1.0) * 100.0

    def agreement(self, side: str) -> float:
        """Share of forecast steps that move in ``side``'s direction.

        ``side`` is ``"long"``/``"buy"`` or ``"short"``/``"sell"``.
        """
        normalized = normalize_side(side)
        if self.values.size == 0:
            return 0.0
        if normalized == "long":
            moves = self.values > self.last_value
        else:
            moves = self.values < self.last_value
        return float(np.mean(moves))


def normalize_side(side: str) -> str:
    value = str(side).strip().lower()
    if value in {"long", "buy", "b"}:
        return "long"
    if value in {"short", "sell", "s"}:
        return "short"
    raise ValueError(f"side must be long/buy or short/sell, got {side!r}")
