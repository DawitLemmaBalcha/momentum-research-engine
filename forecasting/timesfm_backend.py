# forecasting/timesfm_backend.py

"""Thin adapter around Google Research's TimesFM.

The engine only needs a point path for a horizon it already uses for exits.
TimesFM 2.5 is the default because its pretrained weights are Apache-2.0;
TimesFM 3.0 weights are non-commercial, so they are not the default here.

Importing this module does not import ``torch`` or ``timesfm``. Model loading
happens on the first forecast so a backtest that never enables the gate pays
nothing.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from typing import Protocol

import numpy as np

from forecasting.types import Forecast


class TimesFMUnavailable(RuntimeError):
    """Raised when TimesFM cannot be imported, loaded, or run."""


class ForecastBackend(Protocol):
    """Low-level backend used by :class:`TimesFMForecaster`."""

    def forecast(self, closes: np.ndarray, horizon: int) -> tuple[np.ndarray, np.ndarray | None]:
        """Return ``(point_path, quantiles)`` for the next ``horizon`` steps."""


class Forecaster(Protocol):
    """What the gate consumes: something that returns a :class:`Forecast`."""

    def forecast(self, closes: Sequence[float], horizon: int) -> Forecast:
        """Return a forecast over the next ``horizon`` steps."""


ModelRunner = Callable[[np.ndarray, int], tuple[np.ndarray, np.ndarray | None]]


class TimesFMForecaster:
    """TimesFM-backed forecaster with lazy model loading.

    Parameters
    ----------
    checkpoint:
        Hugging Face checkpoint. The 2.5 200M PyTorch checkpoint is the
        default because its weights are Apache-2.0.
    max_context:
        Number of trailing observations passed to the model.
    max_horizon:
        Upper bound compiled into the model.
    backend:
        Optional object implementing :class:`ForecastBackend`. Intended for
        tests and for swapping in another model without touching the gate.
    """

    MIN_CONTEXT = 32
    CHECKPOINT = "google/timesfm-2.5-200m-pytorch"

    def __init__(
        self,
        checkpoint: str = CHECKPOINT,
        max_context: int = 512,
        max_horizon: int = 64,
        backend: ForecastBackend | None = None,
    ) -> None:
        if max_context < self.MIN_CONTEXT:
            raise ValueError(f"max_context must be at least {self.MIN_CONTEXT}")
        if max_horizon < 1:
            raise ValueError("max_horizon must be positive")
        self.checkpoint = checkpoint
        self.max_context = int(max_context)
        self.max_horizon = int(max_horizon)
        self._backend = backend
        self._runner: ModelRunner | None = None

    def forecast(self, closes: Sequence[float], horizon: int) -> Forecast:
        if horizon < 1:
            raise ValueError("horizon must be positive")
        if horizon > self.max_horizon:
            raise ValueError(f"horizon {horizon} exceeds max_horizon {self.max_horizon}")

        series = np.asarray(list(closes), dtype=np.float64)
        if series.ndim != 1 or series.size < self.MIN_CONTEXT:
            raise ValueError(f"need at least {self.MIN_CONTEXT} observations, got {series.size}")
        if not np.isfinite(series).all():
            raise ValueError("series contains non-finite values")
        series = series[-self.max_context :]

        if self._backend is not None:
            point, quantiles = self._backend.forecast(series, horizon)
        else:
            point, quantiles = self._run_model(series, horizon)

        point = np.asarray(point, dtype=np.float64).reshape(-1)
        if point.size < horizon:
            raise TimesFMUnavailable(f"model returned {point.size} steps for horizon {horizon}")
        point = point[:horizon]
        if not np.isfinite(point).all():
            raise TimesFMUnavailable("model returned non-finite forecasts")

        quantile_array: np.ndarray | None = None
        if quantiles is not None:
            quantile_array = np.asarray(quantiles, dtype=np.float64)
            if quantile_array.ndim == 3:
                quantile_array = quantile_array[0]
            if quantile_array.ndim != 2 or quantile_array.shape[0] < horizon:
                raise TimesFMUnavailable("model returned an unexpected quantile shape")
            quantile_array = quantile_array[:horizon]

        return Forecast(
            values=point,
            quantiles=quantile_array,
            horizon=horizon,
            checkpoint=self.checkpoint,
            context_length=int(series.size),
            last_value=float(series[-1]),
        )

    def _run_model(self, series: np.ndarray, horizon: int) -> tuple[np.ndarray, np.ndarray | None]:
        if self._runner is None:
            self._runner = self._load_timesfm()
        try:
            return self._runner(series, horizon)
        except TimesFMUnavailable:
            raise
        except Exception as exc:  # pragma: no cover - depends on installed model
            raise TimesFMUnavailable(f"TimesFM inference failed: {exc}") from exc

    def _load_timesfm(self) -> ModelRunner:
        try:
            import timesfm
        except ImportError as exc:
            raise TimesFMUnavailable(
                "timesfm is not installed. Install the optional extra with "
                "`pip install -r requirements-timesfm.txt`."
            ) from exc

        try:
            model = timesfm.TimesFM_2p5_200M_torch.from_pretrained(self.checkpoint)
            model.compile(
                timesfm.ForecastConfig(
                    max_context=self.max_context,
                    max_horizon=self.max_horizon,
                    normalize_inputs=True,
                    use_continuous_quantile_head=True,
                    force_flip_invariance=True,
                    infer_is_positive=True,
                    fix_quantile_crossing=True,
                )
            )
        except Exception as exc:
            raise TimesFMUnavailable(
                f"could not load TimesFM checkpoint {self.checkpoint!r}: {exc}"
            ) from exc

        def run(closes: np.ndarray, horizon: int) -> tuple[np.ndarray, np.ndarray | None]:
            point, quantiles = model.forecast(
                horizon=horizon,
                inputs=[np.asarray(closes, dtype=np.float32)],
            )
            point = np.asarray(point, dtype=np.float64)
            if point.ndim == 2:
                point = point[0]
            quantiles = np.asarray(quantiles, dtype=np.float64)
            if quantiles.ndim == 3:
                quantiles = quantiles[0]
            return point, quantiles

        return run
