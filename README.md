# Momentum-Confirmed Volatility-Adjusted Engine

A Python engine for backtesting, optimising and deploying a trend-anchored,
momentum-confirmed strategy on BTC/USDT perpetuals. It was built around three
ideas: enter on pullbacks into an adaptive anchor grid, require short-timeframe
confirmation before filling an anchor, and exit with volatility-scaled stops.

The repository contains the strategy core, a vectorised-enough event backtester,
a walk-forward optimiser, an ensemble runner, and a live deployment loop. It is
research code, not a turnkey trading product.

## How the strategy works

- **Asymmetric anchors.** A grid of buy and sell levels is placed around the
  current price. Step size follows Bollinger-band width, and the buy/sell split
  follows the detected 30-minute trend regime.
- **Momentum confirmation.** An anchor only fills when 1-minute RSI, MACD and
  relative volume agree with the anchor direction.
- **Volatility-scaled exits.** An initial ATR stop, a trailing stop activated by
  ATR, and a maximum holding time.
- **Position sizing.** Kelly fraction capped by a maximum position percentage,
  with a pyramid decay for later entries in the same anchor set.
- **Drawdown de-risking.** Sizing is cut as equity drawdown approaches the
  configured maximum, and restored gradually after a new peak.
- **Walk-forward optimisation.** Optuna studies over rolling windows, with
  PostgreSQL storage so trials can run concurrently.

## Layout

| Path | Purpose |
|---|---|
| `config.py` | Default strategy and risk parameters |
| `strategies/cluster_*.py` | Parameter sets for ensemble members |
| `core/trading_bot_base.py` | Shared anchor, exit, sizing and risk logic |
| `core/trading_bot.py` | Standalone bot (model A) |
| `core/trading_bot_shared.py` | Portfolio-shared bot (model C) |
| `simulation/backtest_runner.py` | Single-bot event backtest |
| `simulation/backtest_runner_shared.py` | Multi-bot ensemble backtest |
| `simulation/data_fetcher.py` | Bybit OHLCV download and indicator preparation |
| `analysis/` | Walk-forward and ensemble optimisation |
| `deployment/` | Live ensemble loop |
| `forecasting/` | Optional TimesFM entry gate |
| `tests/` | Tests for the forecast gate and its integration |

## Quick start: backtest

Python 3.10-3.12 is required. The pinned requirements were resolved on 3.11.

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env        # add BYBIT_API_KEY and BYBIT_API_SECRET
python main.py
```

`main.py` downloads the configured date range, runs the event backtest and
writes a log to `logs/`. The symbol, timeframe and date range live in
`config.py`. Read-only market data does not require trading permissions; a
public Bybit connection is enough for many endpoints, but the connector expects
credentials to be present.

## Configuration

`config.py` holds the defaults. Each file under `strategies/` is a complete
parameter set for one ensemble member; `deployment/ensemble_config.py` decides
which ones run and how capital is allocated.

The parameters that matter most:

| Parameter | Meaning |
|---|---|
| `NUM_ANCHORS`, `ANCHOR_BAND_MULTIPLIER` | Grid density and spacing |
| `RSI_BUY_CONFIRMATION`, `RSI_SELL_CONFIRMATION` | Confirmation thresholds |
| `VOLUME_CONFIRMATION_FACTOR` | Relative-volume requirement |
| `ATR_INITIAL_STOP_MULTIPLIER`, `ATR_TRAILING_STOP_MULTIPLIER` | Stop distances |
| `MAX_DRAWDOWN_PERCENT`, `DE_RISK_*` | Drawdown response |
| `MAX_POSITION_SIZE_PERCENT`, `KELLY_FRACTION` | Sizing ceilings |

## Walk-forward optimisation

```bash
python -m analysis.walk_forward_ensemble_optimizer
```

The optimiser writes results under `results/` and logs under `logs/`. It expects
a PostgreSQL instance for Optuna storage; the connection string is defined in
the optimiser. Point it at your own database before running.

## Live deployment

`deployment/ensemble_main.py` runs the shared-portfolio loop against Bybit. This
is the part to treat with the most care: it places real orders, and the risk
controls are only as good as the parameters and the exchange connection. Run it
on a demo account first and keep `BYBIT_DEMO` enabled until the behaviour is
understood.

## Optional TimesFM entry gate

The engine can consult Google Research's [TimesFM](https://github.com/google-research/timesfm)
before filling an anchor: if the model's short-horizon path does not support the
anchor direction, the entry is vetoed. The feature is off by default.

**Why TimesFM 2.5.** The adapter targets `google/timesfm-2.5-200m-pytorch`.
TimesFM 2.5 weights are Apache-2.0; TimesFM 3.0 weights are distributed under a
non-commercial licence, so they are not the default here.

**Install.**

```bash
pip install -r requirements-timesfm.txt   # timesfm[torch], Python 3.10-3.12
```

The first forecast downloads the checkpoint from Hugging Face.

**Enable.**

```python
TIMESFM_ENABLED = True
TIMESFM_HORIZON = 5              # 1-minute bars ahead
TIMESFM_MIN_AGREEMENT = 0.6      # share of steps that must agree
TIMESFM_MIN_CONTEXT = 64
TIMESFM_CACHE_BARS = 5
TIMESFM_FAIL_OPEN = False        # veto entries when the model is unavailable
```

**Behaviour.**

- The gate sees only the close history up to the decision bar; the backtest
  runners call `bot.observe_close(timestamp, close)` once per bar before the
  tick loop, and the live loop does the same. There is no look-ahead.
- A forecast is cached for `TIMESFM_CACHE_BARS` bars, and invalidated as soon as
  the latest close changes.
- If the model is missing or fails, entries are vetoed by default. Set
  `TIMESFM_FAIL_OPEN = True` to trade without the gate instead.
- The gate never changes position size or exits; it only accepts or rejects an
  entry.

The unit tests use a deterministic stub backend, so they run without PyTorch or
model weights. Loading the real checkpoint is not covered by CI.

## Tests

```bash
pip install numpy pandas pytest python-dotenv
python -m pytest tests -q
```

The suite covers gate agreement, caching, fail-open/fail-closed behaviour,
history ordering, and the entry hook in the bot.

## Limitations

- The backtester models fees and slippage approximately; results are sensitive
  to the data range and to the parameters chosen by the optimiser.
- Walk-forward results are not a guarantee of future performance.
- The live loop depends on exchange connectivity and on correct API permissions.
- The ensemble allocation logic is tuned for the included strategy sets, not a
  general portfolio optimiser.

## License

MIT. See `LICENSE`.
