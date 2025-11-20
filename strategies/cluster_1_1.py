# config.py

import os
from dotenv import load_dotenv

load_dotenv()

# -- API Credentials --
API_KEY = os.getenv('BYBIT_API_KEY')
API_SECRET = os.getenv('BYBIT_API_SECRET')

# -- Data Fetching Parameters --
SYMBOL = 'BTC/USDT'
TIMEFRAME = '30m'
START_DATE = '2025-08-01 00:00:00'
END_DATE = '2025-09-01 00:00:00'

# -- Bot Core Parameters --
INITIAL_CAPITAL = 100000.0
MAX_POSITIONS = 10


BYBIT_TAKER_FEE = 0.00055
ANCHOR_SETUP_COOLDOWN_SECONDS = 1800 # 30 minutes

#4 hours 14400
# -- Optimizable Parameters -----------------
ATR_INITIAL_STOP_MULTIPLIER = 0.5987695431534483
ATR_TRAILING_STOP_ACTIVATION_MULTIPLIER = 0.6501006568428815
TRAILING_RATIO = 0.5060598554306665
ANCHOR_BAND_MULTIPLIER = 0.10688209044215045
CONFIRMATION_RSI_PERIOD = 30
CONFIRMATION_VOLUME_MA_PERIOD = 31
ATR_TRAILING_STOP_MULTIPLIER = 0.37482594631928206
#----------------------------------------------

PYRAMID_SCALING_FACTOR = 0.5

NUM_ANCHORS = 6
ANCHOR_SIGNAL_WINDOW_SECONDS = 14400  # 3 hours
BOLLINGER_PERIOD = 20
BOLLINGER_STD_DEV = 2.0

RSI_BUY_CONFIRMATION = 45
RSI_SELL_CONFIRMATION = 35
VOLUME_CONFIRMATION_FACTOR = 0.5


# Ratio for a standard, EMA-detected trend
TREND_PYRAMID_RATIO_NORMAL = 0.6  # 60% of anchors in trend direction

# Ratio for a strong, multi-indicator confirmed trend
TREND_PYRAMID_RATIO_STRONG = 0.8  # 80% of anchors in trend direction


CONFIRMATION_MACD_FAST_PERIOD = 12
CONFIRMATION_MACD_SLOW_PERIOD = 26
CONFIRMATION_MACD_SIGNAL_PERIOD = 9



MAX_POSITION_DURATION_SECONDS = 43200


# Min/Max % step size for anchor signals
ANCHOR_VOLATILITY_SCALING_MIN = 0.08 # Note: This param is not used in the base file
ANCHOR_VOLATILITY_SCALING_MAX = 2.00 # Note: This param is not used in the base file
ANCHOR_STEP_MIN_PERCENT = 0.0007
ANCHOR_STEP_MAX_PERCENT = 0.03

# -- Risk Management & Exits --
TAKE_PROFIT_PERCENT = 1000.0  # Disabled to rely on trailing stops

ATR_PERIOD = 14
  

# -- Position Sizing Parameters --
MAX_POSITION_SIZE_PERCENT = 0.5
KELLY_FRACTION = 1
KELLY_LOOKBACK = 20
KELLY_MIN_FRACTION = 0.6

# -- Technical Indicator Parameters --
RSI_PERIOD = 14
SHORT_EMA_PERIOD = 10
LONG_EMA_PERIOD = 20
MACD_FAST_PERIOD = 12
MACD_SLOW_PERIOD = 26
MACD_SIGNAL_PERIOD = 9
VOLUME_MA_PERIOD = 20

# -- Trend Detection Parameters (for 30m) --
TREND_EMA_THRESHOLD_PERCENT = 0.2 # How far apart EMAs need to be to declare a trend
BB_REGIME_THRESHOLD = 0.015       # BB width threshold to define "trending" vs "ranging"
RSI_OVERSOLD = 30
RSI_OVERBOUGHT = 70

# -- Backtest Priming Lookback --
REQUIRED_LOOKBACK = max(MACD_SLOW_PERIOD + MACD_SIGNAL_PERIOD, VOLUME_MA_PERIOD, ATR_PERIOD)




MAX_TRADES = 1000
MAX_PYRAMID_ENTRIES = NUM_ANCHORS # Max number of simultaneous entries per signal


MAX_DRAWDOWN_PERCENT = 7.0
# -- Dynamic Risk Parameters --

# How aggressively to cut size as drawdown increases.
# 1.0 = linear (5% DD in a 10% max = 50% cut)
# 0.5 = half aggression (5% DD in a 10% max = 25% cut)
DE_RISK_AGGRESSION = 1

# The maximum % to cut sizing, no matter how bad the drawdown.
# 0.8 = "At worst, cut sizing by 80% (to 20% of original)"
DE_RISK_MAX_CUT = 0.8

# The minimum sizing multiplier, to prevent sizing from going to zero.
# 0.1 = "Never let position size fall below 10% of its original setting"
DE_RISK_LOWER_BOUND = 0.1

# How fast to restore sizing when the bot is performing well (at a new peak).
# 0.05 = Restore sizing by 5% of its current level per 30m candle.
RE_RISK_AGGRESSION = 0.5