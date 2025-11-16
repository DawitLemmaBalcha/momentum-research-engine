# analysis/data_preparer.py

import pandas as pd
import pandas_ta as ta
import logging
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from analysis.optimize import ConfigContainer

logger = logging.getLogger(__name__)

def prepare_static_data(df_1m_full: pd.DataFrame, config_module) -> pd.DataFrame:
    """
    Prepares all "static" data that does NOT depend on an optimized parameter.
    This includes resampling to 30m, all 30m indicators, and any non-optimized
    1m indicators. This function is run ONCE before the optimization loop.
    """
    logger.info("Preparing static data for optimization...")
    df_1m_full = df_1m_full.reset_index(drop=True)

    # --- 1. Resample 30m data ---
    df_30m = df_1m_full.set_index('timestamp').resample('30min').agg(
        {'open': 'first', 'high': 'max', 'low': 'min', 'close': 'last', 'volume': 'sum'}
    ).reset_index().dropna()

    # --- 2. Calculate all 30m (static) indicators ---
    df_30m['short_ema'] = ta.ema(df_30m['close'], length=config_module.SHORT_EMA_PERIOD)
    df_30m['long_ema'] = ta.ema(df_30m['close'], length=config_module.LONG_EMA_PERIOD)
    df_30m['rsi'] = ta.rsi(df_30m['close'], length=config_module.RSI_PERIOD)
    df_30m['atr'] = ta.atr(df_30m['high'], df_30m['low'], df_30m['close'], length=config_module.ATR_PERIOD)
    
    macd_30m = ta.macd(df_30m['close'], fast=config_module.MACD_FAST_PERIOD, slow=config_module.MACD_SLOW_PERIOD, signal=config_module.MACD_SIGNAL_PERIOD)
    df_30m['macd'] = macd_30m[f'MACD_{config_module.MACD_FAST_PERIOD}_{config_module.MACD_SLOW_PERIOD}_{config_module.MACD_SIGNAL_PERIOD}']

    bb_period = config_module.BOLLINGER_PERIOD
    bb_std = config_module.BOLLINGER_STD_DEV
    bbands_30m = ta.bbands(df_30m['close'], length=bb_period, std=bb_std)

    # Find bbands columns dynamically (robust fix from backtest_runner)
    bbl_col_name = next((col for col in bbands_30m.columns if col.startswith('BBL_')), None)
    bbu_col_name = next((col for col in bbands_30m.columns if col.startswith('BBU_')), None)
    
    if bbl_col_name and bbu_col_name:
        df_30m['bb_lower'] = bbands_30m[bbl_col_name]
        df_30m['bb_upper'] = bbands_30m[bbu_col_name]
    else:
        raise KeyError(f"Could not find Bollinger Bands columns in {bbands_30m.columns}.")

    # --- 3. Calculate 1m (static) indicators ---
    df_1m = df_1m_full.copy()
    
    # macd_1m is static (not optimized)
    macd_1m = ta.macd(df_1m['close'], fast=config_module.CONFIRMATION_MACD_FAST_PERIOD, slow=config_module.CONFIRMATION_MACD_SLOW_PERIOD, signal=config_module.CONFIRMATION_MACD_SIGNAL_PERIOD)
    df_1m['macd_1m'] = macd_1m[f'MACD_{config_module.CONFIRMATION_MACD_FAST_PERIOD}_{config_module.CONFIRMATION_MACD_SLOW_PERIOD}_{config_module.CONFIRMATION_MACD_SIGNAL_PERIOD}'].shift(1)
    
    # NOTE: We intentionally DO NOT calculate rsi_1m or volume_ma_1m here
    # as they depend on optimized parameters.
    
    # --- 4. Shift 30m indicators and merge ---
    indicator_cols = ['timestamp', 'short_ema', 'long_ema', 'rsi', 'atr', 'macd', 'bb_lower', 'bb_upper']
    df_30m_shifted = df_30m[indicator_cols].copy()
    
    for col in indicator_cols:
        if col != 'timestamp':
            df_30m_shifted[col] = df_30m_shifted[col].shift(1)
            
    df_merged = pd.merge_asof(
        df_1m.sort_values('timestamp'), 
        df_30m_shifted.sort_values('timestamp'), 
        on='timestamp', 
        direction='backward'
    )
    
    logger.info("Static data preparation complete.")
    return df_merged

def add_dynamic_indicators(df_static: pd.DataFrame, trial_config: 'ConfigContainer') -> pd.DataFrame:
    """
    Adds "dynamic" indicators to the DataFrame that depend on
    Optuna-optimized parameters. This function is run INSIDE
    every optimization trial.
    """
    # We operate on a copy to avoid modifying the static cache
    df_trial = df_static.copy() 
    
    # 1. Add rsi_1m (dynamic)
    df_trial['rsi_1m'] = ta.rsi(
        df_trial['close'], 
        length=trial_config.CONFIRMATION_RSI_PERIOD
    ).shift(1)
    
    # 2. Add volume_ma_1m (dynamic)
    df_trial['volume_ma_1m'] = df_trial['volume'].rolling(
        window=trial_config.CONFIRMATION_VOLUME_MA_PERIOD
    ).mean().shift(1)
    
    # 3. Drop NaNs created by the new indicators
    return df_trial.dropna()