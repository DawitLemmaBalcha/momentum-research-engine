# simulation/backtest_runner.py

import pandas as pd
import numpy as np
import pandas_ta as ta
import logging
# import random <-- 1. REMOVED (Original file)
from abc import ABC, abstractmethod
from typing import Dict, Any, Callable, List, Optional
from core.trading_bot import MomentumAnchorBot # <-- This is the original bot
from core.core_types import SimulationClock, Tick, Position
import optuna 

# --- NEW: Importing the standardized DummyConnector ---
from connectors.dummy_connector import DummyConnector

# --- REMOVED: The private, internal DummyConnector class was removed from here ---


class SimulationEngine:
    """
    Handles the time-synchronized, concurrent simulation of one or more trading bots.
    (This is the Model A Engine)
    """
    def __init__(self, data: pd.DataFrame, bots: List[MomentumAnchorBot], logger=None, prepared: bool = False):
        self.data = data 
        self.is_prepared = prepared
        
        self.bots = bots
        self.logger = logger or logging.getLogger(__name__)
        self.portfolio_equity_curve = [] # --- FIX: Will be initialized in run() ---
        self.initial_portfolio_capital = sum(bot.initial_capital for bot in self.bots)

    def _prepare_data(self):
        # ... (no changes in this function) ...
        if self.is_prepared:
            return self.data 
            
        df_1m_full = self.data.reset_index(drop=True)

        bot_config = self.bots[0].config
        
        df_30m = df_1m_full.set_index('timestamp').resample('30min').agg({'open': 'first', 'high': 'max', 'low': 'min', 'close': 'last', 'volume': 'sum'}).reset_index().dropna()
        df_30m['short_ema'] = ta.ema(df_30m['close'], length=bot_config.SHORT_EMA_PERIOD)
        df_30m['long_ema'] = ta.ema(df_30m['close'], length=bot_config.LONG_EMA_PERIOD)
        df_30m['rsi'] = ta.rsi(df_30m['close'], length=bot_config.RSI_PERIOD)
        df_30m['atr'] = ta.atr(df_30m['high'], df_30m['low'], df_30m['close'], length=bot_config.ATR_PERIOD)
        macd_30m = ta.macd(df_30m['close'], fast=bot_config.MACD_FAST_PERIOD, slow=bot_config.MACD_SLOW_PERIOD, signal=bot_config.MACD_SIGNAL_PERIOD)
        df_30m['macd'] = macd_30m[f'MACD_{bot_config.MACD_FAST_PERIOD}_{bot_config.MACD_SLOW_PERIOD}_{bot_config.MACD_SIGNAL_PERIOD}']

        bb_period = bot_config.BOLLINGER_PERIOD
        bb_std = bot_config.BOLLINGER_STD_DEV
        
        bbands_30m = ta.bbands(df_30m['close'], length=bb_period, std=bb_std)

        bbl_col_name = None
        bbu_col_name = None

        patterns_to_try = [
            f'BBL_{bb_period}_{bb_std}',      
            f'BBL_{bb_period}_{int(bb_std)}', 
            f'BBL_{bb_period}'              
        ]
        
        for pattern in patterns_to_try:
            if pattern in bbands_30m.columns:
                bbl_col_name = pattern
                bbu_col_name = pattern.replace('BBL', 'BBU') 
                break
        
        if bbl_col_name is None:
            try:
                bbl_col_name = [col for col in bbands_30m.columns if col.startswith('BBL_')][0]
                bbu_suffix = bbl_col_name[4:] 
                bbu_col_name = f'BBU{bbu_suffix}'
                if bbu_col_name not in bbands_30m.columns:
                    bbu_col_name = [col for col in bbands_30m.columns if col.startswith('BBU_')][0]
            except IndexError:
                raise KeyError(f"Could not find Bollinger Bands columns in {bbands_30m.columns}.")

        df_30m['bb_lower'] = bbands_30m[bbl_col_name]
        df_30m['bb_upper'] = bbands_30m[bbu_col_name]

        df_1m = df_1m_full.copy()
        df_1m['rsi_1m'] = ta.rsi(df_1m['close'], length=bot_config.CONFIRMATION_RSI_PERIOD).shift(1)
        df_1m['volume_ma_1m'] = df_1m['volume'].rolling(window=bot_config.CONFIRMATION_VOLUME_MA_PERIOD).mean().shift(1)
        macd_1m = ta.macd(df_1m['close'], fast=bot_config.CONFIRMATION_MACD_FAST_PERIOD, slow=bot_config.CONFIRMATION_MACD_SLOW_PERIOD, signal=bot_config.CONFIRMATION_MACD_SIGNAL_PERIOD)
        df_1m['macd_1m'] = macd_1m[f'MACD_{bot_config.CONFIRMATION_MACD_FAST_PERIOD}_{bot_config.CONFIRMATION_MACD_SLOW_PERIOD}_{bot_config.CONFIRMATION_MACD_SIGNAL_PERIOD}'].shift(1)
        
        indicator_cols = ['timestamp', 'short_ema', 'long_ema', 'rsi', 'atr', 'macd', 'bb_lower', 'bb_upper']
        for col in indicator_cols:
            if col != 'timestamp':
                df_30m[col] = df_30m[col].shift(1)
                
        df_merged = pd.merge_asof(df_1m.sort_values('timestamp'), df_30m[indicator_cols].sort_values('timestamp'), on='timestamp', direction='backward')
        
        return df_merged.dropna()

    def run(self, trial: Optional[optuna.trial.Trial] = None):
        
        try:
            df_merged = self._prepare_data()
        except KeyError as e:
            self.logger.error(f"Failed to prepare data due to missing column: {e}")
            return None, []
        
        if len(df_merged) < 100:
            self.logger.warning("Backtest failed: Insufficient data after indicator calculation.")
            return None, []

        sim_start_time = df_merged['timestamp'].iloc[0].timestamp()
        clock = SimulationClock(start_time=sim_start_time)
        
        # --- *** FIX: Store (timestamp, capital) tuples *** ---
        # Initialize the equity curve with the starting time and capital
        self.portfolio_equity_curve = [(sim_start_time, self.initial_portfolio_capital)]
        # --- *** END FIX *** ---

        dummy_connector = DummyConnector(clock, {})

        first_row = df_merged.iloc[0]
        # Set clock to first row *before* bot init
        clock.current_time = first_row['timestamp'].timestamp()
        clock.current_price = first_row['close']
        
        for bot in self.bots:
            bot.clock = clock
            bot.connector = dummy_connector
            bot.initialize_state(first_row.to_dict())
            
            fee = getattr(bot.config, 'BYBIT_TAKER_FEE', 0.00055)
            bot.connector.params['BYBIT_TAKER_FEE'] = fee


        # --- HIGH-SPEED CONCURRENT SIMULATION LOOP ---
        for i in range(1, len(df_merged)):
            current_row, prev_row = df_merged.iloc[i], df_merged.iloc[i-1]
            
            # --- Equity/Peak Update (using prev_row to prevent lookahead) ---
            prev_candle_close_price = prev_row['close']
            current_portfolio_equity = 0.0
            for bot in self.bots:
                bot.update_equity_on_1m(prev_candle_close_price)
                current_portfolio_equity += bot.capital + bot.unrealized_pnl
            
            
            # Update 30-minute strategies (now uses fresh state)
            if current_row['timestamp'].floor('30min') > prev_row['timestamp'].floor('30min'):
                for bot in self.bots:
                    bot.update_strategy_on_30m(prev_row.to_dict())
            
            o, h, l, c, v, base_ts = current_row['open'], current_row['high'], current_row['low'], current_row['close'], current_row['volume'], current_row['timestamp'].timestamp()
            
            if c > o:
                key_prices = [o, l, h, c]
            else:
                key_prices = [o, h, l, c]
            
            
            rsi_1m_val = current_row['rsi_1m']
            macd_1m_val = current_row['macd_1m']
            volume_ma_1m_val = current_row['volume_ma_1m']
            atr_val = current_row['atr'] 
            volume_1m_val = prev_row['volume'] 


            for price in key_prices:
                tick = Tick(timestamp=base_ts, price=price, volume=0, candle_volume=v) 
                clock.current_time, clock.current_price = tick.timestamp, price
                
                for bot in self.bots:
                    bot.check_confirmation_on_tick(tick, rsi_1m_val, macd_1m_val, volume_1m_val, volume_ma_1m_val, atr_val)
                    bot.check_exits_on_1m(price, atr_val)

            
            # --- *** FIX: Store (timestamp, capital) tuple *** ---
            # Append the 1-minute candle's timestamp and the calculated equity
            self.portfolio_equity_curve.append((current_row['timestamp'].timestamp(), current_portfolio_equity))
            # --- *** END FIX *** ---
            
            
            # --- Pruning Check ---
            if trial and i > 30000 and i % 1000 == 0:
                trial.report(current_portfolio_equity, i) # Report equity
                if trial.should_prune():
                    raise optuna.TrialPruned()


        # --- Final Performance Calculation ---
        portfolio_results = self._calculate_portfolio_metrics()
        
        if trial:
            trial.report(portfolio_results['final_capital'], len(df_merged))
        
        if self.bots:
            # Assumes 1 bot, or pulls from the first bot for individual stats
            realized_equity_curve_tuples = self.bots[0].realized_equity_curve
            
            # --- *** START BUG FIX *** ---
            if len(realized_equity_curve_tuples) > 1:
                # Create a Series for the realized (on-close) equity curve
                realized_ts_raw = pd.Series(
                    [val for ts, val in realized_equity_curve_tuples],
                    index=pd.to_datetime([ts for ts, val in realized_equity_curve_tuples], unit='s')
                )
                
                # Handle duplicate index labels by taking the *last*
                # recorded capital for any given timestamp.
                if not realized_ts_raw.index.is_unique:
                    realized_equity_series = realized_ts_raw.groupby(realized_ts_raw.index).last()
                else:
                    realized_equity_series = realized_ts_raw
                
                realized_peak_series = realized_equity_series.expanding().max()
                realized_drawdown_series = (realized_equity_series - realized_peak_series) / realized_peak_series
                portfolio_results['realized_max_drawdown'] = abs(realized_drawdown_series.min()) * 100
            else:
                portfolio_results['realized_max_drawdown'] = 0.0
            # --- *** END BUG FIX *** ---
        else:
            portfolio_results['realized_max_drawdown'] = 0.0

        individual_results = [bot.log_performance(print_log=False) for bot in self.bots]
        
        return portfolio_results, individual_results

    def _calculate_portfolio_metrics(self):
        # ... (no changes in this function except processing tuples) ...
        
        # --- *** FIX: Correctly process list of tuples *** ---
        if not self.portfolio_equity_curve:
            final_equity = self.initial_portfolio_capital
            max_drawdown = 0.0
        else:
            equity_series = pd.Series(
                [val for ts, val in self.portfolio_equity_curve],
                index=pd.to_datetime([ts for ts, val in self.portfolio_equity_curve], unit='s')
            )
            peak = equity_series.expanding().max()
            drawdown = (equity_series - peak) / peak
            max_drawdown = abs(drawdown.min()) * 100
            final_equity = equity_series.iloc[-1]
        # --- *** END FIX *** ---
        
        total_pnl = final_equity - self.initial_portfolio_capital
        total_return = (total_pnl / self.initial_portfolio_capital) * 100 if self.initial_portfolio_capital > 0 else 0
        
        return {
            "initial_capital": self.initial_portfolio_capital,
            "final_capital": final_equity,
            "pnl_cash": total_pnl,
            "total_return_pct": total_return,
            "max_drawdown": max_drawdown,
        }

# --- Wrapper functions (prepare_data, run_from_prepared, run_single) ---
# ... (no changes needed in these wrapper functions) ...
def prepare_data_for_simulation(df_1m_full: pd.DataFrame, config_module) -> pd.DataFrame:
    dummy_bot = MomentumAnchorBot(
        initial_capital=0,
        simulation_clock=SimulationClock(0),
        config_module=config_module,
        connector=None,
        silent=True
    )
    temp_engine = SimulationEngine(df_1m_full, [dummy_bot], prepared=False)
    try:
        df_prepared = temp_engine._prepare_data()
        return df_prepared
    except Exception as e:
        logging.getLogger(__name__).error(f"Failed during data preparation: {e}", exc_info=True)
        return pd.DataFrame() 

def run_simulation_from_prepared_data(
    df_prepared: pd.DataFrame, 
    config_module, 
    verbose: bool = False, 
    logger=None, 
    trial: Optional[optuna.trial.Trial] = None
):
    bot_config = config_module
    fee = getattr(bot_config, 'BYBIT_TAKER_FEE', 0.00055)
    dummy_connector = DummyConnector(SimulationClock(0), { "BYBIT_TAKER_FEE": fee })
    bot = MomentumAnchorBot(
        initial_capital=bot_config.INITIAL_CAPITAL,
        simulation_clock=SimulationClock(0), 
        config_module=bot_config,
        connector=dummy_connector,
        silent=not verbose
    )
    engine = SimulationEngine(df_prepared, [bot], logger, prepared=True)
    portfolio_results, individual_results = engine.run(trial=trial)
    final_metrics = individual_results[0] if individual_results else {}
    
    # --- MERGE CORRECT DRAWDOWN FROM PORTFOLIO RESULTS ---
    if final_metrics and portfolio_results:
        final_metrics['max_drawdown'] = portfolio_results.get('max_drawdown', 0.0)
        final_metrics['realized_max_drawdown'] = portfolio_results.get('realized_max_drawdown', 0.0)
        # We must also re-generate the log string so it contains the correct values
        final_metrics['performance_log_str'] = bot._get_performance_log_str(final_metrics)
    # --- END MERGE ---

    # --- [NEW] ATTACH EQUITY CURVE FOR OPTIMIZER EFFICIENCY ---
    if engine.portfolio_equity_curve:
        final_metrics['equity_curve'] = engine.portfolio_equity_curve
    # --- END NEW ---

    if verbose and logger and final_metrics:
        logger.info(final_metrics.get("performance_log_str", "Performance log not available."))
    elif not final_metrics and portfolio_results:
        return portfolio_results
    return final_metrics

def run_single_backtest(
    df_1m_full: pd.DataFrame, 
    config_module, 
    verbose: bool = False, 
    logger=None, 
    trial: Optional[optuna.trial.Trial] = None 
):
    bot_config = config_module
    fee = getattr(bot_config, 'BYBIT_TAKER_FEE', 0.00055)
    dummy_connector = DummyConnector(SimulationClock(0), { "BYBIT_TAKER_FEE": fee })
    bot = MomentumAnchorBot(
        initial_capital=bot_config.INITIAL_CAPITAL,
        simulation_clock=SimulationClock(0), 
        config_module=bot_config,
        connector=dummy_connector,
        silent=not verbose
    )
    engine = SimulationEngine(df_1m_full, [bot], logger, prepared=False)
    portfolio_results, individual_results = engine.run(trial=trial)
    final_metrics = individual_results[0] if individual_results else {}

    # --- ADDED: Merge correct drawdown from portfolio results into final metrics ---
    if final_metrics and portfolio_results:
        final_metrics['max_drawdown'] = portfolio_results.get('max_drawdown', 0.0)
        final_metrics['realized_max_drawdown'] = portfolio_results.get('realized_max_drawdown', 0.0)
        # We must also re-generate the log string so it contains the correct values
        final_metrics['performance_log_str'] = bot._get_performance_log_str(final_metrics)
    # --- END ADDED ---
    
    if verbose and logger and final_metrics:
        logger.info(final_metrics.get("performance_log_str", "Performance log not available."))
    elif not final_metrics and portfolio_results:
        return portfolio_results
    return final_metrics