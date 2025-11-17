# simulation/backtest_runner_shared.py

import pandas as pd
import numpy as np
import pandas_ta as ta
import logging
from abc import ABC, abstractmethod
from typing import Dict, Any, Callable, List, Optional
# --- *** IMPORT THE NEW SHARED BOT *** ---
from core.trading_bot_shared import MomentumAnchorBotShared
# --- *** END IMPORT *** ---
from core.core_types import SimulationClock, Tick, Position
import optuna 

# --- NEW: Importing the standardized DummyConnector ---
from connectors.dummy_connector import DummyConnector


class SimulationEngine:
    """
    Handles the time-synchronized, concurrent simulation of one or more trading bots
    using a SHARED CAPITAL POOL (Model C).
    """
    def __init__(self, data: pd.DataFrame, bots: List[MomentumAnchorBotShared], logger=None, prepared: bool = False):
        self.data = data 
        self.is_prepared = prepared
        self.bots = bots
        self.logger = logger or logging.getLogger(__name__)
        
        if not self.bots:
            raise ValueError("SimulationEngine requires at least one bot.")
        
        self.initial_per_bot_capital = self.bots[0].initial_capital
        # --- FIX: Calculate portfolio capital from ALL bots' initial capital ---
        self.initial_portfolio_capital = self.bots[0].initial_capital
        # --- END FIX ---
        self.master_peak_capital = self.initial_portfolio_capital # Equity peak
        
        # --- FIX: Store (timestamp, capital) tuples ---
        self.portfolio_equity_curve = [] # Will be initialized in run()
        self.realized_portfolio_equity_curve = [] # Will be initialized in run()
        # --- END FIX ---
        
        # --- *** FIX 1 of 4: Add a master variable for realized capital *** ---
        # This is the single source of truth for the portfolio's REALIZED capital.
        # It is only updated by the P&L from the DummyConnector.
        self.master_realized_capital = self.initial_portfolio_capital
        # --- *** END FIX *** ---

    def _prepare_data(self):
        """
        --- *** MODIFIED FOR "MASTER DATA" *** ---
        Prepares a single, comprehensive DataFrame containing ALL unique
        indicator calculations required by ANY bot in the ensemble.
        """
        
        if self.is_prepared:
            self.logger.info("Data is already prepared. Skipping.")
            return self.data 
            
        df_1m_full = self.data.reset_index(drop=True)
        if df_1m_full.empty:
            self.logger.warning("Empty 1m dataframe provided to _prepare_data.")
            return pd.DataFrame()

        # --- NEW: Collect all unique parameter combinations from all bots ---
        self.logger.info(f"Preparing master data for {len(self.bots)} bots...")
        params_30m = {
            'short_ema': set(), 'long_ema': set(), 'rsi': set(), 'atr': set(),
            'macd': set(), 'bbands': set()
        }
        params_1m = {
            'rsi_1m': set(), 'volume_ma_1m': set(), 'macd_1m': set()
        }

        for bot in self.bots:
            c = bot.config
            params_30m['short_ema'].add(c.SHORT_EMA_PERIOD)
            params_30m['long_ema'].add(c.LONG_EMA_PERIOD)
            params_30m['rsi'].add(c.RSI_PERIOD)
            params_30m['atr'].add(c.ATR_PERIOD)
            params_30m['macd'].add((c.MACD_FAST_PERIOD, c.MACD_SLOW_PERIOD, c.MACD_SIGNAL_PERIOD))
            params_30m['bbands'].add((c.BOLLINGER_PERIOD, c.BOLLINGER_STD_DEV))
            
            params_1m['rsi_1m'].add(c.CONFIRMATION_RSI_PERIOD)
            params_1m['volume_ma_1m'].add(c.CONFIRMATION_VOLUME_MA_PERIOD)
            params_1m['macd_1m'].add((c.CONFIRMATION_MACD_FAST_PERIOD, c.CONFIRMATION_MACD_SLOW_PERIOD, c.CONFIRMATION_MACD_SIGNAL_PERIOD))

        # --- 1. Calculate all unique 30m indicators ---
        df_30m = df_1m_full.set_index('timestamp').resample('30min').agg(
            {'open': 'first', 'high': 'max', 'low': 'min', 'close': 'last', 'volume': 'sum'}
        ).reset_index().dropna()

        for p in params_30m['short_ema']: df_30m[f'short_ema_{p}'] = ta.ema(df_30m['close'], length=p)
        for p in params_30m['long_ema']: df_30m[f'long_ema_{p}'] = ta.ema(df_30m['close'], length=p)
        for p in params_30m['rsi']: df_30m[f'rsi_{p}'] = ta.rsi(df_30m['close'], length=p)
        for p in params_30m['atr']: df_30m[f'atr_{p}'] = ta.atr(df_30m['high'], df_30m['low'], df_30m['close'], length=p)
        
        for f, s, sig in params_30m['macd']:
            macd_df = ta.macd(df_30m['close'], fast=f, slow=s, signal=sig)
            df_30m[f'macd_{f}_{s}_{sig}'] = macd_df[f'MACD_{f}_{s}_{sig}']
            
        for p, std in params_30m['bbands']:
            bbands_df = ta.bbands(df_30m['close'], length=p, std=std)
            bbl_col = next((col for col in bbands_df.columns if col.startswith('BBL_')), None)
            bbu_col = next((col for col in bbands_df.columns if col.startswith('BBU_')), None)
            if bbl_col and bbu_col:
                df_30m[f'bb_lower_{p}_{std}'] = bbands_df[bbl_col]
                df_30m[f'bb_upper_{p}_{std}'] = bbands_df[bbu_col]
            else:
                self.logger.error(f"Could not find BBands columns for {p}_{std}")

        # --- 2. Calculate all unique 1m indicators ---
        df_1m = df_1m_full.copy()
        for p in params_1m['rsi_1m']: df_1m[f'rsi_1m_{p}'] = ta.rsi(df_1m['close'], length=p).shift(1)
        for p in params_1m['volume_ma_1m']: df_1m[f'volume_ma_1m_{p}'] = df_1m['volume'].rolling(window=p).mean().shift(1)
        
        for f, s, sig in params_1m['macd_1m']:
            macd_df = ta.macd(df_1m['close'], fast=f, slow=s, signal=sig)
            df_1m[f'macd_1m_{f}_{s}_{sig}'] = macd_df[f'MACD_{f}_{s}_{sig}'].shift(1)
        
        # --- 3. Shift all 30m indicators and merge ---
        indicator_cols = [col for col in df_30m.columns if col not in ['timestamp', 'open', 'high', 'low', 'close', 'volume']]
        df_30m_shifted = df_30m[['timestamp'] + indicator_cols].copy()
        for col in indicator_cols:
            df_30m_shifted[col] = df_30m_shifted[col].shift(1)
            
        df_merged = pd.merge_asof(
            df_1m.sort_values('timestamp'), 
            df_30m_shifted.sort_values('timestamp'), 
            on='timestamp', 
            direction='backward'
        )
        
        self.logger.info("Master data preparation complete.")
        return df_merged.dropna()

    # --- *** NEW METHOD (Dynamic Allocation Fix) *** ---
    def _update_dynamic_allocations(self):
        """
        Re-calculates and injects new capital_allocation weights for each bot
        based on their current sizing_multiplier (performance).
        """
        
        # 1. Get all current performance weights (sizing_multiplier)
        weights = [bot.sizing_multiplier for bot in self.bots]
        total_weight = sum(weights)
        
        if total_weight <= 0:
            # Emergency case: all bots are de-risked to zero.
            # Fall back to equal allocation to prevent division by zero.
            num_bots = len(self.bots)
            if num_bots > 0:
                equal_allocation = 1.0 / num_bots
                for bot in self.bots:
                    bot.capital_allocation = equal_allocation
            return

        # 2. Calculate and inject new allocations
        for i, bot in enumerate(self.bots):
            new_allocation = weights[i] / total_weight
            bot.capital_allocation = new_allocation
            
        # Optional: Log the new allocation
        if self.logger and self.clock.time() % 1800 == 0: # Log roughly every 30m
            alloc_strs = [f"Bot{i}: {alloc:.2%}" for i, alloc in enumerate([b.capital_allocation for b in self.bots])]
            self.logger.info(f"Dynamic Allocation Update: {', '.join(alloc_strs)}")
    # --- *** END NEW METHOD *** ---

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
        self.clock = clock
        
        # --- *** FIX: Store (timestamp, capital) tuples *** ---
        self.portfolio_equity_curve = [(sim_start_time, self.initial_portfolio_capital)]
        self.realized_portfolio_equity_curve = [(sim_start_time, self.initial_portfolio_capital)]
        # --- *** END FIX *** ---
        
        # --- *** FIX 2 of 4: Pass the engine instance 'self' to the connector *** ---
        # This allows the connector to update the engine's master capital.
        dummy_connector = DummyConnector(clock, {}, engine=self)

        first_row = df_merged.iloc[0]
        clock.current_time = first_row['timestamp'].timestamp()
        clock.current_price = first_row['close']
        

        for bot in self.bots:
            # This is the bot's own initial capital (e.g., $100,000)
            bot.initial_capital = bot.initial_capital # Use the bot's own defined capital
            
            # --- ADD THIS LINE ---
            # This is the portfolio's initial capital (e.g., $200,000)
            bot.portfolio_initial_capital = self.initial_portfolio_capital
            # --- END ADD ---
            
            bot.capital = self.master_realized_capital 
            bot.peak_capital = self.master_peak_capital 
            
            # --- FIX: Store (timestamp, capital) tuples ---
            # This is for the bot's *realized* equity curve.
            # The *notional* curve is handled inside the bot itself.
            bot.realized_equity_curve = [(sim_start_time, self.initial_portfolio_capital)]
            # --- END FIX ---
            
            bot.clock = clock
            bot.connector = dummy_connector
            # --- *** MODIFIED: Pass the full row, not a dict *** ---
            bot.initialize_state(first_row)
            
            fee = getattr(bot.config, 'BYBIT_TAKER_FEE', 0.00055)
            bot.connector.params['BYBIT_TAKER_FEE'] = fee


        # --- HIGH-SPEED CONCURRENT SIMULATION LOOP ---
        for i in range(1, len(df_merged)):
            current_row, prev_row = df_merged.iloc[i], df_merged.iloc[i-1]
            
            # --- *** MODIFIED SYNC BLOCK (Notional Capital Fix) *** ---
            
            # 1. Get GLOBAL realized capital
            total_realized_capital = self.master_realized_capital 
            
            # 2. Calculate GLOBAL and INDIVIDUAL (Notional) UPL
            total_unrealized_pnl = 0.0
            for bot in self.bots:
                bot_upl = 0.0
                for pos in bot.open_positions:
                    bot_upl += pos.calculate_profit(clock.current_price)
                
                # Set the bot's individual UPL
                bot.notional_unrealized_pnl = bot_upl
                total_unrealized_pnl += bot_upl
            
            current_total_equity = total_realized_capital + total_unrealized_pnl
            
            # 3. Update GLOBAL equity peak
            self.master_peak_capital = max(self.master_peak_capital, current_total_equity)
            
            # --- FIX: Store (timestamp, capital) tuple ---
            self.realized_portfolio_equity_curve.append((self.clock.time(), total_realized_capital))
            # --- END FIX ---
            
            # 4. Update INDIVIDUAL (Notional) peak equity
            for bot in self.bots:
                notional_equity = bot.notional_capital + bot.notional_unrealized_pnl
                bot.notional_peak_equity = max(bot.notional_peak_equity, notional_equity)

            # 5. Inject GLOBAL state into ALL bots (for sizing)
            for bot in self.bots:
                bot.capital = total_realized_capital        
                bot.unrealized_pnl = total_unrealized_pnl   
                bot.peak_capital = self.master_peak_capital 
                # This pass-through is for logging ONLY. The bot doesn't use it.
                bot.realized_equity_curve = self.realized_portfolio_equity_curve 
            # --- *** END MODIFIED SYNC BLOCK *** ---
            
            
            # Update 30-minute strategies
            if current_row['timestamp'].floor('30min') > prev_row['timestamp'].floor('30min'):
                for bot in self.bots:
                    # --- *** MODIFIED: Pass the full row, not a dict *** ---
                    bot.update_strategy_on_30m(prev_row)
                
                # --- ** Dynamic Allocation Fix (Part 2) ** ---
                # After bots update their risk (sizing_multiplier),
                # we immediately re-balance the capital allocations.
                self._update_dynamic_allocations()
                # --- ** End Fix ** ---
            
            o, h, l, c, v, base_ts = current_row['open'], current_row['high'], current_row['low'], current_row['close'], current_row['volume'], current_row['timestamp'].timestamp()
            
            if c > o: key_prices = [o, l, h, c]
            else: key_prices = [o, h, l, c]
            
            # --- Indicators are no longer needed here, the bot looks them up ---

            for price in key_prices:
                tick = Tick(timestamp=base_ts, price=price, volume=0, candle_volume=v) 
                clock.current_time, clock.current_price = tick.timestamp, price
                
                for bot in self.bots:
                    # --- *** MODIFIED: Pass the full current_row *** ---
                    # --- *** AND (THE FIX) PASS prev_row *** ---
                    bot.check_confirmation_on_tick(tick, current_row, prev_row)
                    
                    # --- *** THE FIX *** ---
                    # Look up the specific ATR value this bot needs from the master row
                    try:
                        atr_period = bot.config.ATR_PERIOD
                        atr_val = current_row[f'atr_{atr_period}']
                    except KeyError:
                        # self.logger.warning(f"Could not find atr_{atr_period} for bot. Using 0.0")
                        atr_val = 0.0 # Default to 0 if ATR not found
                    
                    # Pass the float value (atr_val), not the whole row
                    bot.check_exits_on_1m(price, atr_val)
                    # --- *** END FIX *** ---

            # --- FIX: Store (timestamp, capital) tuple ---
            self.portfolio_equity_curve.append((current_row['timestamp'].timestamp(), current_total_equity))
            # --- END FIX ---
            
            if trial and i > 30000 and i % 1000 == 0:
                trial.report(current_total_equity, i) 
                if trial.should_prune():
                    raise optuna.TrialPruned()


        # --- Final Performance Calculation ---
        portfolio_results = self._calculate_portfolio_metrics()
        
        # --- FIX: Correctly process list of tuples ---
        # --- ** MODIFIED: Handle duplicate timestamps from multiple closes ** ---
        realized_ts_raw = pd.Series(
            [val for ts, val in self.realized_portfolio_equity_curve],
            index=pd.to_datetime([ts for ts, val in self.realized_portfolio_equity_curve], unit='s')
        )
        if not realized_ts_raw.index.is_unique:
            realized_equity_series = realized_ts_raw.groupby(realized_ts_raw.index).last()
        else:
            realized_equity_series = realized_ts_raw
        # --- ** END MODIFIED ** ---
        
        realized_peak_series = realized_equity_series.expanding().max()
        realized_drawdown_series = (realized_equity_series - realized_peak_series) / realized_peak_series
        portfolio_results['realized_max_drawdown'] = abs(realized_drawdown_series.min()) * 100
        
        if trial:
            trial.report(portfolio_results['final_capital'], len(df_merged))
        
        individual_results = [bot.log_performance(print_log=False) for bot in self.bots]
        
        return portfolio_results, individual_results

    def _calculate_portfolio_metrics(self):
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
        # --- *** END FIX *** ---
        
        # --- *** THE FIX *** ---
        # DO NOT use the equity curve for final PnL. It is one step behind.
        # Use the master variables, which are 100% up-to-date after the loop.
        
        # Get the final realized capital
        final_realized_capital = self.master_realized_capital
        
        # Get the final unrealized PnL (at the last known price)
        final_unrealized_pnl = 0.0
        if self.clock.current_price > 0:
            for bot in self.bots:
                for pos in bot.open_positions:
                    # Use the clock's last price, which is the close of the last candle
                    final_unrealized_pnl += pos.calculate_profit(self.clock.current_price)
        
        # The true final equity is the sum of these two.
        final_equity = final_realized_capital + final_unrealized_pnl
        # --- *** END FIX *** ---

        total_pnl = final_equity - self.initial_portfolio_capital
        total_return = (total_pnl / self.initial_portfolio_capital) * 100 if self.initial_portfolio_capital > 0 else 0
        
        return {
            "initial_capital": self.initial_portfolio_capital,
            "final_capital": final_equity,      # <-- Now this is correct
            "pnl_cash": total_pnl,              # <-- Now this is correct
            "total_return_pct": total_return,   # <-- Now this is correct
            "max_drawdown": max_drawdown,       # <-- This is now correct
        }

# --- Wrapper functions (prepare_data, run_from_prepared, run_single) ---

# --- *** MODIFIED: This function now accepts a list of bots *** ---
def prepare_data_for_simulation(df_1m_full: pd.DataFrame, bots: List[MomentumAnchorBotShared], logger=None) -> pd.DataFrame:
    if logger is None:
        logger = logging.getLogger(__name__)
        
    if not bots:
        logger.error("No bots provided to prepare_data_for_simulation.")
        return pd.DataFrame()
        
    # --- THIS WAS THE BUG: It said [bot] instead of [dummy_bot] ---
    # --- NOW: It uses the provided list of bots ---
    temp_engine = SimulationEngine(df_1m_full, bots, logger, prepared=False) 
    # --- END BUG FIX ---
    try:
        df_prepared = temp_engine._prepare_data()
        return df_prepared
    except Exception as e:
        logger.error(f"Failed during data preparation: {e}", exc_info=True)
        return pd.DataFrame() 

# --- *** REMOVED `run_simulation_from_prepared_data` *** ---

# --- *** MODIFIED: `run_single_backtest` *** ---
def run_single_backtest(
    df_1m_full: pd.DataFrame, 
    bots: List[MomentumAnchorBotShared], # <-- Takes a list of bots
    verbose: bool = False, 
    logger=None, 
    trial: Optional[optuna.trial.Trial] = None 
):
    if logger is None:
        logger = logging.getLogger(__name__)
    
    if not bots:
        logger.error("run_single_backtest requires a list of at least one bot.")
        return {}
        
    for bot in bots:
        bot.silent = not verbose

    # --- MODIFIED: Pass the list of bots directly to the engine ---
    engine = SimulationEngine(df_1m_full, bots, logger, prepared=False)
    portfolio_results, individual_results = engine.run(trial=trial)
    
    # --- MODIFIED: Return portfolio results for ensemble, or individual if only one bot ---
    if len(individual_results) == 1:
        final_metrics = individual_results[0] # This is the bot's report
            
        # --- *** THIS IS THE FIX *** ---
        # The engine's 'portfolio_results' has the correct drawdown.
        # The bot's 'final_metrics' has the correct PnL, trades, etc.
        # We must merge them.
        if portfolio_results:
            final_metrics['max_drawdown'] = portfolio_results.get('max_drawdown', 0.0)
            
            # --- ** ADDED REALIZED DD MERGE ** ---
            final_metrics['realized_max_drawdown'] = portfolio_results.get('realized_max_drawdown', 0.0)
            
            # We also update the log string to reflect the correct value
            # Note: We must also update Calmar Ratio, which depends on max_drawdown
            
            calmar_str = "999.00" # Default for positive return, zero DD
            if final_metrics['max_drawdown'] > 0:
                calmar = final_metrics.get('total_return_pct', 0) / final_metrics['max_drawdown']
                calmar_str = f"{calmar:.2f}"
            elif final_metrics.get('total_return_pct', 0) <= 0:
                calmar_str = "0.00"

            log_str = final_metrics.get("performance_log_str", "")
            
            log_str = log_str.replace(
                "Max Equity Drawdown:   -0.00% (incl. UPL)",
                f"Max Equity Drawdown:   -{final_metrics['max_drawdown']:.2f}% (incl. UPL)"
            )
            # --- ** ADDED REALIZED DD STRING REPLACE ** ---
            log_str = log_str.replace(
                "Max Realized Drawdown: -0.00% (prop firm style)",
                f"Max Realized Drawdown: -{final_metrics['realized_max_drawdown']:.2f}% (prop firm style)"
            )
            log_str = log_str.replace(
                "Calmar Ratio:    999.00",
                f"Calmar Ratio:    {calmar_str}"
            )
            final_metrics["performance_log_str"] = log_str
        # --- *** END FIX *** ---
        
        if verbose and logger and final_metrics:
            logger.info(final_metrics.get("performance_log_str", "Performance log not available."))
        return final_metrics
    
    # For ensemble runs, return the portfolio metrics
    return portfolio_results