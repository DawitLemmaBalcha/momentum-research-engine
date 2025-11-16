# core/trading_bot.py

import logging
import time
import numpy as np
import pandas as pd
from typing import List, Tuple
from core.core_types import Position, Order, SimulationClock, TradeLog, Tick
from connectors.base_connector import BaseConnector
from core.trading_bot_base import MomentumAnchorBotBase, format_time # <-- IMPORT NEW BASE


class MomentumAnchorBot(MomentumAnchorBotBase):
    """
    Model A (Standalone) Bot Implementation.
    
    This bot manages its own capital, calculates its own UPL,
    and handles its own equity/drawdown updates. It is used
    for optimization and single-bot backtests.
    """
    
    def __init__(self, initial_capital: float, simulation_clock: SimulationClock, config_module, connector: BaseConnector, silent: bool = False):
        # Initialize all common properties from the base class
        super().__init__(initial_capital, simulation_clock, config_module, connector, silent)
        # Model A specific properties
        self.capital_allocation = 1.0 # Model A always uses 100% of its own capital

        # --- FIX: Store (timestamp, capital) tuples ---
        # We get the start time from the simulation_clock
        self.realized_equity_curve = [(self.clock.time(), initial_capital)]
        # --- END FIX ---

    # --- Model A specific implementations ---

    def update_equity_on_1m(self, current_price: float):
        """
        [MODEL A]
        Calculates and updates the bot's equity, peak equity (incl. UPL), 
        and drawdown state every 1 minute.
        """
        is_backtest = self.connector.__class__.__name__ == 'DummyConnector'
        current_total_equity = 0.0
        
        if is_backtest:
            # In (Model A) backtest, we are the "master" and must calculate our own UPL
            unrealized_pnl = 0.0
            if self.clock.current_price > 0:
                for pos in self.open_positions:
                    # Use the 1-minute close price for MTM
                    unrealized_pnl += pos.calculate_profit(current_price) 
            
            # Set the bot's UPL state
            self.unrealized_pnl = unrealized_pnl
            current_total_equity = self.capital + self.unrealized_pnl

            # This is the EQUITY peak (incl. UPL)
            if current_total_equity > self.peak_capital:
                self.peak_capital = current_total_equity
            
            # Calculate EQUITY drawdown
            current_drawdown = (self.peak_capital - current_total_equity) / self.peak_capital * 100 if self.peak_capital > 0 else 0
            self.max_drawdown = max(self.max_drawdown, current_drawdown)
            
            # --- *** THIS IS THE FIX *** ---
            # DO NOT check for de-risking here.
            # This call is moved to update_strategy_on_30m in the base class,
            # so both Model A and Model C update risk at the same frequency.
            # if current_drawdown > self.config.MAX_DRAWDOWN_PERCENT: 
            #    self.update_risk_parameters() # <-- THIS LINE IS REMOVED
            # --- *** END FIX *** ---
        
        else:
            # In live mode, we are a "slave" and use the synced values
            current_total_equity = self.capital + self.unrealized_pnl
            current_drawdown = (self.peak_capital - current_total_equity) / self.peak_capital * 100 if self.peak_capital > 0 else 0
            self.max_drawdown = max(self.max_drawdown, current_drawdown)
            
            # --- *** THIS IS THE FIX *** ---
            # if current_drawdown > self.config.MAX_DRAWDOWN_PERCENT: 
            #    self.update_risk_parameters() # <-- THIS LINE IS REMOVED
            # --- *** END FIX *** ---

    def check_confirmation_on_tick(self, tick: Tick, rsi_1m: float, macd_1m: float, volume_1m: float, volume_ma_1m: float, atr: float):
        """
        [MODEL A]
        Checks Anchor entries based on indicator values passed from the runner.
        """
        for order in self.pending_anchor_signals[:]:
            if (order.order_type == "buy" and tick.price >= order.price) or (order.order_type == "sell" and tick.price <= order.price):

                volume_ok = volume_1m > (volume_ma_1m * getattr(self.config, 'VOLUME_CONFIRMATION_FACTOR', 1.0))

                if order.order_type == "buy": rsi_ok, macd_ok = rsi_1m > self.config.RSI_BUY_CONFIRMATION, macd_1m > 0
                else: rsi_ok, macd_ok = rsi_1m < self.config.RSI_SELL_CONFIRMATION, macd_1m < 0

                if rsi_ok and macd_ok and volume_ok:
                    if not self.silent: self.logger.info(f"{format_time(tick.timestamp)} - Anchor level triggered: {order.order_type} @ {order.price:,.2f} by tick price {tick.price:,.2f}")
                    
                    size, size_details_str = self.calculate_position_size(tick.price, return_details=True)
                    if not self.silent:
                        self.logger.info(f"{format_time(tick.timestamp)} - {size_details_str}")
                    
                    details = {
                        "price_ok": f"{tick.price:,.2f} {' >=' if order.order_type == 'buy' else ' <='} {order.price:,.2f}",
                        "rsi_ok": rsi_ok, "rsi_val": rsi_1m, "rsi_req": getattr(self.config, f'RSI_{"BUY" if order.order_type == 'buy' else "SELL"}_CONFIRMATION'),
                        "macd_ok": macd_ok, "macd_val": macd_1m,
                        "volume_ok": volume_ok,
                        "volume_val": volume_1m,
                        "volume_req": volume_ma_1m * getattr(self.config, 'VOLUME_CONFIRMATION_FACTOR', 1.0)
                    }
                    
                    self.execute_pyramid_entry(order, size, details, atr)
                    self.pending_anchor_signals.remove(order) 

    def calculate_position_size(self, price: float, return_details: bool = False) -> float | Tuple[float, str]:
        """
        [MODEL A]
        Calculates position size based on its *own* capital and UPL.
        """
        if price <= 0:
            details_str = "Pos Size Calc: Price <= 0, returning 0.0"
            return (0.0, details_str) if return_details else 0.0

        is_backtest = self.connector.__class__.__name__ == 'DummyConnector'
        effective_capital_base = 0.0
        details_log_header = ""

        if is_backtest:
            # self.unrealized_pnl is set by this bot's update_equity_on_1m
            effective_capital_base = self.capital + self.unrealized_pnl
            effective_capital = min(effective_capital_base, self.peak_capital)
            
            details_log_header = (f"Pos Size Calc (Backtest): RealizedCap={self.capital:,.2f}, UPL={self.unrealized_pnl:,.2f} -> "
                                  f"EffCapBase={effective_capital_base:,.2f}, PeakCap={self.peak_capital:,.2f} -> ")
        else:
            # Live mode assumes UPL is synced by an external process
            effective_capital_base = self.capital + self.unrealized_pnl
            effective_capital = min(effective_capital_base, self.peak_capital)
            
            details_log_header = (f"Pos Size Calc (Live): SyncedRealizedCap={self.capital:,.2f}, SyncedUPL={self.unrealized_pnl:,.2f} -> "
                                  f"EffCapBase={effective_capital_base:,.2f}, SyncedPeakCap={self.peak_capital:,.2f} -> ")

        effective_capital = max(0.0, effective_capital) 

        # Note: self.max_position_size and self.kelly_fraction are now
        # dynamically set by the update_risk_parameters() function
        max_value_limit = effective_capital * self.max_position_size
        kelly_value = effective_capital * self.kelly_fraction
        final_pos_value = min(max_value_limit, kelly_value)

        base_amount = final_pos_value / price if price > 0 else 0.0
        
        trade_count_in_set = self.trades_since_anchor_reset 
        
        pyramid_multiplier = pow(self.PYRAMID_SCALING_FACTOR, trade_count_in_set)
        
        final_amount = base_amount * pyramid_multiplier
        
        details_str = (f"{details_log_header}"
                       f"EffCapital={effective_capital:,.2f}, MaxPosVal={max_value_limit:,.2f}, KellyPosVal={kelly_value:,.2f} "
                       f"(MaxSz={self.max_position_size:.3f}, KellyFrac={self.kelly_fraction:.3f}) "
                       f"-> FinalVal={final_pos_value:,.2f} -> BaseAmount={base_amount:.5f}\n"
                       f"             -> Degrading: Factor={self.PYRAMID_SCALING_FACTOR}, TradeCountInSet={trade_count_in_set} -> Multiplier={pyramid_multiplier:.4f}\n"
                       f"             -> Final Amount={final_amount:.5f}")

        return (final_amount, details_str) if return_details else final_amount

    def close_position(self, position: Position, current_price: float, reason: str):
        """
        [MODEL A]
        Closes a position and *updates this bot's own capital*.
        """
        current_time_str = format_time(self.clock.time()) 
        capital_before = self.capital
        
        if not self.silent: self.logger.info(f"{current_time_str} - Attempting to close position {position.id} via connector. Reason: {reason}")

        close_result = self.connector.close_position(position)
        if not close_result or not close_result.get("success"):
            if not self.silent: self.logger.error(f"{current_time_str} - Connector failed to close position {position.id}")
            return

        profit = close_result['pnl']
        
        is_backtest = self.connector.__class__.__name__ == 'DummyConnector'
        if is_backtest:
            # Model A updates its own capital
            self.capital += profit
            
            # --- FIX: Store (timestamp, capital) tuple ---
            # Log the change to the realized equity curve with the current time
            self.realized_equity_curve.append((self.clock.time(), self.capital))
            # --- END FIX ---
            
        profit_percent = (profit / (position.entry_price * position.amount)) * 100 if position.entry_price * position.amount != 0 else 0
        self.trade_history.append(TradeLog(entry_price=position.entry_price, close_price=close_result['close_price'], entry_time=position.entry_time, close_time=self.clock.time(), pnl_percent=profit_percent, pnl_cash=profit, reason=reason, direction='LONG' if position.is_long else 'SHORT', size=position.amount, kelly_risk_usd=position.kelly_risk_usd, atr_risk_usd=position.atr_risk_usd, entry_regime=position.entry_regime))
        if position in self.open_positions: self.open_positions.remove(position)

        if not self.silent:
            duration_s = self.clock.time() - position.entry_time
            duration_str = time.strftime('%Hh %Mm %Ss', time.gmtime(duration_s)) if duration_s >=0 else "N/A"
            gross_pnl = profit + close_result.get('commission', 0)
            
            capital_update_log = ""
            if is_backtest:
                capital_update_log = f"${self.capital:,.2f}" 
            else:
                capital_update_log = "(Waiting for next sync...)" 
                
            self.logger.info(f"\n{current_time_str} - [TRADE CLOSE] | {'LONG' if position.is_long else 'SHORT'} | Reason: {reason}\n"
                                f"  - Position ID:      {position.id}\n"
                                f"  - Duration:         {duration_str}\n"
                                f"  - Entry Price:      {position.entry_price:,.2f}\n"
                                f"  - Close Price:      {close_result['close_price']:,.2f}\n"
                                f"  - Gross PnL:        ${gross_pnl:,.2f}\n"
                                f"  - Commission Paid:  -${close_result.get('commission', 0):,.2f}\n"
                                f"  - Net PnL:          ${profit:,.2f} ({profit_percent:.2f}%)\n"
                                f"  - Portfolio Impact: Capital ${capital_before:,.2f} -> {capital_update_log}")