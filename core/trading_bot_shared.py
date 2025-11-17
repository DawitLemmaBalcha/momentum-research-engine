# core/trading_bot_shared.py

import logging
import time
import numpy as np
import pandas as pd
from typing import List, Tuple
from core.core_types import Position, Order, SimulationClock, TradeLog, Tick
from connectors.base_connector import BaseConnector
from core.trading_bot_base import MomentumAnchorBotBase, format_time # <-- IMPORT NEW BASE


class MomentumAnchorBotShared(MomentumAnchorBotBase):
    """
    Model C (Shared Portfolio) Bot Implementation.
    
    This bot is designed to run in an ensemble. It receives its capital,
    UPL, and peak equity from a master (SimulationEngine or LiveDataManager).
    It uses a capital_allocation factor for sizing.
    """

    def __init__(self, initial_capital: float, simulation_clock: SimulationClock, config_module, connector: BaseConnector, silent: bool = False):
        # Initialize all common properties from the base class
        super().__init__(initial_capital, simulation_clock, config_module, connector, silent)
        
        # --- ** Notional Capital Fix (Part 1) ** ---
        # Initialize the bot's *own* capital ledger, separate from the portfolio.
        # We read the bot's configured starting capital (e.g., $20,000).
        self.notional_capital = getattr(self.config, 'INITIAL_CAPITAL', self.initial_capital)
        self.notional_peak_equity = self.notional_capital
        self.notional_unrealized_pnl = 0.0
        # --- ** End Fix ** ---
        
        # Model C properties are set by the engine/runner
        self.portfolio_initial_capital = initial_capital
        self.capital_allocation = 1.0 # Will be overwritten by engine

    # --- Model C specific implementations ---

    def update_equity_on_1m(self, current_price: float):
        """
        [MODEL C]
        This method is not used. The SimulationEngine (Model C)
        manages all equity updates externally.
        """
        pass # Explicitly do nothing.


    def check_confirmation_on_tick(self, tick: Tick, data_row: pd.Series, prev_data_row: pd.Series):
        """
        [MODEL C]
        Checks Anchor entries based on indicator values looked up from the data_row.
        
        *** THIS CONTAINS THE LOOKAHEAD BIAS FIX ***
        """
        
        # --- Config & Data Lookups ---
        c = self.config
        try:
            # 1m Confirmaton Lookups
            conf_rsi_p = c.CONFIRMATION_RSI_PERIOD
            conf_vol_ma_p = c.CONFIRMATION_VOLUME_MA_PERIOD
            conf_macd_f, conf_macd_s, conf_macd_sig = c.CONFIRMATION_MACD_FAST_PERIOD, c.CONFIRMATION_MACD_SLOW_PERIOD, c.CONFIRMATION_MACD_SIGNAL_PERIOD
            
            rsi_1m = data_row[f'rsi_1m_{conf_rsi_p}']
            volume_ma_1m = data_row[f'volume_ma_1m_{conf_vol_ma_p}']
            macd_1m = data_row[f'macd_1m_{conf_macd_f}_{conf_macd_s}_{conf_macd_sig}']
            
            # --- *** THE FIX *** ---
            # Use the PREVIOUS candle's volume for confirmation, not the current one.
            volume_1m = prev_data_row['volume']
            # --- *** END FIX *** ---
            
            # 30m ATR Lookup (for stop loss calculation in execute_pyramid_entry)
            atr_p = c.ATR_PERIOD
            atr = data_row[f'atr_{atr_p}']
        except KeyError as e:
            # self.logger.warning(f"Missing 1m/30m data in check_confirmation_on_tick: {e}. Skipping entry.")
            return # Not enough data
        # --- END NEW ---

        # --- ** THIS IS THE MODIFIED LOOP ** ---
        for order in self.pending_anchor_signals[:]:
            if (order.order_type == "buy" and tick.price >= order.price) or (order.order_type == "sell" and tick.price <= order.price):

                # --- FIX 1 (SPAM): Remove anchor immediately on trigger. ---
                # This stops it from being re-triggered on every tick.
                self.pending_anchor_signals.remove(order)

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
                    
                    # --- FIX 2 (DUPLICATE): The duplicate call has been removed. ---
                    # This is the one and only call.
                    trade_successful = self.execute_pyramid_entry(order, size, details, atr)

                    # We no longer check `if trade_successful:` to remove the order,
                    # as it was already removed in FIX 1.
                
                else:
                    # --- START FIX ---
                    reasons = []
                    if not rsi_ok: reasons.append(f"RSI FAILED ({rsi_1m:.1f})")
                    if not macd_ok: reasons.append(f"MACD FAILED ({macd_1m:.2f})")

                    vol_confirm_factor = getattr(self.config, 'VOLUME_CONFIRMATION_FACTOR', 1.0)
                    if not volume_ok: reasons.append(f"VOLUME FAILED ({volume_1m:,.0f} < {volume_ma_1m * vol_confirm_factor:,.0f})")

                    rejection_string = ", ".join(reasons)

                    # Only log if the rejection reason is new
                    if not self.silent and rejection_string and rejection_string != order.last_rejection_reason:
                        self.logger.info(
                            f"🚫 [ENTRY_REJECTED] | "
                            f"Anchor price {order.price:,.2f} hit for {order.order_type}. "
                            f"Reason(s): {rejection_string}"
                        )

                    # Update the last known reason
                    order.last_rejection_reason = rejection_string



    def calculate_position_size(self, price: float, return_details: bool = False) -> float | Tuple[float, str]:
        """
        [MODEL C]
        Calculates position size based on INJECTED portfolio capital
        and this bot's specific `capital_allocation`.
        """
        if price <= 0:
            details_str = "Pos Size Calc: Price <= 0, returning 0.0"
            return (0.0, details_str) if return_details else 0.0

        # In Model C, self.capital, self.unrealized_pnl, and self.peak_capital
        # are INJECTED by the master engine.
        effective_capital_base = self.capital + self.unrealized_pnl
        effective_capital = min(effective_capital_base, self.peak_capital)
        
        details_log_header = (f"Pos Size Calc (Model C/Live): SyncedTotalRealized={self.capital:,.2f}, SyncedTotalUPL={self.unrealized_pnl:,.2f} -> "
                              f"EffCapBase={effective_capital_base:,.2f}, SyncedTotalPeak={self.peak_capital:,.2f} -> ")

        effective_capital = max(0.0, effective_capital) 
        
        # Apply this bot's (now dynamic) allocation
        allocated_capital_for_sizing = effective_capital * self.capital_allocation

        max_value_limit = allocated_capital_for_sizing * self.max_position_size
        kelly_value = allocated_capital_for_sizing * self.kelly_fraction
        final_pos_value = min(max_value_limit, kelly_value)

        base_amount = final_pos_value / price if price > 0 else 0.0
        
        trade_count_in_set = self.trades_since_anchor_reset 
        
        pyramid_multiplier = pow(self.PYRAMID_SCALING_FACTOR, trade_count_in_set)
        
        final_amount = base_amount * pyramid_multiplier
        
        details_str = (f"{details_log_header}"
                       f"AllocFactor={self.capital_allocation:.2f} -> AllocCapForSizing={allocated_capital_for_sizing:,.2f}\n"
                       f"             -> MaxVal={max_value_limit:,.2f}, KellyVal={kelly_value:,.2f} (MaxSz={self.max_position_size:.3f}, KellyFrac={self.kelly_fraction:.3f}) \n"
                       f"             -> FinalVal={final_pos_value:,.2f} -> BaseAmount={base_amount:.5f}\n"
                       f"             -> Degrading: Factor={self.PYRAMID_SCALING_FACTOR}, TradeCountInSet={trade_count_in_set} -> Multiplier={pyramid_multiplier:.4f}\n"
                       f"             -> Final Amount={final_amount:.5f}")

        return (final_amount, details_str) if return_details else final_amount

    def close_position(self, position: Position, current_price: float, reason: str):
        """
        [MODEL C]
        Closes a position. *Does NOT update self.capital*.
        The master engine is responsible for capital updates via the connector.
        """
        current_time_str = format_time(self.clock.time()) 
        capital_before = self.capital # Log the *global* capital before
        
        if not self.silent: self.logger.info(f"{current_time_str} - Attempting to close position {position.id} via connector. Reason: {reason}")

        close_result = self.connector.close_position(position)
        if not close_result or not close_result.get("success"):
            if not self.silent: self.logger.error(f"{current_time_str} - Connector failed to close position {position.id}")
            return

        profit = close_result['pnl']
        
        # --- ** Notional Capital Fix (Part 1) ** ---
        # Update the bot's *own* realized capital ledger.
        # The engine updates the notional peak equity,
        # but the realized notional capital is updated here on trade close.
        self.notional_capital += profit
        # --- ** End Fix ** ---
        
        # In Model C, the connector itself updates the master capital
        # in the engine. This bot *does not* touch self.capital.
        
        profit_percent = (profit / (position.entry_price * position.amount)) * 100 if position.entry_price * position.amount != 0 else 0
        self.trade_history.append(TradeLog(entry_price=position.entry_price, close_price=close_result['close_price'], entry_time=position.entry_time, close_time=self.clock.time(), pnl_percent=profit_percent, pnl_cash=profit, reason=reason, direction='LONG' if position.is_long else 'SHORT', size=position.amount, kelly_risk_usd=position.kelly_risk_usd, atr_risk_usd=position.atr_risk_usd, entry_regime=position.entry_regime))
        if position in self.open_positions: self.open_positions.remove(position)

        if not self.silent:
            duration_s = self.clock.time() - position.entry_time
            duration_str = time.strftime('%Hh %Mm %Ss', time.gmtime(duration_s)) if duration_s >=0 else "N/A"
            gross_pnl = profit + close_result.get('commission', 0)
            
            # Log that we are waiting for the next sync
            capital_update_log = "(Waiting for next sync...)" 
                
            self.logger.info(f"\n{current_time_str} - [TRADE CLOSE] | {'LONG' if position.is_long else 'SHORT'} | Reason: {reason}\n"
                                f"  - Position ID:      {position.id}\n"
                                f"  - Duration:         {duration_str}\n"
                                f"  - Entry Price:      {position.entry_price:,.2f}\n"
                                f"  - Close Price:      {close_result['close_price']:,.2f}\n"
                                f"  - Gross PnL:        ${gross_pnl:,.2f}\n"
                                f"  - Commission Paid:  -${close_result.get('commission', 0):,.2f}\n"
                                f"  - Net PnL:          ${profit:,.2f} ({profit_percent:.2f}%)\n"
                                f"  - Portfolio Impact: Global Capital ${capital_before:,.2f} -> {capital_update_log}")