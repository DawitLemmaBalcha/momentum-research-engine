# core/trading_bot_base.py

import logging
import time
import numpy as np
import pandas as pd
from typing import List, Tuple
from dataclasses import dataclass
from collections import defaultdict, deque
from abc import ABC, abstractmethod
from core.core_types import Position, Order, SimulationClock, TradeLog, Tick
from connectors.base_connector import BaseConnector

default_logger = logging.getLogger(__name__)

def format_time(timestamp: float) -> str:
    return pd.to_datetime(timestamp, unit='s').strftime('%Y-%m-%d %H:%M:%S')

class MomentumAnchorBotBase(ABC):
    """
    Abstract Base Class for the Trend-Anchored, Momentum-Confirmed Bot.
    """
    def __init__(self, initial_capital: float, simulation_clock: SimulationClock, config_module, connector: BaseConnector, silent: bool = False):
        self.config = config_module
        self.logger = default_logger
        self.initial_capital = initial_capital
        self.clock = simulation_clock
        self.connector = connector
        self.silent = silent
        self.symbol = self.config.SYMBOL
        self.open_positions, self.pending_anchor_signals, self.trade_history = [], [], []
        self.capital = initial_capital
        self.peak_capital = initial_capital
        self.unrealized_pnl = 0.0
        self.realized_equity_curve = [(self.clock.time(), initial_capital)]
        self.max_drawdown = 0.0
        self.kelly_fraction = self.config.KELLY_FRACTION
        self.max_position_size = self.config.MAX_POSITION_SIZE_PERCENT
        self.PYRAMID_SCALING_FACTOR = getattr(self.config, 'PYRAMID_SCALING_FACTOR', 1.0) 
        self.trades_since_anchor_reset = 0
        self.last_anchor_setup_time = 0
        self.anchor_step_size = 0.01
        self.trend_regime = "uninitialized"
        self.sizing_multiplier = 1.0
        self.total_anchors_set = 0
        self.total_anchors_triggered = 0
        self.portfolio_initial_capital = initial_capital
        self.capital_allocation = 1.0
        timesfm_context = int(getattr(self.config, 'TIMESFM_CONTEXT', 512))
        self._close_history = deque(maxlen=timesfm_context)
        self._close_history_times = deque(maxlen=timesfm_context)
        self.forecast_gate = self._build_forecast_gate()

    def _build_forecast_gate(self):
        """Build the optional TimesFM entry gate. Disabled unless configured."""
        if not getattr(self.config, 'TIMESFM_ENABLED', False):
            return None
        from forecasting import ForecastGate, TimesFMForecaster

        horizon = int(getattr(self.config, 'TIMESFM_HORIZON', 5))
        forecaster = TimesFMForecaster(
            checkpoint=getattr(self.config, 'TIMESFM_CHECKPOINT', 'google/timesfm-2.5-200m-pytorch'),
            max_context=int(getattr(self.config, 'TIMESFM_CONTEXT', 512)),
            max_horizon=max(horizon, int(getattr(self.config, 'TIMESFM_MAX_HORIZON', 64))),
        )
        return ForecastGate(
            forecaster,
            horizon=horizon,
            min_context=int(getattr(self.config, 'TIMESFM_MIN_CONTEXT', 64)),
            min_agreement=float(getattr(self.config, 'TIMESFM_MIN_AGREEMENT', 0.6)),
            min_expected_return_pct=float(getattr(self.config, 'TIMESFM_MIN_EXPECTED_RETURN_PCT', 0.0)),
            cache_bars=int(getattr(self.config, 'TIMESFM_CACHE_BARS', 5)),
            bar_seconds=float(getattr(self.config, 'TIMESFM_BAR_SECONDS', 60.0)),
            fail_open=bool(getattr(self.config, 'TIMESFM_FAIL_OPEN', False)),
        )

    def observe_close(self, timestamp: float, close: float) -> None:
        """Record one closed bar for the forecast gate.

        Bars are appended in time order and duplicate or older timestamps are
        ignored so a tick loop can call this safely once per bar.
        """
        if self.forecast_gate is None:
            return
        try:
            ts = float(timestamp)
            price = float(close)
        except (TypeError, ValueError):
            return
        if not np.isfinite(ts) or not np.isfinite(price) or price <= 0:
            return
        if self._close_history_times and ts <= self._close_history_times[-1]:
            return
        self._close_history.append(price)
        self._close_history_times.append(ts)

    def _entry_allowed(self, order: Order) -> bool:
        gate = self.forecast_gate
        if gate is None or len(self._close_history) < gate.min_context:
            return True
        decision = gate.evaluate(list(self._close_history), side=order.order_type, timestamp=self.clock.time())
        if decision.allowed:
            return True
        if not self.silent:
            detail = decision.reason
            if decision.agreement is not None:
                detail += f" (agreement={decision.agreement:.2f}"
                if decision.expected_return_pct is not None:
                    detail += f", expected_return={decision.expected_return_pct:.3f}%"
                detail += ")"
            self.logger.info(
                f"{format_time(self.clock.time())} - [TIMESFM VETO] {order.order_type} @ {order.price:,.2f}: {detail}"
            )
        return False

    @abstractmethod
    def calculate_position_size(self, price: float, return_details: bool = False) -> float | Tuple[float, str]:
        pass

    @abstractmethod
    def close_position(self, position: Position, current_price: float, reason: str):
        pass

    def initialize_state(self, initial_data_row: dict):
        self.trend_regime = self._get_trend_regime(initial_data_row)
        self.setup_asymmetric_anchors("Initial Setup", initial_data_row)

    def _get_trend_regime(self, data_row: dict) -> str:
        c = self.config
        try:
            short_ma = data_row.get(f'short_ema_{c.SHORT_EMA_PERIOD}', data_row.get('short_ema'))
            long_ma = data_row.get(f'long_ema_{c.LONG_EMA_PERIOD}', data_row.get('long_ema'))
            rsi = data_row.get(f'rsi_{c.RSI_PERIOD}', data_row.get('rsi'))
            macd = data_row.get(f'macd_{c.MACD_FAST_PERIOD}_{c.MACD_SLOW_PERIOD}_{c.MACD_SIGNAL_PERIOD}', data_row.get('macd'))
            bb_upper = data_row.get(f'bb_upper_{c.BOLLINGER_PERIOD}_{c.BOLLINGER_STD_DEV}', data_row.get('bb_upper'))
            bb_lower = data_row.get(f'bb_lower_{c.BOLLINGER_PERIOD}_{c.BOLLINGER_STD_DEV}', data_row.get('bb_lower'))
            current_price = data_row['close']
            
            if any(v is None for v in [short_ma, long_ma, rsi, macd, bb_upper, bb_lower]):
                return "neutral"
        except KeyError:
            return "neutral"

        trend = "neutral"
        if short_ma > long_ma * (1 + self.config.TREND_EMA_THRESHOLD_PERCENT / 100): trend = "up"
        elif short_ma < long_ma * (1 - self.config.TREND_EMA_THRESHOLD_PERCENT / 100): trend = "down"
        if rsi > self.config.RSI_OVERBOUGHT: trend = "strong_up"
        elif rsi < self.config.RSI_OVERSOLD: trend = "strong_down"
        if macd > 0 and "up" in trend: trend = "strong_up"
        elif macd < 0 and "down" in trend: trend = "strong_down"
        bb_width = (bb_upper - bb_lower) / current_price if current_price != 0 else 0
        regime = "trending" if bb_width > self.config.BB_REGIME_THRESHOLD else "ranging"
        return f"{trend}_{regime}"

    def setup_asymmetric_anchors(self, reason: str, data_row: dict):
        current_time = self.clock.time()
        current_time_str = format_time(current_time)
        if current_time - self.last_anchor_setup_time < getattr(self.config, 'ANCHOR_SETUP_COOLDOWN_SECONDS', 1800): return
        
        c = self.config
        try:
            current_price = data_row['close']
            bb_upper = data_row.get(f'bb_upper_{c.BOLLINGER_PERIOD}_{c.BOLLINGER_STD_DEV}', data_row.get('bb_upper'))
            bb_lower = data_row.get(f'bb_lower_{c.BOLLINGER_PERIOD}_{c.BOLLINGER_STD_DEV}', data_row.get('bb_lower'))
            if bb_upper is None or bb_lower is None: return 
        except KeyError: return 

        self.pending_anchor_signals.clear() 
        self.trades_since_anchor_reset = 0
        
        old_anchor_step_size = self.anchor_step_size
        bb_width = (bb_upper - bb_lower) / current_price if current_price > 0 else 0
        self.anchor_step_size = max(self.config.ANCHOR_STEP_MIN_PERCENT, min(bb_width * getattr(self.config, 'ANCHOR_BAND_MULTIPLIER', 0.5), self.config.ANCHOR_STEP_MAX_PERCENT))
        
        total_anchors = self.config.NUM_ANCHORS
        if "ranging" in self.trend_regime: total_anchors = 2 
        
        ratio_normal = getattr(self.config, 'TREND_PYRAMID_RATIO_NORMAL', 0.6)
        ratio_strong = getattr(self.config, 'TREND_PYRAMID_RATIO_STRONG', 0.8)
        
        if "strong_up" in self.trend_regime: buy_anchors, sell_anchors = int(round(total_anchors * ratio_strong)), total_anchors - int(round(total_anchors * ratio_strong))
        elif "up" in self.trend_regime: buy_anchors, sell_anchors = int(round(total_anchors * ratio_normal)), total_anchors - int(round(total_anchors * ratio_normal))
        elif "strong_down" in self.trend_regime: sell_anchors, buy_anchors = int(round(total_anchors * ratio_strong)), total_anchors - int(round(total_anchors * ratio_strong))
        elif "down" in self.trend_regime: sell_anchors, buy_anchors = int(round(total_anchors * ratio_normal)), total_anchors - int(round(total_anchors * ratio_normal))
        else: buy_anchors, sell_anchors = total_anchors // 2, total_anchors - (total_anchors // 2)

        new_anchor_signals = [] 
        for i in range(1, buy_anchors + 1):
            price = current_price * (1 + i * self.anchor_step_size)
            new_signal = Order(price, "buy")
            new_anchor_signals.append(new_signal)
            if not self.silent: self.logger.info(f"{current_time_str} - Signals ➡️ Created internal anchor signal: buy @ {new_signal.price:,.2f}")
        for i in range(1, sell_anchors + 1):
            price = current_price * (1 - i * self.anchor_step_size)
            new_signal = Order(price, "sell")
            new_anchor_signals.append(new_signal)
            if not self.silent: self.logger.info(f"{current_time_str} - Signals ⬅️ Created internal anchor signal: sell @ {new_signal.price:,.2f}")

        self.pending_anchor_signals = new_anchor_signals 
        if self.pending_anchor_signals: self.total_anchors_set += len(self.pending_anchor_signals)

        if not self.silent:
            buy_anchors_list = [o for o in self.pending_anchor_signals if o.order_type == 'buy']
            sell_anchors_list = [o for o in self.pending_anchor_signals if o.order_type == 'sell']
            self.logger.info(f"\n{current_time_str} - [⚓ ANCHORS RESET] Reason: {reason}\n"
                            f"  - Center Price:    {current_price:,.2f}\n"
                            f"  - Trend/Regime:    {self.trend_regime}\n"
                            f"  - Anchor Step:     {self.anchor_step_size:.3%} (was {old_anchor_step_size:.3%})\n"
                            f"  - Setup:           {len(buy_anchors_list)} Buy Anchors | {len(sell_anchors_list)} Sell Anchors\n"
                            f"  - Buy Levels:      [ {', '.join([f'{o.price:,.2f}' for o in buy_anchors_list])} ]\n"
                            f"  - Sell Levels:     [ {', '.join([f'{o.price:,.2f}' for o in sell_anchors_list])} ]")
        self.last_anchor_setup_time = current_time

    def check_exits_on_1m(self, current_price: float, atr: float):
        current_time = self.clock.time()
        current_time_str = format_time(current_time) 
        for position in self.open_positions[:]:
            if current_time - position.entry_time > getattr(self.config, 'MAX_POSITION_DURATION_SECONDS', 86400):
                if not self.silent: self.logger.info(f"{current_time_str} - ⌛ Triggering Time Stop for position {position.id}")
                self.close_position(position, current_price, "Time Stop"); continue
            
            if not (atr > 0 and pd.notna(atr)): continue
            
            if position.stop_loss is None: 
                position.stop_loss = position.entry_price - (atr * self.config.ATR_INITIAL_STOP_MULTIPLIER) if position.is_long else position.entry_price + (atr * self.config.ATR_INITIAL_STOP_MULTIPLIER)
            
            activation_price = position.entry_price + (atr * self.config.ATR_TRAILING_STOP_ACTIVATION_MULTIPLIER) if position.is_long else position.entry_price - (atr * self.config.ATR_TRAILING_STOP_ACTIVATION_MULTIPLIER)

            if not position.is_trailing_active and ((position.is_long and current_price >= activation_price) or (not position.is_long and current_price <= activation_price)):
                position.is_trailing_active = True
                if not self.silent: self.logger.info(f"{current_time_str} - 🎯 Trailing stop activated for position {position.id} at price {current_price:,.2f}.")

            if position.is_trailing_active:
                trailing_distance = atr * self.config.ATR_TRAILING_STOP_MULTIPLIER
                if position.is_long:
                    position.peak_price = max(position.peak_price, current_price)
                    new_stop = max(position.stop_loss, position.peak_price - trailing_distance)
                else:
                    position.peak_price = min(position.peak_price, current_price)
                    new_stop = min(position.stop_loss, position.peak_price + trailing_distance)
                
                if position.last_logged_stop_loss == 0.0:
                    position.last_logged_stop_loss = position.stop_loss if position.stop_loss else new_stop

                if position.last_logged_stop_loss > 0:
                    change_pct = abs(new_stop - position.last_logged_stop_loss) / position.last_logged_stop_loss
                else:
                    change_pct = 1.0 # Force log if invalid

                # Only log if moved > 0.05% (0.0005)
                if change_pct > 0.0005:
                    if not self.silent: 
                        self.logger.info(f"{current_time_str} - 🔒 Trailing SL moved to {new_stop:,.2f} for position {position.id}")
                    position.last_logged_stop_loss = new_stop
                
                position.stop_loss = new_stop


            if position.stop_loss is not None and ((position.is_long and current_price <= position.stop_loss) or (not position.is_long and current_price >= position.stop_loss)):
                exit_reason = "Trail Stop" if position.is_trailing_active else "ATR Stop"
                emoji = "🟢" if position.is_trailing_active else "🛑" 
                if not self.silent: self.logger.info(f"{current_time_str} - {emoji} Triggering {exit_reason} for position {position.id} at price {current_price:,.2f} (Stop: {position.stop_loss:,.2f})")
                self.close_position(position, current_price, exit_reason)

    def execute_pyramid_entry(self, order: Order, amount: float, confirmation_details: dict, atr: float):
        current_time_str = format_time(self.clock.time()) 
        
        if len(self.open_positions) >= self.config.MAX_POSITIONS or amount <= 0: 
            if amount <= 0 and not self.silent:
                self.logger.info(f"{current_time_str} - ⏩ Skipping trade. Calculated size is {amount:.5f}.")
            return False
        
        if not self._entry_allowed(order):
            return False

        direction = 'LONG' if order.order_type == 'buy' else 'SHORT'
        if not self.silent: self.logger.info(f"{current_time_str} - 📲 Attempting to place order via connector: {order.order_type} {amount:.5f} {self.symbol}")

        result = self.connector.place_order(symbol=self.symbol, side=order.order_type, order_type="Market", qty=amount, price=order.price)
        if not result or not result.get("success"):
            if not self.silent: self.logger.error(f"{current_time_str} - ❌ Connector failed to place order.")
            return False

        self.total_anchors_triggered += 1
        self.trades_since_anchor_reset += 1

        entry_price = result['entry_price']
        
        # --- FIX: Use the actual filled quantity from the connector if available ---
        # This prevents the "Qty invalid" error when closing because of decimal mismatches
        filled_amount = result.get('qty', amount) 
        
        new_position = Position(entry_price, filled_amount, direction == 'LONG')
        new_position.symbol = self.symbol 
        new_position.id, new_position.entry_time, new_position.entry_regime = result['trade_id'], self.clock.time(), self.trend_regime 
        
        # Store entry commission if the connector provided it (for accurate PnL)
        if 'commission' in result:
            new_position.entry_commission = result['commission']

        stop_distance = atr * self.config.ATR_INITIAL_STOP_MULTIPLIER if (atr > 0 and pd.notna(atr)) else entry_price * 0.01 
        new_position.kelly_risk_usd, new_position.atr_risk_usd = entry_price * new_position.amount, stop_distance * new_position.amount
        self.open_positions.append(new_position)

        if not self.silent:
            initial_stop_price = (entry_price - stop_distance) if new_position.is_long else (entry_price + stop_distance)
            self.logger.info(f"\n{current_time_str} - [🚀 TRADE OPEN] | {direction} @ {entry_price:,.2f}\n"
                            f"  - Position ID:    {new_position.id}\n"
                            f"  - Entry Regime:   {new_position.entry_regime}\n"
                            f"  - Risk Profile:\n"
                            f"    - Position Value (Kelly Risk): ${new_position.kelly_risk_usd:,.2f}\n"
                            f"    - Stop-Loss Risk (ATR Risk):  -${new_position.atr_risk_usd:,.2f}\n"
                            f"    - Initial Stop Price:          ${initial_stop_price:,.2f}\n"
                            f"  - Justification (1m Confirmation):\n"
                            f"    - Price ({confirmation_details['price_ok']}) -> [PASS]\n"
                            f"    - RSI ({confirmation_details['rsi_val']:.2f} {'>' if direction == 'LONG' else '<'} {confirmation_details['rsi_req']}) -> [{'✅' if confirmation_details['rsi_ok'] else '❌'}]\n"
                            f"    - MACD ({confirmation_details['macd_val']:.2f} {'> 0' if direction == 'LONG' else '< 0'}) -> [{'✅' if confirmation_details['macd_ok'] else '❌'}]\n"
                            f"    - Volume ({confirmation_details['volume_val']:,.2f} > {confirmation_details['volume_req']:,.2f}) -> [{'✅' if confirmation_details['volume_ok'] else '❌'}]")
        return True

    def update_risk_parameters(self, log_status: bool = False):
        bot_realized_capital = getattr(self, 'notional_capital', self.capital)
        bot_unrealized_pnl = getattr(self, 'notional_unrealized_pnl', self.unrealized_pnl)
        bot_peak_equity = getattr(self, 'notional_peak_equity', self.peak_capital)
        current_equity = bot_realized_capital + bot_unrealized_pnl
        current_peak = bot_peak_equity
        
        drawdown_trigger_pct = getattr(self.config, 'MAX_DRAWDOWN_PERCENT', 10.0) / 100.0
        aggression = getattr(self.config, 'DE_RISK_AGGRESSION', 1.0)
        max_cut_pct = getattr(self.config, 'DE_RISK_MAX_CUT', 0.8)
        lower_bound_mult = getattr(self.config, 'DE_RISK_LOWER_BOUND', 0.1)
        re_risk_aggression = getattr(self.config, 'RE_RISK_AGGRESSION', 0.25)

        current_drawdown_pct = 0.0
        if current_peak > 0 and current_equity < current_peak:
            current_drawdown_pct = (current_peak - current_equity) / current_peak
        
        target_multiplier = 1.0 

        if current_drawdown_pct > 0 and drawdown_trigger_pct > 0:
            drawdown_progress = current_drawdown_pct / drawdown_trigger_pct
            scaled_reduction = drawdown_progress * aggression
            capped_reduction = min(max_cut_pct, scaled_reduction)
            target_multiplier = 1.0 - capped_reduction
            
        target_multiplier = max(lower_bound_mult, target_multiplier)

        current_multiplier = self.sizing_multiplier 
        
        if target_multiplier < current_multiplier:
            self.sizing_multiplier = target_multiplier
            if log_status and not self.silent:
                self.logger.warning(
                    f"📉 [RISK_DECREASED] | "
                    f"Drawdown {current_drawdown_pct:.2%} hit. "
                    f"Sizing multiplier SNAPPED DOWN from {current_multiplier:.2%} to {self.sizing_multiplier:.2%}"
                )
        elif target_multiplier > current_multiplier:
            gap_to_recover = target_multiplier - current_multiplier
            increase_amount = gap_to_recover * re_risk_aggression
            new_multiplier = current_multiplier + increase_amount
            self.sizing_multiplier = min(new_multiplier, 1.0)
            
            if log_status and not self.silent and self.sizing_multiplier > current_multiplier:
                self.logger.info(
                    f"📈 [RISK_INCREASED] | "
                    f"Drawdown healing. Sizing SNAPPED UP from {current_multiplier:.2%} to {self.sizing_multiplier:.2%} "
                    f"(Target: {target_multiplier:.2%}, Aggression: {re_risk_aggression:.0%})"
                )
        else:
            if log_status and not self.silent:
                self.logger.info(
                    f"🛡️  [RISK_STABLE] | "
                    f"Drawdown {current_drawdown_pct:.2%}. "
                    f"Sizing multiplier remains {self.sizing_multiplier:.2%}"
                )

        self.max_position_size = self.config.MAX_POSITION_SIZE_PERCENT * self.sizing_multiplier
        self.kelly_fraction = self.config.KELLY_FRACTION * self.sizing_multiplier

    def update_kelly_fraction(self):
        if len(self.trade_history) < self.config.KELLY_LOOKBACK: return
        recent_pnls = [t.pnl_percent / 100 for t in self.trade_history[-self.config.KELLY_LOOKBACK:]]
        wins = [p for p in recent_pnls if p > 0]
        if not wins:
            self.kelly_fraction = self.config.KELLY_MIN_FRACTION; return 
        win_rate = len(wins) / len(recent_pnls)
        avg_win = np.mean(wins)
        losses = [p for p in recent_pnls if p <= 0]
        avg_loss = abs(np.mean(losses)) if losses else 0.00001 

        win_loss_ratio = avg_win / avg_loss if avg_loss > 0 else avg_win / 0.00001 
        kelly_raw = win_rate - (1 - win_rate) / win_loss_ratio if win_loss_ratio > 0 else 0.0
        
        self.config.KELLY_FRACTION = max(self.config.KELLY_MIN_FRACTION, min(kelly_raw, 1.0)) 
        self.kelly_fraction = self.config.KELLY_FRACTION * self.sizing_multiplier

    def update_strategy_on_30m(self, data_row: dict):
        old_trend_regime = self.trend_regime
        self.trend_regime = self._get_trend_regime(data_row)
        rebuild_reason = None
        
        if self.trend_regime != old_trend_regime: 
            rebuild_reason = f"Strategy Shift: {old_trend_regime} -> {self.trend_regime}"
        elif not self.open_positions and self.clock.time() - self.last_anchor_setup_time > getattr(self.config, 'ANCHOR_SIGNAL_WINDOW_SECONDS', 14400): 
            rebuild_reason = "Anchor Signal Window Expired"
            
        if rebuild_reason: 
            if not self.silent:
                self.logger.info(f"🧠 [STRATEGY_UPDATE] | {rebuild_reason}. Rebuilding anchors.")
            self.setup_asymmetric_anchors(rebuild_reason, data_row)
        else:
            if not self.silent:
                self.logger.info(f"🧠 [STRATEGY_UPDATE] | Regime stable ({self.trend_regime}). No anchor reset needed.")
            
        self.update_kelly_fraction()
        self.update_risk_parameters(log_status=True)

    def log_performance(self, print_log: bool = False, logger=None) -> dict:
        if logger is None: logger = self.logger
        is_backtest = self.connector.__class__.__name__ == 'DummyConnector'
        is_model_c_run = is_backtest and (self.capital_allocation != 1.0)

        if is_model_c_run:
            pnl_cash = self.capital - self.portfolio_initial_capital
            total_return_pct = (pnl_cash / self.portfolio_initial_capital) * 100 if self.portfolio_initial_capital > 0 else 0
        else:
            pnl_cash = self.capital - self.initial_capital
            total_return_pct = (pnl_cash / self.initial_capital) * 100 if self.initial_capital > 0 else 0

        metrics = {
            "initial_capital": self.initial_capital,
            "final_capital": self.capital,
            "total_return_pct": total_return_pct, 
            "pnl_cash": pnl_cash, 
            "max_drawdown": self.max_drawdown, 
            "sharpe_ratio": 0, "sortino_ratio": 0, "calmar_ratio": 0, 
            "win_rate": 0, "profit_factor": 0, 
            "total_trades": len(self.trade_history), 
            "regime_performance": {},
            "total_anchors_set": self.total_anchors_set,
            "total_anchors_triggered": self.total_anchors_triggered,
            "anchor_trigger_rate": (self.total_anchors_triggered / self.total_anchors_set * 100) if self.total_anchors_set > 0 else 0,
            "realized_max_drawdown": 0.0 
        }
                   
        if len(self.trade_history) > 1:
            metrics['win_rate'] = sum(1 for t in self.trade_history if t.pnl_cash > 0) / len(self.trade_history) * 100
            gross_profit = sum(t.pnl_cash for t in self.trade_history if t.pnl_cash > 0) or 1
            gross_loss = abs(sum(t.pnl_cash for t in self.trade_history if t.pnl_cash < 0)) or 1
            metrics['profit_factor'] = gross_profit / gross_loss
            
            trade_df = pd.DataFrame([vars(t) for t in self.trade_history])
            trade_df['close_datetime'] = pd.to_datetime(trade_df['close_time'], unit='s')
            trade_df['daily_return'] = trade_df['pnl_percent'] / 100
            daily_returns = trade_df.set_index('close_datetime')['daily_return'].resample('D').sum() 
            
            if len(daily_returns) > 1:
                avg_daily_return = daily_returns.mean()
                std_daily_return = daily_returns.std()
                metrics['sharpe_ratio'] = (avg_daily_return / std_daily_return) * np.sqrt(365) if std_daily_return > 0 else 0 

                neg_daily_returns = daily_returns[daily_returns < 0]
                downside_dev_daily = neg_daily_returns.std() if len(neg_daily_returns) > 1 else 0
                metrics['sortino_ratio'] = (avg_daily_return / downside_dev_daily) * np.sqrt(365) if downside_dev_daily > 0 else (999.0 if avg_daily_return > 0 else 0)

            if self.max_drawdown > 0: metrics['calmar_ratio'] = total_return_pct / self.max_drawdown
            elif total_return_pct > 0: metrics['calmar_ratio'] = 999.0
            else: metrics['calmar_ratio'] = 0.0

            if len(self.realized_equity_curve) > 1:
                try:
                    realized_equity_series = pd.Series(
                        [val for ts, val in self.realized_equity_curve],
                        index=pd.to_datetime([ts for ts, val in self.realized_equity_curve], unit='s')
                    )
                except (TypeError, ValueError):
                    realized_equity_series = pd.Series(self.realized_equity_curve)

                realized_peak_series = realized_equity_series.expanding().max()
                realized_drawdown_series = (realized_equity_series - realized_peak_series) / realized_peak_series
                metrics['realized_max_drawdown'] = abs(realized_drawdown_series.min()) * 100

            regime_trades = defaultdict(list)
            for t in self.trade_history: regime_trades[t.entry_regime].append(t)
            for regime, trades in regime_trades.items():
                if not trades: continue 
                regime_pnl = sum(t.pnl_cash for t in trades)
                regime_wins = sum(1 for t in trades if t.pnl_cash > 0)
                regime_gross_profit = sum(t.pnl_cash for t in trades if t.pnl_cash > 0)
                regime_gross_loss = abs(sum(t.pnl_cash for t in trades if t.pnl_cash < 0))
                if regime_gross_loss < 0.01:
                    regime_profit_factor_str = f"{regime_gross_profit / 1.0:.2f}" if regime_gross_profit > 0 else "0.00"
                else:
                    regime_profit_factor_str = f"{regime_gross_profit / regime_gross_loss:.2f}"

                metrics["regime_performance"][regime] = {
                    "total_trades": len(trades), 
                    "win_rate": f"{(regime_wins / len(trades)) * 100 if trades else 0:.2f}%", 
                    "profit_factor": regime_profit_factor_str,
                    "net_pnl": f"${regime_pnl:,.2f}"
                }

        metrics["performance_log_str"] = self._get_performance_log_str(metrics) 
        if print_log: logger.info(metrics["performance_log_str"])
        return metrics

    def _get_performance_log_str(self, metrics: dict) -> str:
        winning_trades, losing_trades = [t.pnl_cash for t in self.trade_history if t.pnl_cash > 0], [t.pnl_cash for t in self.trade_history if t.pnl_cash < 0]
        wins, losses = len(winning_trades), len(losing_trades)
        avg_winner_usd, avg_loser_usd = np.mean(winning_trades) if wins > 0 else 0, np.mean(losing_trades) if losses > 0 else 0
        avg_pos_value = np.mean([t.kelly_risk_usd for t in self.trade_history]) if self.trade_history else 0
        open_longs, open_shorts = sum(1 for p in self.open_positions if p.is_long), sum(1 for p in self.open_positions if not p.is_long)

        is_backtest = self.connector.__class__.__name__ == 'DummyConnector'
        capital_source = "(Backtest Model A)"
        if not is_backtest:
            capital_source = "(Synced from exchange)"
        elif self.capital_allocation != 1.0:
            capital_source = "(Synced from Model C Engine)"

        log_msg = (f"\n{'='*20} 📊 PORTFOLIO STATUS | {format_time(self.clock.time())} {'='*20}\n" 
                   f"  📋 Core Metrics:\n"
                   f"    - Total Return:    {metrics['total_return_pct']:,.2f}%\n"
                   f"    - PnL:             ${metrics['pnl_cash']:,.2f}\n"
                   f"    - Current Capital: ${self.capital:,.2f} {capital_source}\n"
                   f"  📈 Risk & Performance:\n"
                   f"    - Peak Equity:     ${self.peak_capital:,.2f} {capital_source}\n"
                   f"    - Max Equity Drawdown:   -{metrics['max_drawdown']:.2f}% (incl. UPL)\n"
                   f"    - Max Realized Drawdown: -{metrics['realized_max_drawdown']:.2f}% (prop firm style)\n"
                   f"    - Sharpe Ratio:    {metrics['sharpe_ratio']:.2f} (Daily Approx.)\n"
                   f"    - Sortino Ratio:   {metrics['sortino_ratio']:.2f} (Daily Approx.)\n"
                   f"    - Calmar Ratio:    {metrics['calmar_ratio']:.2f}\n"
                   f"  ⚔️ Trade Stats (This Bot Only):\n" 
                   f"    - Total Trades:    {metrics['total_trades']}\n"
                   f"    - Win Rate:        {metrics.get('win_rate', 0):.2f}% ({wins} W / {losses} L)\n"
                   f"    - Profit Factor:   {metrics.get('profit_factor', 0):.2f}\n"
                   f"    - Avg Winner ($):  ${avg_winner_usd:,.2f}\n"
                   f"    - Avg Loser ($):   ${avg_loser_usd:,.2f}\n"
                   f"  ⚓ Anchor Stats (This Bot Only):\n" 
                   f"    - Total Anchors Set: {metrics['total_anchors_set']}\n"
                   f"    - Total Anchors Triggered:{metrics['total_anchors_triggered']}\n"
                   f"    - Trigger Rate:          {metrics['anchor_trigger_rate']:.2f}%\n"
                   f"  ⚖️ Risk Stats (This Bot Only):\n" 
                   f"    - Avg Position Value: ${avg_pos_value:,.2f}\n"
                   f"  ⚙️ Current State:\n"
                   f"    - Open Positions:  {len(self.open_positions)} ({open_longs} L, {open_shorts} S)\n"
                   f"    - Trend/Regime:    {self.trend_regime}\n"
                   f"    - Kelly Fraction:  {self.kelly_fraction:.3f} (Base: {self.config.KELLY_FRACTION:.3f} * Multi: {self.sizing_multiplier:.2%})\n"
                   f"    - Cap Allocation:  {self.capital_allocation:.2%}\n")

        if metrics["regime_performance"]:
            log_msg += f"  {'='*20} 🌪️ Regime Performance Breakdown (This Bot Only) {'='*11}\n" 
            log_msg += f"  {'Regime':<25} | {'Trades':>8} | {'Win Rate':>10} | {'Profit Factor':>15} | {'Net PnL':>15}\n"
            log_msg += f"  {'-'*78}\n"
            for regime, stats in sorted(metrics["regime_performance"].items()):
                log_msg += f"  {regime:<25} | {stats['total_trades']:>8} | {stats['win_rate']:>10} | {stats['profit_factor']:>15} | {stats['net_pnl']:>15}\n"

        log_msg += f"{'='*78}" 
        return log_msg