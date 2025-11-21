# deployment/ensemble_main.py
import sys
import os

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

import importlib.util
import time
import logging
import threading
import pandas as pd
import pandas_ta as ta
from collections import deque, defaultdict
from dotenv import load_dotenv
from datetime import datetime, timedelta

dotenv_path = os.path.join(project_root, 'api.env')
load_dotenv(dotenv_path=dotenv_path)

from typing import List, Dict, Callable

from deployment.ensemble_config import VENUES
from core.trading_bot_shared import MomentumAnchorBotShared
from core.core_types import SimulationClock, Tick, Position

logging.Formatter.converter = time.gmtime
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

log_timestamp = time.strftime("%Y-%m-%d_%H-%M-%S", time.gmtime())
log_dir = os.path.join(project_root, 'logs')
os.makedirs(log_dir, exist_ok=True) 

main_log_file = os.path.join(log_dir, f"ensemble_log_{log_timestamp}.log")
file_handler = logging.FileHandler(main_log_file)
file_handler.setFormatter(
    logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
)

logging.getLogger().addHandler(file_handler)

logger = logging.getLogger(__name__)

# --- CONFIGURABLE CONSTANTS ---
PORTFOLIO_SYNC_INTERVAL_SECONDS = 10  
INDICATOR_PRIMING_CANDLES = 1000 
MAX_DATAFRAME_LENGTH = 1500      
WATCHDOG_TIMEOUT_SECONDS = 60 

class LiveDataManager:
    """
    Manages all live data streams (ticks, 1m, 30m) for a single venue.
    Calculates indicators in real-time and routes data to the correct bots.
    """
    def __init__(self, bots: List[MomentumAnchorBotShared], connector):
        self.bots = bots
        self.connector = connector
        self.bots_by_symbol: Dict[str, List[MomentumAnchorBotShared]] = {}
        
        self.indicator_state_by_symbol: Dict[str, Dict] = {}
        self.dataframes: Dict[str, pd.DataFrame] = {}
        
        # --- NEW: Multi-Channel Liveness Tracker ---
        # Keys: "{symbol}_{stream_type}" (e.g., "BTC/USDT_1m")
        # Values: timestamp of last update
        self.liveness_tracker: Dict[str, float] = {} 
        # -------------------------------------------

        self.unique_1m_rsi_periods_by_symbol = defaultdict(set)
        self.unique_1m_vol_ma_periods_by_symbol = defaultdict(set)
        
        for bot in self.bots:
            if bot.symbol not in self.bots_by_symbol:
                self.bots_by_symbol[bot.symbol] = []
            self.bots_by_symbol[bot.symbol].append(bot)
            self.indicator_state_by_symbol[bot.symbol] = {} 
            
            self.unique_1m_rsi_periods_by_symbol[bot.symbol].add(bot.config.CONFIRMATION_RSI_PERIOD)
            self.unique_1m_vol_ma_periods_by_symbol[bot.symbol].add(bot.config.CONFIRMATION_VOLUME_MA_PERIOD)

            # Initialize liveness keys for this symbol
            self.liveness_tracker[f"{bot.symbol}_1m"] = time.time()
            self.liveness_tracker[f"{bot.symbol}_30m"] = time.time()
            self.liveness_tracker[f"{bot.symbol}_tick"] = time.time()

    def _parse_candle(self, candle_data) -> dict:
        if isinstance(candle_data, dict):
            return {
                "timestamp": pd.to_datetime(int(candle_data['start']), unit='ms'),
                "open": float(candle_data['open']),
                "high": float(candle_data['high']),
                "low": float(candle_data['low']),
                "close": float(candle_data['close']),
                "volume": float(candle_data['volume']),
            }
        else:
            return {
                "timestamp": pd.to_datetime(int(candle_data[0]), unit='ms'),
                "open": float(candle_data[1]),
                "high": float(candle_data[2]),
                "low": float(candle_data[3]),
                "close": float(candle_data[4]),
                "volume": float(candle_data[5]),
            }

    def prime_all_data(self):
        logger.info(f"Priming indicators for {len(self.bots_by_symbol.keys())} symbol(s)...")
        for symbol in self.bots_by_symbol.keys():
            symbol_bybit = symbol.replace('/', '')
            
            try:
                kline_1m = self.connector.session.get_kline(
                    category="linear", 
                    symbol=symbol_bybit, 
                    interval="1", 
                    limit=INDICATOR_PRIMING_CANDLES
                )['result']['list']
                
                kline_30m = self.connector.session.get_kline(
                    category="linear", 
                    symbol=symbol_bybit, 
                    interval="30", 
                    limit=INDICATOR_PRIMING_CANDLES
                )['result']['list']

                self.dataframes[f"{symbol}_1m"] = pd.DataFrame([self._parse_candle(c) for c in kline_1m]).sort_values(by="timestamp")
                self.dataframes[f"{symbol}_30m"] = pd.DataFrame([self._parse_candle(c) for c in kline_30m]).sort_values(by="timestamp")
                
                self._calculate_1m_indicators(symbol)
                
                initial_30m_anchor_data = self._calculate_30m_indicators(symbol)
                self.indicator_state_by_symbol[symbol]['30m_row'] = initial_30m_anchor_data
                logger.info(f"Successfully primed data for {symbol}")

                if initial_30m_anchor_data:
                    logger.info(f"Initializing bot state for {symbol}...")
                    for bot in self.bots_by_symbol[symbol]:
                        bot.initialize_state(initial_30m_anchor_data)
                else:
                    logger.error(f"Failed to get initial 30m data for {symbol}, bots will not start with active anchors.")
                
                # Initialize watchdog timestamps on success
                now = time.time()
                self.liveness_tracker[f"{symbol}_1m"] = now
                self.liveness_tracker[f"{symbol}_30m"] = now
                self.liveness_tracker[f"{symbol}_tick"] = now

            except Exception as e:
                logger.error(f"Failed to prime data for {symbol}: {e}. This symbol will not trade.")
                
    def _calculate_1m_indicators(self, symbol: str):
        df = self.dataframes.get(f"{symbol}_1m")
        if df is None or df.empty:
            return

        rsi_periods = self.unique_1m_rsi_periods_by_symbol[symbol]
        vol_ma_periods = self.unique_1m_vol_ma_periods_by_symbol[symbol]

        config = self.bots_by_symbol[symbol][0].config

        for p in rsi_periods:
            df[f'rsi_1m_{p}'] = ta.rsi(df['close'], length=p).shift(1)

        for p in vol_ma_periods:
            df[f'volume_ma_1m_{p}'] = df['volume'].rolling(window=p).mean().shift(1)

        macd_1m = ta.macd(df['close'], fast=config.CONFIRMATION_MACD_FAST_PERIOD, slow=config.CONFIRMATION_MACD_SLOW_PERIOD, signal=config.CONFIRMATION_MACD_SIGNAL_PERIOD)
        
        df[f'macd_1m_{config.CONFIRMATION_MACD_FAST_PERIOD}_{config.CONFIRMATION_MACD_SLOW_PERIOD}_{config.CONFIRMATION_MACD_SIGNAL_PERIOD}'] = macd_1m[f'MACD_{config.CONFIRMATION_MACD_FAST_PERIOD}_{config.CONFIRMATION_MACD_SLOW_PERIOD}_{config.CONFIRMATION_MACD_SIGNAL_PERIOD}'].shift(1)

    def _calculate_30m_indicators(self, symbol: str) -> dict:
        df = self.dataframes.get(f"{symbol}_30m")
        if df is None or df.empty or len(df) < 2:
            return {}
            
        config = self.bots_by_symbol[symbol][0].config
        
        df['short_ema'] = ta.ema(df['close'], length=config.SHORT_EMA_PERIOD)
        df['long_ema'] = ta.ema(df['close'], length=config.LONG_EMA_PERIOD)
        df['rsi'] = ta.rsi(df['close'], length=config.RSI_PERIOD)
        
        atr_val = ta.atr(df['high'], df['low'], df['close'], length=config.ATR_PERIOD)
        df['atr'] = atr_val 
        df[f'atr_{config.ATR_PERIOD}'] = atr_val 
        
        macd = ta.macd(df['close'], fast=config.MACD_FAST_PERIOD, slow=config.MACD_SLOW_PERIOD, signal=config.MACD_SIGNAL_PERIOD)
        df['macd'] = macd[f'MACD_{config.MACD_FAST_PERIOD}_{config.MACD_SLOW_PERIOD}_{config.MACD_SIGNAL_PERIOD}']
        bbands = ta.bbands(df['close'], length=config.BOLLINGER_PERIOD, std=config.BOLLINGER_STD_DEV)
        
        bbl_col = [col for col in bbands.columns if col.startswith('BBL_')][0]
        bbu_col = [col for col in bbands.columns if col.startswith('BBU_')][0]
        df['bb_lower'] = bbands[bbl_col]
        df['bb_upper'] = bbands[bbu_col]
        
        return df.iloc[-1].to_dict()

    def _get_trend_regime(self, data_row: dict, symbol: str) -> str:
        try:
            config = self.bots_by_symbol[symbol][0].config
        except (KeyError, IndexError):
            return "neutral"
        
        c = config
        try:
            short_ma = data_row.get('short_ema')
            long_ma = data_row.get('long_ema')
            rsi = data_row.get('rsi')
            macd = data_row.get('macd')
            bb_upper = data_row.get('bb_upper')
            bb_lower = data_row.get('bb_lower')
            current_price = data_row.get('close')
            
            if any(v is None for v in [short_ma, long_ma, rsi, macd, bb_upper, bb_lower, current_price]):
                return "neutral"
        except KeyError:
            return "neutral"

        trend = "neutral"
        if short_ma > long_ma * (1 + c.TREND_EMA_THRESHOLD_PERCENT / 100): trend = "up"
        elif short_ma < long_ma * (1 - c.TREND_EMA_THRESHOLD_PERCENT / 100): trend = "down"
        if rsi > c.RSI_OVERBOUGHT: trend = "strong_up"
        elif rsi < c.RSI_OVERSOLD: trend = "strong_down"
        if macd > 0 and "up" in trend: trend = "strong_up"
        elif macd < 0 and "down" in trend: trend = "strong_down"
        
        bb_width = (bb_upper - bb_lower) / current_price if current_price != 0 else 0
        regime = "trending" if bb_width > c.BB_REGIME_THRESHOLD else "ranging"
        
        return f"{trend}_{regime}"


    def start_all_streams(self):
        logger.info(f"Starting data streams for all symbols...")
        for symbol in self.bots_by_symbol.keys():
            try:
                self.connector.start_public_stream(
                    symbol=symbol,
                    on_tick_callback=self._on_tick
                )
                self.connector.start_kline_stream(
                    symbol=symbol,
                    interval="1",
                    on_kline_callback=self._on_1m_candle
                )
                self.connector.start_kline_stream(
                    symbol=symbol,
                    interval="30",
                    on_kline_callback=self._on_30m_candle
                )
                logger.info(f"Successfully subscribed to Public Tick, 1m, and 30m streams for {symbol}")
            except Exception as e:
                logger.error(f"Failed to start public streams for {symbol}: {e}")

        try:
            self.connector.start_private_stream(
                on_private_callback=self._on_private_message
            )
        except Exception as e:
            logger.error(f"CRITICAL: Failed to start private account stream: {e}")


    def _on_private_message(self, message: dict):
        # Private messages don't block a specific symbol's liveness, 
        # but we can track them if needed. For now, just log.
        logger.info(f"Received Private WebSocket Data: {message}")


    def _on_1m_candle(self, candle_data, symbol: str):
        # Update liveness tracker REGARDLESS of confirmation
        # This proves the socket is alive and sending data
        self.liveness_tracker[f"{symbol}_1m"] = time.time()
        
        if not candle_data.get('confirm', False):
            return

        try:
            candle_dict = self._parse_candle(candle_data)
            df = self.dataframes.get(f"{symbol}_1m", pd.DataFrame())
            
            new_df = pd.concat([df, pd.DataFrame([candle_dict])], ignore_index=True)
            if len(new_df) > MAX_DATAFRAME_LENGTH:
                new_df = new_df.iloc[-MAX_DATAFRAME_LENGTH:]
            
            self.dataframes[f"{symbol}_1m"] = new_df
            
            self._calculate_1m_indicators(symbol)
            
            updated_df = self.dataframes[f"{symbol}_1m"]

            if len(updated_df) > 1:
                self.indicator_state_by_symbol[symbol]['1m_row'] = updated_df.iloc[-1]
            if len(updated_df) > 2:
                self.indicator_state_by_symbol[symbol]['prev_1m_row'] = updated_df.iloc[-2]

        except Exception as e:
            logger.error(f"Error processing 1m candle for {symbol}: {e}")

    def _on_30m_candle(self, candle_data, symbol: str):
        # Update liveness tracker REGARDLESS of confirmation
        self.liveness_tracker[f"{symbol}_30m"] = time.time()
        
        if not candle_data.get('confirm', False):
            return

        try:
            candle_dict = self._parse_candle(candle_data)
            df = self.dataframes[f"{symbol}_30m"]

            new_df = pd.concat([df, pd.DataFrame([candle_dict])], ignore_index=True)
            if len(new_df) > MAX_DATAFRAME_LENGTH:
                new_df = new_df.iloc[-MAX_DATAFRAME_LENGTH:]
            self.dataframes[f"{symbol}_30m"] = new_df
            
            strategy_data_row = self._calculate_30m_indicators(symbol)
            if not strategy_data_row:
                return
            
            self.indicator_state_by_symbol[symbol]['30m_row'] = strategy_data_row

            new_regime = self._get_trend_regime(strategy_data_row, symbol)
            logger.info(f"🕯️  [DATA_MANAGER] | New 30m Candle for {symbol}. New Regime: {new_regime}")

            for bot in self.bots_by_symbol[symbol]:
                bot.update_strategy_on_30m(strategy_data_row)
                
        except Exception as e:
            logger.error(f"Error processing 30m candle for {symbol}: {e}")

    def _on_tick(self, tick: Tick, symbol: str):
        # Update liveness for ticks
        self.liveness_tracker[f"{symbol}_tick"] = time.time()
        
        try:
            state = self.indicator_state_by_symbol.get(symbol, {})
            current_1m_row = state.get('1m_row')   
            prev_1m_row = state.get('prev_1m_row')    
            current_30m_row = state.get('30m_row') 

            if not isinstance(current_1m_row, pd.Series) or not current_30m_row:
                return 

            # --- FIX: Data Stitch Validator (Staleness Check) ---
            ts_30m = current_30m_row.get('timestamp')
            ts_1m = current_1m_row.get('timestamp')
            
            if ts_30m and ts_1m:
                time_diff_seconds = (ts_1m - ts_30m).total_seconds()
                if time_diff_seconds > 3720: # 62 mins
                    # Rate limit logging for this warning
                    # (Optimization: Could add a timer here to avoid flooding logs)
                    return 
            # ----------------------------------------------------

            merged_current_row = pd.concat([pd.Series(current_30m_row), current_1m_row])
            atr_30m = current_30m_row.get('atr', 0.0)

            for bot in self.bots_by_symbol[symbol]:
                bot.clock.current_time = tick.timestamp
                bot.clock.current_price = tick.price
                
                bot.check_confirmation_on_tick(
                    tick,
                    merged_current_row, 
                    current_1m_row      
                )
                
                bot.check_exits_on_1m(tick.price, atr_30m)

        except Exception as e:
            logger.error(f"Error processing tick for {symbol}: {e}", exc_info=True)


class Venue:
    def __init__(self, name, config):
        self.name = name
        self.config = config
        self.connector = None
        self.bots: List[MomentumAnchorBotShared] = []
        self.is_ready = False
        
        self.master_peak_capital = 0.0
        self.data_manager: LiveDataManager = None
        self.portfolio_sync_thread: threading.Thread = None
        
        self.is_ready = self._setup()

    def _setup(self):
        connector_class = self.config["connector_class"]
        if not connector_class:
            logger.warning(f"Connector for venue '{self.name}' not available. It will be skipped.")
            return False

        connection_params = self.config["connection_params"]
        if 'api_key' in connection_params and not connection_params['api_key']:
             logger.error(f"API key for venue '{self.name}' is missing. Check .env.")
             return False

        self.connector = connector_class(**connection_params)

        live_balance = self.connector.get_wallet_balance(coin="USDT")
        if live_balance <= 0:
            logger.error(f"Venue '{self.name}' has zero/invalid balance ({live_balance}). Cannot start bots.")
            return False
        
        self.master_peak_capital = live_balance 

        num_bots_in_venue = len(self.config["strategies"])
        if num_bots_in_venue == 0:
            logger.warning(f"Venue '{self.name}' has no strategies assigned.")
            return False
            
        total_requested_capital = 0
        strategy_configs = []
        
        for strategy_path in self.config["strategies"]:
            spec = importlib.util.spec_from_file_location(strategy_path, strategy_path)
            strategy_module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(strategy_module)
            strategy_configs.append(strategy_module)
            total_requested_capital += strategy_module.INITIAL_CAPITAL

        logger.info(f"Venue '{self.name}' live balance is {live_balance:,.2f} USDT. Sizing allocation based on strategy weights.")

        for strategy_module in strategy_configs:
            clock = SimulationClock(start_time=time.time())
            bot_allocation = strategy_module.INITIAL_CAPITAL / total_requested_capital if total_requested_capital > 0 else 1.0 / num_bots_in_venue

            bot = MomentumAnchorBotShared(
                initial_capital=live_balance,
                simulation_clock=clock,
                config_module=strategy_module,
                connector=self.connector,
                silent=False
            )
            bot.capital_allocation = bot_allocation 
            bot.portfolio_initial_capital = live_balance 
            
            strategy_file_path = strategy_module.__file__
            strategy_name = strategy_file_path.split('/')[-1].replace('.py', '')
            bot.bot_id = f"{self.name}_{strategy_name}"
            
            bot_logger = logging.getLogger(bot.bot_id)
            bot_logger.setLevel(logging.INFO)
            bot_logger.propagate = False

            log_directory = os.path.join(project_root, 'logs', self.name)
            os.makedirs(log_directory, exist_ok=True)
            log_file_path = os.path.join(log_directory, f"{strategy_name}.log")
            bot_file_handler = logging.FileHandler(log_file_path)
            bot_file_handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
            if not bot_logger.hasHandlers():
                bot_logger.addHandler(bot_file_handler)
            bot.logger = bot_logger

            self.bots.append(bot)
            logger.info(f"Initialized bot: {bot.bot_id} on Venue: {self.name} for symbol {bot.symbol} with {bot.capital_allocation:.2%} allocation.")

        try:
            logger.info(f"[{self.name}] Hydrating bot state from exchange...")
            exchange_positions = self.connector.get_open_positions()
            
            for ex_pos in exchange_positions:
                symbol = ex_pos['symbol']
                qty = ex_pos['qty']
                side = ex_pos['side']
                entry_price = ex_pos['entry_price']
                
                found_bot = False
                for bot in self.bots:
                    if bot.symbol.replace('/', '') == symbol:
                        logger.warning(f"Found orphaned position for {symbol}: {side} {qty} @ {entry_price}. Adopting.")
                        
                        new_pos = Position(
                            entry_price=entry_price,
                            amount=qty,
                            is_long=(side == 'Buy')
                        )
                        new_pos.id = "orphaned_" + str(int(time.time()))
                        new_pos.entry_time = time.time()
                        new_pos.symbol = bot.symbol
                        
                        # --- FIX: Robust Orphan Adoption ---
                        new_pos.entry_regime = "adopted_orphan" 
                        # -----------------------------------
                        
                        bot.open_positions.append(new_pos)
                        found_bot = True
                
                if not found_bot:
                    logger.error(f"Found orphaned position for {symbol} but no bot is configured to trade it!")
                    
            logger.info(f"[{self.name}] State hydration complete. Found {len(exchange_positions)} open positions.")
            
        except Exception as e:
            logger.error(f"CRITICAL: Failed to hydrate bot state: {e}. Positions may be orphaned.", exc_info=True)
        
        self.data_manager = LiveDataManager(self.bots, self.connector)
        return True

    def connect_and_run_streams(self):
        if self.connector:
            logger.info(f"Connecting Venue: {self.name}...")
            self.connector.connect()
            self.data_manager.prime_all_data()
            self.data_manager.start_all_streams()

    def start_portfolio_sync(self):
        logger.info(f"Starting portfolio sync thread for venue '{self.name}'...")
        self.portfolio_sync_thread = threading.Thread(target=self._portfolio_sync_loop, daemon=True)
        self.portfolio_sync_thread.start()
        
    def _update_dynamic_allocations(self):
        weights = [bot.sizing_multiplier for bot in self.bots]
        total_weight = sum(weights)
        if total_weight <= 0:
            num_bots = len(self.bots)
            if num_bots > 0:
                equal_allocation = 1.0 / num_bots
                for bot in self.bots:
                    bot.capital_allocation = equal_allocation
            return

        for i, bot in enumerate(self.bots):
            new_allocation = weights[i] / total_weight
            bot.capital_allocation = new_allocation

    def _portfolio_sync_loop(self):
        last_risk_update_minute = -1
        while True:
            try:
                realized_capital = self.connector.get_wallet_balance("USDT")
                unrealized_pnl = 0.0
                try:
                    open_positions = self.connector.get_open_positions()
                    for pos in open_positions:
                        symbol = pos['symbol']
                        entry_price = float(pos['entry_price'])
                        qty = float(pos['qty'])
                        side = pos['side']
                        mark_price = self.connector.get_mark_price(symbol)
                        if mark_price > 0:
                            if side == 'Buy': unrealized_pnl += (mark_price - entry_price) * qty
                            else: unrealized_pnl += (entry_price - mark_price) * qty
                except Exception as e:
                    logger.warning(f"Could not calculate UPL on this cycle: {e}. UPL will be 0.")
                    unrealized_pnl = 0.0

                current_total_equity = realized_capital + unrealized_pnl
                self.master_peak_capital = max(self.master_peak_capital, current_total_equity)

                for bot in self.bots:
                    bot.capital = realized_capital
                    bot.unrealized_pnl = unrealized_pnl
                    bot.peak_capital = self.master_peak_capital
                
                current_dt = datetime.now()
                if current_dt.minute % 5 == 0 and current_dt.minute != last_risk_update_minute:
                    logger.info("❤️ [HEARTBEAT] Updating Risk Parameters & Allocations...")
                    for bot in self.bots:
                        bot.update_risk_parameters()
                    self._update_dynamic_allocations()
                    last_risk_update_minute = current_dt.minute
                
                alloc_strs = []
                for bot in self.bots:
                    try: bot_name = bot.bot_id.split(f"{self.name}_")[-1]
                    except Exception: bot_name = bot.bot_id
                    alloc_strs.append(f"{bot_name}: {bot.capital_allocation:.1%}")
                alloc_summary = ", ".join(alloc_strs)

                logger.info(
                    f"💰 [PORTFOLIO_SYNC] | "
                    f"Equity: ${current_total_equity:,.2f} "
                    f"(Realized: ${realized_capital:,.2f}, UPL: ${unrealized_pnl:,.2f}) | "
                    f"Peak: ${self.master_peak_capital:,.2f} | "
                    f"Allocations: [{alloc_summary}]"
                )

            except Exception as e:
                logger.error(f"CRITICAL: Portfolio sync CYCLE failed for venue '{self.name}': {e}")
            
            time.sleep(PORTFOLIO_SYNC_INTERVAL_SECONDS)

    def disconnect(self):
        if self.connector:
            logger.info(f"Disconnecting Venue: {self.name}...")
            self.connector.disconnect()

class EnsembleTrader:
    def __init__(self, venues_config: Dict):
        self.venues: List[Venue] = []
        for name, config in venues_config.items():
            if config.get("mode") in ["live", "paper_trading"]:
                venue = Venue(name, config)
                if venue.is_ready:
                    self.venues.append(venue)

    def run(self):
        if not self.venues:
            logger.error("No valid venues configured or ready to run.")
            return

        try:
            for venue in self.venues:
                logger.info(f"--- Starting Venue: {venue.name} ---")
                venue.connect_and_run_streams()
                venue.start_portfolio_sync()

            logger.info("All venues running. Waiting for exit signal (Ctrl+C).")
            
            # --- FIX: Multi-Channel Watchdog Main Loop ---
            while True:
                time.sleep(1) 
                
                current_time = time.time()
                for venue in self.venues:
                    if not venue.data_manager: continue
                    
                    # Iterate over all tracked streams (e.g., BTC/USDT_1m, BTC/USDT_30m)
                    for stream_key, last_ts in list(venue.data_manager.liveness_tracker.items()):
                        if current_time - last_ts > WATCHDOG_TIMEOUT_SECONDS:
                            logger.error(f"🚨 [WATCHDOG] Stream '{stream_key}' died! Last update: {int(current_time - last_ts)}s ago. Triggering Venue Reconnect...")
                            
                            try:
                                venue.connector.disconnect()
                                time.sleep(2)
                                venue.connector.connect()
                                venue.data_manager.start_all_streams() 
                                
                                # Reset all keys to now to prevent infinite loops
                                now = time.time()
                                for k in venue.data_manager.liveness_tracker.keys():
                                    venue.data_manager.liveness_tracker[k] = now
                                    
                                logger.info(f"✅ [WATCHDOG] Venue '{venue.name}' reconnected successfully.")
                                break # Break inner loop to restart venue monitoring
                            except Exception as e:
                                 logger.error(f"❌ [WATCHDOG] Reconnect failed: {e}")
                                 # Update timestamp to prevent log flooding, hope next cycle fixes it
                                 venue.data_manager.liveness_tracker[stream_key] = time.time() 
                # -------------------------------

        except KeyboardInterrupt:
            logger.info("Shutdown signal received (Ctrl+C).")
        except Exception as e:
            logger.error(f"Fatal error in main loop: {e}", exc_info=True)
        finally:
            # --- FIX: Shutdown Protocol (Panic Close) ---
            logger.warning("🛑 [SHUTDOWN PROTOCOL] Initiating Emergency Shutdown...")
            for venue in self.venues:
                if not venue.bots: continue
                
                logger.warning(f"Checking open positions for venue {venue.name}...")
                for bot in venue.bots:
                    if bot.open_positions:
                        logger.warning(f"⚠️  Closing {len(bot.open_positions)} open positions for bot {bot.bot_id}...")
                        for pos in list(bot.open_positions):
                            try:
                                close_price = bot.clock.current_price if bot.clock.current_price > 0 else pos.entry_price
                                bot.close_position(pos, close_price, "Emergency Shutdown")
                                time.sleep(0.2) 
                            except Exception as e:
                                logger.error(f"Failed to panic close position {pos.id}: {e}")
            # --------------------------------------------
            
            logger.info("Disconnecting all venues...")
            for venue in self.venues:
                venue.disconnect()
            logger.info("Ensemble trader shut down cleanly.")


if __name__ == "__main__":
    dotenv_path = os.path.join(project_root, 'api.env')
    load_dotenv(dotenv_path=dotenv_path)

    if len(sys.argv) < 2:
        print("ERROR: Please specify which venue to run.")
        print("Usage: python deployment/ensemble_main.py <venue_name>")
        print("\nAvailable venues in ensemble_config.py:")
        for name, config in VENUES.items():
            if config.get("mode") in ["live", "paper_trading"]:
                print(f"  - {name}")
        sys.exit(1)

    venue_to_run_name = sys.argv[1]
    venues_to_run = {name: config for name, config in VENUES.items() if name == venue_to_run_name}
    
    if not venues_to_run:
        print(f"ERROR: Venue '{venue_to_run_name}' not found in ensemble_config.py.")
        sys.exit(1)

    ensemble = EnsembleTrader(venues_to_run)
    ensemble.run()