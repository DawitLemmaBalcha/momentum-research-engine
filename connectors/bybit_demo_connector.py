# connectors/bybit_demo_connector.py

import logging
import time
from pybit.unified_trading import HTTP, WebSocket
from .base_connector import BaseConnector
from typing import Dict, Any, Callable, List
from core.core_types import Tick, Position
import math

class BybitDemoConnector(BaseConnector):
    """
    A dedicated, fully functional connector for paper trading on the Bybit Mainnet Demo.
    """

    def __init__(self, api_key: str, api_secret: str, **kwargs):
        self.logger = logging.getLogger(__name__)
        self.api_key = api_key
        self.api_secret = api_secret
        self.ws_public = None
        self.ws_private = None
        
        self.logger.warning("Initializing Bybit MAINNET DEMO Connector. This will trade on your UTA Demo Account.")
        
        self.session = HTTP(
            testnet=False,
            demo=True,
            api_key=self.api_key,
            api_secret=self.api_secret,
            recv_window=10000 
        )

    def connect(self):
        self.logger.info("Connecting to Bybit Mainnet (for Demo Account)...")
        pass

    def disconnect(self):
        self.logger.info("Disconnecting from Bybit Mainnet...")
        if self.ws_public:
            self.ws_public.exit()
        if self.ws_private:
            self.ws_private.exit()

    def place_order(self, symbol: str, side: str, order_type: str, qty: float, price: float = None, stop_loss: float = None) -> Dict[str, Any]:
        """
        Places a market order on Bybit Demo.
        """
        side_bybit = "Buy" if side == "buy" else "Sell"
        symbol_bybit = symbol.replace('/', '')

        # --- QUANTITY PRECISION FIX ---
        # Define precision (e.g., 3 decimal places for BTC).
        # In a real deployment, fetch this from exchange info (lotSizeFilter).
        precision = 3 
        factor = 10 ** precision
        
        # Round down to ensure we don't exceed balance or limits
        rounded_qty = math.floor(qty * factor) / factor
        qty_str = f"{rounded_qty:.{precision}f}"

        self.logger.info(f"PLACING MAINNET DEMO ORDER: {side_bybit} {qty_str} {symbol_bybit} (Original: {qty})")
        
        try:
            response = self.session.place_order(
                category="linear",
                symbol=symbol_bybit,
                side=side_bybit,
                orderType="Market",
                qty=qty_str,
            )
            
            if response and response.get('retCode') == 0:
                order_id = response['result']['orderId']
                time.sleep(1) # Short sleep to allow fill
                
                # Try to fetch fill price
                trade_history = self.session.get_order_history(category="linear", orderId=order_id, limit=1)
                
                entry_price = 0.0
                commission = 0.0
                
                if trade_history and trade_history['result']['list']:
                    trade = trade_history['result']['list'][0]
                    entry_price = float(trade.get('avgPrice', 0.0))
                    commission = float(trade.get('execFee', 0.0))
                
                # Fallback if avgPrice is not yet available (unlikely for market order but possible)
                if entry_price == 0.0:
                    entry_price = price if price else 0.0 
                    self.logger.warning(f"Order {order_id} placed but avgPrice unavailable immediately. Using fallback.")

                self.logger.info(f"Successfully placed demo order {order_id} at avg price {entry_price}")
                
                return {
                    "success": True, 
                    "trade_id": order_id, 
                    "entry_price": entry_price,
                    "commission": commission,
                    "qty": float(qty_str) # <--- CRITICAL FIX: Return the ACTUAL ordered quantity
                }
            else:
                error_msg = response.get('retMsg', 'Unknown error')
                self.logger.error(f"Failed to place demo order: {error_msg}")
                return {"success": False, "error": error_msg, "commission": 0.0}

        except Exception as e:
            self.logger.error(f"An exception occurred while placing demo order: {e}")
            return {"success": False, "error": str(e), "commission": 0.0}

    def close_position(self, position: Position) -> Dict[str, Any]:
        """
        Closes a position on Bybit Demo by placing an opposing market order.
        """
        symbol_bybit = position.symbol.replace('/', '')
        side_to_close = "Sell" if position.is_long else "Buy"
        
        # Use the amount stored in the position (which should now be correct)
        self.logger.info(f"CLOSING MAINNET DEMO POSITION: {side_to_close} {position.amount} {symbol_bybit}")

        try:
            response = self.session.place_order(
                category="linear",
                symbol=symbol_bybit,
                side=side_to_close,
                orderType="Market",
                qty=str(position.amount),
                reduce_only=True
            )
            
            if response and response.get('retCode') == 0:
                order_id = response['result']['orderId']
                time.sleep(0.5)
                
                trade_history = self.session.get_order_history(category="linear", orderId=order_id, limit=1)
                
                close_price = 0.0
                exit_commission = 0.0
                
                if trade_history and trade_history['result']['list']:
                    trade = trade_history['result']['list'][0]
                    close_price = float(trade.get('avgPrice', 0.0))
                    exit_commission = float(trade.get('execFee', 0.0))
                
                # Fallback
                if close_price == 0.0:
                    self.logger.warning("Close order placed but avgPrice unavailable. PnL calc may be inaccurate.")
                    close_price = position.entry_price # Prevent crash

                entry_commission = getattr(position, 'entry_commission', 0.0)
                total_commission = entry_commission + exit_commission
                
                gross_pnl = (close_price - position.entry_price) * position.amount if position.is_long else (position.entry_price - close_price) * position.amount
                net_pnl = gross_pnl - total_commission

                self.logger.info(f"Successfully closed demo position {position.id} at avg price {close_price}")
                return {
                    "success": True, 
                    "close_price": close_price, 
                    "pnl": net_pnl,
                    "commission": total_commission
                }
            else:
                error_msg = response.get('retMsg', 'Unknown error')
                self.logger.error(f"Failed to close demo position: {error_msg}")
                return {"success": False, "error": error_msg, "pnl": 0.0, "commission": 0.0}
                
        except Exception as e:
            self.logger.error(f"An exception occurred while closing demo position: {e}")
            return {"success": False, "error": str(e), "pnl": 0.0, "commission": 0.0}

    def modify_stop_loss(self, symbol: str, trade_id: str, new_stop_price: float) -> bool:
        self.logger.info(f"MODIFYING DEMO STOP for {trade_id} to {new_stop_price} (not implemented)")
        return True

    def start_kline_stream(self, symbol: str, interval: str, on_kline_callback: Callable[[dict, str], None]):
        if not self.ws_public:
            try:
                self.start_public_stream(symbol, lambda tick, s: None)
                self.logger.info("Implicitly initialized ws_public for kline stream.")
            except Exception as e:
                self.logger.error(f"Failed to initialize ws_public for kline stream: {e}")
                return

        self.logger.info(f"Subscribing to PUBLIC kline stream: {symbol} ({interval} interval)...")
        symbol_bybit = symbol.replace('/', '')
        self.ws_public.kline_stream(
            interval=interval,
            symbol=symbol_bybit,
            callback=lambda msg: on_kline_callback(msg['data'][0], symbol)
        )

    def start_public_stream(self, symbol: str, on_tick_callback: Callable[[Tick, str], None]):
        self.logger.info(f"Initializing Bybit Mainnet WebSocket for PUBLIC trades on {symbol}...")
        self.ws_public = WebSocket(
            testnet=False,
            demo=False, # Must be False for live public data
            channel_type="linear"
        )
        symbol_bybit = symbol.replace('/', '')
        self.ws_public.trade_stream(
            symbol=symbol_bybit,
            callback=lambda msg: self._handle_websocket_message(msg, on_tick_callback, symbol)
        )

    def start_private_stream(self, on_private_callback: Callable[[dict], None]):
        self.logger.info(f"Initializing Bybit Mainnet Demo WebSocket for PRIVATE account data...")
        self.ws_private = WebSocket(
            testnet=False,
            demo=True, # Must be True for demo private data
            api_key=self.api_key,
            api_secret=self.api_secret,
            channel_type="private"
        )
        self.ws_private.position_stream(callback=on_private_callback)
        self.ws_private.order_stream(callback=on_private_callback)

    def _handle_websocket_message(self, msg: Dict, on_tick_callback: Callable, symbol: str):
        try:
            if 'data' in msg:
                for trade in msg['data']:
                    tick = Tick(
                        timestamp=float(trade['T']) / 1000,
                        price=float(trade['p']),
                        volume=float(trade['v']),
                        candle_volume=0
                    )
                    on_tick_callback(tick, symbol)
        except Exception as e:
            self.logger.error(f"Error processing WebSocket message: {e}")

    def get_wallet_balance(self, coin: str) -> float:
        try:
            response = self.session.get_wallet_balance(accountType="UNIFIED")
            if response and response.get('retCode') == 0:
                result_data = response.get('result', {})
                result_list = result_data.get('list', [])
                if result_list and len(result_list) > 0:
                    account_info = result_list[0]
                    balance = float(account_info['totalEquity'])
                    self.logger.info(f"Successfully fetched total equity: {balance} USD")
                    return balance
            self.logger.error("Failed to fetch wallet balance.")
            return 0.0
        except Exception as e:
            self.logger.error(f"An exception occurred while fetching wallet balance: {e}")
            return 0.0

    def get_mark_price(self, symbol: str) -> float:
        symbol_bybit = symbol.replace('/', '')
        try:
            response = self.session.get_tickers(category="linear", symbol=symbol_bybit)
            if response and response.get('retCode') == 0:
                return float(response['result']['list'][0]['markPrice'])
            return 0.0
        except Exception as e:
            self.logger.error(f"Exception fetching mark price: {e}")
            return 0.0

    def get_open_positions(self, symbol: str = None) -> List[Dict[str, Any]]:
        params = {"category": "linear", "settleCoin": "USDT"}
        if symbol: params["symbol"] = symbol.replace('/', '')
        try:
            response = self.session.get_positions(**params)
            if response and response.get('retCode') == 0:
                positions = response['result']['list']
                open_positions = [p for p in positions if float(p.get('size', 0.0)) > 0]
                standardized_list = []
                for pos in open_positions:
                    standardized_list.append({
                        "symbol": pos['symbol'],
                        "entry_price": float(pos['avgPrice']),
                        "qty": float(pos['size']),
                        "side": pos['side']
                    })
                return standardized_list
            return []
        except Exception as e:
            self.logger.error(f"Exception fetching open positions: {e}")
            return []