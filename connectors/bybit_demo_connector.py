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
    --- MODIFIED: This now points to the MAINNET to use the UTA Demo Account ---
    """

    def __init__(self, api_key: str, api_secret: str, **kwargs):
        self.logger = logging.getLogger(__name__)
        self.api_key = api_key
        self.api_secret = api_secret
        self.ws_public = None  # <-- NEW: Renamed for public stream
        self.ws_private = None # <-- NEW: Added for private stream
        
        # --- *** MODIFICATION: Changed to MAINNET DEMO *** ---
        self.logger.warning("Initializing Bybit MAINNET DEMO Connector. This will trade on your UTA Demo Account.")
        
        # HTTP session is hardcoded to testnet=False
        self.session = HTTP(
            testnet=False, # <-- Keep this as False
            demo=True,     # <-- THIS IS THE FIX 
            api_key=self.api_key,
            api_secret=self.api_secret,
            recv_window=10000 
        )
        # --- *** END MODIFICATION *** ---

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

        # Define precision (e.g., 5 decimal places for BTC).
        # This should be dynamic or a config lookup in a real system.
        precision = 3
        factor = 10 ** precision

        # We use math.floor to round DOWN to the nearest valid precision
        rounded_qty = math.floor(qty * factor) / factor

        # Convert the correctly rounded float to a string
        qty_str = f"{rounded_qty:.{precision}f}"

        self.logger.info(f"PLACING MAINNET DEMO ORDER: {side_bybit} {qty} {symbol_bybit}")
        
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
                time.sleep(2) # Allow time for the market order to fill
                
                trade_history = self.session.get_order_history(category="linear", orderId=order_id, limit=1)
                
                if trade_history and trade_history['result']['list']:
                    trade = trade_history['result']['list'][0]
                    entry_price = float(trade['avgPrice'])
                    commission = float(trade.get('execFee', 0.0))
                    
                    self.logger.info(f"Successfully placed demo order {order_id} at avg price {entry_price}")
                    return {
                        "success": True, 
                        "trade_id": order_id, 
                        "entry_price": entry_price,
                        "commission": commission
                    }
                else:
                    self.logger.error(f"Placed order {order_id} but could not fetch execution price.")
                    return {"success": False, "error": "Could not fetch execution price", "commission": 0.0}
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
        symbol_bybit = position.symbol.replace('/', '') # Assuming position object has a symbol
        side_to_close = "Sell" if position.is_long else "Buy"
        
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
                
                if trade_history and trade_history['result']['list']:
                    trade = trade_history['result']['list'][0]
                    close_price = float(trade['avgPrice'])
                    
                    # --- *** MODIFIED: FULL PNL & COMMISSION LOGIC *** ---
                    exit_commission = float(trade.get('execFee', 0.0))
                    entry_commission = getattr(position, 'entry_commission', 0.0) # Assumes bot stored this
                    total_commission = entry_commission + exit_commission
                    
                    gross_pnl = (close_price - position.entry_price) * position.amount if position.is_long else (position.entry_price - close_price) * position.amount
                    net_pnl = gross_pnl - total_commission # <-- This is the NET PNL

                    self.logger.info(f"Successfully closed demo position {position.id} at avg price {close_price}")
                    return {
                        "success": True, 
                        "close_price": close_price, 
                        "pnl": net_pnl, # <-- Return NET PNL
                        "commission": total_commission
                    }
                    # --- *** END MODIFIED *** ---
                else:
                    return {"success": False, "error": "Could not fetch execution price for closing order", "pnl": 0.0, "commission": 0.0}
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
        """
        Starts a WebSocket connection for PUBLIC kline (candle) data.
        This uses the ws_public client, which is connected to the
        LIVE endpoint (demo=False) as required.
        """
        if not self.ws_public:
            # This check is crucial. We must ensure the public stream is ready.
            try:
                self.start_public_stream(symbol, lambda tick, s: None) # A dummy callback to init
                self.logger.info("Implicitly initialized ws_public for kline stream.")
            except Exception as e:
                self.logger.error(f"Failed to initialize ws_public for kline stream: {e}")
                return

        self.logger.info(f"Subscribing to PUBLIC kline stream: {symbol} ({interval} interval)...")
        
        symbol_bybit = symbol.replace('/', '')
        
        # Use the pybit library's kline_stream method on the public client
        self.ws_public.kline_stream(
            interval=interval,
            symbol=symbol_bybit,
            callback=lambda msg: on_kline_callback(msg['data'][0], symbol) # Pass back the candle data and symbol
        )

    def start_public_stream(self, symbol: str, on_tick_callback: Callable[[Tick, str], None]):
        """
        Starts a WebSocket connection for PUBLIC data (trades).
        This must connect to the LIVE endpoint (demo=False).
        """
        self.logger.info(f"Initializing Bybit Mainnet WebSocket for PUBLIC trades on {symbol}...")
        self.ws_public = WebSocket(
            testnet=False,
            demo=False,    # <-- Must be False to get live public data [cite: 109, 119]
            channel_type="linear"
        )
        
        symbol_bybit = symbol.replace('/', '')
        
        self.ws_public.trade_stream(
            symbol=symbol_bybit,
            callback=lambda msg: self._handle_websocket_message(msg, on_tick_callback, symbol)
        )
        self.logger.info(f"WebSocket subscribed to PUBLIC trades for {symbol_bybit}")

    def start_private_stream(self, on_private_callback: Callable[[dict], None]):
        """
        Starts a WebSocket connection for PRIVATE data (orders, positions).
        This must connect to the DEMO endpoint (demo=True).
        """
        self.logger.info(f"Initializing Bybit Mainnet Demo WebSocket for PRIVATE account data...")
        self.ws_private = WebSocket(
            testnet=False,
            demo=True,     # <-- Correct
            api_key=self.api_key,
            api_secret=self.api_secret,
            channel_type="private" # <-- THIS IS THE FINAL FIX
        )
        
        # Subscribe to private streams
        self.ws_private.position_stream(callback=on_private_callback)
        self.ws_private.order_stream(callback=on_private_callback)
        
        self.logger.info(f"WebSocket subscribed to PRIVATE position and order streams.")

    def _handle_websocket_message(self, msg: Dict, on_tick_callback: Callable, symbol: str):
        """
        Processes messages from the PUBLIC WebSocket and formats them into a standard Tick object.
        """
        try:
            if 'data' in msg:
                for trade in msg['data']:
                    tick = Tick(
                        timestamp=float(trade['T']) / 1000,
                        price=float(trade['p']),
                        volume=float(trade['v']),
                        candle_volume=0
                    )
                    # Pass both the tick and the original symbol to the callback
                    on_tick_callback(tick, symbol)
        except Exception as e:
            self.logger.error(f"Error processing WebSocket message: {e}")

    def get_wallet_balance(self, coin: str) -> float:
        """
        --- *** MODIFIED: THIS IS THE CRITICAL FIX *** ---
        Fetches the TOTAL EQUITY of the Unified Trading Account.
        The demo account is a UTA, so we ONLY check for "UNIFIED".
        """
        try:
            response = self.session.get_wallet_balance(
                accountType="UNIFIED" 
            )
            
            if response and response.get('retCode') == 0:
                
                # --- THIS IS THE FIX ---
                # The data is inside response['result']['list'][0]
                result_data = response.get('result', {})
                result_list = result_data.get('list', [])
                
                if result_list and len(result_list) > 0:
                    account_info = result_list[0]
                    balance = float(account_info['totalEquity'])
                    self.logger.info(f"Successfully fetched total equity: {balance} USD (display value)")
                    return balance
                else:
                    self.logger.error("API call successful but 'list' was empty in response.")
                    return 0.0
                # --- END FIX ---

            else:
                self.logger.error(f"Failed to fetch wallet balance: {response.get('retMsg', 'Unknown error')}")
                return 0.0
        except Exception as e:
            self.logger.error(f"An exception occurred while fetching wallet balance: {e}")
            return 0.0

    def get_mark_price(self, symbol: str) -> float:
        """
        Fetches the current mark price for a specific symbol.
        """
        symbol_bybit = symbol.replace('/', '')
        try:
            response = self.session.get_tickers(
                category="linear",
                symbol=symbol_bybit
            )
            if response and response.get('retCode') == 0:
                price = float(response['result']['list'][0]['markPrice'])
                return price
            else:
                self.logger.warning(f"Could not fetch mark price for {symbol}: {response.get('retMsg')}")
                return 0.0
        except Exception as e:
            self.logger.error(f"Exception fetching mark price for {symbol}: {e}")
            return 0.0

    def get_open_positions(self, symbol: str = None) -> List[Dict[str, Any]]:
        """
        Fetches all open positions from the Bybit Unified Trading Account.
        """
        # --- MODIFIED: Must specify settleCoin for UTA ---
        params = {"category": "linear", "settleCoin": "USDT"} # Or "USDC" if that's your collateral
        if symbol:
            params["symbol"] = symbol.replace('/', '')

        try:
            response = self.session.get_positions(**params)
            
            if response and response.get('retCode') == 0:
                positions = response['result']['list']
                
                # Filter for positions that are actually open
                open_positions = [p for p in positions if float(p.get('size', 0.0)) > 0]
                
                # Standardize the output
                standardized_list = []
                for pos in open_positions:
                    standardized_list.append({
                        "symbol": pos['symbol'],
                        "entry_price": float(pos['avgPrice']),
                        "qty": float(pos['size']),
                        "side": pos['side'] # 'Buy' or 'Sell'
                    })
                return standardized_list
            else:
                self.logger.error(f"Could not fetch open positions: {response.get('retMsg')}")
                return []
        except Exception as e:
            self.logger.error(f"Exception fetching open positions: {e}")
            return []
    # --- *** END OF NEW METHODS *** ---