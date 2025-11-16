# connectors/base_connector.py

from abc import ABC, abstractmethod
from typing import Dict, Any, Callable, List
from core.core_types import Position, Tick

class BaseConnector(ABC):
    """
    Abstract Base Class for all platform connectors.
    Defines the standard interface for the trading bot to interact with any exchange.
    """
    
    @abstractmethod
    def connect(self):
        pass

    @abstractmethod
    def disconnect(self):
        pass

    @abstractmethod
    def place_order(self, symbol: str, side: str, order_type: str, qty: float, price: float = None, stop_loss: float = None) -> Dict[str, Any]:
        """
        Places an order on the exchange.
        
        MUST return a dictionary with at least:
        {
            "success": bool,
            "trade_id": str,
            "entry_price": float,
            "commission": float,
            "error": str (if success=False)
        }
        """
        pass

    @abstractmethod
    def close_position(self, position: Position) -> Dict[str, Any]:
        """
        Closes a position on the exchange.
        
        MUST return a dictionary with at least:
        {
            "success": bool,
            "close_price": float,
            "pnl": float (This MUST be NET PnL, after fees),
            "commission": float,
            "error": str (if success=False)
        }
        """
        pass

    @abstractmethod
    def modify_stop_loss(self, symbol: str, trade_id: str, new_stop_price: float) -> bool:
        pass
    
    # --- *** MODIFIED: Replaced start_data_stream with bifurcated methods *** ---

    @abstractmethod
    def start_public_stream(self, symbol: str, on_tick_callback: Callable[[Tick, str], None]):
        """
        Starts the flow of PUBLIC market data (e.g., trades) for a symbol.
        """
        pass
        
    @abstractmethod
    def start_kline_stream(self, symbol: str, interval: str, on_kline_callback: Callable[[dict, str], None]):
        """
        Starts the flow of PUBLIC kline (candle) data for a symbol.
        """
        pass

    @abstractmethod
    def start_private_stream(self, on_private_callback: Callable[[dict], None]):
        """
        Starts the flow of PRIVATE account data (e.g., positions, orders).
        """
        pass

    # --- *** END MODIFIED *** ---

    @abstractmethod
    def get_wallet_balance(self, coin: str) -> float:
        """Fetches the available balance for a specific coin."""
        pass
        
    @abstractmethod
    def get_open_positions(self, symbol: str = None) -> List[Dict[str, Any]]:
        """
        Fetches all open positions from the exchange.
        """
        pass

    @abstractmethod
    def get_mark_price(self, symbol: str) -> float:
        """
        Fetches the current mark price for a specific symbol.
        """
        pass