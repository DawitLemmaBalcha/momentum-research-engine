# connectors/dummy_connector.py

from abc import ABC
from typing import Dict, Any, Callable, List
from core.core_types import SimulationClock, Tick, Position

class DummyConnector(ABC):
    """
    A simulated connector for backtesting purposes.
    Updated to match the full BaseConnector interface.
    """
    SLIPPAGE_MULTIPLIER = 0.001
    
    def __init__(self, clock: SimulationClock, params: dict, engine=None):
        self.clock, self.params, self._trade_id_counter = clock, params, 0
        self.engine = engine # Engine holds master capital
    
    def connect(self): 
        pass
    def disconnect(self): 
        pass

    def place_order(self, symbol: str, side: str, order_type: str, qty: float, price: float = None, stop_loss: float = None) -> Dict[str, Any]:
        self._trade_id_counter += 1
        base_entry_price = price if price is not None else self.clock.current_price
        slipped_entry_price = base_entry_price * (1 + self.SLIPPAGE_MULTIPLIER) if side == 'buy' else base_entry_price * (1 - self.SLIPPAGE_MULTIPLIER)
        
        # --- MODIFIED: Calculate and return commission ---
        commission_rate = self.params.get('BYBIT_TAKER_FEE', 0.00055)
        entry_commission = (slipped_entry_price * qty) * commission_rate
        
        return {
            "success": True, 
            "trade_id": f"dummy_{self._trade_id_counter}", 
            "entry_price": slipped_entry_price,
            "commission": entry_commission # <-- ADDED
        }

    def close_position(self, position: Position) -> Dict[str, Any]:
        base_close_price = self.clock.current_price
        slipped_close_price = base_close_price * (1 - self.SLIPPAGE_MULTIPLIER) if position.is_long else base_close_price * (1 + self.SLIPPAGE_MULTIPLIER)
        gross_pnl = (slipped_close_price - position.entry_price) * position.amount if position.is_long else (position.entry_price - slipped_close_price) * position.amount
        
        commission_rate = self.params.get('BYBIT_TAKER_FEE', 0.00055)
        
        # --- MODIFIED: Get entry commission from position object if it was stored ---
        # We assume the bot stores this. If not, we calculate both.
        entry_commission = getattr(position, 'entry_commission', (position.entry_price * position.amount) * commission_rate)
        exit_commission = (slipped_close_price * position.amount) * commission_rate
        total_commission = entry_commission + exit_commission
        
        net_pnl = gross_pnl - total_commission
        
        if self.engine:
            # This line adds the NET P&L to the engine's master variable
            self.engine.master_realized_capital += net_pnl
        
        return {
            "success": True, 
            "close_price": slipped_close_price, 
            "pnl": net_pnl, # <-- This is already NET PNL
            "commission": total_commission
        }

    def modify_stop_loss(self, symbol: str, trade_id: str, new_stop_price: float) -> bool: 
        return True

    def start_data_stream(self, on_tick_callback: Callable[[Tick], None]): 
        pass

    # --- *** NEW: IMPLEMENTED DUMMY METHODS FOR INTERFACE MATCHING *** ---
    
    def get_open_positions(self, symbol: str = None) -> List[Dict[str, Any]]:
        """
        Dummy implementation. Returns an empty list because the
        SimulationEngine is the source of truth for open positions.
        """
        return []

    def get_mark_price(self, symbol: str) -> float:
        """
        Dummy implementation. Returns the clock's current price.
        """
        return self.clock.current_price
    # --- *** END OF NEW METHODS *** ---