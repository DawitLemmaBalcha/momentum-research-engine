# core/core_types.py

import time
from typing import List
from dataclasses import dataclass

# --- NEW: A standard object for individual market ticks ---
@dataclass
class Tick:
    timestamp: float
    price: float
    volume: float
    candle_volume: float

@dataclass
class TradeLog:
    entry_price: float
    close_price: float
    entry_time: float
    close_time: float
    pnl_percent: float
    pnl_cash: float
    reason: str
    direction: str
    size: float
    kelly_risk_usd: float
    atr_risk_usd: float
    entry_regime: str 

class Position:
    def __init__(self, entry_price: float, amount: float, is_long: bool):
        self.id: str = "" # To be filled by the exchange trade ID
        self.entry_price = entry_price
        self.amount = amount
        self.is_long = is_long
        self.stop_loss = None
        self.peak_price = entry_price
        self.entry_time: float = 0.0
        self.kelly_risk_usd: float = 0.0
        self.atr_risk_usd: float = 0.0
        self.is_trailing_active: bool = False
        self.entry_regime: str = ""
        
        # --- ADDED FIELDS ---
        self.entry_commission: float = 0.0   # Tracks entry fees for true Net PnL
        self.last_logged_stop_loss: float = 0.0 # Prevents log spam for trailing stops
        # --------------------

    def calculate_profit(self, current_price: float) -> float:
        return (current_price - self.entry_price) * self.amount if self.is_long else (self.entry_price - current_price) * self.amount


class Order:
    def __init__(self, price: float, order_type: str):
        self.price = price
        self.order_type = order_type
        self.last_rejection_reason = ""

class SimulationClock:
    def __init__(self, start_time):
        self.current_time = start_time
        self.current_price = 0.0 

    def advance(self, seconds):
        self.current_time += seconds

    def time(self):
        return self.current_time