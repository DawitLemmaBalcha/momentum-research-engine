# ensemble_config.py
import os

# --- Connector Imports ---
try:
    from connectors.mt5_connector import MetaTraderConnector
    from connectors.bybit_connector import BybitConnector
except ImportError:
    MetaTraderConnector = BybitConnector = None

from connectors.bybit_demo_connector import BybitDemoConnector
from connectors.dummy_connector import DummyConnector

demo_key = os.getenv("BYBIT_DEMO_API_KEY")
demo_secret = os.getenv("BYBIT_DEMO_API_SECRET") # <-- ADD THIS
print(f"--- DEBUG: Attempting to load BYBIT_DEMO_API_KEY. Value found: '{demo_key}' ---")
print(f"--- DEBUG: Attempting to load BYBIT_DEMO_API_SECRET. Value found: '{demo_secret}' ---")

VENUES = {
    # --- Backtest Venue ---
    "ensemble_backtest": {
        "mode": "backtest",
        "connector_class": DummyConnector,
        "connection_params": {},
        "strategies": [
            "strategies/cluster_1_1.py",
            "strategies/cluster_1_2.py",
            "strategies/cluster_2_1.py",
            "strategies/cluster_2_2.py",
            "strategies/cluster_3_1.py",
            "strategies/cluster_3_2.py",
            "strategies/cluster_4_1.py",
            "strategies/cluster_4_2.py",
            "strategies/cluster_4_3.py",
            "strategies/cluster_4_4.py",
            "strategies/cluster_4_5.py" 
        ]
    },
    
    # --- Paper Trading Venue ---
    "bybit_paper_trading": {
        "mode": "live",
        "connector_class": BybitDemoConnector,
        "connection_params": {
             "api_key": demo_key,
             "api_secret": demo_secret,
        },
        "strategies": [
            "strategies/cluster_1_1.py",
            "strategies/cluster_1_2.py",
            "strategies/cluster_2_1.py",
            "strategies/cluster_2_2.py",
            "strategies/cluster_3_1.py",
            "strategies/cluster_3_2.py",
            "strategies/cluster_4_1.py",
            "strategies/cluster_4_2.py" 
        ]
    },
    
    # --- Live Trading Venues ---
    "live_prop_firm_A": {
        "mode": "live",
        "connector_class": MetaTraderConnector,
        "connection_params": {
            "login": 123456, "password": "YOUR_MT5_PASSWORD", "server": "PropFirm-Server"
        },
        "strategies": ["strategies/b_conservative_trend.py"]
    },
    "live_bybit_main": {
        "mode": "live",
        "connector_class": BybitConnector,
        "connection_params": {
             "api_key": os.getenv("BYBIT_MAINNET_API_KEY"),
             "api_secret": os.getenv("BYBIT_MAINNET_API_SECRET"),
        },
        "strategies": ["strategies/a_aggressive_scalper.py"]
    }
}