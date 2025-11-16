# main_shared.py

import warnings
warnings.filterwarnings("ignore", category=UserWarning, message="pkg_resources is deprecated")
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=RuntimeWarning)
import pandas as pd
import logging
from datetime import datetime
import config  # It still reads the same base config.py for dates/symbols

# --- *** ADDED THIS MISSING IMPORT *** ---
from simulation.data_fetcher import fetch_all_dataframes
# --- *** END CHANGE *** ---

# --- We import from the new _shared runner, which uses the _shared bot ---
from simulation.backtest_runner_shared import run_single_backtest

# --- *** NEW IMPORTS TO CREATE THE BOT *** ---
# --- *** MODIFIED: Import refactored Model C bot *** ---
from core.trading_bot_shared import MomentumAnchorBotShared
from core.core_types import SimulationClock
# --- *** END NEW IMPORTS *** ---


def setup_logging():
    """Configures logging for a shared backtest run."""
    log_filename = f"logs/backtest_SHARED_log_{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}.log"
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    for handler in logger.handlers[:]:
        logger.removeHandler(handler)
    file_handler = logging.FileHandler(log_filename)
    file_handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(logging.Formatter('%(message)s'))
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)
    logging.info(f"Logging configured. Output will be saved to: {log_filename}")

def run_backtest(print_log=True):
    """
    Main function to set up and run a single backtest using the 
    SHARED (Model C) runner.
    """
    logging.info("--- Inside run_backtest (SHARED) ---")
    logging.info("1. Fetching dataframes...")
    _, df_1m, _ = fetch_all_dataframes() 
    logging.info("2. Data fetching complete.")

    if df_1m.empty:
        logging.error("EXIT POINT: 1m dataframe is empty.")
        return {}

    logging.info("3. Running efficient backtest via SHARED runner (Model C)...")

    # --- *** FIX: CREATE THE BOT INSTANCE *** ---
    # We must create the bot object first, as the runner
    # now expects a list of bots, not a config module.
    bot = MomentumAnchorBotShared( 
        initial_capital=config.INITIAL_CAPITAL,
        simulation_clock=SimulationClock(0), # Will be replaced by engine
        config_module=config,
        connector=None, # Will be set by the engine
        silent=not print_log
    )
    # For a single bot run, its allocation is 100%
    bot.capital_allocation = 1.0 
    # --- *** END FIX *** ---

    # This now calls the run_single_backtest from backtest_runner_shared.py
    # --- *** MODIFIED: Pass [bot] (a list) instead of config *** ---
    performance = run_single_backtest(df_1m, [bot], verbose=print_log, logger=logging)
    
    if not performance:
        logging.error("EXIT POINT: Backtest failed (insufficient data).")
        return {}
    
    logging.info("9. Simulation finished.")
    logging.info("10. Performance logged.")
    return performance

if __name__ == "__main__":
    setup_logging()
    logging.info("Script execution started (SHARED - Model C).")
    run_backtest()
    logging.info("Script execution finished (SHARED - Model C).")