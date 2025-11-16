import warnings
warnings.filterwarnings("ignore", category=UserWarning, message="pkg_resources is deprecated")
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=RuntimeWarning)
import logging
from datetime import datetime
import importlib.util
import sys
import os
import pandas as pd

# --- START OF "GOLDEN HEADER" ---
# 1. Add the project root to sys.path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# 2. Load the .env file from the root *before* other imports
from dotenv import load_dotenv
dotenv_path = os.path.join(project_root, 'api.env')
load_dotenv(dotenv_path=dotenv_path)
# --- END OF "GOLDEN HEADER" ---

# Now, all other project imports will work correctly
from deployment.ensemble_config import VENUES
from simulation.data_fetcher import fetch_all_dataframes
# --- MODIFIED: Import Model A components ---
from simulation.backtest_runner import SimulationEngine, prepare_data_for_simulation
from core.trading_bot import MomentumAnchorBot
# --- END MODIFIED ---
from core.core_types import SimulationClock
import config # Import base config

def setup_logging():
    log_filename = f"ensemble_backtest_log_{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}.log"
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
    logger.info(f"Logging configured. Output will be saved to: {log_filename}")

def log_prop_firm_summary(portfolio_metrics: dict):
    logger = logging.getLogger()
    if not portfolio_metrics:
        logger.warning("No portfolio metrics to summarize.")
        return
    # --- MODIFIED: Changed title to reflect Model A ---
    logger.info("\n" + "="*25 + " CONSOLIDATED PORTFOLIO SUMMARY (MODEL A) " + "="*24)
    logger.info("   (Simulating strategies with SEPARATE capital pools)")
    logger.info(f"  Initial Portfolio Capital: ${portfolio_metrics['initial_capital']:,.2f}")
    logger.info(f"  Final Portfolio Capital:   ${portfolio_metrics['final_capital']:,.2f}")
    logger.info(f"  Total Net PnL:             ${portfolio_metrics['pnl_cash']:,.2f}")
    logger.info(f"  Total Portfolio Return:    {portfolio_metrics['total_return_pct']:.2f}%")
    logger.info(f"  Max Equity Drawdown:       {portfolio_metrics['max_drawdown']:.2f}% (approx. sum of equity)")
    # --- MODIFIED: Added Realized DD (will be calculated below) ---
    logger.info(f"  Max Realized Drawdown:     {portfolio_metrics.get('realized_max_drawdown', 0.0):.2f}% (approx. sum of equity)")
    logger.info("="*87)


def run_ensemble_backtest():
    """
    Orchestrates a CONCURRENT backtest for all strategies in 'backtest' venues.
    --- MODIFIED: This now runs each bot in a SEPARATE simulation ---
    """
    logger = logging.getLogger()
    logger.info("--- Starting Concurrent Ensemble Backtest (SEPARATE CAPITAL - MODEL A) ---")
    
    backtest_venues = {name: config for name, config in VENUES.items() if config.get("mode") == "backtest"}
    if not backtest_venues:
        logger.error("No venues with mode: 'backtest' found in ensemble_config.py. Exiting.")
        return

    all_bots = []
    for venue_name, venue_config in backtest_venues.items():
        for strategy_path in venue_config["strategies"]:
            spec = importlib.util.spec_from_file_location(strategy_path, strategy_path)
            strategy_module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(strategy_module)
            
            bot = MomentumAnchorBot( # Using Model A bot
                initial_capital=strategy_module.INITIAL_CAPITAL, # Uses its own capital
                simulation_clock=SimulationClock(0),
                config_module=strategy_module,
                connector=None,
                silent=True
            )
            bot.bot_id = f"{venue_name}_{strategy_path.split('/')[-1].replace('.py', '')}"
            all_bots.append(bot)

    if not all_bots:
        logger.error("No bots were created from the backtest venues. Exiting.")
        return

    first_bot_config = all_bots[0].config
    # Set main config dates from the first bot
    config.SYMBOL = first_bot_config.SYMBOL
    config.START_DATE = first_bot_config.START_DATE
    config.END_DATE = first_bot_config.END_DATE
    
    logger.info(f"1. Fetching data for {config.SYMBOL} from {config.START_DATE} to {config.END_DATE}...")
    _, df_1m, _ = fetch_all_dataframes()

    if df_1m.empty:
        logger.error(f"FATAL: 1m dataframe is empty for the specified date range.")
        return
    
    logger.info(f"2. Running {len(all_bots)} separate backtests (Model A)...")
    
    individual_results = []
    portfolio_equity_curves = []
    portfolio_realized_curves = []
    total_initial_capital = 0.0
    
    # --- NEW LOGIC: Loop and run each bot separately ---
    for bot in all_bots:
        logger.info(f"\n--- Running backtest for: {bot.bot_id} ---")
        
        # 1. Prepare data using THIS bot's config
        logger.info(f"Preparing data using {bot.bot_id} parameters...")
        df_prepared = prepare_data_for_simulation(df_1m, bot.config)
        
        if df_prepared.empty:
            logger.error(f"Failed to prepare data for {bot.bot_id}. Skipping.")
            continue
            
        # 2. Create a new engine just for this bot
        engine = SimulationEngine(df_prepared, [bot], logger, prepared=True)
        
        # 3. Run the simulation
        _, bot_individual_results = engine.run()
        
        if not bot_individual_results:
            logger.error(f"Backtest failed for {bot.bot_id}. Skipping.")
            continue
            
        # 4. Store results for aggregation
        bot_metrics = bot_individual_results[0]
        individual_results.append(bot_metrics)
        
        # --- *** FIX 1 of 2: Handle (ts, val) tuples for Equity *** ---
        # Create a Series for the 1-minute equity curve
        equity_ts = pd.Series(
            [val for ts, val in engine.portfolio_equity_curve],
            index=pd.to_datetime([ts for ts, val in engine.portfolio_equity_curve], unit='s')
        )
        # 1-min candle timestamps are unique, so no extra step needed
        portfolio_equity_curves.append(equity_ts)
        
        # --- *** FIX 2 of 2: Handle (ts, val) tuples AND Duplicates for Realized *** ---
        # Create a Series for the realized (on-close) equity curve
        realized_ts_raw = pd.Series(
            [val for ts, val in bot.realized_equity_curve],
            index=pd.to_datetime([ts for ts, val in bot.realized_equity_curve], unit='s')
        )
        
        # --- THIS IS THE FIX ---
        # If multiple trades close on the same timestamp, this creates duplicate
        # index labels. We must consolidate them by taking the *last*
        # recorded capital for any given timestamp.
        if not realized_ts_raw.index.is_unique:
            realized_ts = realized_ts_raw.groupby(realized_ts_raw.index).last()
        else:
            realized_ts = realized_ts_raw
        # --- END FIX ---
        
        portfolio_realized_curves.append(realized_ts)
        # --- *** END ALL FIXES *** ---

        total_initial_capital += bot.initial_capital
        
        # 5. Log individual performance
        logger.info(bot_metrics.get("performance_log_str", "Performance log not available."))

    logger.info("\n--- All Backtests Finished. Generating Reports ---")
    
    # --- NEW LOGIC: Manually aggregate portfolio results ---
    if not individual_results:
        logger.error("No bots ran successfully. Cannot generate summary.")
        return

    # --- *** FIX: Correctly combine timestamp-indexed Series *** ---
    
    # 1. Combine all 1-minute equity curves.
    #    concat(axis=1) aligns them by timestamp.
    #    .ffill() carries forward the last known equity for each bot.
    #    .sum(axis=1) adds all bot equities at that timestamp.
    df_portfolio_equity = pd.concat(portfolio_equity_curves, axis=1).ffill()
    df_portfolio_equity_sum = df_portfolio_equity.sum(axis=1)
    
    # 2. Combine all realized (on-close) equity curves.
    #    Same logic. This correctly sums the realized capital over time.
    df_portfolio_realized = pd.concat(portfolio_realized_curves, axis=1).ffill()
    df_portfolio_realized_sum = df_portfolio_realized.sum(axis=1)

    # 3. Calculate combined PnL
    final_portfolio_equity = df_portfolio_equity_sum.iloc[-1]
    total_pnl = final_portfolio_equity - total_initial_capital
    total_return = (total_pnl / total_initial_capital) * 100 if total_initial_capital > 0 else 0
    
    # 4. Calculate combined equity drawdown (from 1-minute curve)
    peak_equity = df_portfolio_equity_sum.expanding().max()
    drawdown_equity = (df_portfolio_equity_sum - peak_equity) / peak_equity
    max_drawdown_equity = abs(drawdown_equity.min()) * 100
    
    # 5. Calculate combined realized drawdown (from on-close curve)
    #    This is the FIX for the 77% error.
    peak_realized = df_portfolio_realized_sum.expanding().max()
    drawdown_realized = (df_portfolio_realized_sum - peak_realized) / peak_realized
    max_drawdown_realized = abs(drawdown_realized.min()) * 100
    # --- *** END FIX *** ---

    portfolio_results = {
        "initial_capital": total_initial_capital,
        "final_capital": final_portfolio_equity,
        "pnl_cash": total_pnl,
        "total_return_pct": total_return,
        "max_drawdown": max_drawdown_equity,
        "realized_max_drawdown": max_drawdown_realized # Now correct
    }
    
    log_prop_firm_summary(portfolio_results)
    
    # Individual stats were already logged inside the loop
    logger.info("\n" + "="*28 + " INDIVIDUAL STRATEGY STATS " + "="*27)
    logger.info("(See full stats for each bot above)")


if __name__ == "__main__":
    setup_logging()
    run_ensemble_backtest()