import sys
import os
import logging
from datetime import datetime
import importlib.util

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
from simulation.backtest_runner_shared import SimulationEngine # Note: Using the _shared runner
from core.trading_bot_shared import MomentumAnchorBot # Note: Using the _shared bot
from core.core_types import SimulationClock
import config # Import base config

# --- (Copied from ensemble_backtest_runner.py, but using SHARED components) ---

def setup_logging():
    log_filename = f"ensemble_backtest_SHARED_log_{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}.log"
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
    logger.info("\n" + "="*25 + " CONSOLIDATED PORTFOLIO SUMMARY (MODEL C) " + "="*25)
    logger.info("   (Simulating all strategies on a single SHARED account)")
    logger.info(f"  Initial Portfolio Capital: ${portfolio_metrics['initial_capital']:,.2f}")
    logger.info(f"  Final Portfolio Capital:   ${portfolio_metrics['final_capital']:,.2f}")
    logger.info(f"  Total Net PnL:             ${portfolio_metrics['pnl_cash']:,.2f}")
    logger.info(f"  Total Portfolio Return:    {portfolio_metrics['total_return_pct']:.2f}%")
    logger.info(f"  Max Equity Drawdown:       {portfolio_metrics['max_drawdown']:.2f}% (incl. UPL)")
    logger.info(f"  Max Realized Drawdown:     {portfolio_metrics.get('realized_max_drawdown', 0.0):.2f}% (Prop Firm Style)")
    logger.info("="*87)


def run_ensemble_backtest():
    """Orchestrates a concurrent backtest for all strategies in 'backtest' venues."""
    logger = logging.getLogger()
    logger.info("--- Starting Concurrent Ensemble Backtest (SHARED CAPITAL - MODEL C) ---")
    
    backtest_venues = {name: config for name, config in VENUES.items() if config.get("mode") == "backtest"}
    if not backtest_venues:
        logger.error("No venues with mode: 'backtest' found in ensemble_config.py. Exiting.")
        return

    all_bots = []
    num_bots = 0
    for venue_name, venue_config in backtest_venues.items():
        num_bots += len(venue_config["strategies"])
        for strategy_path in venue_config["strategies"]:
            spec = importlib.util.spec_from_file_location(strategy_path, strategy_path)
            strategy_module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(strategy_module)
            
            bot = MomentumAnchorBot( # Using _shared bot
                initial_capital=strategy_module.INITIAL_CAPITAL, # This will be overwritten by engine
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

    # Set the capital allocation for each bot
    # This uses the bot's own configured capital as a *weight*
    total_requested_capital = sum(bot.initial_capital for bot in all_bots)
    
    # Use the main config.py for the *actual* total capital to start with
    main_portfolio_capital = config.INITIAL_CAPITAL
    logger.info(f"Main config INITIAL_CAPITAL of ${main_portfolio_capital:,.2f} will be used as the total shared pool.")

    for bot in all_bots:
        # Calculate allocation based on its requested capital as a fraction of total
        bot.capital_allocation = (bot.initial_capital / total_requested_capital) if total_requested_capital > 0 else (1.0 / num_bots)
        # Set the bot's initial capital to the *portfolio* capital for the engine's init
        bot.initial_capital = main_portfolio_capital
        
        logger.info(f"  - Bot '{bot.bot_id}' allocated {bot.capital_allocation:.2%} of sizing.")


    first_bot_config = all_bots[0].config
    # We must overwrite the main config's dates to match the strategy's dates
    # (using the first bot as the source of truth)
    config.SYMBOL = first_bot_config.SYMBOL
    config.START_DATE = first_bot_config.START_DATE
    config.END_DATE = first_bot_config.END_DATE
    
    logger.info(f"1. Fetching data for {config.SYMBOL} from {config.START_DATE} to {config.END_DATE}...")
    _, df_1m, _ = fetch_all_dataframes()

    if df_1m.empty:
        logger.error(f"FATAL: 1m dataframe is empty for the specified date range.")
        return
    
    logger.info(f"2. Initializing Shared SimulationEngine with {len(all_bots)} bots...")
    engine = SimulationEngine(df_1m, all_bots, logger) # Using _shared engine
    portfolio_results, individual_results = engine.run()

    logger.info("\n--- Backtest Finished. Generating Reports ---")
    
    log_prop_firm_summary(portfolio_results)
    
    logger.info("\n" + "="*28 + " INDIVIDUAL STRATEGY STATS (MODEL A DIAGNOSTICS) " + "="*21)
    
# --- Run Model A diagnostics ---
    try:
        from simulation.backtest_runner import run_simulation_from_prepared_data as run_model_a_simulation
        from simulation.backtest_runner import prepare_data_for_simulation
        
        # --- FIX: REMOVED data preparation from here ---
        
        for bot in all_bots:
            logger.info(f"\n--- Performance for Bot: {bot.bot_id} (run separately) ---")
            
            # --- FIX: Prepare data INSIDE the loop using bot.config ---
            logger.info(f"Preparing data using {bot.bot_id} parameters...")
            df_1m_prepared_model_a = prepare_data_for_simulation(df_1m, bot.config)
            
            # Reset initial capital for the Model A test
            bot.config.INITIAL_CAPITAL = 20000.0 # Or some standard test amount
            
            model_a_metrics = run_model_a_simulation(
                df_1m_prepared_model_a,
                bot.config,
                verbose=False,
                logger=logger
            )

            if model_a_metrics:
                logger.info(model_a_metrics.get("performance_log_str", "Performance log not available."))
            else:
                logger.warning("Could not generate Model A diagnostic stats for this bot.")
                
    except ImportError:
        logger.error("Could not import Model A runner for diagnostics.")
    except Exception as e:
        logger.error(f"Failed to run Model A diagnostics: {e}")


if __name__ == "__main__":
    setup_logging()
    run_ensemble_backtest()
