# analysis/walk_forward_ensemble_optimizer.py
import sys
import os

# Add the project root directory to the Python path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)
import logging
from datetime import datetime
import pandas as pd
import numpy as np
import optuna
import json

# --- MODIFIED: Replaced sklearn KMeans with tslearn ---
from tslearn.clustering import TimeSeriesKMeans
from tslearn.preprocessing import TimeSeriesScalerMeanVariance
# --- END MODIFIED ---

# Import necessary components
import config
from simulation.data_fetcher import fetch_all_dataframes
# --- MODIFIED: Import objective, new preparer, Model A runner, and Model A bot ---
from analysis.optimize import ConfigContainer, objective
from analysis.data_preparer import prepare_static_data, add_dynamic_indicators
from core.trading_bot import MomentumAnchorBot as ModelABot # <-- Renamed to avoid conflict
# --- *** FIX 1 of 2: IMPORT SimulationClock *** ---
from core.core_types import SimulationClock
# --- *** END FIX *** ---
from simulation.backtest_runner import (
    SimulationEngine as ModelAEngine, # <-- Renamed to be explicit
    run_simulation_from_prepared_data,
    prepare_data_for_simulation # <-- This is the Model A preparer
)
# --- END MODIFIED ---

# --- Global list to store results from all walks ---
all_walks_data = []

# --- MODIFIED: Helper function to create a *single* wide objective ---
def create_wide_objective_function():
    """
    Creates a tailored objective function for Optuna with wide parameter ranges
    to allow the optimizer to explore a large landscape.
    """
    # --- MODIFIED: Accepts df_static_prepared ---
    def wide_objective(trial, df_static_prepared: pd.DataFrame, opt_logger):
        try:
            trial_config = ConfigContainer()
            for attr in dir(config):
                if attr.isupper():
                    setattr(trial_config, attr, getattr(config, attr))

            # --- MODIFIED: Use wide parameter ranges ---
            initial_stop_multiplier = trial.suggest_float("ATR_INITIAL_STOP_MULTIPLIER", 0.5, 2.0)
            activation_multiplier = trial.suggest_float("ATR_TRAILING_STOP_ACTIVATION_MULTIPLIER", low=initial_stop_multiplier, high=initial_stop_multiplier * 4.0)
            trailing_ratio = trial.suggest_float("TRAILING_RATIO", 0.1, 0.9)
            
            trial_config.ANCHOR_BAND_MULTIPLIER = trial.suggest_float("ANCHOR_BAND_MULTIPLIER", 0.03, 0.3)
            trial_config.ATR_INITIAL_STOP_MULTIPLIER = initial_stop_multiplier
            trial_config.ATR_TRAILING_STOP_ACTIVATION_MULTIPLIER = activation_multiplier
            trial_config.ATR_TRAILING_STOP_MULTIPLIER = activation_multiplier * trailing_ratio

            trial_config.CONFIRMATION_RSI_PERIOD = trial.suggest_int("CONFIRMATION_RSI_PERIOD", 15, 50)
            trial_config.CONFIRMATION_VOLUME_MA_PERIOD = trial.suggest_int("CONFIRMATION_VOLUME_MA_PERIOD", 15, 50)
            # --- END MODIFIED ---

            # --- MODIFIED: Use hybrid data prep ---
            df_trial_prepared = add_dynamic_indicators(df_static_prepared, trial_config)
            
            if df_trial_prepared.empty:
                return -1.0
            # --- END MODIFIED ---

            performance_metrics = run_simulation_from_prepared_data(
                df_trial_prepared, # <-- Use trial-specific data
                trial_config, 
                verbose=False, 
                logger=opt_logger,
                trial=trial 
            )

            if not performance_metrics: return -1.0
            
            trial.set_user_attr("performance", performance_metrics)
            return performance_metrics.get("sortino_ratio", -1.0)
        
        except optuna.TrialPruned:
            opt_logger.info(f"--- Trial {trial.number} Pruned ---")
            return -1.0
        except Exception as e:
            opt_logger.error(f"TRIAL FAILED: {e}", exc_info=False)
            return -1.0
    return wide_objective
# --- END MODIFIED ---

# --- *** NEW HELPER FUNCTION *** ---
def get_resampled_equity_curve(trial: optuna.Trial, df_static_prepared: pd.DataFrame, base_config, logger) -> np.ndarray:
    """
    Runs a full Model A backtest for a given trial and returns
    its daily-resampled, mean-normalized equity curve.
    """
    try:
        # 1. Create the bot's config from the trial
        trial_config = ConfigContainer()
        for attr in dir(base_config):
            if attr.isupper():
                setattr(trial_config, attr, getattr(base_config, attr))
        
        for key, value in trial.params.items():
            setattr(trial_config, key, value)
        
        trial_config.ATR_TRAILING_STOP_MULTIPLIER = trial_config.ATR_TRAILING_STOP_ACTIVATION_MULTIPLIER * trial_config.TRAILING_RATIO

        # 2. Prepare this trial's specific data
        df_trial_prepared = add_dynamic_indicators(df_static_prepared, trial_config)
        if df_trial_prepared.empty:
            return None

        # 3. Setup and run the Model A SimulationEngine
        # We must use the Model A bot and engine to get the isolated curve
        bot = ModelABot(
            initial_capital=trial_config.INITIAL_CAPITAL,
            simulation_clock=SimulationClock(0), 
            config_module=trial_config,
            connector=None, # DummyConnector will be set by engine
            silent=True
        )
        engine = ModelAEngine(df_trial_prepared, [bot], logger, prepared=True)
        
        # Run and get the *1-minute* equity curve
        # This is the list of (timestamp, capital) tuples
        portfolio_results, _ = engine.run()
        equity_tuples = engine.portfolio_equity_curve

        if not portfolio_results or not equity_tuples:
            return None

        # 4. Resample the 1-minute curve to Daily
        equity_series = pd.Series(
            [val for ts, val in equity_tuples],
            index=pd.to_datetime([ts for ts, val in equity_tuples], unit='s')
        )
        
        # Resample to daily, getting the last value of each day
        # and forward-filling weekends/holidays
        daily_equity_series = equity_series.resample('D').last().ffill()

        # 5. Normalize for clustering
        # We normalize by the mean. This keeps the shape but removes
        # the magnitude (e.g., 20% return vs 30% return).
        # We cluster on SHAPE, not on total P&L.
        # Handle potential zero-mean (flat curve)
        mean_val = daily_equity_series.mean()
        if mean_val == 0:
            return daily_equity_series.values # Return as-is
            
        normalized_curve = daily_equity_series.values / mean_val

        return normalized_curve

    except Exception as e:
        logger.error(f"Failed to generate equity curve for trial {trial.number}: {e}", exc_info=False)
        return None
# --- *** END NEW HELPER FUNCTION *** ---


# --- *** MODIFIED: Main walk function *** ---
def run_ensemble_walk(walk_number: int, is_start: str, is_end: str, oos_end: str, n_trials: int, n_clusters: int, top_n_per_cluster: int, opt_logger):
    """
    Executes a full walk-forward cycle with a single wide optimization
    and K-Means clustering for diverse ensemble selection.
    """
    opt_logger.info(f"--- Starting Walk #{walk_number} | IS: {is_start} to {is_end} | OOS: {is_end} to {oos_end} ---")

    # 1. --- Fetch and slice data ---
    config.START_DATE, config.END_DATE = is_start, oos_end
    try:
        _, df_full, _ = fetch_all_dataframes()
        df_is_raw = df_full[(df_full['timestamp'] >= pd.to_datetime(is_start)) & (df_full['timestamp'] < pd.to_datetime(is_end))].copy()
        df_oos_raw = df_full[(df_full['timestamp'] >= pd.to_datetime(is_end)) & (df_full['timestamp'] < pd.to_datetime(oos_end))].copy()
        if df_is_raw.empty or df_oos_raw.empty:
            opt_logger.warning("Skipping walk due to empty in-sample or out-of-sample data.")
            return
            
        # --- NEW: Prepare static data ONCE for IS and OOS ---
        opt_logger.info(f"Preparing {len(df_is_raw)} IS records (static)...")
        df_is_static_prepared = prepare_static_data(df_is_raw, config)
        
        opt_logger.info(f"Preparing {len(df_oos_raw)} OOS records (static)...")
        df_oos_static_prepared = prepare_static_data(df_oos_raw, config)
        
        if df_is_static_prepared.empty or df_oos_static_prepared.empty:
            opt_logger.warning("Skipping walk due to empty prepared data (IS or OOS).")
            return
        # --- END NEW ---
            
    except Exception as e:
        opt_logger.error(f"Failed to fetch or slice data: {e}", exc_info=True)
        return

    # 2. --- Run ONE Single, Wide Optimization ---
    optuna.logging.set_verbosity(optuna.logging.ERROR)
    
    opt_logger.info(f"--- Running WIDE optimization ({n_trials} trials)... ---")
    objective_func = create_wide_objective_function()
    study = optuna.create_study(direction="maximize")
    
    study.optimize(
        lambda trial: objective_func(trial, df_is_static_prepared, opt_logger), # <-- Use static IS data
        n_trials=n_trials, 
        n_jobs=-1, 
        show_progress_bar=True
    )

    # 3. --- Filter, Cluster, and Select Top Performers ---
    completed_trials = []
    for t in study.trials:
        if t.state == optuna.trial.TrialState.COMPLETE and t.value is not None:
            perf = t.user_attrs.get("performance", {})
            sortino = perf.get("sortino_ratio", -1.0)
            total_trades = perf.get("total_trades", 0)
            
            if sortino > 1 and total_trades > 20:
                completed_trials.append(t)

    opt_logger.info(f"Optimization complete. Found {len(completed_trials)} high-quality trials (Sortino > 1, Trades > 20).")

    # --- *** MODIFIED: Replaced parameter KMeans with DTW Equity Curve Clustering *** ---
    
    opt_logger.info(f"Generating equity curves for {len(completed_trials)} winning trials...")
    
    equity_curves = []
    valid_trials_for_clustering = []

    # 1. Generate resampled equity curves for all winners
    for trial in completed_trials:
        # We use df_is_static_prepared (In-Sample data) for clustering
        daily_curve = get_resampled_equity_curve(
            trial, 
            df_is_static_prepared, # <-- Use In-Sample data
            config, 
            opt_logger
        )
        
        if daily_curve is not None:
            equity_curves.append(daily_curve)
            valid_trials_for_clustering.append(trial)

    if len(equity_curves) < n_clusters:
        opt_logger.warning(f"Only {len(equity_curves)} valid equity curves generated, which is less than {n_clusters} clusters. Using all valid trials.")
        ensemble_bots_trials = valid_trials_for_clustering
        # Create a dummy df for logging
        best_trials_from_clusters = pd.DataFrame([{'trial_object': t, 'cluster': 0, 'sortino': t.value} for t in ensemble_bots_trials])
    else:
        opt_logger.info(f"Clustering {len(equity_curves)} equity curves using TimeSeriesKMeans (DTW)...")

        # 2. Normalize and Reshape for tslearn
        # TimeSeriesScalerMeanVariance centers the data, which is good for DTW
        scaler = TimeSeriesScalerMeanVariance(mu=0., std=1.)
        # tslearn expects (n_samples, n_timestamps, n_features)
        # Our curves are (n_samples, n_timestamps), so we add a feature dim
        scaled_curves = scaler.fit_transform(equity_curves)

        # 3. Run DTW Clustering
        model = TimeSeriesKMeans(
            n_clusters=n_clusters,
            metric="dtw",
            max_iter=10, # Keep it low for speed
            random_state=42,
            n_jobs=-1
        )
        cluster_labels = model.fit_predict(scaled_curves)
        
        opt_logger.info("Clustering complete. Selecting best trial from each cluster...")

        # 4. Rebuild the DataFrame to select the champions
        df_data = []
        for i, trial in enumerate(valid_trials_for_clustering):
            df_data.append({
                'trial_object': trial,
                'cluster': cluster_labels[i],
                'sortino': trial.value # The In-Sample Sortino
            })
        
        df = pd.DataFrame(df_data)
        
        # Select the trial with the highest Sortino from each cluster
        best_trials_from_clusters = df.groupby('cluster').apply(
            lambda x: x.nlargest(top_n_per_cluster, 'sortino')
        ).reset_index(drop=True)
        
        ensemble_bots_trials = best_trials_from_clusters['trial_object'].tolist()
    # --- *** END MODIFICATION *** ---

    if not ensemble_bots_trials:
        opt_logger.error("No successful trials were selected. Cannot run OOS test.")
        return

    opt_logger.info(f"--- Selected {len(ensemble_bots_trials)} Trials for OOS Ensemble ---")
    for _, trial_row in best_trials_from_clusters.sort_values(by=['cluster', 'sortino'], ascending=[True, False]).iterrows():
        trial = trial_row['trial_object']
        cluster = trial_row['cluster']
        sortino = trial_row['sortino']
        
        opt_logger.info(f"  [Cluster {cluster}] - Trial {trial.number} | In-Sample Sortino: {sortino:.4f}")
        
        params_to_log = trial.params.copy()
        params_to_log['ATR_TRAILING_STOP_MULTIPLIER'] = params_to_log.get("ATR_TRAILING_STOP_ACTIVATION_MULTIPLIER", 1.0) * params_to_log.get("TRAILING_RATIO", 0.5)
        opt_logger.info(f"    Params: {json.dumps(params_to_log, indent=2)}")

        is_perf = trial.user_attrs.get("performance", {})
        is_perf_str = (
            f"    IS Perf: Return={is_perf.get('total_return_pct', 0):.2f}%, "
            f"Trades={is_perf.get('total_trades', 0)}, "
            f"WinRate={is_perf.get('win_rate', 0):.2f}%, "
            f"PF={is_perf.get('profit_factor', 0):.2f}"
        )
        opt_logger.info(is_perf_str)

    # 4. --- Run Out-of-Sample Ensemble Backtest (Model A) ---
    opt_logger.info(f"--- Preparing {len(ensemble_bots_trials)} bots for OOS ensemble... ---")
    ensemble_bots = []
    
    for trial in ensemble_bots_trials:
        bot_config = ConfigContainer()
        for attr in dir(config):
            if attr.isupper():
                setattr(bot_config, attr, getattr(config, attr))
        
        for key, value in trial.params.items():
            setattr(bot_config, key, value)
        
        bot_config.ATR_TRAILING_STOP_MULTIPLIER = bot_config.ATR_TRAILING_STOP_ACTIVATION_MULTIPLIER * bot_config.TRAILING_RATIO
        
        bot = ModelABot( # <-- Uses refactored Model A bot
            initial_capital=bot_config.INITIAL_CAPITAL, 
            # --- *** FIX 2 of 2: Pass a dummy SimulationClock *** ---
            simulation_clock=SimulationClock(0),
            # --- *** END FIX *** ---
            config_module=bot_config,
            connector=None,
            silent=True
        )
        ensemble_bots.append(bot)

    opt_logger.info(f"--- Running OOS ENSEMBLE backtest with {len(ensemble_bots)} bots... ---")
    
    # --- MODIFIED: We must prepare data for EACH bot in the ensemble ---
    all_individual_results = []
    
    for bot in ensemble_bots:
        opt_logger.info(f"Running OOS for bot with RSI={bot.config.CONFIRMATION_RSI_PERIOD}, VolMA={bot.config.CONFIRMATION_VOLUME_MA_PERIOD}...")
        
        # Prepare data for this specific bot using its config
        df_oos_bot_prepared = add_dynamic_indicators(df_oos_static_prepared, bot.config)
        
        if df_oos_bot_prepared.empty:
            opt_logger.warning("OOS data empty for this bot, skipping.")
            continue
            
        # Run a *single bot* simulation
        bot_metrics = run_simulation_from_prepared_data(
            df_oos_bot_prepared,
            bot.config,
            verbose=False,
            logger=opt_logger
        )
        if bot_metrics:
            all_individual_results.append(bot_metrics)
    
    if not all_individual_results:
        opt_logger.error("Ensemble backtest failed to produce any results.")
        return
        
    # --- 5. Log and Save Results (Aggregated Model A) ---
    opt_logger.info("\n" + "="*25 + f" OOS ENSEMBLE PERFORMANCE (Model A) | WALK #{walk_number} " + "="*25)
    
    # Manually aggregate the results
    total_initial_capital = sum(res.get('initial_capital', config.INITIAL_CAPITAL) for res in all_individual_results)
    total_final_capital = sum(res.get('final_capital', config.INITIAL_CAPITAL) for res in all_individual_results)
    total_pnl = total_final_capital - total_initial_capital
    total_return_pct = (total_pnl / total_initial_capital) * 100 if total_initial_capital > 0 else 0
    
    # Note: Drawdown is tricky to aggregate. We'll just average it.
    avg_max_drawdown = np.mean([res.get('max_drawdown', 0) for res in all_individual_results])
    avg_realized_drawdown = np.mean([res.get('realized_max_drawdown', 0) for res in all_individual_results])

    portfolio_results = {
        'initial_capital': total_initial_capital,
        'final_capital': total_final_capital,
        'pnl_cash': total_pnl,
        'total_return_pct': total_return_pct,
        'max_drawdown': avg_max_drawdown, # Mark as average
        'realized_max_drawdown': avg_realized_drawdown # Mark as average
    }
    
    opt_logger.info(f"  --- Portfolio Summary (Sum of all bots) ---")
    opt_logger.info(f"  Initial Portfolio Capital: ${portfolio_results['initial_capital']:,.2f}")
    opt_logger.info(f"  Final Portfolio Capital:   ${portfolio_results['final_capital']:,.2f}")
    opt_logger.info(f"  Total Net PnL:             ${portfolio_results['pnl_cash']:,.2f}")
    opt_logger.info(f"  Total Portfolio Return:    {portfolio_results['total_return_pct']:.2f}%")
    opt_logger.info(f"  Portfolio Max Drawdown:    {portfolio_results['max_drawdown']:.2f}% (Average of individual bot DDs)")
    opt_logger.info(f"  Portfolio Realized DD:     {portfolio_results.get('realized_max_drawdown', 0.0):.2f}% (Average of individual bot DDs)")
    
    opt_logger.info(f"\n  --- Individual Bot Performance (Model A) ---")
    for i, bot_metrics in enumerate(all_individual_results):
        bot_id_str = f"Bot {i+1} (Trial {ensemble_bots_trials[i].number})"
        opt_logger.info(f"\n--- Stats for {bot_id_str} ---")
        
        log_str = bot_metrics.get("performance_log_str", "Performance log not available.")
        
        log_lines = log_str.splitlines()
        if len(log_lines) > 2:
            for line in log_lines[1:-1]:
                opt_logger.info(line)
        else:
            opt_logger.info(log_str)
    
    opt_logger.info("="*87 + "\n")

    portfolio_results['walk'] = walk_number
    all_walks_data.append(portfolio_results)


if __name__ == "__main__":
    for env_var in ['OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'VECLIB_MAXIMUM_THREADS', 'NUMEXPR_NUM_THREADS']:
        os.environ[env_var] = '1'
    
    log_filename = f"logs/walk_forward_ensemble_log_{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}.log"
    main_logger = logging.getLogger('wf_ensemble_logger')
    main_logger.setLevel(logging.INFO)
    file_handler = logging.FileHandler(log_filename)
    file_handler.setFormatter(logging.Formatter('%(asctime)s - %(message)s'))
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(logging.Formatter('%(message)s'))
    main_logger.addHandler(file_handler)
    main_logger.addHandler(console_handler)
    
    walk_forward_schedule = [
        {'is_start': '2025-01-01 00:00:00', 'is_end': '2025-06-01 00:00:00', 'oos_end': '2025-09-01 00:00:00'},
        #{'is_start': '2024-09-01 00:00:00', 'is_end': '2024-12-01 00:00:00', 'oos_end': '2025-01-01 00:00:00'},
        #{'is_start': '2024-10-01 00:00:00', 'is_end': '2025-01-01 00:00:00', 'oos_end': '2025-02-01 00:00:00'},
        #{'is_start': '2024-11-01 00:00:00', 'is_end': '2025-02-01 00:00:00', 'oos_end': '2025-03-01 00:00:00'},
        #{'is_start': '2024-12-01 00:00:00', 'is_end': '2025-03-01 00:00:00', 'oos_end': '2025-04-01 00:00:00'},
        #{'is_start': '2025-01-01 00:00:00', 'is_end': '2025-04-01 00:00:00', 'oos_end': '2025-05-01 00:00:00'},
        #{'is_start': '2025-02-01 00:00:00', 'is_end': '2025-05-01 00:00:00', 'oos_end': '2025-06-01 00:00:00'},
        #{'is_start': '2025-03-01 00:00:00', 'is_end': '2025-06-01 00:00:00', 'oos_end': '2025-07-01 00:00:00'},
        #{'is_start': '2025-04-01 00:00:00', 'is_end': '2025-07-01 00:00:00', 'oos_end': '2025-08-01 00:00:00'},
        #{'is_start': '2025-05-01 00:00:00', 'is_end': '2025-08-01 00:00:00', 'oos_end': '2025-09-01 00:00:00'},
    ]
    
    trials_per_optimization_walk = 500 
    number_of_clusters = 3           
    top_n_per_cluster = 3              

    main_logger.info("--- Starting Full Walk-Forward Ensemble Analysis (with K-Means Clustering) ---")
    for i, walk_dates in enumerate(walk_forward_schedule):
        walk_num = i + 1
        run_ensemble_walk(
            walk_number=walk_num,
            is_start=walk_dates['is_start'],
            is_end=walk_dates['is_end'],
            oos_end=walk_dates['oos_end'],
            n_trials=trials_per_optimization_walk,
            n_clusters=number_of_clusters,
            top_n_per_cluster=top_n_per_cluster,
            opt_logger=main_logger
        )
    main_logger.info("\n--- All Walk-Forward Ensemble Analysis runs are complete. ---")

    if all_walks_data:
        results_df = pd.DataFrame(all_walks_data)
        results_filename = f"results/walk_forward_ensemble_results_{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}.csv"
        os.makedirs("results", exist_ok=True)
        results_df.to_csv(results_filename, index=False)
        main_logger.info(f"\nSuccessfully saved all walk-forward ensemble data to: {results_filename}")