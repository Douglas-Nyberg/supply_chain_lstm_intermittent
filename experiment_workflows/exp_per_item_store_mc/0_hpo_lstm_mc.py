#////////////////////////////////////////////////////////////////////////////////#
# File:         0_hpo_lstm_mc.py                                                 #
# Author:       Douglas Nyberg                                                   #
# Email:        douglas1.nyberg@gmail.com                                        #
# Date:         2025-06-05                                                       #
#////////////////////////////////////////////////////////////////////////////////#
#!/usr/bin/env python3
"""
Runs Bayesian Optimization (BO) for Per-Series LSTM models on Monte Carlo synthetic data
to find optimal hyperparameters for forecasting.

This script adapts the existing HPO framework for Monte Carlo synthetic data:
1. Uses simplified feature set (no pricing, events, SNAP, holidays)
2. Optimizes for RMSSE performance on synthetic intermittent demand
3. Searches reasonable parameter space for synthetic data complexity
4. Evaluates on subset of synthetic series for efficiency
"""

# Import necessary libraries
import os
import sys
import argparse
import json
import logging
from pathlib import Path
from datetime import datetime
import numpy as np

# Set up project path
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.append(str(PROJECT_ROOT))
if str(PROJECT_ROOT / "src") not in sys.path:
    sys.path.append(str(PROJECT_ROOT / "src"))

# Import project-specific modules
from src.hpo.bayesian_optimizer import BayesianOptimizer
from src.hpo.lstm_fitness import create_lstm_fitness_function
from src.hpo.search_spaces import get_lstm_param_space_limited

# Import Monte Carlo master configuration (same directory)
import config_mc_master as config

# Define custom JSON encoder for numpy types
class NumpyEncoder(json.JSONEncoder):
    """Custom JSON encoder to handle serialization of numpy data types."""
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, np.bool_):
            return bool(obj)
        elif hasattr(obj, '__float__'):
            return float(obj)
        elif hasattr(obj, '__int__'):
            return int(obj)
        return super().default(obj)

def get_mc_lstm_param_space():
    """
    Get LSTM search space from master configuration.
    """
    return config.HPO_CONFIG["search_space"]

def main():
    """Main function to set up and run Bayesian Optimization for Monte Carlo LSTM."""
    
    # Get master configuration for defaults
    hpo_config = config.HPO_CONFIG
    
    # Parse command-line arguments
    parser = argparse.ArgumentParser(
        description='Run Bayesian Optimization for LSTM Models on Monte Carlo Synthetic Data.'
    )
    
    # Configuration profile
    parser.add_argument(
        '--config-profile', type=str, default="default",
        choices=["default", "quick_test", "moderate", "production", "hpo_intensive"],
        help='Configuration profile to use.'
    )
    
    # Data configuration
    parser.add_argument(
        '--data-dir', type=str, 
        default=str(PROJECT_ROOT / "data" / "preprocessed" / "exp_mc_synthetic_demand_comparison"),
        help='Directory containing preprocessed Monte Carlo data.'
    )
    parser.add_argument(
        '--limit-series', type=int, default=hpo_config["limit_series"],
        help=f'Limit number of series for evaluating each hyperparameter set (default: {hpo_config["limit_series"]}).'
    )
    
    # HPO configuration
    parser.add_argument(
        '--n-trials', type=int, default=hpo_config["n_trials"],
        help=f'Number of Bayesian optimization trials to run (default: {hpo_config["n_trials"]}).'
    )
    parser.add_argument(
        '--n-initial-points', type=int, default=hpo_config["n_initial_points"],
        help=f'Number of random initial points before Bayesian optimization (default: {hpo_config["n_initial_points"]}).'
    )
    parser.add_argument(
        '--cv-folds', type=int, default=hpo_config["cv_folds"],
        help=f'Number of cross-validation folds for each evaluation (default: {hpo_config["cv_folds"]}).'
    )
    
    # Output configuration
    parser.add_argument(
        '--output-dir', type=str, 
        default=str(PROJECT_ROOT / "results" / "hpo_mc_lstm"),
        help='Directory to save HPO results.'
    )
    parser.add_argument(
        '--run-name', type=str, 
        default=f"mc_lstm_hpo_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
        help='Name for this HPO run.'
    )
    
    # Device configuration  
    default_device = config.DEVICE_CONFIG["device_name"]
    parser.add_argument(
        '--device', type=str, default=default_device,
        help=f'Device to use for training (cpu or cuda). Default: {default_device}.'
    )
    
    args = parser.parse_args()
    
    # Use the simplified config
    selected_hpo_config = config.HPO_CONFIG
    
    # Override with profile-specific defaults if not explicitly provided
    if not any('--limit-series' in arg for arg in sys.argv):
        args.limit_series = selected_hpo_config["limit_series"]
    if not any('--n-trials' in arg for arg in sys.argv):
        args.n_trials = selected_hpo_config["n_trials"]
    if not any('--cv-folds' in arg for arg in sys.argv):
        args.cv_folds = selected_hpo_config["cv_folds"]
    
    # Setup logging
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    log_file = output_dir / f"{args.run_name}.log"
    logging.basicConfig(
        level=logging.INFO,  # Back to INFO
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler(sys.stdout)
        ]
    )
    logger = logging.getLogger(__name__)
    
    logger.info("=" * 70)
    logger.info(f"STARTING MONTE CARLO LSTM HPO: {args.run_name}")
    logger.info("=" * 70)
    logger.info(f"Data directory: {args.data_dir}")
    logger.info(f"Limit series: {args.limit_series}")
    logger.info(f"Trials: {args.n_trials}")
    logger.info(f"CV folds: {args.cv_folds}")
    logger.info(f"Device: {args.device}")
    logger.info(f"Output directory: {output_dir}")
    
    try:
        # Define search space from master config
        search_space = get_mc_lstm_param_space()
        logger.info(f"Search space defined with {len(search_space)} parameters using profile '{args.config_profile}'")
        
        # Create LSTM fitness evaluation function using standard infrastructure
        from src.hpo.lstm_fitness import create_lstm_fitness_function
        
        logger.info("Creating LSTM fitness function for Monte Carlo data")
        
        # Create args compatible with what fitness function expects
        # Map MC HPO args to expected names
        class HPOArgs:
            def __init__(self, mc_args):
                self.limit_items = mc_args.limit_series  # Map limit_series to limit_items
                self.use_cv = True  # Always use CV for HPO
                self.cv_splits = mc_args.cv_folds  # Map cv_folds to cv_splits
                self.cv_initial_train_size = 50  # Smaller for MC test data
                self.device = mc_args.device
                self.random_seed = 42  # Fixed seed for reproducibility
                # Pass through other attributes
                for attr in ['data_dir', 'output_dir', 'run_name']:
                    if hasattr(mc_args, attr):
                        setattr(self, attr, getattr(mc_args, attr))
        
        hpo_args = HPOArgs(args)
        
        # Use simplified config directly
        mc_config = config
        
        # Set logging level for fitness function to DEBUG
        import logging as fitness_logging
        fitness_logging.getLogger('src.hpo.lstm_fitness').setLevel(fitness_logging.DEBUG)
        
        fitness_function = create_lstm_fitness_function(
            splice_file_path=args.data_dir,  # MC uses data_dir instead of splice_file_path
            forecast_horizon=28,  # Standard M5 forecast horizon
            validation_days=28,   # Standard validation period
            hpo_runner_args=hpo_args,  # Use mapped args
            global_project_config=mc_config  # Use properly structured config
        )
        
        # Initialize Bayesian optimizer
        optimizer = BayesianOptimizer(
            param_space=search_space,
            fitness_function=fitness_function,
            n_iterations=args.n_trials,
            n_initial_points=args.n_initial_points,
            maximize=False,  # We want to minimize RMSSE
            random_seed=42,  # For reproducibility
            verbose=True
        )
        
        logger.info("Starting Bayesian optimization...")
        start_time = datetime.now()
        
        # Run optimization
        best_params, best_score, history = optimizer.optimize()
        
        end_time = datetime.now()
        duration = end_time - start_time
        logger.info(f"HPO completed in {duration}")
        
        # Save results
        results_file = output_dir / f"{args.run_name}_results.json"
        with open(results_file, 'w') as f:
            json.dump({
                'run_info': {
                    'run_name': args.run_name,
                    'start_time': start_time.isoformat(),
                    'end_time': end_time.isoformat(),
                    'duration_seconds': duration.total_seconds(),
                    'n_trials': args.n_trials,
                    'limit_series': args.limit_series,
                    'cv_folds': args.cv_folds,
                    'device': args.device
                },
                'search_space': search_space,
                'best_params': best_params,
                'best_score': best_score,
                'optimization_history': history
            }, f, indent=2, cls=NumpyEncoder)
        
        # Create best config file for easy use
        best_config_file = output_dir / f"{args.run_name}_best_config.py"
        with open(best_config_file, 'w') as f:
            f.write("# Best hyperparameters found by Monte Carlo HPO\n")
            f.write(f"# Run: {args.run_name}\n")
            f.write(f"# Best RMSSE: {best_score:.4f}\n")
            f.write(f"# Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            f.write("BEST_LSTM_CONFIG = {\n")
            for key, value in best_params.items():
                if isinstance(value, str):
                    f.write(f"    '{key}': '{value}',\n")
                else:
                    f.write(f"    '{key}': {value},\n")
            f.write("}\n")
        
        logger.info("=" * 70)
        logger.info("HPO RESULTS SUMMARY")
        logger.info("=" * 70)
        logger.info(f"Best RMSSE score: {best_score:.4f}")
        logger.info("Best parameters:")
        for key, value in best_params.items():
            logger.info(f"  {key}: {value}")
        logger.info(f"Results saved to: {results_file}")
        logger.info(f"Best config saved to: {best_config_file}")
        logger.info("=" * 70)
        
    except Exception as e:
        logger.error(f"HPO failed with error: {e}", exc_info=True)
        raise

if __name__ == "__main__":
    main()