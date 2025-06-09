#////////////////////////////////////////////////////////////////////////////////#
# File:         0_run_hpo_lstm_bo.py                                             #
# Author:       Douglas Nyberg                                                   #
# Email:        douglas1.nyberg@gmail.com                                        #
# Date:         2025-04-18                                                       #
#////////////////////////////////////////////////////////////////////////////////#
#!/usr/bin/env python3
"""
Runs Bayesian Optimization (BO) for Per-Series (Item-Store) LSTM models
to find optimal hyperparameters for forecasting.

This script orchestrates the HPO process by:
1.  Defining command-line arguments for configuring the HPO run (data, BO settings, CV).
2.  Loading the appropriate LSTM hyperparameter search space definition.
3.  Instantiating the LSTMFitnessEvaluator, which handles training and evaluating
    LSTM models on subsets of series for given hyperparameter sets.
4.  Utilizing the BayesianOptimizer class to search the hyperparameter space.
5.  Saving HPO results, including the best parameters found and optimization history.
6.  Optionally generating a configuration snippet or script for training with best parameters.
"""

# Import necessary libraries
import os
import sys
import argparse
import json
import logging
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, Tuple, Optional
import numpy as np

# Set up project path
# Scripts now in PROJECT_ROOT/experiment_workflows/exp1_item_level_unified/
# and src is PROJECT_ROOT/src/
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.append(str(PROJECT_ROOT))
# Explicitly add src to path if it's not automatically found
if str(PROJECT_ROOT / "src") not in sys.path:
    sys.path.append(str(PROJECT_ROOT / "src"))

# Import project-specific modules
from src.hpo.bayesian_optimizer import BayesianOptimizer
from src.hpo.lstm_fitness import create_lstm_fitness_function # Factory for LSTM fitness
from src.hpo.search_spaces import ( # LSTM search space definitions
    get_lstm_param_space,
    get_lstm_param_space_limited,
    get_lstm_param_space_focused
    # get_unified_param_space # If you create this one later
)
# Import the main experiment configuration to access shared constants
# config_exp1 is now in the same directory
import config_exp1 as global_project_config

# Define custom JSON encoder for numpy types
class NumpyEncoder(json.JSONEncoder):
    """Custom JSON encoder to handle serialization of numpy data types."""
    def default(self, obj):
        """
        Override default JSON encoder to handle NumPy types.
        
        Args:
            obj: Object to encode
            
        Returns:
            JSON-serializable representation of the object
        """
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, np.bool_):
            return bool(obj)
        elif hasattr(obj, '__float__'): # For some specific float-like types
            return float(obj)
        elif hasattr(obj, '__int__'): # For some specific int-like types
            return int(obj)
        return super().default(obj)

# Main HPO execution function
def main():
    """Main function to set up and run Bayesian Optimization for LSTMs."""
    
    # Parse command-line arguments
    parser = argparse.ArgumentParser(
        description='Run Bayesian Optimization for Per-Series (Item-Store) LSTM Models.'
    )
    
    # Arguments for Data and Model Configuration
    parser.add_argument(
        '--splice-file-path', type=str, required=True,
        help='Path to the data splice file (e.g., a feature-rich CSV from 1_create_splices) to be used for HPO evaluations.'
    )
    parser.add_argument(
        '--limit-items', type=int, default=None, # Using limit-items to match other scripts
        help='Limit number of unique series (item-store IDs) for evaluating each hyperparameter set. Speeds up HPO.'
    )
    parser.add_argument(
        '--device', type=str, default=None, choices=['cuda', 'cpu'],
        help='Target device for LSTM training during HPO (e.g., cuda, cpu). Defaults to cuda if available, else cpu.'
    )
    
    # Arguments for Bayesian Optimization Process
    parser.add_argument(
        '--n-iterations', type=int, default=30, # Adjusted default for LSTMs
        help='Total number of Bayesian optimization iterations to run.'
    )
    parser.add_argument(
        '--n-initial-points', type=int, default=5, # Adjusted default
        help='Number of initial random hyperparameter sets to evaluate before BO starts modeling the surface.'
    )
    parser.add_argument(
        '--exploration-weight', type=float, default=0.01, # Corresponds to 'xi' in EI acquisition
        help='Exploration weight (xi) for the Expected Improvement acquisition function.'
    )
    parser.add_argument(
        '--search-space-type', type=str, default='limited', # Default to 'limited' for faster initial HPO
        choices=['standard', 'limited', 'focused'], # Ensure these match functions in src.hpo.search_spaces
        help='Which predefined LSTM hyperparameter search space definition to use.'
    )
    
    # Arguments for Cross-Validation (within each fitness evaluation)
    parser.add_argument(
        '--use-cv', action='store_true',
        help='Use rolling cross-validation within fitness evaluation instead of a single train/validation split.'
    )
    parser.add_argument(
        '--cv-splits', type=int, default=3,
        help='Number of rolling CV splits if --use-cv is active.'
    )
    parser.add_argument(
        '--cv-initial-train-size', type=int, default=365, # Example: 1 year
        help='Initial training size (in days/time periods) for the first CV fold.'
    )
    
    # General Parameters for HPO run
    parser.add_argument(
        '--forecast-horizon', type=int, default=global_project_config.FORECAST_CONFIG["horizon"],
        help='Forecast horizon length for evaluation during HPO.'
    )
    parser.add_argument(
        '--validation-days', type=int, default=global_project_config.DATA_CONFIG["validation_days"],
        help='Number of validation days for HPO train/validation splits (or size of each CV validation fold).'
    )
    parser.add_argument(
        '--output-dir', type=str, default=None,
        help='Directory to save HPO results. Defaults to "hyperopt_results/bo_lstm_opt_TIMESTAMP".'
    )
    parser.add_argument(
        '--random-seed', type=int, default=global_project_config.RANDOM_SEED,
        help='Random seed for reproducibility of the HPO process and subset selection.'
    )
    parser.add_argument(
        '--verbose-bo', action='store_true', # Renamed for clarity
        help='Enable verbose output from the BayesianOptimizer class itself.'
    )
    parser.add_argument(
        '--log-level-fitness', type=str, default="INFO", # Renamed for clarity
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Logging level for the LSTMFitnessEvaluator module."
    )

    args = parser.parse_args()
    
    # Setup logging
    # Configure the main HPO runner's logger
    logging.basicConfig(
        level=logging.DEBUG if args.verbose_bo else logging.INFO, # Use verbose_bo for this logger
        format='%(asctime)s - HPO_LSTM_RUNNER - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(sys.stdout) # Default to stdout for runner
        ]
    )
    main_hpo_runner_logger = logging.getLogger("HPO_LSTM_RUNNER")
    
    # Configure the logger for the LSTMFitnessEvaluator module
    lstm_fitness_module_logger = logging.getLogger("src.hpo.lstm_fitness") # Matches __name__ in lstm_fitness.py
    lstm_fitness_module_logger.setLevel(getattr(logging, args.log_level_fitness.upper()))
    # If main runner also logs to file, fitness logger might inherit that. Add specific handlers if needed.
    # Example: Adding a stream handler for fitness logger if not already propagated
    if not lstm_fitness_module_logger.handlers:
        ch = logging.StreamHandler(sys.stdout)
        ch.setFormatter(logging.Formatter('%(asctime)s - LSTM_FITNESS - %(levelname)s - %(message)s'))
        lstm_fitness_module_logger.addHandler(ch)
        lstm_fitness_module_logger.propagate = False # Avoid double logging if root has handler

    # Determine output directory
    hpo_output_dir_path: Path
    if args.output_dir is None:
        current_timestamp_str = datetime.now().strftime('%Y%m%d_%H%M%S')
        hpo_output_dir_path = Path(f'hyperopt_results/bo_lstm_opt_{current_timestamp_str}')
    else:
        hpo_output_dir_path = Path(args.output_dir)
    
    hpo_output_dir_path.mkdir(parents=True, exist_ok=True)
    main_hpo_runner_logger.info(f"HPO results will be saved to: {hpo_output_dir_path.resolve()}")
    
    # Save HPO run configuration
    hpo_run_config_dict = vars(args)
    hpo_config_file_path = hpo_output_dir_path / 'hpo_run_configuration.json'
    try:
        with open(hpo_config_file_path, 'w') as f_config_out:
            json.dump(hpo_run_config_dict, f_config_out, indent=2, cls=NumpyEncoder)
        main_hpo_runner_logger.info(f"HPO run configuration saved to: {hpo_config_file_path}")
    except Exception as e:
        main_hpo_runner_logger.error(f"Could not save HPO run configuration: {e}")

    # Select LSTM hyperparameter search space
    selected_lstm_param_space: Dict[str, Any]
    if args.search_space_type == 'standard':
        selected_lstm_param_space = get_lstm_param_space()
    elif args.search_space_type == 'limited':
        selected_lstm_param_space = get_lstm_param_space_limited()
    elif args.search_space_type == 'focused':
        selected_lstm_param_space = get_lstm_param_space_focused()
    else:
        # This should be caught by argparse 'choices', but defensive check
        error_msg = f"Unknown LSTM search space type specified: {args.search_space_type}"
        main_hpo_runner_logger.error(error_msg)
        raise ValueError(error_msg)
    main_hpo_runner_logger.info(f"Using LSTM search space type: '{args.search_space_type}'")
    main_hpo_runner_logger.debug(f"Search space details: {selected_lstm_param_space}")
    
    # Create LSTM fitness evaluation function
    # The factory function create_lstm_fitness_function will instantiate LSTMFitnessEvaluator
    # and return its 'evaluate' method.
    lstm_fitness_function_for_bo = create_lstm_fitness_function(
        splice_file_path=args.splice_file_path,
        forecast_horizon=args.forecast_horizon,
        validation_days=args.validation_days,
        hpo_runner_args=args, # Pass all parsed HPO runner arguments
        global_project_config=global_project_config # Pass the main config_exp1 module
    )
    
    # Instantiate and run the Bayesian optimizer
    bayesian_opt_instance = BayesianOptimizer(
        param_space=selected_lstm_param_space,
        fitness_function=lstm_fitness_function_for_bo,
        n_iterations=args.n_iterations,
        n_initial_points=args.n_initial_points,
        exploration_weight=args.exploration_weight,
        maximize=False,  # Typically minimizing a loss metric (e.g., val_loss, WRMSSE)
        output_dir=str(hpo_output_dir_path), # BayesianOptimizer expects string for path
        random_seed=args.random_seed,
        verbose=args.verbose_bo # Verbosity for the optimizer itself
    )
    
    main_hpo_runner_logger.info("Starting Bayesian Optimization for LSTM hyperparameters...")
    best_found_hyperparams, best_achieved_fitness_score, full_optimization_history = \
        bayesian_opt_instance.optimize()
    
    # Save final HPO results summary
    final_hpo_run_summary_dict = {
        'best_hyperparameters_found': best_found_hyperparams,
        'best_fitness_score_achieved': float(best_achieved_fitness_score), # Ensure JSON serializable
        'model_optimized': 'lstm',
        'search_space_used': args.search_space_type,
        'total_iterations_completed': len(full_optimization_history),
        'cv_used_in_fitness_eval': args.use_cv,
        # Lighter summary of history for quick overview; full history is saved by BayesianOptimizer
        'optimization_progress_summary': [
            {
                'iteration': trial_entry['iteration'], 
                'fitness_achieved': trial_entry['fitness'],
                'best_fitness_up_to_iteration': trial_entry['best_fitness']
            } for trial_entry in full_optimization_history
        ]
    }
    
    final_hpo_results_file_path = hpo_output_dir_path / 'hpo_final_run_summary.json'
    try:
        with open(final_hpo_results_file_path, 'w') as f_results_out:
            json.dump(final_hpo_run_summary_dict, f_results_out, indent=2, cls=NumpyEncoder)
        main_hpo_runner_logger.info(f"Final HPO summary saved to: {final_hpo_results_file_path}")
    except Exception as e:
        main_hpo_runner_logger.error(f"Could not save final HPO summary: {e}")

    main_hpo_runner_logger.info(f"LSTM Hyperparameter Optimization complete!")
    main_hpo_runner_logger.info(f"Best Fitness Score achieved (e.g., Validation Loss/WRMSSE): {best_achieved_fitness_score:.6f}")
    main_hpo_runner_logger.info(f"Best Hyperparameters found: {best_found_hyperparams}")
    
    # Save best hyperparameters separately for easy use
    best_hyperparams_file_path = hpo_output_dir_path / 'best_lstm_hyperparameters_config.json'
    try:
        with open(best_hyperparams_file_path, 'w') as f_best_hp_out:
            json.dump(best_found_hyperparams, f_best_hp_out, indent=2, cls=NumpyEncoder)
        main_hpo_runner_logger.info(f"Best LSTM hyperparameters also saved separately to: {best_hyperparams_file_path}")
        main_hpo_runner_logger.info(
            "To use these optimal parameters, you can either manually update 'LSTM_CONFIG' "
            "in your 'config_exp1.py' file, or modify your main training script "
            "('3_train_lstm_models.py') to optionally load hyperparameters from a JSON file."
        )
    except Exception as e:
        main_hpo_runner_logger.error(f"Could not save best hyperparameters config: {e}")

# Script execution guard
if __name__ == '__main__':
    main()