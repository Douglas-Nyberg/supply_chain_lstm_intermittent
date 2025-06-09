#////////////////////////////////////////////////////////////////////////////////#
# File:         2_train_classical_models.py                                      #
# Author:       Douglas Nyberg                                                   #
# Email:        douglas1.nyberg@gmail.com                                        #
# Date:         2025-03-22                                                       #
#////////////////////////////////////////////////////////////////////////////////#
#!/usr/bin/env python3
"""
Train Classical Forecasting Models for Experiment 1 (Per-Item-Per-Store).

This script trains classical forecasting models using the statsforecast library.
Each unique item-store combination gets its own model instance.

Main Steps:
1. Load preprocessed training data from Step 1
2. Validate each series has sufficient data for training
3. Train specified classical models (ARIMA, Croston, TSB, etc.)
4. Save trained models and metadata
5. Generate training summary report
"""

import argparse
import json
import logging
import pickle
import sys
import time
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime

import numpy as np
import pandas as pd
from tqdm import tqdm

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent.parent
sys.path.append(str(PROJECT_ROOT))


# Import our statsforecast-based classical models
from src.models.classical import (
    ARIMAForecaster, 
    CrostonsMethod,
    CrostonSBA, 
    TSBForecaster,
    MovingAverage,
    SimpleExponentialSmoothing,
    HoltWintersForecaster,
    ETS,
    ADIDAForecaster,
    IMAPAForecaster
)

from experiment_workflows.exp1_per_item_store_models import config_exp1 as config
from src.utils import set_random_seed
from src.cross_validation.expanding_window_cv import WalkForwardValidator

config.LOGS_DIR.mkdir(parents=True, exist_ok=True)

for handler in logging.root.handlers[:]:
    logging.root.removeHandler(handler)

logging.basicConfig(
    level=getattr(logging, config.LOGGING_CONFIG["level"]),
    format=config.LOGGING_CONFIG["format"],
    handlers=[
        logging.FileHandler(str(config.LOGGING_CONFIG["log_file"]), mode='a'),
        logging.StreamHandler(sys.stdout)
    ]
)

logger = logging.getLogger(__name__)

# helpers
def parse_arguments() -> argparse.Namespace:
    """
    Parse command-line arguments for training classical models.
    
    Returns:
        argparse.Namespace: Parsed command-line arguments
    """
    parser = argparse.ArgumentParser(
        description="Train classical models (Per-Series) for Unified Experiment 1",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    parser.add_argument(
        "--input-dir", 
        type=str,
        default=config.DATA_CONFIG["preprocessed_output_dir"],
        help="Directory containing preprocessed data from 1_preprocess_exp1.py"
    )

    parser.add_argument(
        "--models-to-train", 
        type=str, 
        nargs='+',
        default=config.DEFAULT_CLASSICAL_MODELS_TO_RUN,
        choices=config.AVAILABLE_MODELS["classical"],
        help="List of classical models to train (e.g., 'arima', 'croston', 'tsb')"
    )

    parser.add_argument(
        "--parallel-jobs", 
        type=int,
        default=config.WORKFLOW_CONFIG.get("parallel_jobs_classical", 
                                         config.WORKFLOW_CONFIG["parallel_jobs"]),
        help="Number of parallel training jobs"
    )

    parser.add_argument(
        "--limit-series", 
        type=int,
        default=config.DATA_CONFIG["limit_items"],
        help="Limit number of unique series to train (None = all)"
    )

    parser.add_argument(
        "--force", 
        action="store_true",
        help="Force retraining even if models already exist"
    )
    
    parser.add_argument(
        "--use-cv",
        action="store_true",
        help="Enable expanding window cross-validation (overrides config setting)"
    )
    
    parser.add_argument(
        "--cv-folds",
        type=int,
        default=config.CV_CONFIG.get("max_splits", 3),
        help="Number of cross-validation folds to use"
    )
    
    return parser.parse_args()


def sanitize_for_path(text: str) -> str:
    """
    Sanitize a string to be safe for use in file paths.
    
    Args:
        text: String to sanitize
        
    Returns:
        Sanitized string safe for file paths
    """
    # replace problematic chars with underscores
    replacements = {
        '/': '_',
        '\\': '_',
        ':': '_',
        '*': '_',
        '?': '_',
        '"': '_',
        '<': '_',
        '>': '_',
        '|': '_',
        ' ': '_'
    }
    
    sanitized = text
    for char, replacement in replacements.items():
        sanitized = sanitized.replace(char, replacement)
    
    return sanitized


def load_preprocessed_data(
    input_dir: Path
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """
    Load preprocessed training data and metadata from Step 1.
    
    Args:
        input_dir: Directory containing preprocessed data
        
    Returns:
        Tuple containing:
        - train_val_data: DataFrame with training/validation data
        - metadata: Dictionary with preprocessing metadata
        
    Raises:
        FileNotFoundError: If required files are missing
    """
    logger.info("\n" + "=" * 60)
    logger.info("LOADING PREPROCESSED DATA")
    logger.info("=" * 60)
    logger.info(f"Loading from: {input_dir}")
    
    # Define file paths
    train_file = input_dir / config.DATA_CONFIG["train_file_name"]
    val_file = input_dir / config.DATA_CONFIG["validation_file_name"]
    metadata_file = input_dir / config.DATA_CONFIG["metadata_file_name"]
    
    # check if files exsit
    required_files = [train_file, val_file, metadata_file]
    missing_files = [f for f in required_files if not f.exists()]
    
    if missing_files:
        error_msg = (
            f"Required files not found: {missing_files}\n"
            f"Please run 1_preprocess_exp1.py first."
        )
        logger.error(error_msg)
        raise FileNotFoundError(error_msg)
    
    logger.info(f"Loading training data from {train_file.name}...")
    train_data = pd.read_csv(train_file, parse_dates=['date'])
    logger.info(f"  - Loaded {len(train_data)} training rows")

    logger.info(f"Loading validation data from {val_file.name}...")
    val_data = pd.read_csv(val_file, parse_dates=['date'])
    logger.info(f"  - Loaded {len(val_data)} validation rows")
    
    logger.info("Combining training and validation data...")
    train_val_data = pd.concat([train_data, val_data], ignore_index=True)
    train_val_data = train_val_data.sort_values(['id', 'date']).reset_index(drop=True)
    
    logger.info(f"  - Combined shape: {train_val_data.shape}")
    logger.info(f"  - Unique series: {train_val_data['id'].nunique()}")
    logger.info(f"  - Date range: {train_val_data['date'].min()} to {train_val_data['date'].max()}")

    logger.info(f"Loading metadata from {metadata_file.name}...")
    with open(metadata_file, 'r') as f:
        metadata = json.load(f)
    logger.info("  - Metadata loaded successfully")
    
    return train_val_data, metadata


def validate_series_for_training(
    series_data: pd.DataFrame, 
    series_id: str,
    min_periods: int = 20,
    min_non_zero: int = 2
) -> bool:
    """
    Validate if a series has sufficient data for classical model training.
    
    Classical models need:
    - Minimum number of historical periods
    - Some non-zero sales (for intermittent demand models)
    - Some variability in the data
    
    Args:
        series_data: DataFrame for a single series
        series_id: ID of the series
        min_periods: Minimum required periods
        min_non_zero: Minimum required non-zero periods
        
    Returns:
        True if series is valid for training, False otherwise
    """
    sales_values = series_data['sales'].values
    
    if len(sales_values) < min_periods:
        logger.debug(
            f"Series '{series_id}': Too short ({len(sales_values)} < {min_periods})"
        )
        return False
    
    non_zero_count = np.sum(sales_values > 0)
    if non_zero_count < min_non_zero:
        logger.debug(
            f"Series '{series_id}': Too few non-zero values ({non_zero_count} < {min_non_zero})"
        )
        return False
    
    unique_values = len(np.unique(sales_values))
    if unique_values < 2:
        logger.debug(
            f"Series '{series_id}': No variability (all values constant)"
        )
        return False
    
    return True


def get_model_instance(model_type: str, hyperparams: Dict) -> Any:
    """
    Create an instance of the specified classical model.
    
    Args:
        model_type: Type of model ('arima', 'croston', 'tsb', etc.)
        hyperparams: Model-specific hyperparameters
        
    Returns:
        Instantiated model object
        
    Raises:
        ValueError: If model type is not supported
    """
    # Map model type strings to class constructors
    model_mapping = {
        "arima": ARIMAForecaster,
        "croston": CrostonsMethod,
        "croston_sba": CrostonSBA,
        "tsb": TSBForecaster,
        "moving_average": MovingAverage,
        "ses": SimpleExponentialSmoothing,
        "holt_winters": HoltWintersForecaster,
        "ets": ETS,
        "adida": ADIDAForecaster,
        "imapa": IMAPAForecaster
    }
    
    if model_type not in model_mapping:
        raise ValueError(f"Unsupported model type: {model_type}")
    
    # Create model instance with hyperparameters
    model_class = model_mapping[model_type]
    return model_class(**hyperparams)


def perform_cross_validation_for_series(
    series_data: pd.DataFrame,
    model_type: str,
    hyperparams: Dict,
    cv_config: Dict
) -> Tuple[Any, Dict[str, Any]]:
    """
    Perform expanding window cross-validation for a single series.
    
    Args:
        series_data: Training data for this series (sorted by date)
        model_type: Type of model to train
        hyperparams: Model hyperparameters
        cv_config: Cross-validation configuration
        
    Returns:
        Tuple containing:
        - best_model: The best performing model from CV
        - cv_results: Dictionary with CV metrics and fold details
    """
    # Extract sales values for CV
    sales_values = series_data['sales'].values
    
    # Initialize walk-forward validator
    validator = WalkForwardValidator(
        initial_train_size=cv_config['initial_train_size'],
        step_size=cv_config['step_size'],
        max_splits=cv_config['max_splits']
    )
    
    # Get CV splits
    cv_splits = validator.split(sales_values.reshape(1, -1))
    
    if not cv_splits:
        # Not enough data for CV, train on all data
        model = get_model_instance(model_type, hyperparams)
        model.fit(sales_values)
        return model, {"cv_used": False, "reason": "insufficient_data"}
    
    # Perform CV folds
    fold_results = []
    fold_models = []
    
    for fold_idx, split in enumerate(cv_splits):
        # Extract train and validation data for this fold
        train_data = split.train_data[0]  # First (and only) series
        val_data = split.val_data[0]
        
        # Train model on this fold
        fold_model = get_model_instance(model_type, hyperparams)
        fold_model.fit(train_data)
        
        # Generate predictions for validation period
        val_predictions = fold_model.predict(h=len(val_data))
        
        # Calculate validation metrics
        val_actuals = val_data
        
        # Handle case where predictions might be shorter than validation data
        min_length = min(len(val_actuals), len(val_predictions))
        val_actuals = val_actuals[:min_length]
        val_predictions = val_predictions[:min_length]
        
        if min_length > 0:
            fold_metrics = {
                'fold': fold_idx + 1,
                'train_size': len(train_data),
                'val_size': len(val_data),
                'rmse': float(np.sqrt(np.mean((val_actuals - val_predictions) ** 2))),
                'mae': float(np.mean(np.abs(val_actuals - val_predictions))),
                'mape': float(np.mean(np.abs((val_actuals - val_predictions) / 
                                           (val_actuals + 1e-8))) * 100)
            }
        else:
            # Handle edge case where no validation data
            fold_metrics = {
                'fold': fold_idx + 1,
                'train_size': len(train_data),
                'val_size': 0,
                'rmse': np.inf,
                'mae': np.inf,
                'mape': np.inf
            }
        
        fold_results.append(fold_metrics)
        fold_models.append(fold_model)
    
    # Select best model based on RMSE
    valid_folds = [i for i, r in enumerate(fold_results) if np.isfinite(r['rmse'])]
    
    if valid_folds:
        best_fold_idx = min(valid_folds, key=lambda i: fold_results[i]['rmse'])
        best_model = fold_models[best_fold_idx]
    else:
        # All folds failed, train on all data as fallback
        best_model = get_model_instance(model_type, hyperparams)
        best_model.fit(sales_values)
        best_fold_idx = -1
    
    # Aggregate CV results
    if valid_folds:
        avg_metrics = {
            'avg_rmse': float(np.mean([fold_results[i]['rmse'] for i in valid_folds])),
            'avg_mae': float(np.mean([fold_results[i]['mae'] for i in valid_folds])),
            'avg_mape': float(np.mean([fold_results[i]['mape'] for i in valid_folds])),
            'std_rmse': float(np.std([fold_results[i]['rmse'] for i in valid_folds])),
            'std_mae': float(np.std([fold_results[i]['mae'] for i in valid_folds])),
            'std_mape': float(np.std([fold_results[i]['mape'] for i in valid_folds]))
        }
    else:
        avg_metrics = {
            'avg_rmse': np.inf,
            'avg_mae': np.inf,
            'avg_mape': np.inf,
            'std_rmse': 0.0,
            'std_mae': 0.0,
            'std_mape': 0.0
        }
    
    cv_results = {
        'cv_used': True,
        'n_folds': len(cv_splits),
        'n_valid_folds': len(valid_folds),
        'best_fold': best_fold_idx + 1 if best_fold_idx >= 0 else None,
        'fold_results': fold_results,
        'aggregated_metrics': avg_metrics
    }
    
    return best_model, cv_results


def train_model_for_series(
    series_id: str,
    series_data: pd.DataFrame,
    model_type: str,
    hyperparams: Dict,
    force_retrain: bool
) -> Dict[str, Any]:
    """
    Train a single classical model for a single series.
    
    Supports both regular training and cross-validation based on configuration.
    
    Args:
        series_id: Unique identifier for the series
        series_data: Training data for this series
        model_type: Type of model to train
        hyperparams: Model hyperparameters
        force_retrain: Whether to retrain existing models
        
    Returns:
        Dictionary with training results and metadata
    """
    # Prep file paths
    model_dir = config.get_model_storage_dir(model_type)
    model_dir.mkdir(parents=True, exist_ok=True)
    
    safe_series_id = sanitize_for_path(series_id)
    model_file = model_dir / f"{safe_series_id}_{model_type}_model.pkl"
    metadata_file = model_dir / f"{safe_series_id}_{model_type}_metadata.json"
    
    # Check if model already exists
    if model_file.exists() and metadata_file.exists() and not force_retrain:
        logger.debug(f"Model exists for '{series_id}' ({model_type}), skipping")
        return {
            "status": "skipped_exists",
            "model_file": str(model_file)
        }
    
    
    logger.debug(f"Training {model_type} for series '{series_id}'...")
    
    try:
        # Sort data by date
        series_data_sorted = series_data.sort_values('date')
        sales_values = series_data_sorted['sales'].values
        
        start_time = time.time()
        
        use_cv = config.CV_CONFIG.get("use_cv", False)
        cv_results = None
        
        if use_cv:
            logger.debug(f"  Using cross-validation for {series_id}")
            
            # Perform cross validation
            model, cv_results = perform_cross_validation_for_series(
                series_data=series_data_sorted,
                model_type=model_type,
                hyperparams=hyperparams,
                cv_config=config.CV_CONFIG
            )
        else:
            # Regular training without CV
            model = get_model_instance(model_type, hyperparams)
            model.fit(sales_values)
        
        training_time = time.time() - start_time
        
        # Save trained model
        with open(model_file, 'wb') as f:
            pickle.dump(model, f)
        
        # Prep metadata
        metadata = {
            "series_id": series_id,
            "model_type": model_type,
            "training_timestamp": datetime.now().isoformat(),
            "hyperparameters": hyperparams,
            "training_time_seconds": training_time,
            "cross_validation_used": use_cv,
            "training_data_stats": {
                "n_periods": len(sales_values),
                "n_non_zero": int(np.sum(sales_values > 0)),
                "mean_sales": float(np.mean(sales_values)),
                "std_sales": float(np.std(sales_values)),
                "min_sales": float(np.min(sales_values)),
                "max_sales": float(np.max(sales_values))
            },
            "model_file": model_file.name,
            "model_path_relative": str(model_file.relative_to(PROJECT_ROOT))
        }
        
        # Add CV results if available
        if cv_results is not None:
            metadata["cross_validation_results"] = cv_results
        
    
        with open(metadata_file, 'w') as f:
            json.dump(metadata, f, indent=2)
        
        logger.debug(f"Successfully trained {model_type} for '{series_id}' in {training_time:.2f}s")
        
        return {
            "status": "success",
            "training_time": training_time,
            "model_file": str(model_file),
            "metadata_file": str(metadata_file),
            "cv_used": use_cv,
            "cv_results": cv_results
        }
        
    except Exception as e:
        logger.warning(f"Error training {model_type} for '{series_id}': {str(e)}")
        return {
            "status": "error",
            "error_message": str(e)
        }


def train_all_models_for_series(
    series_id: str,
    series_data: pd.DataFrame,
    models_to_train: List[str],
    force_retrain: bool
) -> Dict[str, Dict]:
    """
    Train all specified models for a single series.
    
    Args:
        series_id: Unique identifier for the series
        series_data: Training data for this series
        models_to_train: List of model types to train
        force_retrain: Whether to retrain existing models
        
    Returns:
        Dictionary mapping model types to their training results
    """
    results = {}
    
    for model_type in models_to_train:
        # Get model hyperparameters from config
        hyperparams = config.get_model_config_params(model_type)
        
        # Train 
        model_results = train_model_for_series(
            series_id=series_id,
            series_data=series_data,
            model_type=model_type,
            hyperparams=hyperparams,
            force_retrain=force_retrain
        )
        
        results[model_type] = model_results
    
    return results

def train_all_series_parallel(
    train_data: pd.DataFrame,
    models_to_train: List[str],
    n_jobs: int,
    limit_series: Optional[int],
    force_retrain: bool
) -> Dict[str, Dict[str, Dict]]:
    """
    Train classical models for all series, optionally in parallel.
    
    Args:
        train_data: Full training dataset with all series
        models_to_train: List of model types to train
        n_jobs: Number of parallel jobs
        limit_series: Optional limit on number of series
        force_retrain: Whether to retrain existing models
        
    Returns:
        Nested dictionary: {series_id: {model_type: results}}
    """
    logger.info("\n" + "=" * 60)
    logger.info("PREPARING SERIES FOR TRAINING")
    logger.info("=" * 60)
    
    all_series_ids = train_data['id'].unique()
    
    # Apply series limit if given
    if limit_series is not None and limit_series > 0:
        all_series_ids = all_series_ids[:limit_series]
        logger.info(f"Limiting to first {len(all_series_ids)} series")
    else:
        logger.info(f"Processing all {len(all_series_ids)} series")
    
    
    valid_tasks = []
    skipped_series = {}
    
    logger.info("\nValidating series...")
    for series_id in all_series_ids:
        # get data for this series
        series_data = train_data[train_data['id'] == series_id]
        
        # validate series
        min_periods = config.DATA_CONFIG.get('min_train_periods_classical', 20)
        min_non_zero = config.DATA_CONFIG.get('min_non_zero_periods', 2)
        
        if validate_series_for_training(series_data, series_id, min_periods, min_non_zero):
            valid_tasks.append((series_id, series_data))
        else:
            # Record why series was skipped
            skipped_series[series_id] = {
                model_type: {"status": "skipped_validation"} 
                for model_type in models_to_train
            }
    
    # TODO: make sure LSTM uses same filtering criteria - it's training on 500 series vs our 499
    # TODO: probably some sparse household series that we skip but LSTM tries to train on
    # TODO: that might be causing the crazy MASE values (1600+) in evaluation
    logger.info(f"  - Valid series: {len(valid_tasks)}")
    logger.info(f"  - Skipped series: {len(skipped_series)}")
    
    if not valid_tasks:
        logger.warning("No valid series found for training!")
        return skipped_series
    
    # Train models
    logger.info("\n" + "=" * 60)
    logger.info("TRAINING CLASSICAL MODELS")
    logger.info("=" * 60)
    
    all_results = {}
    all_results.update(skipped_series)
    
    if n_jobs > 1 and len(valid_tasks) > 1:
        # Parallel training
        logger.info(f"Training in parallel with {n_jobs} workers...")
        
        with ProcessPoolExecutor(max_workers=n_jobs) as executor:
            # Submit tasks
            future_to_series = {
                executor.submit(
                    train_all_models_for_series,
                    series_id,
                    series_data,
                    models_to_train,
                    force_retrain
                ): series_id
                for series_id, series_data in valid_tasks
            }
            
            # progress bar
            with tqdm(total=len(valid_tasks), desc="Training Progress") as pbar:
                for future in as_completed(future_to_series):
                    series_id = future_to_series[future]
                    try:
                        results = future.result()
                        all_results[series_id] = results
                    except Exception as e:
                        logger.error(f"Failed to train models for '{series_id}': {e}")
                        all_results[series_id] = {
                            model_type: {"status": "error", "error_message": str(e)}
                            for model_type in models_to_train
                        }
                    pbar.update(1)
    else:
        logger.info("Training sequentially...")
        
        for series_id, series_data in tqdm(valid_tasks, desc="Training Progress"):
            results = train_all_models_for_series(
                series_id=series_id,
                series_data=series_data,
                models_to_train=models_to_train,
                force_retrain=force_retrain
            )
            all_results[series_id] = results
    
    return all_results

def save_training_summary(
    results: Dict[str, Dict[str, Dict]],
    models_trained: List[str],
    total_runtime: float,
    args: argparse.Namespace
) -> Path:
    """
    Save a comprehensive summary of the training run.
    
    Args:
        results: Training results for all series and models
        models_trained: List of model types that were trained
        total_runtime: Total runtime in seconds
        args: Command-line arguments used
        
    Returns:
        Path to saved summary file
    """
    # 
    summary_dir = config.RESULTS_BASE_DIR / "classical_training_summary"
    summary_dir.mkdir(parents=True, exist_ok=True)
    
    # make filename with timestamp
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    summary_file = summary_dir / f"classical_training_summary_{timestamp}.json"
    
    counts = {
        "successful": 0,
        "errors": 0,
        "skipped_exists": 0,
        "skipped_validation": 0
    }
    
    for series_id, model_results in results.items():
        for model_type, result in model_results.items():
            status = result.get("status", "unknown")
            
            if status == "success":
                counts["successful"] += 1
            elif status == "error":
                counts["errors"] += 1
            elif status == "skipped_exists":
                counts["skipped_exists"] += 1
            elif status == "skipped_validation":
                counts["skipped_validation"] += 1
    
    summary = {
        "run_timestamp": datetime.now().isoformat(),
        "runtime_seconds": total_runtime,
        "configuration": {
            "models_trained": models_trained,
            "input_directory": str(args.input_dir),
            "parallel_jobs": args.parallel_jobs,
            "series_limit": args.limit_series,
            "force_retrain": args.force
        },
        "statistics": {
            "total_series": len(results),
            "total_model_instances": len(results) * len(models_trained),
            "successful_trains": counts["successful"],
            "failed_trains": counts["errors"],
            "skipped_existing": counts["skipped_exists"],
            "skipped_validation": counts["skipped_validation"]
        },
        "detailed_results": results
    }
    
    
    with open(summary_file, 'w') as f:
        json.dump(summary, f, indent=2)
    
    logger.info(f"\nSaved training summary to: {summary_file}")
    
    return summary_file


def main():
    """
    Main workflow for training classical models.
    
    This function handles the entire training process:
    1. Parse command-line arguments
    2. Load preprocessed data
    3. Train models for each series
    4. Save results and summary
    """ 
    args = parse_arguments()
    set_random_seed(config.RANDOM_SEED)
    
    logger.info("=" * 60)
    logger.info("STARTING CLASSICAL MODEL TRAINING")
    logger.info("=" * 60)
    logger.info(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    logger.info(f"Models to train: {args.models_to_train}")
    logger.info(f"Input directory: {args.input_dir}")
    logger.info(f"Parallel jobs: {args.parallel_jobs}")
    logger.info(f"Series limit: {args.limit_series}")
    logger.info(f"Force retrain: {args.force}")
    
    #TODO figure out override
    # Override CV configuration with command-line arguments
    if args.use_cv:
        # Temporarily override config for this run
        config.CV_CONFIG["use_cv"] = True
        config.CV_CONFIG["max_splits"] = args.cv_folds
    
    use_cv = config.CV_CONFIG.get("use_cv", False)
    if use_cv:
        logger.info("\nCross-Validation Configuration:")
        logger.info(f"  - CV enabled: {use_cv}")
        logger.info(f"  - Initial train size: {config.CV_CONFIG['initial_train_size']} days")
        logger.info(f"  - Step size: {config.CV_CONFIG['step_size']} days")
        logger.info(f"  - Max splits: {config.CV_CONFIG['max_splits']}")
        logger.info(f"  - Gap: {config.CV_CONFIG.get('gap', 0)} days")
        if args.use_cv:
            logger.info(f"  - CV enabled")  # cross validation
    else:
        logger.info(f"\nCross-validation disabled")
    
    
    start_time = time.time()
    
    # Load preprocessed data 
    try:
        train_data, metadata = load_preprocessed_data(Path(args.input_dir))
    except FileNotFoundError as e:
        logger.error(str(e))
        sys.exit(1)
    
    # Train models for all series
    results = train_all_series_parallel(
        train_data=train_data,
        models_to_train=args.models_to_train,
        n_jobs=args.parallel_jobs,
        limit_series=args.limit_series,
        force_retrain=args.force
    )
    
    total_runtime = time.time() - start_time
    summary_file = save_training_summary(
        results=results,
        models_trained=args.models_to_train,
        total_runtime=total_runtime,
        args=args
    )
    
    logger.info("\n" + "=" * 60)
    logger.info("TRAINING COMPLETED")
    logger.info("=" * 60)
    
    # Count final statistics
    successful = sum(1 for s_results in results.values() 
                    for r in s_results.values() 
                    if r.get("status") == "success")
    errors = sum(1 for s_results in results.values() 
                for r in s_results.values() 
                if r.get("status") == "error")
    skipped = sum(1 for s_results in results.values() 
                 for r in s_results.values() 
                 if r.get("status", "").startswith("skipped"))
    
    logger.info(f"Total runtime: {total_runtime:.2f} seconds")
    logger.info(f"Successful model trains: {successful}")
    logger.info(f"Failed model trains: {errors}")
    logger.info(f"Skipped model trains: {skipped}")

if __name__ == "__main__":
    main()