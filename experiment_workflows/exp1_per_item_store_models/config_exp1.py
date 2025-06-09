#////////////////////////////////////////////////////////////////////////////////#
# File:         config_exp1.py                                                   #
# Author:       Douglas Nyberg                                                   #
# Email:        douglas1.nyberg@gmail.com                                        #
# Date:         2025-03-28                                                       #
#////////////////////////////////////////////////////////////////////////////////#
#!/usr/bin/env python3
"""
Config for Exp1: Item-level forecasting comparison.

Supports LSTM and classical methods with same data setup.
"""

# imports
import os
from pathlib import Path
import torch # For checking GPU availability
from typing import Dict, List, Tuple, Optional, Any

# experiment metadata
EXPERIMENT_NAME = "unified_item_level_exp1"
EXPERIMENT_TAG = "per_item_lstm_vs_classical" # More specific tag
EXPERIMENT_DESCRIPTION = """
Unified item-level forecasting comparison:
- LSTM deep learning models (trained PER ITEM, using rich features and embeddings)
- Classical methods (ARIMA, Croston, TSB, trained PER ITEM)
All models use M5 competition data, with identical data splits and evaluation metrics.
"""

# base paths
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
DATA_DIR = PROJECT_ROOT / "data"
LOGS_DIR = PROJECT_ROOT / "logs"
# Store results under a directory that reflects the experiment tag
RESULTS_BASE_DIR = PROJECT_ROOT / "results" / f"exp1_{EXPERIMENT_TAG}"
MODELS_BASE_DIR = PROJECT_ROOT / "trained_models" / f"exp1_{EXPERIMENT_TAG}"
PREDICTIONS_BASE_DIR = PROJECT_ROOT / "predictions" / f"exp1_{EXPERIMENT_TAG}"

# Ensure all necessary directories exist
for directory in [LOGS_DIR, RESULTS_BASE_DIR, MODELS_BASE_DIR, PREDICTIONS_BASE_DIR]:
    directory.mkdir(parents=True, exist_ok=True)

# data config
DEFAULT_SPLICE_FILE = DATA_DIR / "m5_splices" / "all_stores_500_500_items_features.csv"
PREPROCESSED_DATA_OUTPUT_DIR = DATA_DIR / "preprocessed" / f"exp1_{EXPERIMENT_TAG}"
PREPROCESSED_DATA_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

DATA_CONFIG = {
    "splice_file": str(DEFAULT_SPLICE_FILE),
    "preprocessed_output_dir": str(PREPROCESSED_DATA_OUTPUT_DIR),
    "train_file_name": "train_data.csv",  # Training data
    "validation_file_name": "validation_data.csv",  # Validation data
    "train_val_file_name": "train_data.csv",  # Combined train+val for LSTM (uses train_data.csv)
    "test_file_name": "test_data.csv",  # Holdout test set
    "metadata_file_name": "preprocessing_metadata.json",
    "embedding_info_pickle_name": "embedding_info_full.pkl", # For LSTM per-item model setup
    "limit_items": None,  # e.g., 10 for quick testing, None for all.
    "min_non_zero_periods": 5,
    "validation_days": 28, # Days for validation set (kept for backward compatibility)
    "test_days": 28, # Days for test set (M5 horizon)
    "use_m5_official_split": True,  # Use M5 competition train/test methodology
}

# LSTM config
# Optimized hyperparameters for intermittent demand forecasting
LSTM_CONFIG = {
    "model_name_prefix": "item_lstm", # Prefix for per-item model files
    "hidden_dim": 64, # increased capacity for complex patterns
    "num_layers": 2,  # Publication-optimized: deeper architecture
    "dropout": 0.2, # Publication-optimized: balanced regularization
    "bidirectional": False, # Keep simple for stability
    "use_attention": True, # Publication-optimized: enable attention for better pattern capture

    "batch_size": 32, # Publication-optimized: larger batches for stability
    "learning_rate": 0.001, # Publication-optimized: higher learning rate for better convergence
    "weight_decay": 0.0001, # Keep minimal weight decay
    "epochs": 75, # Publication-optimized: more epochs for thorough training
    "patience": 10, # Publication-optimized: increased patience for complex patterns
    "gradient_clip": 1.0,

    "sequence_length": 28, # HPO optimized: confirmed optimal
    "forecast_horizon": 28,

    # HPO optimized regularization
    "l1_reg": 1.4321698289111523e-05,  # L1 regularization weight
    "l2_reg": 1.1430983876313214e-05,  # L2 regularization weight

    "quantile_output": True, # Each per-item LSTM will output quantiles
    "quantiles": [0.005, 0.025, 0.05, 0.1, 0.165, 0.250, 0.500, 0.750, 0.835, 0.9, 0.95, 0.975, 0.995],
    
    # Feature selection flags from HPO
    "use_price_features": True,
    "use_lag_features": True,
    "use_rolling_features": True,
    "use_calendar_features": False, # Optimized: calendar features disabled
    "use_event_features": True,
}

# forecast config
FORECAST_CONFIG = {
    "horizon": 28,
    "quantiles": LSTM_CONFIG["quantiles"],
    "point_forecast_method": "median",
}

# classical model configs
ARIMA_CONFIG = {
    "order": (1, 1, 1),  # (p, d, q) parameters
    "seasonal_order": (0, 0, 0, 0)  # (P, D, Q, s) parameters
}
CROSTON_CONFIG = { 
    "alpha": 0.094
}  # Statsforecast CrostonClassic only takes alpha parameter (optimized from HPO)
TSB_CONFIG = { 
    "alpha": 0.1,  # alpha_d parameter (demand smoothing)
    "beta": 0.1    # alpha_p parameter (probability smoothing) 
}

# Additional classical model configurations
CROSTON_SBA_CONFIG = {
    "alpha": 0.1  # Smoothing parameter for SBA variant
}
ADIDA_CONFIG = {
    # ADIDA takes no parameters - optimizes internally
}
IMAPA_CONFIG = {
    # same
}
HOLT_WINTERS_CONFIG = {
    "season_length": 7,  # Weekly seasonality for M5 data
    "error_type": "add",  # Additive error
    "trend_type": "add",  # Additive trend
    "season_type": "add"  # Additive seasonality
}
ETS_CONFIG = {
    "season_length": 7  # Weekly seasonality
}
SES_CONFIG = {
    "alpha": 0.1  # Smoothing parameter
}
MOVING_AVERAGE_CONFIG = {
    "window_size": 7  # 7-day moving average
}

# cross-validation config
CV_CONFIG = {
    "use_cv": True,  # Set to True to enable cross-validation
    "initial_train_size": 90,  # Minimum training days (3+ months recommended)
    "step_size": 28,  # M5 horizon - how much to advance each fold
    "max_splits": 3,  # Number of CV folds (3-5 recommended for M5 data)
    "gap": 0,  # Gap between train and validation (if needed)
}

# forecasting config
FORECAST_CONFIG = {
    "horizon": 28,
    "quantiles": LSTM_CONFIG["quantiles"], # Consistent quantiles
    "point_forecast_method": "median",
}

# evaluation config
EVALUATION_CONFIG = {
    # Removed MAPE and SMAPE - fundamentally unsuitable for intermittent demand forecasting
    # Focus on metrics used in M5 competition and meaningful for retail research
    "metrics": ["wrmsse", "mase", "rmse", "mae"],
    "primary_metric": "wrmsse",
    "seasonality": 7,
    "quantile_metrics": ["quantile_loss", "coverage_rate"],
}

# workflow config
WORKFLOW_CONFIG = {
    "parallel_jobs": min(4, os.cpu_count() or 1), # For both classical and LSTM per-item
    "save_predictions": True, "save_models": True, "generate_plots": True,
    "plot_sample_items": 5,
}

# visualization config
VISUALIZATION_CONFIG = {
    "style": "seaborn-v0_8-darkgrid",
    "figure_size": (12, 6),
    "dpi": 100,
}

# model selection
AVAILABLE_MODELS = {
    "classical": ["arima", "croston", "croston_sba", "tsb", "adida", "imapa", "holt_winters", "ets", "ses", "moving_average"],
    "deep_learning": ["lstm"] # Represents the per-item LSTM strategy
}
DEFAULT_CLASSICAL_MODELS_TO_RUN = ["croston", "tsb"]
DEFAULT_DEEP_LEARNING_MODELS_TO_RUN = ["lstm"]

# logging config
LOG_FILE_PATH = LOGS_DIR / f"{EXPERIMENT_NAME}_{EXPERIMENT_TAG}.log"
LOGGING_CONFIG = {
    "level": "DEBUG",
    "format": "%(asctime)s - %(name)s - %(levelname)s - %(module)s:%(lineno)d - %(message)s",
    "log_file": str(LOG_FILE_PATH)
}

# hardware config
# For per-item LSTMs, CPU parallelization is primary. GPU could be used sequentially if chosen.
DEVICE_CONFIG = {
    "use_gpu_if_available_for_lstm_sequential_train": False, # Set to True to train LSTMs one-by-one on GPU
    "device_name_lstm": "cuda" if torch.cuda.is_available() and True else "cpu", # Target device for LSTMs
    "num_workers_dataloader_lstm": 0, # For per-item, simpler dataloading
    "pin_memory_dataloader_lstm": False,
}

# reproducibility
RANDOM_SEED = 42

# helper functions
def get_model_config_params(model_name: str) -> dict:
    model_name_lower = model_name.lower()
    if model_name_lower == "lstm": return LSTM_CONFIG
    if model_name_lower == "arima": return ARIMA_CONFIG
    if model_name_lower == "croston": return CROSTON_CONFIG
    if model_name_lower == "croston_sba": return CROSTON_SBA_CONFIG
    if model_name_lower == "tsb": return TSB_CONFIG
    if model_name_lower == "adida": return ADIDA_CONFIG
    if model_name_lower == "imapa": return IMAPA_CONFIG
    if model_name_lower == "holt_winters": return HOLT_WINTERS_CONFIG
    if model_name_lower == "ets": return ETS_CONFIG
    if model_name_lower == "ses": return SES_CONFIG
    if model_name_lower == "moving_average": return MOVING_AVERAGE_CONFIG
    raise ValueError(f"Unknown model name for configuration: {model_name}")

def _get_model_category_path_name(model_name: str) -> str:
    if model_name.lower() in AVAILABLE_MODELS["classical"]: return "classical"
    if model_name.lower() in AVAILABLE_MODELS["deep_learning"]: return "deep_learning"
    raise ValueError(f"Model {model_name} not in classical or deep_learning categories.")

def get_model_storage_dir(model_name: str) -> Path:
    category_path = _get_model_category_path_name(model_name)
    # Each model type gets its own subfolder for its per-item models
    return MODELS_BASE_DIR / category_path / model_name.lower()

def get_predictions_output_dir(run_specific_experiment_name: str, model_name: str) -> Path:
    return PREDICTIONS_BASE_DIR / run_specific_experiment_name / model_name.lower()

def get_evaluation_output_dir(run_specific_experiment_name: str, model_name: str) -> Path:
    return RESULTS_BASE_DIR / run_specific_experiment_name / model_name.lower() / "evaluation"

def get_visualization_output_dir(run_specific_experiment_name: str) -> Path:
    return RESULTS_BASE_DIR / run_specific_experiment_name / "visualizations"

# end of config