#////////////////////////////////////////////////////////////////////////////////#
# File:         config_mc_master.py                                             #
# Author:       Douglas Nyberg                                                  #
# Email:        douglas1.nyberg@gmail.com                                       #
# Date:         2025-06-06                                                      #
# Description:  Config for monte carlo synthetic data experiments.              #
# Affiliation:  Physics Department, Purdue University                           #
#////////////////////////////////////////////////////////////////////////////////#

"""
Config for monte carlo synthetic data experiments.

This is basically the same setup as exp1 but for synthetic data instead
of real M5 data. should make it easier to test things.
"""

import os
from pathlib import Path
import torch
from typing import Dict, List, Tuple, Optional, Any

# experiment info
EXPERIMENT_NAME = "monte_carlo_synthetic_exp"
EXPERIMENT_TAG = "synthetic_demand_comparison"
EXPERIMENT_DESCRIPTION = """
monte carlo synthetic data forecasting comparison experiment.
compares LSTM vs classical methods on synthetic intermittent demand.
basically the same as exp1 but with fake data so we can test faster.
"""

# setup paths
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
DATA_DIR = PROJECT_ROOT / "data"
LOGS_DIR = PROJECT_ROOT / "logs"
RESULTS_BASE_DIR = PROJECT_ROOT / "results" / f"exp_mc_{EXPERIMENT_TAG}"
MODELS_BASE_DIR = PROJECT_ROOT / "trained_models" / f"exp_mc_{EXPERIMENT_TAG}"
PREDICTIONS_BASE_DIR = PROJECT_ROOT / "predictions" / f"exp_mc_{EXPERIMENT_TAG}"

# monte carlo data directories
MC_DATA_DIR = DATA_DIR / "synthetic_monte_carlo"
SYNTHETIC_DATA_DIR = MC_DATA_DIR / "m5_format"
PREPROCESSED_DATA_OUTPUT_DIR = MC_DATA_DIR / "preprocessed"

# make sure directories exsit
for directory in [LOGS_DIR, RESULTS_BASE_DIR, MODELS_BASE_DIR, PREDICTIONS_BASE_DIR, MC_DATA_DIR, SYNTHETIC_DATA_DIR, PREPROCESSED_DATA_OUTPUT_DIR]:
    directory.mkdir(parents=True, exist_ok=True)

# data config
DATA_CONFIG = {
    "raw_data_dir": str(SYNTHETIC_DATA_DIR),
    "preprocessed_output_dir": str(PREPROCESSED_DATA_OUTPUT_DIR),
    "train_file_name": "train_data.csv",
    "validation_file_name": "validation_data.csv", 
    "train_val_file_name": "train_val_combined.csv",
    "test_file_name": "test_data.csv",
    "metadata_file_name": "preprocessing_metadata.json",
    "embedding_info_pickle_name": "embedding_info_full.pkl",
    "limit_items": None,  # use all synthetic items
    "min_non_zero_periods": 3,  # lower threshold for synthetic data
    "validation_days": 28,
    "test_days": 28,
    "use_m5_official_split": True,  # keep same split logic
}

# monte carlo data generation config
MC_DATA_CONFIG = {
    "num_items": 50,  # how many synthetic items to generate
    "num_years": 3,   # years of synthetic data
    "height_range": {"LOW": 1, "HIGH": 15},  # demand spike sizes
    "delta_time_range": {"LOW": 7, "HIGH": 60},  # days between spikes
    "max_ease_range": 3,  # falloff spread around peaks
    "default_output_file": str(MC_DATA_DIR / "test_synthetic_data.h5"),  # default output path
    # TODO: maybe add more complex seasonality later
}

# LSTM config (these hyperparams came from HPO on synthetic data)
LSTM_CONFIG = {
    "model_name_prefix": "mc_lstm",
    
    # architecture settings (from hpo)
    "hidden_dim": 112,
    "num_layers": 1,
    "dropout": 0.1,
    "bidirectional": False,
    "use_attention": True,
    
    # training settings (hpo found these work well)
    "batch_size": 32,
    "learning_rate": 0.0005450937646755395,  # weird specific value from hpo
    "weight_decay": 0.00022563110735889096,  # also from hpo
    "epochs": 150,
    "patience": 10,
    "gradient_clip": 2.0,
    
    # regularization (hpo tuned)
    "l1_reg": 1.15761823424808e-06,
    "l2_reg": 1.5395382367241133e-06,
    
    # sequence config
    "sequence_length": 28,
    "forecast_horizon": 28,
    
    # quantile forecasting setup
    "quantile_output": True,
    "quantiles": [0.005, 0.025, 0.05, 0.1, 0.165, 0.250, 0.500, 0.750, 0.835, 0.9, 0.95, 0.975, 0.995],
    
    # feature config (simplified for synthetic data)
    "use_price_features": False,    # no complex pricing in synthetic data
    "use_lag_features": True,       # keep lag features
    "use_rolling_features": True,   # keep rolling stats
    "use_calendar_features": True,  # basic calendar stuff
    "use_event_features": False,    # no events in synthetic data
    "use_snap_features": False,     # no snap benefits
    "use_holiday_features": False,  # no holidays
}

# hpo config
HPO_CONFIG = {
    # bayesian optimization settings
    "n_trials": 50,                                 # number of hpo trials
    "n_initial_points": 10,                         # random init trials
    "random_seed": 42,                              # for reproducability
    
    # evaluation settings
    "limit_series": 20,                             # series to evaluate per trial
    "cv_folds": 3,                                  # cross validation folds
    "hpo_training_epochs": 10,                      # epochs during hpo (not full training)
    
    # search space for lstm hyperparameters
    "search_space": {
        # architecture stuff
        "hidden_dim": (32, 128, 16),                # range with step size
        "num_layers": (1, 3, 1),                    # 1 to 3 layers
        "dropout": (0.1, 0.4, 0.05),               # dropout rate
        "bidirectional": [True, False],             # boolean choice
        "use_attention": [True, False],             # attention mechanism
        "sequence_length": [28, 42],                # sequence lengths to try
        
        # training params
        "batch_size": (16, 64, 16),                 # batch sizes
        "learning_rate": (1e-4, 1e-3, "log"),      # log scale search
        "weight_decay": (1e-5, 1e-3, "log"),       # weight decay
        "epochs": (50, 150, 25),                   # training epochs
        "patience": (10, 20, 5),                   # early stopping patience
        "gradient_clip": (0.5, 2.0, 0.5),          # gradient clipping
        
        # regularization
        "l1_reg": (1e-6, 1e-4, "log"),             # l1 regularization
        "l2_reg": (1e-6, 1e-4, "log"),             # l2 regularization
    }
}

# classical model configs (mostly defaults but some tuned)
ARIMA_CONFIG = {
    "order": (1, 1, 1),  # basic arima config
    "seasonal_order": (0, 0, 0, 0)  # no seasonality for now
}

CROSTON_CONFIG = { 
    "alpha": 0.094  # tuned alpha for croston
}

TSB_CONFIG = { 
    "alpha": 0.1,  # tsb alpha
    "beta": 0.1    # tsb beta
}

CROSTON_SBA_CONFIG = {
    "alpha": 0.1  # croston sba alpha
}

ADIDA_CONFIG = {}  # uses defaults

IMAPA_CONFIG = {}  # uses defaults

HOLT_WINTERS_CONFIG = {
    "season_length": 7,  # weekly seasonality
    "error_type": "add",
    "trend_type": "add",
    "season_type": "add"
}

ETS_CONFIG = {
    "season_length": 7  # weekly seasonality
}

SES_CONFIG = {
    "alpha": 0.1  # simple exponential smoothing alpha
}

MOVING_AVERAGE_CONFIG = {
    "window_size": 7  # 7 day moving average
}

# all classical configs in one place
CLASSICAL_CONFIG = {
    "arima": ARIMA_CONFIG,
    "croston": CROSTON_CONFIG,
    "tsb": TSB_CONFIG,
    "croston_sba": CROSTON_SBA_CONFIG,
    "adida": ADIDA_CONFIG,
    "imapa": IMAPA_CONFIG,
    "holt_winters": HOLT_WINTERS_CONFIG,
    "ets": ETS_CONFIG,
    "ses": SES_CONFIG,
    "moving_average": MOVING_AVERAGE_CONFIG,
}

# cross validation config
CV_CONFIG = {
    "use_cv": True,
    "initial_train_size": 60,  # shorter for synthetic data
    "step_size": 28,
    "max_splits": 3,
    "gap": 0,  # no gap between train/test
}

# forecasting config
FORECAST_CONFIG = {
    "horizon": 28,
    "quantiles": LSTM_CONFIG["quantiles"],  # use same quantiles as lstm
    "point_forecast_method": "median",  # use median for point forecasts
}

# evaluation config
EVALUATION_CONFIG = {
    "metrics": ["rmsse", "mase", "smape", "rmse", "mae"],  # rmsse not wrmsse for synthetic
    "primary_metric": "rmsse",
    "seasonality": 7,  # weekly seasonality
    "quantile_metrics": ["quantile_loss", "coverage_rate"],
}

# logging config
LOGGING_CONFIG = {
    "level": "INFO",
    "format": "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    "log_file": LOGS_DIR / "preprocessing.log",
}

# visualization config
VISUALIZATION_CONFIG = {
    "plot_sample_items": 5,  # how many items to plot
    "figure_size": (12, 8),
    "dpi": 150,
}

# what models are available
AVAILABLE_MODELS = {
    "classical": ["arima", "croston", "tsb", "croston_sba", "adida", "imapa", "holt_winters", "ets", "ses", "moving_average"],
    "deep_learning": ["lstm"]
}

# default models to run (subset for faster testing)
DEFAULT_CLASSICAL_MODELS_TO_RUN = ["arima", "croston", "tsb"]  # just the main ones
DEFAULT_DEEP_LEARNING_MODELS_TO_RUN = ["lstm"]

# workflow config
WORKFLOW_CONFIG = {
    "parallel_jobs": min(4, os.cpu_count() or 1),  # dont use all cores
    "save_predictions": True, 
    "save_models": True, 
    "generate_plots": True,
    "plot_sample_items": 5,
}

# device config (force cpu for testing)
DEVICE_CONFIG = {
    "use_gpu_if_available_for_lstm_sequential_train": False,  # force cpu
    "device_name_lstm": "cpu",                                # cpu only
    "num_workers_dataloader_lstm": 0,
    "pin_memory_dataloader_lstm": False,                      # disable for cpu
    "use_gpu_if_available": False,                            # force cpu
    "device_name": "cpu",                                     # cpu only
    "num_workers_dataloader": 0,
    "pin_memory_dataloader": False,                           # disable for cpu
}

# reproducability
RANDOM_SEED = 42

# helper functions
def get_model_config_params(model_name: str) -> dict:
    """get config params for a specific model"""
    model_name_lower = model_name.lower()
    
    # check each model type
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
    
    # if we get here, unknown model
    raise ValueError(f"dont know model: {model_name}")

def _get_model_category_path_name(model_name: str) -> str:
    """figure out if model is classical or deep learning"""
    if model_name.lower() in AVAILABLE_MODELS["classical"]: 
        return "classical"
    if model_name.lower() in AVAILABLE_MODELS["deep_learning"]: 
        return "deep_learning"
    raise ValueError(f"model {model_name} not in any category")

def get_model_storage_dir(model_name: str) -> Path:
    """get directory where model should be saved"""
    category_path = _get_model_category_path_name(model_name)
    return MODELS_BASE_DIR / category_path / model_name.lower()

def get_predictions_output_dir(run_specific_experiment_name: str, model_name: str) -> Path:
    """get directory for prediction outputs"""
    return PREDICTIONS_BASE_DIR / run_specific_experiment_name / model_name.lower()

def get_evaluation_output_dir(run_specific_experiment_name: str, model_name: str) -> Path:
    """get directory for evaluation results"""
    return RESULTS_BASE_DIR / run_specific_experiment_name / model_name.lower() / "evaluation"

def get_visualization_output_dir(run_specific_experiment_name: str) -> Path:
    """get directory for plots and visualizations"""
    return RESULTS_BASE_DIR / run_specific_experiment_name / "visualizations"

def get_model_names(model_type: str = "all"):
    """get list of available model names"""
    if model_type == "classical":
        return AVAILABLE_MODELS["classical"]
    elif model_type == "deep_learning":
        return AVAILABLE_MODELS["deep_learning"]
    else:  # all or any other value
        return AVAILABLE_MODELS["classical"] + AVAILABLE_MODELS["deep_learning"]

# end of config file