#////////////////////////////////////////////////////////////////////////////////#
# File:         config.py                                                        #
# Author:       Douglas Nyberg                                                   #
# Email:        douglas1.nyberg@gmail.com                                        #
# Date:         2025-03-05                                                       #
# Description:  Configuration settings for supply chain forecasting project.    #
#////////////////////////////////////////////////////////////////////////////////#




"""
Configuration settings for the supply chain forecasting project.
"""
import os
from pathlib import Path

# Project directory structure
PROJECT_ROOT = Path(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
TRAINED_MODELS_DIR = PROJECT_ROOT / "trained_models"
PREDICTIONS_DIR = PROJECT_ROOT / "predictions"
RESULTS_DIR = PROJECT_ROOT / "results"

# M5 competition specific settings
M5_HORIZON = 28  # Forecast horizon for M5 competition
M5_QUANTILE_LEVELS = [0.005, 0.025, 0.165, 0.250, 0.500, 0.750, 0.835, 0.975, 0.995]
M5_WEEKLY_SEASONALITY = 7  # Weekly seasonality for M5 retail data (7 days) - primary seasonality in retail
M5_YEARLY_SEASONALITY = 365  # Yearly seasonality for daily data (365 days) - for long-term seasonal patterns

# Data settings
DEFAULT_SEQUENCE_LENGTH = 28  # Default input sequence length for LSTM model
VALIDATION_DAYS = 56  # Number of days to use for validation (sequence_length + forecast_horizon)

# Training settings
RANDOM_SEED = 42
DEFAULT_BATCH_SIZE = 64
DEFAULT_LEARNING_RATE = 0.0001  # Reduced from 0.001 to help prevent NaN issues
DEFAULT_EPOCHS = 50
DEFAULT_PATIENCE = 10

# Cross-validation settings
CV_N_SPLITS = 3  # Number of time series cross-validation splits for HPO
CV_GAP = 0  # Gap between train and validation periods (days)
CV_MIN_TRAIN_SIZE = 60  # Minimum training window size (days)

# LSTM model settings
DEFAULT_HIDDEN_DIM = 64
DEFAULT_NUM_LAYERS = 2
DEFAULT_DROPOUT = 0.2
USE_BIDIRECTIONAL = False
USE_ATTENTION = True

# Feature engineering settings
USE_PRICE_FEATURES = True
USE_LAG_FEATURES = True
MAX_LAGS = [7, 14, 28]  # Lag values to use
USE_ROLLING_FEATURES = True
ROLLING_WINDOWS = [7, 14, 28]  # Rolling window sizes
USE_CALENDAR_FEATURES = True
USE_EVENT_FEATURES = False  # Disabled for synthetic Monte Carlo data
USE_INTERMITTENCY_FEATURES = True  # Add features for intermittent demand patterns
PROMOTION_THRESHOLD = 0.9  # Price below this fraction of average is considered a promotion

# Data file configuration
DATA_CONFIG = {
    "train_file_name": "train_data.csv",
    "validation_file_name": "validation_data.csv", 
    "test_file_name": "test_data.csv",
    "metadata_file_name": "preprocessing_metadata.json",
    "embedding_file_name": "embedding_info_full.pkl"
}

# Evaluation settings
WRMSSE_WEIGHT_BASED_ON = "sales"  # "sales" or "units" or "equal"
WSPL_WEIGHT_HIGHER_QUANTILES = True  # Give more weight to higher quantiles