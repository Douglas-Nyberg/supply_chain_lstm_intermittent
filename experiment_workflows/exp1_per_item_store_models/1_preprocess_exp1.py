#////////////////////////////////////////////////////////////////////////////////#
# File:         1_preprocess_exp1.py                                             #
# Author:       Douglas Nyberg                                                   #
# Email:        douglas1.nyberg@gmail.com                                        #
# Date:         2025-03-15                                                       #
#////////////////////////////////////////////////////////////////////////////////#
#!/usr/bin/env python3
"""
Unified Data Preprocessing Script for Experiment 1 (Per-Item-Per-Store).

This script prepares M5 competition data for forecasting models, operating at the
individual item-store level. It ensures data quality, applies feature engineering,
and creates train/validation/test splits suitable for both LSTM and classical models.

Main Steps:
1. Load M5 data splice
2. Validate data quality for each item-store combination
3. Apply feature engineering (lag features, rolling statistics, calendar features)
4. Create train_val and test splits
5. Save preprocessed data and metadata
"""

import argparse
import json
import logging
import pickle
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple, Any, Optional

import numpy as np
import pandas as pd


SCRIPT_DIR = Path(__file__).resolve().parent

PROJECT_ROOT = SCRIPT_DIR.parent.parent

sys.path.append(str(PROJECT_ROOT))

from src.data_loader import load_m5_splice
from src.feature_engineering import (
    preprocess_m5_features_for_embeddings,
    get_categorical_feature_specs,
    create_feature_set,
)


from experiment_workflows.exp1_per_item_store_models import config_exp1 as config


# Create logs directory
config.LOGS_DIR.mkdir(parents=True, exist_ok=True)

# avoid duplicate logs
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



# HELPER FUNCTIONS
def parse_arguments() -> argparse.Namespace:
    """
    Parse command-line arguments for the preprocessing script.
    
    Returns:
        argparse.Namespace: Parsed command-line arguments
    """
    parser = argparse.ArgumentParser(
        description="Unified Preprocessing for M5 Experiment 1",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    parser.add_argument(
        "--splice-file", 
        type=str,
        default=config.DATA_CONFIG["splice_file"],
        help="Path to the input M5 data splice file (CSV format)"
    )
    
    parser.add_argument(
        "--output-dir", 
        type=str,
        default=config.DATA_CONFIG["preprocessed_output_dir"],
        help="Directory where preprocessed data will be saved"
    )
    
    parser.add_argument(
        "--limit-items", 
        type=int,
        default=config.DATA_CONFIG["limit_items"],
        help="Limit the number of item-store series to process (None = all)"
    )
    
    parser.add_argument(
        "--force", 
        action="store_true",
        help="Reprocess data even if output files already exist"
    )
    
    return parser.parse_args()


def check_feature_engineering_config_consistency():
    """
    Check if feature engineering settings are consistent between configs.
    
    Sets up feature engineering flags if they don't exist in the experiment config.
    """
    try:
        from src import config as src_config
        
        # List of feature flags to check/set
        feature_flags = [
            'USE_LAG_FEATURES',
            'USE_ROLLING_FEATURES', 
            'USE_CALENDAR_FEATURES',
            'USE_PRICE_FEATURES',
            'USE_EVENT_FEATURES',
            'USE_INTERMITTENCY_FEATURES'
        ]
        
        # Set missing feature flags from src config or use defaults
        missing_flags = []
        for flag in feature_flags:
            if not hasattr(config, flag):
                # Get value from src config or default to True
                src_value = getattr(src_config, flag, True)
                setattr(config, flag, src_value)
                missing_flags.append(f"  - {flag}: set to {src_value}")
        
        if missing_flags:
            logger.info(
                "Added missing feature engineering config:\n" + 
                "\n".join(missing_flags)
            )
            
    except Exception as e:
        logger.warning(f"Could not check config consistency: {e}")
        # Set reasonable defaults if import fails
        if not hasattr(config, 'USE_LAG_FEATURES'):
            config.USE_LAG_FEATURES = True
        if not hasattr(config, 'USE_ROLLING_FEATURES'):
            config.USE_ROLLING_FEATURES = True
        if not hasattr(config, 'USE_CALENDAR_FEATURES'):
            config.USE_CALENDAR_FEATURES = True
        if not hasattr(config, 'USE_PRICE_FEATURES'):
            config.USE_PRICE_FEATURES = True
        if not hasattr(config, 'USE_EVENT_FEATURES'):
            config.USE_EVENT_FEATURES = True
        if not hasattr(config, 'USE_INTERMITTENCY_FEATURES'):
            config.USE_INTERMITTENCY_FEATURES = True
        
        # Also add the feature lists that feature engineering expects
        if not hasattr(config, 'LAG_FEATURES'):
            config.LAG_FEATURES = [7, 14, 28]  # Default lag features
        if not hasattr(config, 'ROLLING_WINDOWS'):
            config.ROLLING_WINDOWS = [7, 14, 28]  # Default rolling windows


def validate_data(
    data: pd.DataFrame,
    sequence_length: int,
    forecast_horizon: int,
    test_days: int,
    cv_folds: int = 3
) -> Tuple[pd.DataFrame, List[str], Dict[str, Any]]:
    """
    Validate that each item-store series has sufficient data for all models.
    
    This function ensures each time series has enough historical data to:
    - Create LSTM sequences of the required length
    - Perform cross-validation with the specified number of folds
    - Reserve data for the test period
    
    Args:
        data: Input dataframe with columns ['id', 'date', 'sales', ...]
        sequence_length: Number of historical days needed for LSTM input
        forecast_horizon: Number of days to forecast ahead
        test_days: Number of days to reserve for testing
        cv_folds: Number of cross-validation folds
        
    Returns:
        Tuple containing:
        - valid_data: DataFrame with only valid time series
        - dropped_ids: List of item-store IDs that were dropped
        - validation_stats: Dictionary with validation statistics
    """
    logger.info("=" * 60)
    logger.info("VALIDATING DATA FOR ALL MODELS")
    logger.info("=" * 60)
    
    # Calculate minimum required data points
    # For LSTM: need at least sequence_length + forecast_horizon for one training sample
    min_train_points = sequence_length + forecast_horizon
    
    # For CV: need enough data for all folds plus test period
    min_total_points = (cv_folds * forecast_horizon) + sequence_length + test_days
    
    logger.info(f"Minimum requirements:")
    logger.info(f"  - Sequence length: {sequence_length} days")
    logger.info(f"  - Forecast horizon: {forecast_horizon} days")
    logger.info(f"  - Test period: {test_days} days")
    logger.info(f"  - CV folds: {cv_folds}")
    logger.info(f"  - Min points per series: {min_total_points}")
    
    # Group by item store ID and check each series
    valid_ids = []
    dropped_ids = []
    series_lengths = []
    
    for item_id, item_data in data.groupby('id'):
        series_length = len(item_data)
        series_lengths.append(series_length)
        
        if series_length >= min_total_points:
            valid_ids.append(item_id)
        else:
            dropped_ids.append(item_id)
            logger.debug(
                f"Dropping {item_id}: has {series_length} points, "
                f"needs {min_total_points}"
            )
    
    # Filter to keep only valid series
    valid_data = data[data['id'].isin(valid_ids)].copy()
    
    # Calculate validation statistics
    validation_stats = {
        'total_series': len(data['id'].unique()),
        'valid_series': len(valid_ids),
        'dropped_series': len(dropped_ids),
        'min_series_length': min(series_lengths) if series_lengths else 0,
        'max_series_length': max(series_lengths) if series_lengths else 0,
        'avg_series_length': np.mean(series_lengths) if series_lengths else 0,
        'min_required_points': min_total_points
    }
    
    # Log validation results
    logger.info(f"\nValidation Results:")
    logger.info(f"  - Total series: {validation_stats['total_series']}")
    logger.info(f"  - Valid series: {validation_stats['valid_series']}")
    logger.info(f"  - Dropped series: {validation_stats['dropped_series']}")
    logger.info(f"  - Series length range: [{validation_stats['min_series_length']}, "
                f"{validation_stats['max_series_length']}]")
    
    if dropped_ids:
        logger.warning(f"Dropped {len(dropped_ids)} series due to insufficient data")
        logger.debug(f"First 10 dropped IDs: {dropped_ids[:10]}")
    
    return valid_data, dropped_ids, validation_stats


def create_train_test_split(
    data: pd.DataFrame,
    use_m5_official_split: bool = True
) -> Tuple[pd.DataFrame, pd.DataFrame, str]:
    """
    Create train/test split using official M5 competition methodology.
    
    M5 Competition Split:
    - Training: Days 1-1913 (from sales_train_validation.csv)  
    - Test: Days 1914-1941 (28-day forecast period)
    
    Args:
        data: Input dataframe with sales data
        use_m5_official_split: If True, use M5 competition split; if False, use last 28 days
        
    Returns:
        Tuple containing:
        - train_data: Training data (days 1-1913 for M5)
        - test_data: Test data (days 1914-1941 for M5)  
        - split_info: Description of the split
    """
    logger.info("\n" + "=" * 60)
    logger.info("CREATING M5 COMPETITION-COMPLIANT TRAIN/TEST SPLIT")
    logger.info("=" * 60)
    
    # Get unique dates and sort them
    unique_dates = sorted(data['date'].unique())
    total_days = len(unique_dates)
    
    logger.info(f"Total unique dates: {total_days}")
    logger.info(f"Date range: {unique_dates[0]} to {unique_dates[-1]}")
    
    if use_m5_official_split:
        # M5 Competition methodology: days 1914-1941 are test period
        # This assumes the data has exactly 1941 days (d_1 to d_1941)
        if total_days >= 1941:
            # Use last 28 days as test (days 1914-1941)
            split_date = unique_dates[-28]
            split_info = "M5 Official Split: Train=days 1-1913, Test=days 1914-1941"
            logger.info("Using M5 competition official split")
            logger.info("Train period: Days 1-1913 (sales_train_validation period)")
            logger.info("Test period: Days 1914-1941 (M5 evaluation period)")
        else:
            # For smaller datasets (like splices), use proportional split  
            test_days = 28
            split_date = unique_dates[-test_days]
            split_info = f"M5-style split on subset: last {test_days} days as test"
            logger.info(f"Dataset has {total_days} days (< 1941), using last 28 days as test")
    else:
        # Legacy split: last 28 days
        test_days = 28
        split_date = unique_dates[-test_days]
        split_info = f"Legacy split: last {test_days} days as test"
        logger.info("Using legacy split (last 28 days)")
    
    logger.info(f"Split date: {split_date}")
    
    # Split the data
    train_data = data[data['date'] < split_date].copy()
    test_data = data[data['date'] >= split_date].copy()
    
    # Verify the split
    train_days = len(train_data['date'].unique())
    test_days_actual = len(test_data['date'].unique())
    
    logger.info(f"\nSplit Results:")
    logger.info(f"  - Train: {len(train_data)} rows, {train_days} days")
    logger.info(f"  - Test: {len(test_data)} rows, {test_days_actual} days")
    logger.info(f"  - Split methodology: {split_info}")
    
    return train_data, test_data, split_info


def create_train_test_split_by_days(
    data: pd.DataFrame,
    test_days: int
) -> Tuple[pd.DataFrame, pd.DataFrame, str]:
    """
    Split data into train_val and test sets based on date.
    
    The test set contains the most recent 'test_days' of data,
    while train_val contains all data before that.
    
    Args:
        data: Input dataframe with a 'date' column
        test_days: Number of days to include in test set
        
    Returns:
        Tuple containing:
        - train_val_data: Training and validation data
        - test_data: Test data (most recent days)
        - split_date: The date where the split occurs
    """
    logger.info("\n" + "=" * 60)
    logger.info("CREATING TRAIN/TEST SPLIT")
    logger.info("=" * 60)
    
    # Get unique dates and sort them
    unique_dates = sorted(data['date'].unique())
    total_days = len(unique_dates)
    
    logger.info(f"Total unique dates: {total_days}")
    logger.info(f"Date range: {unique_dates[0]} to {unique_dates[-1]}")
    
    # Calculate split date
    # The test set gets the last 'test_days' dates
    split_date = unique_dates[-test_days]
    
    logger.info(f"Split date: {split_date}")
    logger.info(f"Test period: {split_date} to {unique_dates[-1]} ({test_days} days)")
    
    # Split the data
    train_val_data = data[data['date'] < split_date].copy()
    test_data = data[data['date'] >= split_date].copy()
    
    # Verify the split
    train_val_days = len(train_val_data['date'].unique())
    test_days_actual = len(test_data['date'].unique())
    
    logger.info(f"\nSplit Results:")
    logger.info(f"  - Train/Val: {len(train_val_data)} rows, {train_val_days} days")
    logger.info(f"  - Test: {len(test_data)} rows, {test_days_actual} days")
    
    # Verify each series appears in both sets
    train_val_ids = set(train_val_data['id'].unique())
    test_ids = set(test_data['id'].unique())
    
    if train_val_ids != test_ids:
        missing_in_test = train_val_ids - test_ids
        missing_in_train = test_ids - train_val_ids
        
        if missing_in_test:
            logger.warning(f"Series in train but not test: {len(missing_in_test)}")
        if missing_in_train:
            logger.warning(f"Series in test but not train: {len(missing_in_train)}")
    
    return train_val_data, test_data, str(split_date)


def reorder_features_for_lstm_compatibility(data: pd.DataFrame) -> pd.DataFrame:
    """
    TEMPORARY: Reorder numerical features to match trained LSTM models.
    
    The trained LSTM models expect features in a specific order. This function
    reorders them to ensure compatibility.
    
    TODO: Remove this when retraining models with flexible feature ordering.
    """
    logger = logging.getLogger(__name__)
    
    # Expected order from trained LSTM models
    expected_numerical_order = [
        'wm_yr_wk', 'wday', 'year', 'snap_CA', 'snap_TX', 'snap_WI', 'sell_price',
        'day_of_week', 'day_of_month', 'day_of_year', 'is_weekend', 'week_of_year',
        'quarter_numeric', 'day_of_week_sin', 'day_of_week_cos', 'month_sin', 'month_cos',
        'lag_7', 'lag_14', 'lag_28', 'rolling_7_mean', 'rolling_7_std', 'rolling_7_max',
        'rolling_7_min', 'rolling_14_mean', 'rolling_14_std', 'rolling_14_max', 'rolling_14_min',
        'rolling_28_mean', 'rolling_28_std', 'rolling_28_max', 'rolling_28_min',
        'price_change', 'price_relative_to_avg', 'price_7d_avg', 'is_promotion',
        'has_event_name_1', 'has_event_type_1', 'has_event_name_2', 'has_event_type_2',
        'days_since_last_sale', 'zero_streak_length', 'intermittency_rate',
        'avg_demand_interval', 'demand_cv'
    ]
    
    # Get non-numerical columns to preserve  
    # Note: 'quarter' is categorical and should NOT be in numerical features
    all_numerical_cols = set(expected_numerical_order)
    non_numerical_cols = [col for col in data.columns if col not in all_numerical_cols]
    
    # Reorder: non-numerical columns + numerical columns in expected order
    present_numerical = [col for col in expected_numerical_order if col in data.columns]
    missing_numerical = [col for col in expected_numerical_order if col not in data.columns]
    
    if missing_numerical:
        logger.warning(f"Missing expected numerical features: {missing_numerical}")
    
    # TEMPORARY: Remove categorical quarter completely for LSTM compatibility
    non_numerical_cols_filtered = [col for col in non_numerical_cols if col != 'quarter']
    
    reordered_columns = non_numerical_cols_filtered + present_numerical
    reordered_data = data[reordered_columns].copy()
    
    logger.info(f"Reordered features for LSTM compatibility: {len(present_numerical)} numerical features")
    
    return reordered_data


def apply_feature_engineering(
    data: pd.DataFrame,
    categorical_specs: Dict[str, Dict]
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """
    Apply feature engineering to the data.
    
    Since the M5 splice already contains engineered features, this function
    primarily processes categorical features for embeddings and creates metadata.
    
    Args:
        data: Input dataframe with pre-engineered features
        categorical_specs: Specifications for categorical features
        
    Returns:
        Tuple containing:
        - engineered_data: DataFrame with all features
        - feature_info: Dictionary with feature metadata
    """
    logger.info("\n" + "=" * 60)
    logger.info("APPLYING FEATURE ENGINEERING")
    logger.info("=" * 60)
    
    # Log feature engineering configuration
    logger.info("Feature Engineering Configuration:")
    logger.info(f"  - Lag features: {config.USE_LAG_FEATURES}")
    logger.info(f"  - Rolling features: {config.USE_ROLLING_FEATURES}")
    logger.info(f"  - Calendar features: {config.USE_CALENDAR_FEATURES}")
    logger.info(f"  - Price features: {config.USE_PRICE_FEATURES}")
    logger.info(f"  - Event features: {config.USE_EVENT_FEATURES}")
    logger.info(f"  - Intermittency features: {config.USE_INTERMITTENCY_FEATURES}")
    
    # Check if intermittency features are missing and need to be generated
    intermittency_features = ['avg_demand_interval', 'days_since_last_sale', 'demand_cv', 'intermittency_rate', 'quarter_numeric', 'zero_streak_length']
    missing_intermittency = [f for f in intermittency_features if f not in data.columns]
    
    if missing_intermittency and config.USE_INTERMITTENCY_FEATURES:
        logger.info(f"Missing intermittency features: {missing_intermittency}")
        logger.info("Running full feature engineering pipeline to generate missing features")
        
        # Run full feature engineering pipeline
        from src.feature_engineering import create_feature_set
        engineered_data = create_feature_set(
            df=data,
            target_col='sales',
            id_col='id',
            date_col='date',
            price_col='sell_price'
        )
        
        # Then process categorical features for embeddings
        engineered_data, feature_info_dict, embedding_info = preprocess_m5_features_for_embeddings(engineered_data)
        
        # TEMPORARY: Reorder features to match trained LSTM models
        engineered_data = reorder_features_for_lstm_compatibility(engineered_data)
        
    else:
        # Data splice already contains engineered features - processing categorical features only
        logger.info("Data splice already contains engineered features - processing categorical features only")
        
        # Use the simple preprocessing function that works with pre-engineered data
        engineered_data, feature_info_dict, embedding_info = preprocess_m5_features_for_embeddings(data)
    
    # Count features by type from the data columns
    all_columns = list(engineered_data.columns)
    
    # Identify categorical and numerical features
    # TEMPORARY: Exclude 'quarter' for compatibility with existing trained LSTM models
    # TODO: Include 'quarter' when retraining models with categorical quarter embeddings
    categorical_features = ['weekday', 'month', 'event_name_1', 'event_type_1', 'event_name_2', 'event_type_2']
    categorical_features = [f for f in categorical_features if f in all_columns]
    
    # All other features (except id, date, sales, ID columns, and day index) are numerical
    # ID columns contain string values and cannot be scaled
    # Note: For per-item-per-store models, ID columns are constants and not useful
    id_columns = ['item_id', 'dept_id', 'cat_id', 'store_id', 'state_id']
    # 'd' column is M5 day index (d_1, d_2, ..., d_1941) - useful for tracking but not for modeling
    exclude_cols = ['id', 'date', 'sales', 'd'] + categorical_features + id_columns
    numerical_features = [f for f in all_columns if f not in exclude_cols]
    
    feature_info = {
        'total_features': len(numerical_features) + len(categorical_features),
        'numerical_features': numerical_features,
        'categorical_features': categorical_features,
        'embedding_specs': embedding_info,
        'feature_counts': {
            'numerical': len(numerical_features),
            'categorical': len(categorical_features),
            'lag': len([f for f in numerical_features if f.startswith('lag_')]),
            'rolling': len([f for f in numerical_features if 'rolling_' in f]),
            'calendar': len([f for f in numerical_features if any(
                cal in f for cal in ['month', 'week', 'dayofweek', 'quarter']
            )]),
        }
    }
    
    logger.info(f"\nFeature Engineering Results:")
    logger.info(f"  - Total features: {feature_info['total_features']}")
    logger.info(f"  - Numerical: {feature_info['feature_counts']['numerical']}")
    logger.info(f"  - Categorical: {feature_info['feature_counts']['categorical']}")
    logger.info(f"  - Lag features: {feature_info['feature_counts']['lag']}")
    logger.info(f"  - Rolling features: {feature_info['feature_counts']['rolling']}")
    
    return engineered_data, feature_info


def save_preprocessed_data(
    train_val_data: pd.DataFrame,
    test_data: pd.DataFrame,
    output_dir: Path,
    metadata: Dict[str, Any]
) -> None:
    """
    Save preprocessed data and metadata to disk.
    
    Args:
        train_val_data: Training and validation data
        test_data: Test data
        output_dir: Directory to save files
        metadata: Preprocessing metadata to save
    """
    logger.info("\n" + "=" * 60)
    logger.info("SAVING PREPROCESSED DATA")
    logger.info("=" * 60)
    
    # Create output directory if it doesn't exist
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Define file paths
    train_val_path = output_dir / "train_data.csv"
    test_path = output_dir / "test_data.csv"
    validation_path = output_dir / "validation_data.csv"
    metadata_path = output_dir / "preprocessing_metadata.json"
    embedding_info_path = output_dir / "embedding_info_full.pkl"
    
    # Save train_val data
    logger.info(f"Saving train_val data to {train_val_path}")
    train_val_data.to_csv(train_val_path, index=False)
    logger.info(f"  - Saved {len(train_val_data)} rows")
    
    # Save test data
    logger.info(f"Saving test data to {test_path}")
    test_data.to_csv(test_path, index=False)
    logger.info(f"  - Saved {len(test_data)} rows")
    
    # For backward compatibility, also save validation data (last part of train_val)
    #TODO cleanup backward compatibility
    validation_days = config.DATA_CONFIG.get('validation_days', 28)
    unique_dates = sorted(train_val_data['date'].unique())
    
    if len(unique_dates) > validation_days:
        val_start_date = unique_dates[-validation_days]
        validation_data = train_val_data[train_val_data['date'] >= val_start_date]
        
        logger.info(f"Saving validation data to {validation_path}")
        validation_data.to_csv(validation_path, index=False)
        logger.info(f"  - Saved {len(validation_data)} rows")
    
    # Save metadata
    logger.info(f"Saving metadata to {metadata_path}")
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=2, default=str)
    
    # Save embedding info 
    if 'embedding_info' in metadata:
        logger.info(f"Saving embedding info to {embedding_info_path}")
        with open(embedding_info_path, 'wb') as f:
            pickle.dump(metadata['embedding_info'], f)
    
    logger.info("\nAll files saved successfully!")

def main():
    """
    Main preprocessing workflow for Experiment 1.
    
    This function runs the entire preprocessing pipeline:
    1. Parse command-line arguments
    2. Check configuration consistency
    3. Load raw data
    4. Validate data quality
    5. Apply feature engineering
    6. Create train/test splits
    7. Save preprocessed data and metadata
    """
    
    args = parse_arguments()
    
    
    check_feature_engineering_config_consistency()
    
    logger.info("=" * 60)
    logger.info("STARTING UNIFIED PREPROCESSING FOR EXPERIMENT 1")
    logger.info("=" * 60)
    logger.info(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    logger.info(f"Splice file: {args.splice_file}")
    logger.info(f"Output directory: {args.output_dir}")
    logger.info(f"Item limit: {args.limit_items}")
    logger.info(f"Force reprocess: {args.force}")
    
    
    splice_path = Path(args.splice_file)
    output_dir = Path(args.output_dir)
    
    # Check if output files already exist
    if not args.force and (output_dir / "train_data.csv").exists():
        logger.info("\nOutput files already exist. Use --force to reprocess.")
        return
    
    # Load raw data 
    logger.info("\n" + "=" * 60)
    logger.info("LOADING RAW DATA")
    logger.info("=" * 60)
    
    if not splice_path.exists():
        logger.error(f"Splice file not found: {splice_path}")
        sys.exit(1)
    
    # Load data using our custom loader
    raw_data = load_m5_splice(
        str(splice_path),
        limit_items=args.limit_items
    )
    
    logger.info(f"Loaded data shape: {raw_data.shape}")
    logger.info(f"Columns: {list(raw_data.columns)}")
    logger.info(f"Unique items: {raw_data['id'].nunique()}")
    logger.info(f"Date range: {raw_data['date'].min()} to {raw_data['date'].max()}")
    
    # Validate data 
    # Get parameters from config
    sequence_length = config.LSTM_CONFIG['sequence_length']
    forecast_horizon = config.LSTM_CONFIG['forecast_horizon']
    test_days = config.DATA_CONFIG['test_days']
    cv_folds = config.CV_CONFIG.get('max_splits', 3) 
    
    valid_data, dropped_ids, validation_stats = validate_data(
        data=raw_data,
        sequence_length=sequence_length,
        forecast_horizon=forecast_horizon,
        test_days=test_days,
        cv_folds=cv_folds
    )
    
    if valid_data.empty:
        logger.error("No valid series found after validation!")
        sys.exit(1)
    
    # Apply feature engineering 
    # Get categorical feature specifications
    # TODO verify embedding features work with lstm
    categorical_specs = get_categorical_feature_specs(valid_data)
    
    engineered_data, feature_info = apply_feature_engineering(
        data=valid_data,
        categorical_specs=categorical_specs
    )
    
    # Create train/test split using M5 competition methodology
    use_m5_split = config.DATA_CONFIG.get('use_m5_official_split', True)
    train_val_data, test_data, split_info = create_train_test_split(
        data=engineered_data,
        use_m5_official_split=use_m5_split
    )
    
    # Prep metadata 
    metadata = {
        'preprocessing_timestamp': datetime.now().isoformat(),
        'config': {
            'splice_file': str(splice_path),
            'output_dir': str(output_dir),
            'limit_items': args.limit_items,
            'sequence_length': sequence_length,
            'forecast_horizon': forecast_horizon,
            'test_days': test_days,
            'cv_folds': cv_folds,
        },
        'data_stats': {
            'raw_shape': list(raw_data.shape),
            'valid_shape': list(valid_data.shape),
            'train_val_shape': list(train_val_data.shape),
            'test_shape': list(test_data.shape),
            'date_range': {
                'start': str(valid_data['date'].min()),
                'end': str(valid_data['date'].max()),
                'split_methodology': split_info
            }
        },
        'validation_stats': validation_stats,
        'feature_info': feature_info,
        'embedding_info': feature_info.get('embedding_specs', {}),
        'dropped_ids': dropped_ids[:100],  # Save first 100 for reference
    }
    
    # Save preprocessed data 
    save_preprocessed_data(
        train_val_data=train_val_data,
        test_data=test_data,
        output_dir=output_dir,
        metadata=metadata
    )
    
    # final summary
    logger.info("\n" + "=" * 60)
    logger.info("PREPROCESSING COMPLETED SUCCESSFULLY")
    logger.info("=" * 60)
    logger.info(f"Output directory: {output_dir}")
    logger.info(f"Total processing time: {datetime.now()}")
  


if __name__ == "__main__":
    main()