#////////////////////////////////////////////////////////////////////////////////#
# File:         cv_utils.py                                                      #
# Author:       Douglas Nyberg                                                   #
# Email:        douglas1.nyberg@gmail.com                                        #
# Date:         2025-04-02                                                       #
# Description:  CV utilities for time series. Index-based validation for models.#
# Affiliation:  Physics Department, Purdue University                            #
#////////////////////////////////////////////////////////////////////////////////#
#!/usr/bin/env python3

# add project root to path for imports
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

"""
Cross-validation utilities for time series forecasting.

This module provides:
1. IndexBasedWalkForwardValidator: Format-agnostic CV split generator
2. Data handlers for classical models (sales only)
3. Data handlers for LSTM models (with features and sequences)

Adheres to the 9 key principles for readability.
"""

import numpy as np
import pandas as pd
import logging
from typing import List, Tuple, Dict, Optional, Any

logger = logging.getLogger(__name__)

class IndexBasedWalkForwardValidator:
    """
    Index-based walk-forward validator that returns indices instead of data.
    
    This makes it format-agnostic and memory efficient. The validator
    generates expanding window splits where training data grows progressively.
    """
    
    def __init__(self, 
                 initial_train_size: int = 90,
                 step_size: int = 28,
                 max_splits: int = 3,
                 gap: int = 0):
        """
        Initialize validator with CV parameters.
        
        Args:
            initial_train_size: Minimum training period length
            step_size: How much to advance each fold (typically forecast horizon)
            max_splits: Maximum number of CV folds
            gap: Gap between train and validation (optional)
        """
        self.initial_train_size = initial_train_size
        self.step_size = step_size
        self.max_splits = max_splits
        self.gap = gap
        
        logger.info(f"Initialized IndexBasedWalkForwardValidator with:")
        logger.info(f"  Initial train size: {initial_train_size}")
        logger.info(f"  Step size: {step_size}")
        logger.info(f"  Max splits: {max_splits}")
        logger.info(f"  Gap: {gap}")
    
    def get_split_indices(self, data_length: int) -> List[Tuple[range, range]]:
        """
        Generate train/validation index ranges for CV splits.
        
        Args:
            data_length: Total length of the time series
            
        Returns:
            List of (train_indices, val_indices) tuples
        """
        splits = []
        
        # Validate minimum data requirements 
        min_required = self.initial_train_size + self.gap + self.step_size
        if data_length < min_required:
            logger.warning(
                f"Insufficient data for CV: have {data_length} days, "
                f"need at least {min_required} days"
            )
            return splits
        
        # Generate splits with expanding training window 
        train_end = self.initial_train_size - 1
        
        for i in range(self.max_splits):
            # Calculate validation start and end
            val_start = train_end + 1 + self.gap
            val_end = min(val_start + self.step_size - 1, data_length - 1)
            
            # Check if we have enough validation data
            if val_start >= data_length:
                logger.debug(f"Stopping at split {i}: no more data for validation")
                break
            
            # Check if validation period is too short
            actual_val_size = val_end - val_start + 1
            if actual_val_size < self.step_size * 0.5:  # Less than half expected size
                logger.warning(
                    f"Split {i}: Validation period too short ({actual_val_size} days), "
                    f"stopping CV generation"
                )
                break
                
            # Create index ranges
            train_indices = range(0, train_end + 1)
            val_indices = range(val_start, val_end + 1)
            
            splits.append((train_indices, val_indices))
            
            logger.debug(
                f"Split {i}: Train [0-{train_end}] ({len(train_indices)} days), "
                f"Val [{val_start}-{val_end}] ({len(val_indices)} days)"
            )
            
            # Expand training window for next split
            train_end = val_end
            
        logger.info(f"Generated {len(splits)} CV splits for series with {data_length} days")
        return splits


# Define Classical Model Data Handler 
def get_classical_fold_data(
    df: pd.DataFrame, 
    series_id: str,
    train_idx: range,
    val_idx: range
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Extract sales data for classical models from a specific CV fold.
    
    Classical models (ARIMA, Croston, TSB, etc...) only use sales history,
    not external features.
    
    Args:
        df: Long-format DataFrame with columns ['id', 'date', 'sales', ...]
        series_id: The specific series to extract
        train_idx: Training indices
        val_idx: Validation indices
        
    Returns:
        Tuple of (train_sales_1d, val_sales_1d) as numpy arrays
    """
    # Filter for specific series and sort by date 
    series_df = df[df['id'] == series_id].sort_values('date').reset_index(drop=True)
    
    if len(series_df) == 0:
        raise ValueError(f"No data found for series '{series_id}'")
    
    # Extract sales values 
    sales_values = series_df['sales'].values.astype(np.float32)
    
    # Validate indices 
    max_idx = len(sales_values) - 1
    if max(train_idx) > max_idx or max(val_idx) > max_idx:
        raise IndexError(
            f"Index out of bounds for series '{series_id}': "
            f"data has {len(sales_values)} points, but requested indices up to "
            f"{max(max(train_idx), max(val_idx))}"
        )
    
    # Apply indices to get fold data 
    train_sales = sales_values[train_idx]
    val_sales = sales_values[val_idx]
    
    logger.debug(
        f"Classical fold data for '{series_id}': "
        f"train shape {train_sales.shape}, val shape {val_sales.shape}"
    )
    
    return train_sales, val_sales


# define lstm model data handler 
def get_lstm_fold_data(
    df: pd.DataFrame,
    series_id: str,
    train_idx: range,
    val_idx: range,
    sequence_length: int,
    numerical_cols: List[str],
    categorical_cols: List[str],
    target_col: str = 'sales'
) -> Dict[str, Any]:
    """
    Extract and prepare sequence data for LSTM models from a CV fold.
    
    This function handles the complexity of creating sequences with proper
    context, especially for validation data that needs historical context.
    
    Args:
        df: Long-format DataFrame
        series_id: The specific series to extract
        train_idx: Training indices
        val_idx: Validation indices  
        sequence_length: Length of input sequences
        numerical_cols: List of numerical feature columns
        categorical_cols: List of categorical feature columns
        target_col: Name of target column (default: 'sales')
        
    Returns:
        Dictionary with prepared sequences and features for training
    """
    #  Filter for series and sort 
    series_df = df[df['id'] == series_id].sort_values('date').reset_index(drop=True)
    
    if len(series_df) == 0:
        raise ValueError(f"No data found for series '{series_id}'")
    
    # Extract relevant data 
    sales = series_df[target_col].values.astype(np.float32)
    
    # Extract numerical features if provided
    numerical_features = None
    if numerical_cols:
        numerical_features = series_df[numerical_cols].values.astype(np.float32)
    
    # Extract categorical features if provided
    categorical_features = {}
    if categorical_cols:
        for col in categorical_cols:
            categorical_features[col] = series_df[col].values.astype(np.int32)
    
    # Create training sequences 
    train_data = _create_sequences_from_indices(
        sales=sales[train_idx],
        numerical_features=numerical_features[train_idx] if numerical_features is not None else None,
        categorical_features={k: v[train_idx] for k, v in categorical_features.items()},
        sequence_length=sequence_length,
        forecast_horizon=28  # M5 forecast horizon
    )
    
    # Create validation sequences with context.
    # Need to include some training data for context.
    context_start = max(0, min(val_idx) - sequence_length)
    context_end = max(train_idx)
    
    val_data = _create_sequences_with_context(
        historical_sales=sales[context_start:context_end + 1],
        val_sales=sales[val_idx],
        historical_numerical=numerical_features[context_start:context_end + 1] if numerical_features is not None else None,
        val_numerical=numerical_features[val_idx] if numerical_features is not None else None,
        historical_categorical={k: v[context_start:context_end + 1] for k, v in categorical_features.items()},
        val_categorical={k: v[val_idx] for k, v in categorical_features.items()},
        sequence_length=sequence_length,
        forecast_horizon=28  # M5 forecast horizon
    )
    
    return {
        'train': train_data,
        'val': val_data,
        'series_id': series_id,
        'train_samples': len(train_data['X']),
        'val_samples': len(val_data['X'])
    }


# Helpers
def _create_sequences_from_indices(
    sales: np.ndarray,
    numerical_features: Optional[np.ndarray],
    categorical_features: Dict[str, np.ndarray],
    sequence_length: int,
    forecast_horizon: int = 28
) -> Dict[str, Any]:
    """
    Create sequences for training from consecutive data.
    
    Args:
        sales: Sales data array
        numerical_features: Numerical features array (optional)
        categorical_features: Dictionary of categorical features
        sequence_length: Length of each sequence
        forecast_horizon: Number of future steps to predict
        
    Returns:
        Dictionary with X (inputs) and y (targets) sequences
    """
    n_samples = len(sales) - sequence_length - forecast_horizon + 1
    
    if n_samples <= 0:
        logger.warning(f"Not enough data for sequences: {len(sales)} points, need {sequence_length + forecast_horizon}")
        return {'X': np.array([]), 'y': np.array([]), 'X_num': None, 'X_cat': {}}
    
    # Create sales sequences 
    X_sales = []
    y = []
    
    for i in range(n_samples):
        X_sales.append(sales[i:i + sequence_length])
        y.append(sales[i + sequence_length:i + sequence_length + forecast_horizon])
    
    X_sales = np.array(X_sales, dtype=np.float32)
    y = np.array(y, dtype=np.float32)
    
    # Create numerical feature sequences 
    X_num = None
    if numerical_features is not None:
        # Handle NaN values before creating sequences
        # Replace NaN with 0 (for lag/price features where missing history)
        numerical_features_clean = np.where(np.isnan(numerical_features), 0.0, numerical_features)
        
        X_num = []
        for i in range(n_samples):
            X_num.append(numerical_features_clean[i:i + sequence_length])
        X_num = np.array(X_num, dtype=np.float32)
    
    # Create categorical feature sequences
    X_cat = {}
    for feat_name, feat_values in categorical_features.items():
        X_cat[feat_name] = []
        for i in range(n_samples):
            X_cat[feat_name].append(feat_values[i:i + sequence_length])
        X_cat[feat_name] = np.array(X_cat[feat_name], dtype=np.int32)
    
    return {
        'X': X_sales,
        'y': y,
        'X_num': X_num,
        'X_cat': X_cat
    }


def _create_sequences_with_context(
    historical_sales: np.ndarray,
    val_sales: np.ndarray,
    historical_numerical: Optional[np.ndarray],
    val_numerical: Optional[np.ndarray],
    historical_categorical: Dict[str, np.ndarray],
    val_categorical: Dict[str, np.ndarray],
    sequence_length: int,
    forecast_horizon: int = 28
) -> Dict[str, Any]:
    """
    Create validation sequences using historical context for multi-step forecasting.
    
    This is crucial for proper validation as we need historical data
    to create the input sequences for predicting validation targets.
    
    Args:
        historical_sales: Historical sales including training period
        val_sales: Validation period sales
        historical_numerical: Historical numerical features
        val_numerical: Validation numerical features
        historical_categorical: Historical categorical features
        val_categorical: Validation categorical features
        sequence_length: Length of input sequences
        forecast_horizon: Number of future steps to predict
        
    Returns:
        Dictionary with validation sequences
    """
    # Combine historical and validation data 
    combined_sales = np.concatenate([historical_sales, val_sales])
    
    combined_numerical = None
    if historical_numerical is not None and val_numerical is not None:
        combined_numerical = np.concatenate([historical_numerical, val_numerical])
        combined_numerical = np.where(np.isnan(combined_numerical), 0.0, combined_numerical)
    
    combined_categorical = {}
    for feat_name in historical_categorical:
        combined_categorical[feat_name] = np.concatenate([
            historical_categorical[feat_name],
            val_categorical[feat_name]
        ])
    
    # Determine validation sequence creation strategy 
    # For multi-step forecasting, we need sequence_length + forecast_horizon points
    # We can create sequences starting from historical data that predict into validation period
    hist_len = len(historical_sales)
    combined_len = len(combined_sales)
    
    # Calculate how many validation sequences we can create
    # Each sequence needs sequence_length input + forecast_horizon output
    min_data_needed = sequence_length + forecast_horizon
    
    if combined_len < min_data_needed:
        logger.warning(f"Insufficient data for validation sequences: {combined_len} points, need {min_data_needed}")
        return {'X': np.array([]), 'y': np.array([]), 'X_num': None, 'X_cat': {}}
    
    # Create sequences that predict validation targets
    # Start from positions where we have enough historical context and can predict into validation
    max_start_pos = combined_len - sequence_length - forecast_horizon
    
    # We want sequences that predict into the validation period
    # The validation period starts at hist_len, so we need sequences that end there or later
    min_start_pos = max(0, hist_len - sequence_length - forecast_horizon + 1)
    
    if min_start_pos > max_start_pos:
        logger.warning(f"Cannot create validation sequences: min_start={min_start_pos}, max_start={max_start_pos}")
        return {'X': np.array([]), 'y': np.array([]), 'X_num': None, 'X_cat': {}}
    
    # Create sequences for validation period 
    X_sales = []
    y = []
    
    # Create sequences that predict into validation period
    for start_pos in range(min_start_pos, max_start_pos + 1):
        # Input sequence
        seq_end = start_pos + sequence_length
        X_sales.append(combined_sales[start_pos:seq_end])
        
        # Target sequence (forecast_horizon steps ahead)
        target_start = seq_end
        target_end = target_start + forecast_horizon
        y.append(combined_sales[target_start:target_end])
    
    if not X_sales:
        logger.warning("No validation sequences could be created")
        return {'X': np.array([]), 'y': np.array([]), 'X_num': None, 'X_cat': {}}
    
    X_sales = np.array(X_sales, dtype=np.float32)
    y = np.array(y, dtype=np.float32)
    
    # Create feature sequences 
    X_num = None
    if combined_numerical is not None:
        X_num = []
        for start_pos in range(min_start_pos, max_start_pos + 1):
            seq_end = start_pos + sequence_length
            X_num.append(combined_numerical[start_pos:seq_end])
        X_num = np.array(X_num, dtype=np.float32)
    
    X_cat = {}
    for feat_name, feat_values in combined_categorical.items():
        X_cat[feat_name] = []
        for start_pos in range(min_start_pos, max_start_pos + 1):
            seq_end = start_pos + sequence_length
            X_cat[feat_name].append(feat_values[start_pos:seq_end])
        X_cat[feat_name] = np.array(X_cat[feat_name], dtype=np.int32)
    
    return {
        'X': X_sales,
        'y': y,
        'X_num': X_num,
        'X_cat': X_cat
    }


# Utility Functions for CV Metrics 
def aggregate_cv_metrics(cv_results: List[Dict[str, float]]) -> Dict[str, Any]:
    """
    Aggregate metrics across CV folds for academic reporting.
    
    Args:
        cv_results: List of dictionaries containing metrics for each fold
        
    Returns:
        Dictionary with comprehensive statistics for paper reporting
    """
    if not cv_results:
        return {}
    
    # Get all metric 
    metric_names = list(cv_results[0].keys())
    
    # Calculate statistics for each metric 
    aggregated = {}
    
    for metric in metric_names:
        values = [fold_result[metric] for fold_result in cv_results]
        values_array = np.array(values)
        
        aggregated[metric] = {
            'mean': float(np.mean(values_array)),

            # Sample std for paper reporting see explanation in expanding_window_cv.py for why sample vs pop. 
            'std': float(np.std(values_array, ddof=1)),  
            'min': float(np.min(values_array)),
            'max': float(np.max(values_array)),
            'all_folds': [float(v) for v in values],  # Convert to regular floats for JSON
            'range': float(np.max(values_array) - np.min(values_array)),
            'best_fold_idx': int(np.argmin(values_array)) if 'loss' in metric.lower() else int(np.argmax(values_array))
        }
    
    # Add overall summary 
    aggregated['cv_summary'] = {
        'n_folds': len(cv_results),
        'best_fold_overall': int(np.argmin([fold['val_loss'] for fold in cv_results])),
        'fold_performance_range': f"{aggregated['val_loss']['min']:.4f} - {aggregated['val_loss']['max']:.4f}"
    }
    
    return aggregated


# Main Module Test 
if __name__ == "__main__":
    # Simple test 
    print("Testing IndexBasedWalkForwardValidator...")
    
    validator = IndexBasedWalkForwardValidator(
        initial_train_size=60,
        step_size=28,
        max_splits=3
    )
    
    for data_length in [50, 100, 150, 200]:
        print(f"\nData length: {data_length}")
        splits = validator.get_split_indices(data_length)
        for i, (train_idx, val_idx) in enumerate(splits):
            print(f"  Split {i}: Train {len(train_idx)} days, Val {len(val_idx)} days")