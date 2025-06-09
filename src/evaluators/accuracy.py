#////////////////////////////////////////////////////////////////////////////////#
# File:         accuracy.py                                                      #
# Author:       Douglas Nyberg                                                   #
# Email:        douglas1.nyberg@gmail.com                                        #
# Date:         2025-03-25                                                       #
#////////////////////////////////////////////////////////////////////////////////#

"""
Accuracy metrics for forecasting eval. WRMSSE implementation. 
"""



import sys
from pathlib import Path
from typing import Dict, List, Union, Tuple, Optional

import numpy as np
import pandas as pd

# Add src to path for config imports
sys.path.append(str(Path(__file__).parent.parent))
from config import M5_WEEKLY_SEASONALITY

from sktime.performance_metrics.forecasting import (
    mean_absolute_error,
    mean_squared_error,
    mean_absolute_scaled_error,
    mean_squared_scaled_error,
    mean_absolute_percentage_error,
    mean_asymmetric_error,
    median_absolute_error,
    median_absolute_scaled_error,
    geometric_mean_absolute_error,
    mean_relative_absolute_error,
    mean_linex_error
)


def calculate_rmse(actuals: np.ndarray, predictions: np.ndarray) -> float:
    """calculate rmse using sktime"""
    # ensure arrays have same length
    n = min(len(actuals), len(predictions))
    actuals = actuals[:n]
    predictions = predictions[:n]
    return mean_squared_error(actuals, predictions, square_root=True)


def calculate_mae(actuals: np.ndarray, predictions: np.ndarray) -> float:
    """calculate mae using sktime"""
    # ensure arrays have same length
    n = min(len(actuals), len(predictions))
    actuals = actuals[:n]
    predictions = predictions[:n]
    return mean_absolute_error(actuals, predictions)


def calculate_mase(actuals: np.ndarray, predictions: np.ndarray, train_series: np.ndarray, m: int = M5_WEEKLY_SEASONALITY) -> float:
    """calculate mase using sktime"""
    # ensure arrays have same length
    n = min(len(actuals), len(predictions))
    actuals = actuals[:n]
    predictions = predictions[:n]
    return mean_absolute_scaled_error(actuals, predictions, y_train=train_series, sp=m)


def calculate_smape(actuals: np.ndarray, predictions: np.ndarray) -> float:
    """calculate smape using sktime"""
    # ensure arrays have same length
    n = min(len(actuals), len(predictions))
    actuals = actuals[:n]
    predictions = predictions[:n]
    return mean_absolute_percentage_error(actuals, predictions, symmetric=True) * 100


def rmsse_scale(train_series_data: np.ndarray) -> float:
    """calculate scaling factor for rmsse"""
    # Handle edge cases
    if len(train_series_data) <= 1:
        return 1e-10  # small positive value for very short series
    
    # check if series is consant
    if np.std(train_series_data) < 1e-10:
        return 1e-10
    
    # calculate the naive forecast errors (shift by 1 period)
    naive_errors = train_series_data[1:] - train_series_data[:-1]
    
    # calculate the scaling factor (rmse of naive forecast)
    denominator = np.sqrt(np.mean(np.square(naive_errors)))
    
    # handle case where series has no variation - follow m5 competition approach
    if denominator < 1e-10:
        # Return small value that's appropriate for the scale of the data
        data_scale = np.mean(np.abs(train_series_data))
        return max(data_scale * 1e-3, 1e-10)
    
    return denominator


def seasonal_naive_rmsse_scale(train_series_data: np.ndarray, m: int = M5_WEEKLY_SEASONALITY) -> float:
    """
    calculate scaling factor for rmsse using seasonal naive forecast.
    this is the proper m5 competition scaling factor.
    
    uses seasonal naive forecast (value from m periods ago) as benchmark.
    official m5 competition scaling approach.
    """
    # Handle edge cases
    if len(train_series_data) <= m:
        # not enough data for seasonal naive, fall back to simple naive
        return rmsse_scale(train_series_data)
    
    # check if series is all zeros or constant
    if np.std(train_series_data) < 1e-10:
        # for constant series, return a small positive value
        # this ensures rmsse is well-defined but heavily penalizes any error
        return 1e-10
    
    # Calculate seasonal naive forecast errors
    # Seasonal naive: y_hat(t) = y(t-m)
    seasonal_naive_forecast = train_series_data[:-m]
    actual_values = train_series_data[m:]
    seasonal_naive_errors = actual_values - seasonal_naive_forecast
    
    # Calculate the scaling factor (RMSE of seasonal naive forecast)
    denominator = np.sqrt(np.mean(np.square(seasonal_naive_errors)))
    
    # handle case where series has no variation
    if denominator < 1e-10:
        # Return small value that's appropriate for the scale of the data
        # use mean absolute value as reference, or 1e-10 if data is all zeros
        data_scale = np.mean(np.abs(train_series_data))
        return max(data_scale * 1e-3, 1e-10)
    
    return denominator


def calculate_rmsse(actual_horizon_data: np.ndarray, 
                   forecast_horizon_data: np.ndarray, 
                   train_series: np.ndarray) -> float:
    """
    calculate rmsse (root mean squared scaled error) for single series using sktime.
    """
    # ensure arrays have same length
    n = min(len(actual_horizon_data), len(forecast_horizon_data))
    actual_horizon_data = actual_horizon_data[:n]
    forecast_horizon_data = forecast_horizon_data[:n]
    return mean_squared_scaled_error(actual_horizon_data, forecast_horizon_data, 
                                   y_train=train_series, square_root=True)


def calculate_wrmsse(actual_values: np.ndarray, 
                    forecast_values: np.ndarray, 
                    train_series: np.ndarray, 
                    weights: np.ndarray = None,
                    m: int = M5_WEEKLY_SEASONALITY,
                    use_seasonal_naive: bool = True) -> float:
    """
    calculate wrmsse (weighted root mean squared scaled error) for time series.
    uses seasonal naive (m5 standard) or simple naive for scaling.
    """
    # ensure arrays have same length
    n = min(len(actual_values), len(forecast_values))
    actual_values = actual_values[:n]
    forecast_values = forecast_values[:n]
    
    # calculate scaling factor using training data
    if use_seasonal_naive:
        scale = seasonal_naive_rmsse_scale(train_series, m)
    else:
        scale = rmsse_scale(train_series)
    
    # calculate rmse
    rmse = np.sqrt(np.mean((actual_values - forecast_values) ** 2))
    
    # scale to get rmsse
    rmsse = rmse / scale
    
    # Return RMSSE directly if no weights provided
    if weights is None:
        return rmsse
    else:
        # For proper WRMSSE, weights should already be normalized and this is just one item's contribution
        # The aggregation happens at a higher level
        return rmsse


def calculate_wrmsse_vectorized(actual_values: pd.DataFrame, 
                              forecast_values: pd.DataFrame, 
                              training_data: pd.DataFrame, 
                              weights: pd.Series,
                              m: int = M5_WEEKLY_SEASONALITY,
                              use_seasonal_naive: bool = True) -> float:
    """
    vectorized calculation of wrmsse for more efficent computation.
    
    args:
        actual_values: DataFrame with rows as series and columns as horizon days.
        forecast_values: DataFrame with rows as series and columns as horizon days.
        training_data: DataFrame with rows as series and columns as training days.
        weights: Series with index as series IDs and values as weights.
        m: Seasonal period for scaling (default: M5_WEEKLY_SEASONALITY=7)
        use_seasonal_naive: Whether to use seasonal naive (M5 standard) or simple naive for scaling
        
    returns:
        The overall WRMSSE score.
    """
    # Ensure that inputs align on index
    common_series = actual_values.index.intersection(
        forecast_values.index.intersection(
            training_data.index.intersection(weights.index)
        )
    )
    
    # Filter to common series
    actual_values = actual_values.loc[common_series]
    forecast_values = forecast_values.loc[common_series]
    training_data = training_data.loc[common_series]
    weights = weights.loc[common_series]
    
    # Normalize weights
    weights = weights / weights.sum()
    
    # Calculate scaling factors for all series
    scales = []
    for idx in training_data.index:
        if use_seasonal_naive:
            scales.append(seasonal_naive_rmsse_scale(training_data.loc[idx].values, m))
        else:
            scales.append(rmsse_scale(training_data.loc[idx].values))
    scales = pd.Series(scales, index=training_data.index)
    
    # Calculate RMSSE for all series
    forecast_errors = actual_values.values - forecast_values.values
    rmse = np.sqrt(np.mean(np.square(forecast_errors), axis=1))
    rmsse = rmse / scales.values
    
    # Calculate weighted RMSSE
    wrmsse = (rmsse * weights.values).sum()
    
    return wrmsse


def compute_series_weights_from_sales(sales_data: pd.DataFrame, 
                                     last_n_days: int = 28) -> pd.Series:
    """
    compute series weights based on dollar sales in the last n days.
    
    args:
        sales_data: DataFrame with rows as series, columns as days, and values as dollar sales.
        last_n_days: Number of last days to consider for weight calculation.
        
    returns:
        Series with index as series IDs and values as normalized weights.
    """
    # extract the last n days of sales data
    last_sales = sales_data.iloc[:, -last_n_days:]
    
    # sum sales for each series over the last n days
    total_sales = last_sales.sum(axis=1)
    
    # Normalize to get weights that sum to 1
    weights = total_sales / total_sales.sum()
    
    return weights


def calculate_asymmetric_error(actuals: np.ndarray, predictions: np.ndarray, 
                             asymmetric_threshold: float = 0.5) -> float:
    """
    calculate asymmetric error that penalizes over/under forecasting diferently.
    useful for supply chain where stockouts vs overstock have different costs.
    
    args:
        actuals: Array of actual values
        predictions: Array of predicted values
        asymmetric_threshold: Threshold between 0 and 1. Values < 0.5 penalize 
                            over-forecasting more, > 0.5 penalize under-forecasting more
        
    returns:
        Asymmetric error value
    """
    # ensure arrays have same length
    n = min(len(actuals), len(predictions))
    actuals = actuals[:n]
    predictions = predictions[:n]
    return mean_asymmetric_error(actuals, predictions, 
                               asymmetric_threshold=asymmetric_threshold)


def calculate_mape(actuals: np.ndarray, predictions: np.ndarray) -> float:
    """
    calculate mean absolute percentage error (mape) using sktime.
    
    args:
        actuals: Array of actual values
        predictions: Array of predicted values
        
    returns:
        MAPE value (as percentage)
    """
    # ensure arrays have same length
    n = min(len(actuals), len(predictions))
    actuals = actuals[:n]
    predictions = predictions[:n]
    return mean_absolute_percentage_error(actuals, predictions, symmetric=False) * 100


def calculate_median_absolute_error(actuals: np.ndarray, predictions: np.ndarray) -> float:
    """
    calculate median absolute error (medae) using sktime.
    more robust to outliers than mae.
    
    args:
        actuals: Array of actual values
        predictions: Array of predicted values
        
    returns:
        Median absolute error value
    """
    n = min(len(actuals), len(predictions))
    actuals = actuals[:n]
    predictions = predictions[:n]
    return median_absolute_error(actuals, predictions)


def calculate_median_absolute_scaled_error(actuals: np.ndarray, predictions: np.ndarray, 
                                         train_series: np.ndarray, m: int = M5_WEEKLY_SEASONALITY) -> float:
    """
    calculate median absolute scaled error (medase) using sktime.
    more robust to outliers than mase.
    
    args:
        actuals: Array of actual values for the test period
        predictions: Array of predicted values for the test period
        train_series: Array of actual values for the training period
        m: Seasonal period (default: M5_WEEKLY_SEASONALITY=7 for M5 retail data)
        
    returns:
        Median absolute scaled error value
    """
    n = min(len(actuals), len(predictions))
    actuals = actuals[:n]
    predictions = predictions[:n]
    
    try:
        # Calculate MedASE with bounds to prevent extreme values
        result = median_absolute_scaled_error(actuals, predictions, y_train=train_series, sp=m)
        
        # Cap extreme values that can occur with near-zero variance in training data
        # Reduced threshold from 1e10 to 1e3 to better handle intermittent demand outliers
        if np.isfinite(result) and result < 1e3:
            return result
        else:
            # Fallback to simple median absolute error scaled by mean absolute value
            mae = np.median(np.abs(actuals - predictions))
            scale = np.mean(np.abs(train_series)) + 1e-8  # Add small epsilon to avoid division by zero
            return min(mae / scale, 1e3)  # Cap fallback at same threshold
    except Exception:
        # Fallback calculation
        mae = np.median(np.abs(actuals - predictions))
        scale = np.mean(np.abs(train_series)) + 1e-8
        return min(mae / scale, 1e3)  # Cap fallback at same threshold


def calculate_geometric_mean_absolute_error(actuals: np.ndarray, predictions: np.ndarray) -> float:
    """
    calculate geometric mean absolute error using sktime.
    less sensitive to large errors than mae.
    
    args:
        actuals: Array of actual values
        predictions: Array of predicted values
        
    returns:
        Geometric mean absolute error value
    """
    n = min(len(actuals), len(predictions))
    actuals = actuals[:n]
    predictions = predictions[:n]
    return geometric_mean_absolute_error(actuals, predictions)


def calculate_mean_relative_absolute_error(actuals: np.ndarray, predictions: np.ndarray, 
                                         benchmark: Optional[np.ndarray] = None) -> float:
    """
    calculate mean relative absolute error using sktime.
    
    args:
        actuals: Array of actual values
        predictions: Array of predicted values
        benchmark: Array of benchmark predictions (defaults to naive forecast)
        
    returns:
        Mean relative absolute error value
    """
    n = min(len(actuals), len(predictions))
    actuals = actuals[:n]
    predictions = predictions[:n]
    
    # Create naive benchmark if not provided
    if benchmark is None:
        benchmark = np.roll(actuals, 1)
        benchmark[0] = actuals[0]  # First value same as actual
    else:
        benchmark = benchmark[:n]
    
    try:
        # Calculate MRAE with bounds to prevent extreme values
        result = mean_relative_absolute_error(actuals, predictions, y_pred_benchmark=benchmark)
        
        # Cap extreme values for intermittent demand series
        if np.isfinite(result) and result < 1e6:
            return result
        else:
            # Fallback: calculate relative to mean absolute value
            mae_pred = np.mean(np.abs(actuals - predictions))
            mae_bench = np.mean(np.abs(actuals - benchmark))
            if mae_bench > 1e-8:
                return mae_pred / mae_bench
            else:
                return mae_pred / (np.mean(np.abs(actuals)) + 1e-8)
    except Exception:
        # Fallback calculation
        mae_pred = np.mean(np.abs(actuals - predictions))
        mae_bench = np.mean(np.abs(actuals - benchmark))
        if mae_bench > 1e-8:
            return mae_pred / mae_bench
        else:
            return mae_pred / (np.mean(np.abs(actuals)) + 1e-8)


def calculate_linex_error(actuals: np.ndarray, predictions: np.ndarray, a: float = 1.0, b: float = 1.0) -> float:
    """
    calculate linex (linear-exponential) error using sktime.
    asymmetric loss function that can penalize over/under-prediction diferently.
    
    args:
        actuals: Array of actual values
        predictions: Array of predicted values
        a: Scale parameter (higher values increase penalty)
        b: Asymmetry parameter (positive values penalize over-prediction more)
        
    returns:
        LINEX error value
    """
    n = min(len(actuals), len(predictions))
    actuals = actuals[:n]
    predictions = predictions[:n]
    return mean_linex_error(actuals, predictions, a=a, b=b)


def calculate_msse(actuals: np.ndarray, predictions: np.ndarray, y_train: Optional[np.ndarray] = None, sp: int = 1) -> float:
    """
    calculate mean squared scaled error (msse) using sktime.
    this is the non-root version of rmsse.
    
    args:
        actuals: Array of actual values
        predictions: Array of predicted values
        y_train: Training data for scaling. If None, uses naive forecast
        sp: Seasonal period for scaling
        
    returns:
        MSSE value (mean squared scaled error)
    """
    n = min(len(actuals), len(predictions))
    actuals = actuals[:n]
    predictions = predictions[:n]
    
    # Note: mean_squared_scaled_error returns MSSE by default (square_root=False)
    return mean_squared_scaled_error(
        actuals, 
        predictions, 
        y_train=y_train, 
        sp=sp,
        square_root=False  # This gives us MSSE instead of RMSSE
    )