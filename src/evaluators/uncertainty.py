#////////////////////////////////////////////////////////////////////////////////#
# File:         uncertainty.py                                                   #
# Author:       Douglas Nyberg                                                   #
# Email:        douglas1.nyberg@gmail.com                                        #
# Date:         2025-04-08                                                       #
#////////////////////////////////////////////////////////////////////////////////#

"""
Uncertainty metrics for probabilistic forecasting evaluation. Implements WSPL (Weighted Scaled Pinball Loss). 
"""
import numpy as np
import pandas as pd
from typing import Dict, List, Union, Tuple, Optional, Any

from sktime.performance_metrics.forecasting.probabilistic import (
    PinballLoss,
    CRPS,
    LogLoss,
    EmpiricalCoverage,
    SquaredDistrLoss,
    ConstraintViolation,
    AUCalibration,
    IntervalWidth
)


def calculate_pinball_loss(actual_value: float,
                          forecast_quantile_value: float,
                          quantile_tau: float) -> float:
    """
    calculate pinball loss for single point and quantile.
    
    args:
        actual_value: The actual observed value.
        forecast_quantile_value: The forecasted value for the given quantile.
        quantile_tau: The quantile level (between 0 and 1).
        
    returns:
        The pinball loss value.
    """
    if actual_value >= forecast_quantile_value:
        return (actual_value - forecast_quantile_value) * quantile_tau
    else:
        return (forecast_quantile_value - actual_value) * (1 - quantile_tau)


def calculate_pinball_loss_sktime(actuals: np.ndarray, 
                                 predictions: Dict[float, np.ndarray], 
                                 quantile_levels: List[float]) -> float:
    """
    calculate pinball loss using sktime.
    
    args:
        actuals: Array of actual values
        predictions: Dictionary mapping quantile levels to prediction arrays
        quantile_levels: List of quantile levels
        
    returns:
        Average pinball loss across all quantiles
    """
    if not predictions or not quantile_levels:
        return 0.0
    
    # Convert actuals to pandas Series for sktime compatibility
    actuals_series = pd.Series(actuals)
    
    # Filter available quantiles and sort them
    available_quantiles = sorted([q for q in quantile_levels if q in predictions])
    if not available_quantiles:
        return 0.0
    
    # Create MultiIndex DataFrame format expected by sktime
    # Format: MultiIndex([('variable', quantile1), ('variable', quantile2), ...])
    variable_name = 'sales'  # Can be any name
    columns = pd.MultiIndex.from_product(
        [[variable_name], available_quantiles],
        names=['variable', 'quantile']
    )
    
    # Build prediction data matrix
    min_len = min(len(actuals), min(len(predictions[q]) for q in available_quantiles))
    data = []
    for i in range(min_len):
        row = [predictions[q][i] for q in available_quantiles]
        data.append(row)
    
    preds_df = pd.DataFrame(data, columns=columns)
    
    # Calculate pinball loss using sktime
    pinball_metric = PinballLoss()
    loss = pinball_metric(actuals_series[:min_len], preds_df)
    
    return float(loss)


def spl_scale(train_series_data: np.ndarray) -> float:
    """
    calculate scaling factor for spl (scaled pinball loss).
    
    args:
        train_series_data: A single training time series.
        
    returns:
        The scaling factor for SPL.
        
    Note:
        Following M5 competition guidelines, if the scaling factor is zero 
        (indicating no variation in historical data), we return a very small number
        instead of zero to avoid division by zero issues while ensuring high penalty
        for errors when history shows no change.
    """
    # handle edge cases
    if len(train_series_data) <= 1:
        return np.finfo(np.float64).eps  # Use machine epsilon for very short series
    
    # calculate naive forecast errors (absolute values of shift by 1 period)
    naive_errors = np.abs(train_series_data[1:] - train_series_data[:-1])
    
    # calculate scaling factor (mean absolute error of naive forecast)
    denominator = np.mean(naive_errors)
    
    # handle case where series has no variation - follow m5 competition approach
    if denominator == 0:
        return np.finfo(np.float64).eps  # Very small number to ensure large but finite SPL
    
    return denominator


def calculate_spl(actual_horizon_data: np.ndarray,
                 forecast_quantiles_horizon_data: Dict[float, np.ndarray],
                 quantile_levels: List[float],
                 scale: float) -> float:
    """
    calculate spl (scaled pinball loss) for single series.
    
    args:
        actual_horizon_data: Actual values for the forecast horizon.
        forecast_quantiles_horizon_data: Dictionary mapping quantile levels to forecasted values.
        quantile_levels: List of quantile levels used in forecasting.
        scale: The pre-calculated scaling factor for this series.
        
    returns:
        The SPL value for this series.
    """
    # handle edge cases
    if len(actual_horizon_data) == 0:
        return 0.0  # no data to evaluate
    
    # note: we assume scale is never exactly zero due to handling in spl_scale
    # that returns small non-zero value instead
    
    total_pinball_loss = 0.0
    total_points = len(actual_horizon_data) * len(quantile_levels)
    
    for tau in quantile_levels:
        quantile_forecast = forecast_quantiles_horizon_data[tau]
        
        for i, actual in enumerate(actual_horizon_data):
            if i < len(quantile_forecast):  # Ensure forecast exists for this point
                total_pinball_loss += calculate_pinball_loss(actual, quantile_forecast[i], tau)
    
    # calculate average pinball loss
    avg_pinball_loss = total_pinball_loss / total_points
    
    # Scale the pinball loss
    spl = avg_pinball_loss / scale
    
    return spl


def calculate_wspl(predictions: Dict[float, np.ndarray],
                  actuals: np.ndarray,
                  quantile_levels: List[float],
                  train_series: Optional[np.ndarray] = None,
                  weights: Optional[np.ndarray] = None) -> float:
    """
    calculate wspl (weighted scaled pinball loss) for quantile forecasts.
    
    args:
        predictions: Dictionary mapping quantile levels to arrays of predicted values
        actuals: Array of actual values
        quantile_levels: List of quantile levels used in forecasting
        train_series: Optional array of training data for scaling (if not provided, no scaling is applied)
        weights: Optional array of weights for different quantiles (defaults to equal weights)
        
    returns:
        The WSPL score (lower is better)
    """
    # Ensure all predicted quantiles are available
    for q in quantile_levels:
        if q not in predictions:
            raise ValueError(f"Quantile {q} not found in predictions")
    
    # Calculate scaling factor if training data is provided
    scale = 1.0
    if train_series is not None and len(train_series) > 1:
        scale = spl_scale(train_series)
    
    # Calculate total pinball loss for all quantiles
    total_pinball_loss = 0.0
    total_points = 0
    
    for q in quantile_levels:
        # Get predictions for this quantile
        q_preds = predictions[q]
        
        # Truncate to match lengths
        min_len = min(len(actuals), len(q_preds))
        q_preds = q_preds[:min_len]
        actuals_trunc = actuals[:min_len]
        
        # Calculate pinball loss for each data point
        for actual, pred in zip(actuals_trunc, q_preds):
            total_pinball_loss += calculate_pinball_loss(actual, pred, q)
            total_points += 1
    
    # calculate average pinball loss
    if total_points == 0:
        return 0.0
    
    avg_pinball_loss = total_pinball_loss / total_points
    
    # Scale the pinball loss
    wspl = avg_pinball_loss / scale
    
    # Apply weights if provided
    if weights is not None:
        if len(weights) != len(quantile_levels):
            raise ValueError(f"Length of weights ({len(weights)}) must match length of quantile_levels ({len(quantile_levels)})")
        
        # Normalize weights
        weights = np.array(weights) / np.sum(weights)
        
        # Calculate weighted average
        wspl = 0.0
        for i, q in enumerate(quantile_levels):
            q_preds = predictions[q]
            min_len = min(len(actuals), len(q_preds))
            q_preds = q_preds[:min_len]
            actuals_trunc = actuals[:min_len]
            
            q_pinball_loss = 0.0
            for actual, pred in zip(actuals_trunc, q_preds):
                q_pinball_loss += calculate_pinball_loss(actual, pred, q)
            
            if min_len > 0:
                avg_q_pinball_loss = q_pinball_loss / min_len
                wspl += weights[i] * (avg_q_pinball_loss / scale)
    
    return wspl


def calculate_wspl_vectorized(actual_values: pd.DataFrame,
                             forecast_quantiles: Dict[float, pd.DataFrame],
                             training_data: pd.DataFrame,
                             weights: pd.Series,
                             quantile_levels: List[float]) -> float:
    """
    vectorized calculation of wspl for more efficent computation.
    
    args:
        actual_values: DataFrame with rows as series and columns as horizon days.
        forecast_quantiles: Dictionary mapping quantile levels to DataFrames with the same structure as actual_values.
        training_data: DataFrame with rows as series and columns as training days.
        weights: Series with index as series IDs and values as weights.
        quantile_levels: List of quantile levels used in forecasting.
        
    returns:
        The overall WSPL score.
    """
    # Ensure that all inputs align on index
    all_series = set(actual_values.index)
    for tau in quantile_levels:
        all_series = all_series.intersection(set(forecast_quantiles[tau].index))
    all_series = all_series.intersection(set(training_data.index))
    all_series = all_series.intersection(set(weights.index))
    all_series = list(all_series)
    
    # Filter to common series
    actual_values = actual_values.loc[all_series]
    forecast_quantiles = {tau: df.loc[all_series] for tau, df in forecast_quantiles.items()}
    training_data = training_data.loc[all_series]
    weights = weights.loc[all_series]
    
    # Normalize weights
    weights = weights / weights.sum()
    
    # Calculate scaling factors for all series
    scales = []
    for idx in training_data.index:
        scales.append(spl_scale(training_data.loc[idx].values))
    scales = pd.Series(scales, index=training_data.index)
    
    # Calculate SPL for all series
    all_spl = []
    for idx in actual_values.index:
        # Extract actual values for this series
        actuals = actual_values.loc[idx].values
        
        # Extract forecasts for all quantiles for this series
        forecasts = {tau: forecast_quantiles[tau].loc[idx].values for tau in quantile_levels}
        
        # Calculate SPL
        series_spl = calculate_spl(actuals, forecasts, quantile_levels, scales[idx])
        all_spl.append(series_spl)
    
    all_spl = pd.Series(all_spl, index=actual_values.index)
    
    # Calculate weighted SPL
    wspl = (all_spl * weights).sum()
    
    return wspl


# Add prediction interval coverage and width calculation functions
def calculate_coverage(predictions: Dict[float, np.ndarray], actuals: np.ndarray, quantile_levels: List[float]) -> Dict[float, float]:
    """
    calculate empirical coverage of prediction intervals.
    
    args:
        predictions: Dictionary mapping quantile levels to arrays of predicted values
        actuals: Array of actual values
        quantile_levels: List of quantile levels for which to calculate coverage
        
    returns:
        Dictionary mapping quantile levels to their empirical coverage rates
    """
    coverage = {}
    
    for q in quantile_levels:
        # Get predictions for this quantile
        q_preds = predictions[q]
        
        # Truncate to match lengths
        min_len = min(len(actuals), len(q_preds))
        q_preds = q_preds[:min_len]
        actuals_trunc = actuals[:min_len]
        
        # Calculate proportion of actual values less than or equal to predictions
        coverage[q] = np.mean(actuals_trunc <= q_preds)
    
    return coverage


def calculate_pi_width(predictions: Dict[float, np.ndarray], quantile_low: float = 0.1, quantile_high: float = 0.9) -> float:
    """
    calculate average width of prediction intervals.
    
    args:
        predictions: Dictionary mapping quantile levels to arrays of predicted values
        quantile_low: Lower quantile level for the prediction interval
        quantile_high: Upper quantile level for the prediction interval
        
    returns:
        Average width of the prediction intervals
    """
    # Make sure the required quantiles are available
    if quantile_low not in predictions or quantile_high not in predictions:
        raise ValueError(f"Required quantiles {quantile_low} and {quantile_high} not found in predictions")
    
    # Get predictions for the lower and upper quantiles
    lower_preds = predictions[quantile_low]
    upper_preds = predictions[quantile_high]
    
    # Truncate to match lengths
    min_len = min(len(lower_preds), len(upper_preds))
    lower_preds = lower_preds[:min_len]
    upper_preds = upper_preds[:min_len]
    
    # Calculate average interval width
    interval_widths = upper_preds - lower_preds
    return np.mean(interval_widths)


def calculate_combined_uncertainty_metric(predictions: Dict[float, np.ndarray], 
                                        actuals: np.ndarray, 
                                        quantile_levels: List[float],
                                        coverage_weight: float = 0.5,
                                        sharpness_weight: float = 0.5) -> float:
    """
    calculate combined uncertainty metric that balances coverage and sharpnes.
    
    args:
        predictions: Dictionary mapping quantile levels to arrays of predicted values
        actuals: Array of actual values
        quantile_levels: List of quantile levels used in forecasting
        coverage_weight: Weight for coverage penalty (0-1)
        sharpness_weight: Weight for interval width (0-1)
        
    returns:
        Combined score balancing coverage and sharpness
    """
    # Calculate coverage
    coverage = calculate_coverage(predictions, actuals, quantile_levels)
    
    # Calculate coverage error (difference from ideal)
    coverage_error = np.mean([abs(coverage[q] - q) for q in quantile_levels])
    
    # Calculate average interval width for different prediction intervals
    # Here we use 50% and 90% prediction intervals
    pi_50_width = calculate_pi_width(predictions, 0.25, 0.75)
    pi_90_width = calculate_pi_width(predictions, 0.05, 0.95)
    
    # Normalize widths based on the range of actuals
    actuals_range = np.max(actuals) - np.min(actuals) if len(actuals) > 0 else 1.0
    normalized_pi_50_width = pi_50_width / actuals_range
    normalized_pi_90_width = pi_90_width / actuals_range
    
    # Calculate average normalized width
    avg_normalized_width = (normalized_pi_50_width + normalized_pi_90_width) / 2
    
    # Combine coverage error and interval width into a single metric
    # Lower is better for both components
    combined_score = coverage_weight * coverage_error + sharpness_weight * avg_normalized_width
    
    return combined_score


# M5 competition specific quantile levels
M5_QUANTILE_LEVELS = [0.005, 0.025, 0.165, 0.250, 0.500, 0.750, 0.835, 0.975, 0.995]


def prepare_m5_forecasts(forecast_df: pd.DataFrame, 
                        horizon: int = 28, 
                        quantile_levels: Optional[List[float]] = None) -> Dict[float, pd.DataFrame]:
    """
    prepare m5 competition style forecasts for evaluation.
    
    args:
        forecast_df: DataFrame with rows as (series_id, quantile) MultiIndex and columns as horizon days.
        horizon: The forecast horizon length.
        quantile_levels: The quantile levels to extract. If None, uses M5 competition levels.
        
    returns:
        Dictionary mapping quantile levels to DataFrames with series_id as index and columns as horizon days.
    """
    if quantile_levels is None:
        quantile_levels = M5_QUANTILE_LEVELS
    
    # Extract forecasts for each quantile
    result = {}
    for tau in quantile_levels:
        # Select rows for this quantile
        tau_forecasts = forecast_df.xs(tau, level=1, drop_level=True)
        
        # Ensure we have only the horizon columns
        if tau_forecasts.shape[1] > horizon:
            tau_forecasts = tau_forecasts.iloc[:, :horizon]
        
        result[tau] = tau_forecasts
    
    return result


def calculate_crps(actuals: np.ndarray, forecast_samples: np.ndarray) -> float:
    """
    calculate continuous rank probability score (crps) using sktime.
    measures quality of distributional forecasts.
    
    args:
        actuals: Array of actual values
        forecast_samples: Array of forecast samples/distribution
        
    returns:
        CRPS value (lower is better)
    """
    crps_metric = CRPS()
    return crps_metric(actuals, forecast_samples)


def calculate_log_loss(actuals: np.ndarray, forecast_probs: np.ndarray) -> float:
    """
    calculate logarithmic loss for probabilistic forecasts using sktime.
    
    args:
        actuals: Array of actual values
        forecast_probs: Array of forecast probabilities/densities
        
    returns:
        Log loss value (lower is better)
    """
    log_loss_metric = LogLoss()
    return log_loss_metric(actuals, forecast_probs)


def calculate_empirical_coverage_sktime(actuals: np.ndarray, 
                                       predictions: Dict[float, np.ndarray], 
                                       quantile_levels: List[float]) -> Dict[float, float]:
    """
    calculate empirical coverage using sktime.
    
    args:
        actuals: Array of actual values
        predictions: Dictionary mapping quantile levels to prediction arrays
        quantile_levels: List of quantile levels
        
    returns:
        Dictionary mapping quantile levels to their empirical coverage
    """
    coverage_results = {}
    
    if not predictions or not quantile_levels:
        return coverage_results
    
    # Convert actuals to pandas Series for sktime compatibility  
    actuals_series = pd.Series(actuals)
    
    # Filter available quantiles and sort them
    available_quantiles = sorted([q for q in quantile_levels if q in predictions])
    if not available_quantiles:
        return coverage_results
    
    # Create MultiIndex DataFrame format expected by sktime
    variable_name = 'sales'  # Can be any name
    columns = pd.MultiIndex.from_product(
        [[variable_name], available_quantiles],
        names=['variable', 'quantile']
    )
    
    # Build prediction data matrix
    min_len = min(len(actuals), min(len(predictions[q]) for q in available_quantiles))
    data = []
    for i in range(min_len):
        row = [predictions[q][i] for q in available_quantiles]
        data.append(row)
    
    preds_df = pd.DataFrame(data, columns=columns)
    
    # Calculate coverage for each quantile level
    coverage_metric = EmpiricalCoverage()
    try:
        # Calculate coverage for all quantiles at once
        coverage_result = coverage_metric(actuals_series[:min_len], preds_df)
        
        # If result is a DataFrame, extract individual quantile results
        if isinstance(coverage_result, pd.DataFrame):
            for q in available_quantiles:
                if (variable_name, q) in coverage_result.columns:
                    coverage_results[q] = float(coverage_result[(variable_name, q)].iloc[0])
        else:
            # If single value, assign to median quantile (0.5 if available)
            if 0.5 in available_quantiles:
                coverage_results[0.5] = float(coverage_result)
            elif available_quantiles:
                coverage_results[available_quantiles[0]] = float(coverage_result)
    except Exception as e:
        # Fallback: try individual quantile calculations
        for tau in available_quantiles:
            try:
                single_q_columns = pd.MultiIndex.from_product(
                    [[variable_name], [tau]],
                    names=['variable', 'quantile']
                )
                single_q_data = [[predictions[tau][i]] for i in range(min_len)]
                single_q_df = pd.DataFrame(single_q_data, columns=single_q_columns)
                
                coverage = coverage_metric(actuals_series[:min_len], single_q_df)
                coverage_results[tau] = float(coverage)
            except:
                coverage_results[tau] = np.nan
    
    return coverage_results


def calculate_constraint_violation(actuals: np.ndarray, 
                                 lower_predictions: np.ndarray,
                                 upper_predictions: np.ndarray) -> float:
    """
    calculate constraint violation percentage using sktime.
    measures how often prediction intervals are violated.
    
    args:
        actuals: Array of actual values
        lower_predictions: Array of lower bound predictions
        upper_predictions: Array of upper bound predictions
        
    returns:
        Violation percentage (lower is better)
    """
    # Create interval predictions format for sktime
    interval_preds = np.column_stack([lower_predictions, upper_predictions])
    violation_metric = ConstraintViolation()
    return violation_metric(actuals, interval_preds)


def calculate_interval_score(actuals: np.ndarray,
                           lower_predictions: np.ndarray,
                           upper_predictions: np.ndarray,
                           alpha: float = 0.1) -> float:
    """
    calculate interval score that combines interval width and coverage.
    lower scores are beter.
    
    args:
        actuals: Array of actual values
        lower_predictions: Array of lower bound predictions
        upper_predictions: Array of upper bound predictions
        alpha: Miscoverage level (e.g., 0.1 for 90% intervals)
        
    returns:
        Interval score (lower is better)
    """
    min_len = min(len(actuals), len(lower_predictions), len(upper_predictions))
    actuals = actuals[:min_len]
    lower_predictions = lower_predictions[:min_len]
    upper_predictions = upper_predictions[:min_len]
    
    # Interval width
    width = upper_predictions - lower_predictions
    
    # Coverage penalties
    lower_penalty = (2 / alpha) * (lower_predictions - actuals) * (actuals < lower_predictions)
    upper_penalty = (2 / alpha) * (actuals - upper_predictions) * (actuals > upper_predictions)
    
    # Total interval score
    interval_scores = width + lower_penalty + upper_penalty
    
    return np.mean(interval_scores)


def calculate_au_calibration(actuals: np.ndarray, forecast_probs: np.ndarray) -> float:
    """
    calculate area under calibration curve using sktime.
    measures how well probabilistic forecasts are calibrted.
    
    args:
        actuals: Array of actual values
        forecast_probs: Array of forecast probabilities/densities
        
    returns:
        AU Calibration score (higher is better for calibration)
    """
    calibration_metric = AUCalibration()
    return calibration_metric(actuals, forecast_probs)


def calculate_squared_distr_loss(actuals: np.ndarray, forecast_samples: np.ndarray) -> float:
    """
    calculate squared distributional loss using sktime.
    measures quality of distributional forecasts using squared loss.
    
    args:
        actuals: Array of actual values
        forecast_samples: Array of forecast samples/distribution
        
    returns:
        Squared distributional loss value (lower is better)
    """
    squared_loss_metric = SquaredDistrLoss()
    return squared_loss_metric(actuals, forecast_samples)


def create_interval_format_dataframe(predictions: Dict[float, np.ndarray], 
                                    interval_levels: List[float] = [0.8, 0.9]) -> pd.DataFrame:
    """
    Create 3-level MultiIndex DataFrame format for sktime interval metrics.
    
    args:
        predictions: Dictionary mapping quantile levels to prediction arrays
        interval_levels: List of coverage levels (e.g., [0.8, 0.9] for 80% and 90% intervals)
        
    returns:
        DataFrame with 3-level MultiIndex: (variable, coverage, bound)
    """
    if not predictions:
        return pd.DataFrame()
    
    variable_name = 'sales'
    columns = []
    data_columns = []
    
    # Find the length of predictions
    first_key = next(iter(predictions.keys()))
    min_len = len(predictions[first_key])
    
    for coverage in interval_levels:
        # Calculate alpha for this coverage level
        alpha = 1 - coverage
        lower_quantile = alpha / 2
        upper_quantile = 1 - alpha / 2
        
        # Debug print
        # print(f"Coverage: {coverage}, alpha: {alpha}, lower_q: {lower_quantile}, upper_q: {upper_quantile}")
        # print(f"Available quantiles: {list(predictions.keys())}")
        
        # Check if we have the required quantiles (with tolerance for floating point precision)
        available_quantiles = list(predictions.keys())
        lower_match = any(abs(q - lower_quantile) < 1e-10 for q in available_quantiles)
        upper_match = any(abs(q - upper_quantile) < 1e-10 for q in available_quantiles)
        
        if lower_match and upper_match:
            # Find the exact matching quantiles
            actual_lower = next(q for q in available_quantiles if abs(q - lower_quantile) < 1e-10)
            actual_upper = next(q for q in available_quantiles if abs(q - upper_quantile) < 1e-10)
            # Add column names for this interval
            columns.extend([
                (variable_name, coverage, 'lower'),
                (variable_name, coverage, 'upper')
            ])
            
            # Add data columns using the actual matched quantiles
            lower_data = predictions[actual_lower][:min_len]
            upper_data = predictions[actual_upper][:min_len]
            data_columns.extend([lower_data, upper_data])
    
    if not columns:
        return pd.DataFrame()
    
    # Create MultiIndex
    multi_columns = pd.MultiIndex.from_tuples(
        columns, 
        names=['variable', 'coverage', 'bound']
    )
    
    # Build data matrix
    data = []
    for i in range(min_len):
        row = [col_data[i] for col_data in data_columns]
        data.append(row)
    
    return pd.DataFrame(data, columns=multi_columns)


def calculate_prediction_interval_width_sktime(predictions: Dict[float, np.ndarray], 
                                             coverage_level: float = 0.8) -> float:
    """
    Calculate prediction interval width using sktime format.
    
    args:
        predictions: Dictionary mapping quantile levels to prediction arrays
        coverage_level: Coverage level for the interval (e.g., 0.8 for 80%)
        
    returns:
        Average interval width
    """
    # Create interval format DataFrame
    interval_df = create_interval_format_dataframe(predictions, [coverage_level])
    
    if interval_df.empty:
        return np.nan
    
    try:
        # Try using sktime IntervalWidth metric
        width_metric = IntervalWidth()
        # Create dummy actuals (not used for width calculation)
        dummy_actuals = pd.Series(np.zeros(len(interval_df)))
        width = width_metric(dummy_actuals, interval_df)
        return float(width)
    except Exception:
        # Fallback to manual calculation
        variable_name = 'sales'
        lower_col = (variable_name, coverage_level, 'lower')
        upper_col = (variable_name, coverage_level, 'upper')
        
        if lower_col not in interval_df.columns or upper_col not in interval_df.columns:
            return np.nan
        
        # Calculate interval widths manually
        widths = interval_df[upper_col] - interval_df[lower_col]
        return float(np.mean(widths))


def calculate_interval_score_sktime(actuals: np.ndarray,
                                  predictions: Dict[float, np.ndarray],
                                  coverage_level: float = 0.8) -> float:
    """
    Calculate interval score using sktime-compatible format.
    
    args:
        actuals: Array of actual values
        predictions: Dictionary mapping quantile levels to prediction arrays
        coverage_level: Coverage level for the interval (e.g., 0.8 for 80%)
        
    returns:
        Interval score (lower is better)
    """
    # Convert actuals to pandas Series
    actuals_series = pd.Series(actuals)
    
    # Create interval format DataFrame
    interval_df = create_interval_format_dataframe(predictions, [coverage_level])
    
    if interval_df.empty:
        return np.nan
    
    # Ensure lengths match
    min_len = min(len(actuals), len(interval_df))
    actuals_series = actuals_series[:min_len]
    interval_df = interval_df.iloc[:min_len]
    
    # Manual interval score calculation using sktime-compatible format
    variable_name = 'sales'
    lower_col = (variable_name, coverage_level, 'lower')
    upper_col = (variable_name, coverage_level, 'upper')
    
    if lower_col not in interval_df.columns or upper_col not in interval_df.columns:
        return np.nan
    
    lower_preds = interval_df[lower_col].values
    upper_preds = interval_df[upper_col].values
    alpha = 1 - coverage_level
    
    # Manual interval score calculation
    width = upper_preds - lower_preds
    lower_penalty = (2 / alpha) * (lower_preds - actuals[:min_len]) * (actuals[:min_len] < lower_preds)
    upper_penalty = (2 / alpha) * (actuals[:min_len] - upper_preds) * (actuals[:min_len] > upper_preds)
    
    interval_scores = width + lower_penalty + upper_penalty
    return float(np.mean(interval_scores))


def calculate_constraint_violation_sktime(actuals: np.ndarray,
                                        predictions: Dict[float, np.ndarray],
                                        coverage_level: float = 0.8) -> float:
    """
    Calculate constraint violation using sktime-compatible format.
    
    args:
        actuals: Array of actual values
        predictions: Dictionary mapping quantile levels to prediction arrays
        coverage_level: Coverage level for the interval (e.g., 0.8 for 80%)
        
    returns:
        Constraint violation rate (lower is better)
    """
    # Convert actuals to pandas Series
    actuals_series = pd.Series(actuals)
    
    # Create interval format DataFrame
    interval_df = create_interval_format_dataframe(predictions, [coverage_level])
    
    if interval_df.empty:
        return np.nan
    
    # Ensure lengths match
    min_len = min(len(actuals), len(interval_df))
    actuals_series = actuals_series[:min_len]
    interval_df = interval_df.iloc[:min_len]
    
    try:
        # Calculate constraint violation using sktime
        violation_metric = ConstraintViolation()
        violation = violation_metric(actuals_series, interval_df)
        return float(violation)
    except Exception as e:
        # Fallback to manual calculation
        variable_name = 'sales'
        lower_col = (variable_name, coverage_level, 'lower')
        upper_col = (variable_name, coverage_level, 'upper')
        
        if lower_col not in interval_df.columns or upper_col not in interval_df.columns:
            return np.nan
        
        lower_preds = interval_df[lower_col].values
        upper_preds = interval_df[upper_col].values
        
        # Count violations (actual outside interval)
        violations = (actuals[:min_len] < lower_preds) | (actuals[:min_len] > upper_preds)
        return float(np.mean(violations))