#////////////////////////////////////////////////////////////////////////////////#
# File:         robust_aggregation.py                                           #
# Author:       Douglas Nyberg                                                   #
# Email:        douglas1.nyberg@gmail.com                                        #
# Date:         2025-06-08                                                       #
#////////////////////////////////////////////////////////////////////////////////#
#!/usr/bin/env python3
"""
Robust aggregation methods for handling outliers in forecasting metrics.

Some metrics like MED_ASE can have extreme outliers when applied to sparse/intermittent 
time series data. This module provides aggregation strategies that are more robust
to such outliers than simple arithmetic mean.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Union
from scipy import stats


def robust_mean(values: np.ndarray, method: str = 'trimmed', trim_pct: float = 0.1) -> float:
    """
    Calculate robust mean that handles outliers better than arithmetic mean.
    
    Args:
        values: Array of values to aggregate
        method: Aggregation method ('trimmed', 'winsorized', 'median', 'huber')
        trim_pct: Percentage to trim from each tail (for trimmed/winsorized methods)
        
    Returns:
        Robust mean value
    """
    # Remove NaN and infinite values
    clean_values = values[np.isfinite(values)]
    
    if len(clean_values) == 0:
        return np.nan
    
    if method == 'trimmed':
        # Remove extreme values from both tails
        return stats.trim_mean(clean_values, trim_pct * 2)  # *2 because it trims from both sides
    
    elif method == 'winsorized':
        # Cap extreme values rather than removing them
        winsorized = stats.mstats.winsorize(clean_values, limits=[trim_pct, trim_pct])
        return np.mean(winsorized)
    
    elif method == 'median':
        # Use median as robust central tendency
        return np.median(clean_values)
    
    elif method == 'huber':
        # Huber robust estimator (requires scipy)
        try:
            from scipy.stats import huber
            return huber(clean_values).location
        except ImportError:
            # Fallback to trimmed mean if scipy not available
            return stats.trim_mean(clean_values, trim_pct * 2)
    
    else:
        # Default to arithmetic mean
        return np.mean(clean_values)


def aggregate_metrics_robust(df: pd.DataFrame, 
                           sensitive_metrics: Optional[List[str]] = None,
                           robust_method: str = 'trimmed') -> Dict[str, float]:
    """
    Aggregate metrics with robust handling for outlier-prone metrics.
    
    Args:
        df: DataFrame with metrics columns
        sensitive_metrics: List of metric names that need robust aggregation
        robust_method: Method to use for robust aggregation
        
    Returns:
        Dictionary of aggregated metrics
    """
    if sensitive_metrics is None:
        # Default outlier-prone metrics in forecasting
        sensitive_metrics = ['med_ase', 'mase', 'mape', 'mrae', 'linex']
    
    result = {}
    numeric_columns = df.select_dtypes(include=[np.number]).columns
    
    for col in numeric_columns:
        if col in sensitive_metrics:
            # Use robust aggregation for sensitive metrics
            result[col] = robust_mean(df[col].values, method=robust_method)
        else:
            # Use standard mean for other metrics
            result[col] = df[col].mean()
    
    return result


def detect_outliers(values: np.ndarray, method: str = 'iqr', threshold: float = 3.0) -> np.ndarray:
    """
    Detect outliers in metric values.
    
    Args:
        values: Array of values to check
        method: Detection method ('iqr', 'zscore', 'modified_zscore')
        threshold: Threshold for outlier detection
        
    Returns:
        Boolean array indicating outliers
    """
    clean_values = values[np.isfinite(values)]
    
    if len(clean_values) == 0:
        return np.zeros(len(values), dtype=bool)
    
    if method == 'iqr':
        # Interquartile range method
        q1, q3 = np.percentile(clean_values, [25, 75])
        iqr = q3 - q1
        lower_bound = q1 - threshold * iqr
        upper_bound = q3 + threshold * iqr
        return (values < lower_bound) | (values > upper_bound)
    
    elif method == 'zscore':
        # Standard Z-score method
        z_scores = np.abs(stats.zscore(clean_values, nan_policy='omit'))
        return z_scores > threshold
    
    elif method == 'modified_zscore':
        # Modified Z-score using median absolute deviation
        median = np.median(clean_values)
        mad = np.median(np.abs(clean_values - median))
        modified_z_scores = 0.6745 * (clean_values - median) / mad
        outlier_mask = np.abs(modified_z_scores) > threshold
        
        # Map back to original array
        result = np.zeros(len(values), dtype=bool)
        finite_mask = np.isfinite(values)
        result[finite_mask] = outlier_mask
        return result
    
    else:
        return np.zeros(len(values), dtype=bool)


def summarize_outliers(df: pd.DataFrame, 
                      metric_col: str,
                      id_col: str = 'id') -> Dict:
    """
    Provide summary of outliers for a specific metric.
    
    Args:
        df: DataFrame with metrics
        metric_col: Name of metric column to analyze
        id_col: Name of identifier column
        
    Returns:
        Dictionary with outlier summary
    """
    values = df[metric_col].values
    outliers = detect_outliers(values, method='modified_zscore')
    
    summary = {
        'total_count': len(values),
        'finite_count': np.sum(np.isfinite(values)),
        'outlier_count': np.sum(outliers),
        'outlier_percentage': np.sum(outliers) / len(values) * 100,
        'mean_with_outliers': np.mean(values[np.isfinite(values)]),
        'mean_without_outliers': np.mean(values[np.isfinite(values) & ~outliers]),
        'median': np.median(values[np.isfinite(values)]),
        'max_value': np.max(values[np.isfinite(values)]),
        'outlier_series': df[outliers][id_col].tolist() if np.sum(outliers) > 0 else []
    }
    
    return summary