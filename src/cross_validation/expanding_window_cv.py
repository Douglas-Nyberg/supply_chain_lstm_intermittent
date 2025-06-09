#////////////////////////////////////////////////////////////////////////////////#
# Author:      Douglas Nyberg                                                    #
# Email:       douglas1.nyberg@gmail.com                                         #
# Date:        2025-05-22                                                        #
# Description: Expanding window time series cross-validation                     #
#////////////////////////////////////////////////////////////////////////////////#
import numpy as np
import pandas as pd
import logging
from typing import List, Tuple, Dict, Any, Optional, Callable
from dataclasses import dataclass
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent))
from config import M5_HORIZON, M5_WEEKLY_SEASONALITY

logger = logging.getLogger(__name__)

@dataclass
class TimeSeriesSplit:
    """single cv split"""
    fold: int
    train_data: np.ndarray  # Shape: (n_series, n_train_days)
    val_data: np.ndarray    # Shape: (n_series, n_val_days)
    train_features: Optional[np.ndarray] = None  # Shape: (n_features, n_train_days)
    val_features: Optional[np.ndarray] = None    # Shape: (n_features, n_val_days)


class WalkForwardValidator:
    """walk forward cv with expanding window"""
    
    def __init__(self, 
                 initial_train_size: int = 60,
                 step_size: int = M5_HORIZON,
                 max_splits: int = 5):
        # init walk forward params
        self.initial_train_size = initial_train_size
        self.step_size = step_size
        self.max_splits = max_splits
        
    def split(self, 
              data: np.ndarray, 
              features: Optional[np.ndarray] = None) -> List[TimeSeriesSplit]:
        """generate walk forward splits"""
        # Validate input shapes
        if data.ndim != 2:
            raise ValueError(
                f"Data must be 2-dimensional (n_series, n_days), but got shape {data.shape}"
            )
        
        n_series, n_days = data.shape
        
        # Validate features shape
        if features is not None:
            if features.ndim != 2:
                raise ValueError(
                    f"Features must be 2-dimensional (n_features, n_days), but got shape {features.shape}"
                )
            if features.shape[1] != n_days:
                raise ValueError(
                    f"Features time dimension ({features.shape[1]}) must match data time dimension ({n_days})"
                )
        
        # Validate minimum data requirements
        min_required_days = self.initial_train_size + self.step_size
        if n_days < min_required_days:
            raise ValueError(
                f"Insufficient data for cross-validation: have {n_days} days, "
                f"but need at least {min_required_days} days "
                f"(initial_train_size={self.initial_train_size} + step_size={self.step_size})"
            )
        
        splits = []
        
        train_end = self.initial_train_size - 1
        
        for i in range(self.max_splits):
            val_start = train_end + 1
            val_end = min(val_start + self.step_size - 1, n_days - 1)
            
            # Check if we have enough data for validation
            if val_start >= n_days:
                break
                
            # Adjust validation end if we don't have enough data
            actual_val_size = val_end - val_start + 1
            if actual_val_size < self.step_size:
                logger.warning(f"Split {i}: Validation size reduced to {actual_val_size}")
            
            # slice data
            train_data = data[:, :train_end + 1]
            val_data = data[:, val_start:val_end + 1]
            
            train_features = None
            val_features = None
            if features is not None:

                train_features = features[:, :train_end + 1]
                val_features = features[:, val_start:val_end + 1]
            
            # Validate split shapes
            if train_data.shape[0] != n_series:
                raise RuntimeError(
                    f"Error: train_data has {train_data.shape[0]} series, expected {n_series}"
                )
            if val_data.shape[0] != n_series:
                raise RuntimeError(
                    f"Error: val_data has {val_data.shape[0]} series, expected {n_series}"
                )
            
            split = TimeSeriesSplit(
                fold=i,
                train_data=train_data,
                val_data=val_data,
                train_features=train_features,
                val_features=val_features
            )
            splits.append(split)
            
            logger.debug(f"Walk-forward split {i}: Train 0-{train_end} "
                        f"({train_data.shape[1]} days), "
                        f"Val {val_start}-{val_end} ({val_data.shape[1]} days)")
            
            # Move to next period
            train_end = val_end
        
        logger.info(f"Created {len(splits)} walk-forward validation splits")
        return splits


def create_expanding_window_splits(data: np.ndarray,
                                 features: Optional[np.ndarray] = None,
                                 n_splits: int = 3,
                                 test_size: int = M5_HORIZON,
                                 initial_train_size: int = 60) -> List[TimeSeriesSplit]:
    """create expanding window splits"""
    validator = WalkForwardValidator(
        max_splits=n_splits,
        step_size=test_size,
        initial_train_size=initial_train_size
    )
    
    return validator.split(data, features)


def calculate_cv_metrics(splits: List[TimeSeriesSplit],
                                       predictions: List[np.ndarray],
                                       metric_func: Callable) -> Dict[str, Any]:
    """calculate metrics across cv folds and return stats"""
    fold_metrics = []
    if len(splits) != len(predictions):
        raise ValueError(
            f"Number of splits ({len(splits)}) must match "
            f"number of prediction arrays ({len(predictions)})"
        )

    number_of_folds = len(splits)
    for i in range(number_of_folds):
        split = splits[i]
        pred = predictions[i]
        
        # get actual validation data
        actual = split.val_data

        # check shapes are same
        if actual.shape != pred.shape:
            raise ValueError(
                f"Shape mismatch in fold {i}: "
                f"actual shape is {actual.shape}, "
                f"but predictions shape is {pred.shape}."
            )

        try:
            metric_value = metric_func(actual, pred)
            fold_metrics.append(metric_value)
        except Exception as e:
            raise RuntimeError(f"error calculating metric for fold {i}: {e}") from e

    # filter out nans
    valid_metrics = [m for m in fold_metrics if not np.isnan(m)]
    if not valid_metrics:
        results = {
            'mean': np.nan,
            'std': np.nan,
            'min': np.nan,
            'max': np.nan,
            'median': np.nan,
            'q1': np.nan,
            'q3': np.nan,
            'iqr': np.nan,
            'cv': np.nan,
            'n_valid': 0,
            'n_total': len(fold_metrics),
            'folds': fold_metrics 
        }
    else:
        # calc stats from valid metrics
        mean_value = np.mean(valid_metrics)
        std_value = np.std(valid_metrics, ddof=1) if len(valid_metrics) > 1 else 0.0
        
        results = {
            'mean': mean_value,
            'std': std_value,
            'min': np.min(valid_metrics),
            'max': np.max(valid_metrics),
            'median': np.median(valid_metrics),
            'q1': np.percentile(valid_metrics, 25),
            'q3': np.percentile(valid_metrics, 75),
            'iqr': np.percentile(valid_metrics, 75) - np.percentile(valid_metrics, 25),
            'cv': std_value / mean_value if mean_value != 0 else np.nan,
            'n_valid': len(valid_metrics),
            'n_total': len(fold_metrics),
            'folds': fold_metrics
        }
    return results


