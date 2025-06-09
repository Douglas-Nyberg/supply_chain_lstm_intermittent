#////////////////////////////////////////////////////////////////////////////////#
# File:         m5_official.py                                                   #
# Author:       Douglas Nyberg                                                   #
# Email:        douglas1.nyberg@gmail.com                                        #
# Date:         2025-05-15                                                       #
#////////////////////////////////////////////////////////////////////////////////#





"""
Official M5 competition evaluation framework.

This module provides evaluation on the official M5 test period (days 1914-1941)
to enable proper comparison with published literature and competition results.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional, Union
from pathlib import Path
import logging

from .accuracy import calculate_wrmsse, calculate_mase
from ..config import M5_WEEKLY_SEASONALITY

logger = logging.getLogger(__name__)

# Official M5 competition constants
M5_OFFICIAL_TRAIN_END = 1913  # Last day of training period
M5_OFFICIAL_TEST_START = 1914  # First day of test period  
M5_OFFICIAL_TEST_END = 1941   # Last day of test period (28 days)
M5_OFFICIAL_HORIZON = 28      # Forecast horizon


class M5OfficialEvaluator:
    """
    Evaluator for official M5 competition test period.
    
    This class handles evaluation on the exact test period used in the M5 competition
    (days 1914-1941) to enable proper comparison with published results.
    """
    
    def __init__(self, data_path: str):
        """
        initialize with m5 data.
        
        args:
            data_path: Path to M5 splice data file
        """
        self.data_path = data_path
        self.data_df = None
        self.train_data = None
        self.test_data = None
        self._load_and_prepare_data()
    
    def _load_and_prepare_data(self):
        """Load and prepare M5 data with official train/test split."""
        logger.info("Loading M5 data for official evaluation...")
        
        # Load the splice data
        raw_data = pd.read_csv(self.data_path)
        logger.info(f"Loaded {len(raw_data)} rows from {self.data_path}")
        
        # Convert to wide format (items × days)
        self.data_df = raw_data.pivot_table(
            index='id',
            columns='date', 
            values='sales',
            aggfunc='first'
        ).fillna(0)
        
        total_days = self.data_df.shape[1]
        logger.info(f"Data shape: {self.data_df.shape} (items × days)")
        
        # Check if we have enough data for official evaluation
        if total_days < M5_OFFICIAL_TEST_END:
            logger.warning(f"Data only has {total_days} days, need {M5_OFFICIAL_TEST_END} for official evaluation")
            # Use available data with proportional split
            test_size = min(M5_OFFICIAL_HORIZON, total_days // 4)  # Use 25% for test or 28 days
            train_end = total_days - test_size
        else:
            # Use official M5 split
            train_end = M5_OFFICIAL_TRAIN_END
            test_size = M5_OFFICIAL_HORIZON
        
        # Create official train/test split
        self.train_data = self.data_df.iloc[:, :train_end]
        self.test_data = self.data_df.iloc[:, train_end:train_end + test_size]
        
        logger.info(f"Official split - Train: {self.train_data.shape}, Test: {self.test_data.shape}")
        logger.info(f"Train period: days 1-{train_end}")
        logger.info(f"Test period: days {train_end + 1}-{train_end + test_size}")
    
    def evaluate_predictions(self, 
                           predictions: np.ndarray,
                           item_ids: List[str] = None) -> Dict[str, float]:
        """
        evaluate predictions using oficial m5 metrics.
        
        args:
            predictions: Array of predictions (n_items × forecast_horizon)
            item_ids: Optional list of item IDs for detailed analysis
            
        returns:
            Dictionary with evaluation metrics
        """
        if predictions.shape != self.test_data.shape:
            raise ValueError(f"Predictions shape {predictions.shape} doesn't match test data shape {self.test_data.shape}")
        
        logger.info("Evaluating predictions on official M5 test period...")
        
        # Calculate metrics for each item
        wrmsse_scores = []
        mase_scores = []
        rmse_scores = []
        mae_scores = []
        
        for i in range(len(self.test_data)):
            # Get data for this item
            train_series = self.train_data.iloc[i].values
            test_actual = self.test_data.iloc[i].values
            test_pred = predictions[i]
            
            # Calculate WRMSSE (primary M5 metric)
            wrmsse = calculate_wrmsse(
                actual_values=test_actual,
                forecast_values=test_pred,
                train_series=train_series,
                m=M5_WEEKLY_SEASONALITY
            )
            wrmsse_scores.append(wrmsse)
            
            # Calculate MASE
            mase = calculate_mase(
                actuals=test_actual,
                predictions=test_pred,
                train_series=train_series,
                m=M5_WEEKLY_SEASONALITY
            )
            mase_scores.append(mase)
            
            # Calculate RMSE
            rmse = np.sqrt(np.mean((test_actual - test_pred) ** 2))
            rmse_scores.append(rmse)
            
            # Calculate MAE
            mae = np.mean(np.abs(test_actual - test_pred))
            mae_scores.append(mae)
        
        # Aggregate results
        results = {
            'wrmsse_mean': float(np.mean(wrmsse_scores)),
            'wrmsse_std': float(np.std(wrmsse_scores)),
            'wrmsse_median': float(np.median(wrmsse_scores)),
            'mase_mean': float(np.mean(mase_scores)),
            'mase_std': float(np.std(mase_scores)),
            'rmse_mean': float(np.mean(rmse_scores)),
            'mae_mean': float(np.mean(mae_scores)),
            'n_items': len(wrmsse_scores),
            'forecast_horizon': self.test_data.shape[1]
        }
        
        # Add per-item results if item_ids provided
        if item_ids is not None:
            results['per_item'] = {
                item_ids[i]: {
                    'wrmsse': float(wrmsse_scores[i]),
                    'mase': float(mase_scores[i]),
                    'rmse': float(rmse_scores[i]),
                    'mae': float(mae_scores[i])
                }
                for i in range(len(item_ids))
            }
        
        logger.info(f"Official M5 evaluation complete:")
        logger.info(f"  WRMSSE: {results['wrmsse_mean']:.6f} ± {results['wrmsse_std']:.6f}")
        logger.info(f"  MASE: {results['mase_mean']:.6f} ± {results['mase_std']:.6f}")
        logger.info(f"  Items evaluated: {results['n_items']}")
        
        return results
    
    def get_train_test_split(self) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        get the official train/test split for model training.
        
        returns:
            Tuple of (train_data, test_data) DataFrames
        """
        return self.train_data.copy(), self.test_data.copy()
    
    def create_naive_baseline(self) -> np.ndarray:
        """
        create naive baseline predictions (last value carried forward).
        
        returns:
            Naive predictions array (n_items × forecast_horizon)
        """
        n_items = len(self.train_data)
        horizon = self.test_data.shape[1]
        
        # Use last non-zero value, or last value if all zeros
        naive_preds = np.zeros((n_items, horizon))
        
        for i in range(n_items):
            train_series = self.train_data.iloc[i].values
            
            # Find last non-zero value
            non_zero_indices = np.where(train_series > 0)[0]
            if len(non_zero_indices) > 0:
                last_value = train_series[non_zero_indices[-1]]
            else:
                last_value = train_series[-1]  # Use last value even if zero
            
            naive_preds[i, :] = last_value
        
        return naive_preds
    
    def create_seasonal_naive_baseline(self) -> np.ndarray:
        """
        create seasonal naive baseline (same day from previous week).
        
        returns:
            Seasonal naive predictions array (n_items × forecast_horizon)
        """
        n_items = len(self.train_data)
        horizon = self.test_data.shape[1]
        
        seasonal_preds = np.zeros((n_items, horizon))
        
        for i in range(n_items):
            train_series = self.train_data.iloc[i].values
            
            for h in range(horizon):
                # Look back 7 days (weekly seasonality)
                lookback_idx = len(train_series) - M5_WEEKLY_SEASONALITY + (h % M5_WEEKLY_SEASONALITY)
                if lookback_idx >= 0:
                    seasonal_preds[i, h] = train_series[lookback_idx]
                else:
                    # Fallback to naive if not enough history
                    seasonal_preds[i, h] = train_series[-1] if len(train_series) > 0 else 0
        
        return seasonal_preds


def evaluate_model_on_official_m5(model, 
                                  data_path: str,
                                  model_type: str = "lstm") -> Dict[str, float]:
    """
    convenience function to evaluate any model on oficial m5 test period.
    
    args:
        model: Trained model with predict() method
        data_path: Path to M5 data file
        model_type: Type of model for logging purposes
        
    returns:
        Dictionary with evaluation results
    """
    evaluator = M5OfficialEvaluator(data_path)
    train_data, test_data = evaluator.get_train_test_split()
    
    # Generate predictions (model-specific logic would go here)
    logger.info(f"Generating {model_type} predictions on official M5 test period...")
    predictions = model.predict(train_data)  # This would need to be implemented per model
    
    # Evaluate
    results = evaluator.evaluate_predictions(predictions)
    results['model_type'] = model_type
    
    return results


def compare_with_m5_baselines(data_path: str) -> Dict[str, Dict[str, float]]:
    """
    compare naive baselines on oficial m5 test period.
    
    args:
        data_path: Path to M5 data file
        
    returns:
        Dictionary with baseline results
    """
    evaluator = M5OfficialEvaluator(data_path)
    
    # Create baselines
    naive_preds = evaluator.create_naive_baseline()
    seasonal_naive_preds = evaluator.create_seasonal_naive_baseline()
    
    # Evaluate baselines
    naive_results = evaluator.evaluate_predictions(naive_preds)
    seasonal_results = evaluator.evaluate_predictions(seasonal_naive_preds)
    
    return {
        'naive': naive_results,
        'seasonal_naive': seasonal_results
    }