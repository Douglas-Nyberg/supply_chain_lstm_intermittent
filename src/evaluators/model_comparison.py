#////////////////////////////////////////////////////////////////////////////////#
# File:         model_comparison.py                                              #
# Author:       Douglas Nyberg                                                   #
# Email:        douglas1.nyberg@gmail.com                                        #
# Date:         2025-06-06                                                       #
#////////////////////////////////////////////////////////////////////////////////#





"""
Model comparison workflows for forecasting evaluation.

This module provides high-level workflows for comparing multiple forecasting models
using M5 competition metrics, particularly WRMSSE. It handles data loading,
prediction loading, and coordinated evaluation across multiple models.
"""

import pandas as pd
import numpy as np
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

from .accuracy import calculate_wrmsse_vectorized, compute_series_weights_from_sales
from ..config import M5_WEEKLY_SEASONALITY, M5_HORIZON

logger = logging.getLogger(__name__)


class ModelComparisonEvaluator:
    """
    High-level evaluator for comparing multiple forecasting models.
    
    This class handles the workflow of loading test data, training data, and predictions
    from multiple models, then calculating comparative metrics like WRMSSE.
    """
    
    def __init__(self, 
                 data_dir: str = "data/preprocessed/exp1_publication_medium_1000",
                 predictions_dir: str = "predictions/exp1_per_item_lstm_vs_classical/exp1_publication_medium_1000"):
        """
        Initialize evaluator with data directories.
        
        Args:
            data_dir: Directory containing test_data.csv and train_data.csv
            predictions_dir: Directory containing model prediction subdirectories
        """
        self.data_dir = Path(data_dir)
        self.predictions_dir = Path(predictions_dir)
        self.test_data = None
        self.train_data = None
        
        # Standard model file mapping
        self.model_file_map = {
            'lstm': 'lstm/lstm_point_forecasts.csv',
            'tsb': 'tsb/tsb_point_forecasts.csv', 
            'croston': 'croston/croston_point_forecasts.csv',
            'arima': 'arima/arima_point_forecasts.csv'
        }
    
    def load_test_data(self) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Load test data and training data for WRMSSE calculation.
        
        Returns:
            Tuple of (test_data, train_data) DataFrames
            
        Raises:
            FileNotFoundError: If required data files are missing
        """
        logger.info("Loading test data for WRMSSE calculation")
        
        # Load preprocessed test data
        test_data_path = self.data_dir / "test_data.csv"
        if not test_data_path.exists():
            raise FileNotFoundError(f"Test data not found at {test_data_path}")
        
        self.test_data = pd.read_csv(test_data_path)
        logger.info(f"Loaded test data: {self.test_data.shape}")
        
        # Load training data for RMSSE scaling
        train_data_path = self.data_dir / "train_data.csv" 
        if not train_data_path.exists():
            raise FileNotFoundError(f"Training data not found at {train_data_path}")
        
        self.train_data = pd.read_csv(train_data_path)
        logger.info(f"Loaded training data: {self.train_data.shape}")
        
        return self.test_data, self.train_data
    
    def load_model_predictions(self, model_name: str) -> pd.DataFrame:
        """
        Load predictions for a specific model.
        
        Args:
            model_name: Name of the model ('lstm', 'tsb', 'croston', 'arima')
            
        Returns:
            DataFrame with model predictions
            
        Raises:
            ValueError: If model name is not supported
            FileNotFoundError: If prediction file is missing
        """
        if model_name not in self.model_file_map:
            raise ValueError(f"Unknown model: {model_name}. Supported: {list(self.model_file_map.keys())}")
        
        pred_file = self.predictions_dir / self.model_file_map[model_name]
        if not pred_file.exists():
            raise FileNotFoundError(f"Predictions not found at {pred_file}")
        
        predictions = pd.read_csv(pred_file)
        logger.info(f"Loaded {model_name} predictions: {predictions.shape}")
        
        return predictions
    
    def prepare_wrmsse_data(self, 
                           test_data: pd.DataFrame, 
                           predictions: pd.DataFrame, 
                           train_data: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.Series]:
        """
        Prepare data in the format expected by WRMSSE function.
        
        Args:
            test_data: Test data with actual values
            predictions: Model predictions in M5 format
            train_data: Training data for RMSSE scaling
            
        Returns:
            Tuple of (actual_values, forecast_values, training_data, weights)
        """
        # Handle ID mapping - test data has "_validation" while predictions have "_validation_evaluation"
        # Create mapping from prediction IDs to test data IDs
        test_base_ids = {id.replace('_validation', '') for id in test_data['id'].unique()}
        pred_to_test_mapping = {}
        
        for pred_id in predictions['id'].unique():
            base_id = pred_id.replace('_validation_evaluation', '')
            if base_id in test_base_ids:
                test_id = base_id + '_validation'
                pred_to_test_mapping[pred_id] = test_id
        
        logger.info(f"Found {len(pred_to_test_mapping)} common series between test data and predictions")
        
        if not pred_to_test_mapping:
            logger.error("No matching series found between test data and predictions!")
            return pd.DataFrame(), pd.DataFrame(), pd.DataFrame(), pd.Series(dtype=float)
        
        # Filter to common series and map IDs
        test_ids = set(pred_to_test_mapping.values())
        pred_ids = set(pred_to_test_mapping.keys())
        
        test_filtered = test_data[test_data['id'].isin(test_ids)]
        pred_filtered = predictions[predictions['id'].isin(pred_ids)]
        train_filtered = train_data[train_data['id'].isin(test_ids)]
        
        # For test data, we need to take the last 28 days as the forecast period
        # Sort by date to ensure proper chronological order
        test_filtered = test_filtered.sort_values(['id', 'date'])
        train_filtered = train_filtered.sort_values(['id', 'date'])
        
        # Get the last 28 days from test data for each series
        test_forecast_period = test_filtered.groupby('id').tail(28)
        
        # Pivot test data to get actual values in proper format
        actual_pivot = test_forecast_period.pivot(index='date', columns='id', values='sales')
        
        # Reshape predictions from M5 format (F1-F28 columns) to long format
        forecast_columns = [f'F{i}' for i in range(1, 29)]
        
        # Melt predictions to long format
        pred_long_list = []
        for _, row in pred_filtered.iterrows():
            series_id = row['id']
            for i, col in enumerate(forecast_columns):
                pred_long_list.append({
                    'id': series_id,
                    'forecast_day': i + 1,
                    'forecast': row[col]
                })
        
        pred_long_df = pd.DataFrame(pred_long_list)
        
        # Create forecast_pivot to match actual_pivot structure
        # We need to match the dates from actual_pivot
        forecast_pivot = pd.DataFrame(
            index=actual_pivot.index,
            columns=actual_pivot.columns
        )
        
        # Fill forecast values by matching series and forecast day order
        # Map prediction IDs to test IDs for proper alignment
        for pred_id, test_id in pred_to_test_mapping.items():
            series_pred = pred_long_df[pred_long_df['id'] == pred_id]['forecast'].values
            if len(series_pred) == 28 and test_id in forecast_pivot.columns:  # Ensure we have complete forecasts
                forecast_pivot[test_id] = series_pred
        
        # For training data, pivot for RMSSE scaling
        train_pivot = train_filtered.pivot(index='date', columns='id', values='sales')
        
        # Calculate weights based on sales volume (M5 standard)
        # Use last 28 days of training data for weight calculation
        # Need to transpose so series are rows and time is columns
        weights = compute_series_weights_from_sales(train_pivot.T, last_n_days=28)
        
        logger.info(f"Prepared data shapes - Actual: {actual_pivot.shape}, Forecast: {forecast_pivot.shape}")
        logger.info(f"Training data shape: {train_pivot.shape}, Weights shape: {weights.shape}")
        
        return actual_pivot, forecast_pivot, train_pivot, weights
    
    def calculate_model_wrmsse(self, model_name: str) -> Optional[float]:
        """
        Calculate WRMSSE for a specific model.
        
        Args:
            model_name: Name of the model to evaluate
            
        Returns:
            WRMSSE score, or None if calculation failed
        """
        logger.info(f"\n{'='*60}")
        logger.info(f"Calculating WRMSSE for {model_name.upper()}")
        logger.info(f"{'='*60}")
        
        try:
            # Ensure data is loaded
            if self.test_data is None or self.train_data is None:
                self.load_test_data()
            
            # Load model predictions
            predictions = self.load_model_predictions(model_name)
            
            # Prepare data for WRMSSE calculation
            actual_values, forecast_values, training_data, weights = self.prepare_wrmsse_data(
                self.test_data, predictions, self.train_data
            )
            
            # Calculate WRMSSE using the project's vectorized function
            # Need to transpose so series are rows and time is columns
            wrmsse = calculate_wrmsse_vectorized(
                actual_values=actual_values.T,      # Transpose: series as rows, time as columns
                forecast_values=forecast_values.T,  # Transpose: series as rows, time as columns
                training_data=training_data.T,      # Transpose: series as rows, time as columns
                weights=weights,
                m=M5_WEEKLY_SEASONALITY,  # 7 for weekly seasonality
                use_seasonal_naive=True   # M5 standard
            )
            
            logger.info(f"✅ {model_name.upper()} WRMSSE: {wrmsse:.6f}")
            return wrmsse
            
        except Exception as e:
            logger.error(f"❌ Error calculating WRMSSE for {model_name}: {str(e)}")
            return None
    
    def compare_all_models(self, 
                          models: Optional[List[str]] = None) -> Dict[str, float]:
        """
        Compare WRMSSE across multiple models.
        
        Args:
            models: List of model names to compare. If None, uses all available models.
            
        Returns:
            Dictionary mapping model names to WRMSSE scores
        """
        if models is None:
            models = ['lstm', 'tsb', 'croston', 'arima']
        
        logger.info("🚀 Starting WRMSSE calculation for all models")
        logger.info("📊 Using M5 competition standard WRMSSE metric")
        
        # Calculate WRMSSE for each model
        results = {}
        for model in models:
            wrmsse = self.calculate_model_wrmsse(model)
            if wrmsse is not None:
                results[model] = wrmsse
        
        # Display final results
        self._display_comparison_results(results)
        
        return results
    
    def _display_comparison_results(self, results: Dict[str, float]) -> None:
        """Display formatted comparison results."""
        logger.info(f"\n{'='*80}")
        logger.info("🏆 FINAL WRMSSE RESULTS (M5 Competition Metric)")
        logger.info(f"{'='*80}")
        
        if results:
            # Sort by WRMSSE (lower is better)
            sorted_results = sorted(results.items(), key=lambda x: x[1])
            
            for rank, (model, wrmsse) in enumerate(sorted_results, 1):
                medal = "🥇" if rank == 1 else "🥈" if rank == 2 else "🥉" if rank == 3 else f"{rank}."
                logger.info(f"{medal} {model.upper():>8}: {wrmsse:.6f} WRMSSE")
            
            # Performance comparison
            if len(sorted_results) > 1:
                best_model, best_wrmsse = sorted_results[0]
                logger.info(f"\n📈 PERFORMANCE ANALYSIS")
                logger.info(f"{'='*50}")
                logger.info(f"Best model: {best_model.upper()} with WRMSSE = {best_wrmsse:.6f}")
                
                for model, wrmsse in sorted_results[1:]:
                    diff = wrmsse - best_wrmsse
                    pct_worse = (diff / best_wrmsse) * 100
                    logger.info(f"{model.upper()} is {diff:.6f} (+{pct_worse:.2f}%) worse than {best_model.upper()}")
        
        else:
            logger.error("❌ No WRMSSE results calculated successfully")
        
        logger.info(f"\n✅ WRMSSE calculation complete!")


def compare_models_wrmsse(data_dir: str = "data/preprocessed/exp1_publication_medium_1000",
                         predictions_dir: str = "predictions/exp1_per_item_lstm_vs_classical/exp1_publication_medium_1000",
                         models: Optional[List[str]] = None) -> Dict[str, float]:
    """
    Convenience function to compare models using WRMSSE.
    
    Args:
        data_dir: Directory containing test and training data
        predictions_dir: Directory containing model predictions
        models: List of models to compare (default: all available)
        
    Returns:
        Dictionary mapping model names to WRMSSE scores
    """
    evaluator = ModelComparisonEvaluator(data_dir, predictions_dir)
    return evaluator.compare_all_models(models)