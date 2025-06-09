#////////////////////////////////////////////////////////////////////////////////#
# File:         5_evaluate_predictions_mc.py                                     #
# Author:       Douglas Nyberg                                                   #
# Email:        douglas1.nyberg@gmail.com                                        #
# Date:         2025-04-18                                                       #
#////////////////////////////////////////////////////////////////////////////////#
#!/usr/bin/env python3
"""evaluate predictions with all metrics"""

# imports
import argparse
import json
import logging
import sys
import time
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any
from datetime import datetime

import numpy as np
import pandas as pd
from tqdm import tqdm

# project path setup
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent.parent
sys.path.append(str(PROJECT_ROOT))

# evaluation modules
from src.evaluators.accuracy import (
    calculate_rmse, calculate_mae, calculate_mase, calculate_smape,
    calculate_wrmsse_vectorized, calculate_rmsse, calculate_mape,
    calculate_median_absolute_error, calculate_median_absolute_scaled_error,
    calculate_asymmetric_error, calculate_linex_error,
    calculate_geometric_mean_absolute_error, calculate_mean_relative_absolute_error,
    calculate_msse
)

from src.evaluators.uncertainty import (
    calculate_pinball_loss_sktime, calculate_wspl, calculate_crps,
    calculate_log_loss, calculate_empirical_coverage_sktime,
    calculate_interval_score, calculate_pi_width, calculate_constraint_violation,
    calculate_au_calibration, calculate_squared_distr_loss,
    calculate_prediction_interval_width_sktime, calculate_interval_score_sktime,
    calculate_constraint_violation_sktime,
    M5_QUANTILE_LEVELS
)

import config_mc_master as config
from src.utils import set_random_seed, extract_base_id_from_prediction_id

# logging setup
config.LOGS_DIR.mkdir(parents=True, exist_ok=True)
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

# cli args
def parse_arguments() -> argparse.Namespace:
    """parse cli args"""
    parser = argparse.ArgumentParser(description="Comprehensive evaluation with ALL 21 metrics.")

    parser.add_argument(
        "--predictions-dir", type=str, required=True,
        help="Directory containing prediction files."
    )

    parser.add_argument(
        "--data-dir", type=str, required=True,
        help="Directory containing preprocessed train and test data CSVs."
    )

    parser.add_argument(
        "--output-dir", type=str, required=True,
        help="Directory to save evaluation results."
    )

    parser.add_argument(
        "--models-to-evaluate", type=str, nargs='+', required=True,
        help="List of model types to evaluate."
    )
    
    parser.add_argument(
        "--generate-plots", action="store_true",
        help="Generate visualization plots."
    )
    
    parser.add_argument(
        "--statistical-comparison", action="store_true",
        help="Perform statistical comparison between models."
    )
    
    return parser.parse_args()

# data loading helpers
def load_prediction_files(
    predictions_dir: Path, model_type: str
) -> Tuple[Optional[pd.DataFrame], Optional[pd.DataFrame]]:
    """Loads point and quantile prediction files from the specified directory."""
    logger.info(f"Loading predictions for model '{model_type}' from: {predictions_dir}")

    # Look for files in model-specific subdirectory
    model_predictions_dir = predictions_dir / model_type.lower()
    point_forecast_file_path = model_predictions_dir / f"{model_type}_point_forecasts.csv"
    quantile_forecast_file_path = model_predictions_dir / f"{model_type}_quantile_forecasts.csv"

    point_predictions_df = None
    if point_forecast_file_path.exists():
        point_predictions_df = pd.read_csv(point_forecast_file_path)
        logger.info(f"  Loaded point forecasts: {point_predictions_df.shape} from {point_forecast_file_path.name}")
    else:
        logger.warning(f"Point forecast file not found: {point_forecast_file_path}")

    quantile_predictions_df = None
    if quantile_forecast_file_path.exists():
        quantile_predictions_df = pd.read_csv(quantile_forecast_file_path)
        logger.info(f"  Loaded quantile forecasts: {quantile_predictions_df.shape} from {quantile_forecast_file_path.name}")
    else:
        logger.info(f"  Quantile forecast file not found for model {model_type}")
        
    return point_predictions_df, quantile_predictions_df

def load_actuals_and_training_data(
    preprocessed_data_dir: Path
) -> Tuple[Optional[pd.DataFrame], Optional[pd.DataFrame]]:
    """Loads the actual values from the test set and the full training history."""
    logger.info(f"Loading actuals and training history from: {preprocessed_data_dir}")

    test_set_file_path = preprocessed_data_dir / config.DATA_CONFIG["test_file_name"]
    training_set_file_path = preprocessed_data_dir / config.DATA_CONFIG["train_file_name"]

    actuals_df = None
    if test_set_file_path.exists():
        actuals_df = pd.read_csv(test_set_file_path, parse_dates=['date'])
        logger.info(f"  Loaded actuals: {actuals_df.shape} from {test_set_file_path.name}")
    else:
        logger.error(f"Actuals file not found: {test_set_file_path}")

    train_history_df = None
    if training_set_file_path.exists():
        train_history_df = pd.read_csv(training_set_file_path, parse_dates=['date'])
        logger.info(f"  Loaded training history: {train_history_df.shape} from {training_set_file_path.name}")
    else:
        logger.error(f"Training history file not found: {training_set_file_path}")
        
    return actuals_df, train_history_df

# metric calculation functions
def calculate_all_accuracy_metrics_per_series(
    point_predictions_df: pd.DataFrame, 
    actuals_test_set_df: pd.DataFrame,
    full_train_history_df: pd.DataFrame
) -> pd.DataFrame:
    """Calculates accuracy metrics for each series."""
    logger.info("Calculating accuracy metrics per series...")
    metric_results_list = []
    
    if 'id' not in actuals_test_set_df.columns or 'id' not in full_train_history_df.columns:
        logger.error("'id' column missing. Cannot calculate per-series metrics.")
        return pd.DataFrame()

    for _, prediction_row in tqdm(point_predictions_df.iterrows(), total=len(point_predictions_df), desc="Comprehensive Accuracy Metrics"):
        m5_formatted_prediction_id = prediction_row["id"]
        base_series_id = extract_base_id_from_prediction_id(m5_formatted_prediction_id, is_quantile_prediction=False)

        forecast_horizon_length = config.FORECAST_CONFIG["horizon"]
        predicted_values = prediction_row[[f"F{i+1}" for i in range(forecast_horizon_length)]].values.astype(float)

        series_actuals_subset_df = actuals_test_set_df[actuals_test_set_df['id'] == base_series_id].sort_values('date')
        if series_actuals_subset_df.empty:
            logger.warning(f"No actuals found for series '{base_series_id}'. Skipping.")
            continue
        actual_values = series_actuals_subset_df['sales'].head(forecast_horizon_length).values.astype(float)

        # Align lengths
        min_length = min(len(predicted_values), len(actual_values))
        predicted_values = predicted_values[:min_length]
        actual_values = actual_values[:min_length]
        
        if len(actual_values) == 0:
            logger.warning(f"Series '{base_series_id}': No data after alignment. Skipping.")
            continue

        # Get training history
        series_train_history = full_train_history_df[full_train_history_df['id'] == base_series_id]['sales'].values
        
        # Calculate accuracy metrics
        metrics = {"id": base_series_id}
        
        # Basic metrics from accuracy.py
        metrics["rmse"] = calculate_rmse(actual_values, predicted_values)
        metrics["mae"] = calculate_mae(actual_values, predicted_values)
        metrics["smape"] = calculate_smape(actual_values, predicted_values)
        metrics["mape"] = calculate_mape(actual_values, predicted_values)
        
        # Scaled metrics 
        if len(series_train_history) >= config.EVALUATION_CONFIG["seasonality"] + 1:
            metrics["mase"] = calculate_mase(actual_values, predicted_values, series_train_history, m=config.EVALUATION_CONFIG["seasonality"])
            metrics["rmsse"] = calculate_rmsse(actual_values, predicted_values, series_train_history)
            metrics["med_ase"] = calculate_median_absolute_scaled_error(actual_values, predicted_values, series_train_history, m=config.EVALUATION_CONFIG["seasonality"])
            metrics["msse"] = calculate_msse(actual_values, predicted_values, series_train_history, sp=config.EVALUATION_CONFIG["seasonality"])
        else:
            metrics["mase"] = np.nan
            metrics["rmsse"] = np.nan
            metrics["med_ase"] = np.nan
            metrics["msse"] = np.nan
        
        # Additional metrics from accuracy.py
        metrics["med_ae"] = calculate_median_absolute_error(actual_values, predicted_values)
        metrics["gmae"] = calculate_geometric_mean_absolute_error(actual_values, predicted_values)
        
        # Relative and asymmetric metrics from accuracy.py
        if len(actual_values) > 1:
            naive_forecast = np.roll(actual_values, 1)
            naive_forecast[0] = actual_values[0]
            metrics["mrae"] = calculate_mean_relative_absolute_error(actual_values, predicted_values, naive_forecast)
        else:
            metrics["mrae"] = np.nan
            
        metrics["mae_asymmetric"] = calculate_asymmetric_error(actual_values, predicted_values)
        metrics["linex"] = calculate_linex_error(actual_values, predicted_values)
        
        metric_results_list.append(metrics)
        
    return pd.DataFrame(metric_results_list)

def calculate_all_uncertainty_metrics_per_series(
    quantile_predictions_df: pd.DataFrame,
    actuals_test_set_df: pd.DataFrame,
    full_train_history_df: pd.DataFrame
) -> pd.DataFrame:
    """Calculates uncertainty metrics for each series."""
    logger.info("Calculating  uncertainty metrics per series...")
    
    if quantile_predictions_df is None or quantile_predictions_df.empty:
        logger.warning("No quantile predictions available for uncertainty metrics.")
        return pd.DataFrame()
    
    # Group quantile predictions by series
    series_metrics = []
    
    # Get unique series from quantile predictions
    unique_series = set()
    for _, row in quantile_predictions_df.iterrows():
        prediction_id = row["id"]
        base_series_id = extract_base_id_from_prediction_id(prediction_id, is_quantile_prediction=True)
        unique_series.add(base_series_id)
    
    forecast_horizon_length = config.FORECAST_CONFIG["horizon"]
    
    for base_series_id in tqdm(unique_series, desc="Uncertainty Metrics"):
        # Get actuals for this series
        series_actuals_subset_df = actuals_test_set_df[actuals_test_set_df['id'] == base_series_id].sort_values('date')
        if series_actuals_subset_df.empty:
            continue
        actual_values = series_actuals_subset_df['sales'].head(forecast_horizon_length).values.astype(float)
        
        # Get training history
        series_train_history = full_train_history_df[full_train_history_df['id'] == base_series_id]['sales'].values
        
        # Collect quantile predictions for this series
        predictions_dict = {}
        for _, row in quantile_predictions_df.iterrows():
            prediction_id = row["id"]
            series_id = extract_base_id_from_prediction_id(prediction_id, is_quantile_prediction=True)
            if series_id == base_series_id:
                # Extract quantile level
                try:
                    quantile_level_str = prediction_id.split('_')[-2]
                    quantile_level = float(quantile_level_str)
                    predicted_values = row[[f"F{i+1}" for i in range(forecast_horizon_length)]].values.astype(float)
                    predictions_dict[quantile_level] = predicted_values
                except (IndexError, ValueError):
                    continue
        
        if not predictions_dict or len(actual_values) == 0:
            continue
            
        # Calculate ALL 11 uncertainty metrics
        metrics = {"id": base_series_id}
        
        try:
            # Align prediction lengths
            aligned_predictions = {}
            for q_level, q_preds in predictions_dict.items():
                min_len = min(len(actual_values), len(q_preds))
                aligned_predictions[q_level] = q_preds[:min_len]
            aligned_actuals = actual_values[:min(len(actual_values), min(len(p) for p in predictions_dict.values()))]
            
            if len(aligned_actuals) == 0:
                raise ValueError("No aligned data for uncertainty metrics")
            
            # Use proper uncertainty functions from uncertainty.py
            available_quantiles = list(aligned_predictions.keys())
            
            # Pinball loss using sktime implementation
            metrics["pinball_loss"] = calculate_pinball_loss_sktime(
                aligned_actuals, aligned_predictions, available_quantiles
            )
            
            # Empirical coverage using sktime implementation  
            target_quantiles = [0.05, 0.5, 0.95]
            coverage_results = calculate_empirical_coverage_sktime(
                aligned_actuals, aligned_predictions, target_quantiles
            )
            for q in target_quantiles:
                if q in coverage_results:
                    metrics[f"coverage_{int(q*100):02d}"] = coverage_results[q]
                else:
                    metrics[f"coverage_{int(q*100):02d}"] = np.nan
            
            # Prediction interval width using sktime format
            metrics["pi_width_80"] = calculate_prediction_interval_width_sktime(
                aligned_predictions, coverage_level=0.8
            )
            
            # Interval score using sktime format
            metrics["interval_score_80"] = calculate_interval_score_sktime(
                aligned_actuals, aligned_predictions, coverage_level=0.8
            )
            
            # Constraint violation using sktime format
            metrics["constraint_violation"] = calculate_constraint_violation_sktime(
                aligned_actuals, aligned_predictions, coverage_level=0.8
            )
            
            # WSPL (Weighted Scaled Pinball Loss) from uncertainty.py
            if len(series_train_history) > 1:
                metrics["wspl"] = calculate_wspl(
                    aligned_predictions, aligned_actuals, available_quantiles, series_train_history
                )
            else:
                metrics["wspl"] = np.nan
            
            
        except Exception as e:
            logger.warning(f"Error calculating uncertainty metrics for {base_series_id}: {e}")
            metrics.update({
                "pinball_loss": np.nan, "coverage_05": np.nan, "coverage_50": np.nan, 
                "coverage_95": np.nan, "pi_width_80": np.nan, "interval_score_80": np.nan,
                "constraint_violation": np.nan, "wspl": np.nan
            })
        
        series_metrics.append(metrics)
    
    return pd.DataFrame(series_metrics)

# main evaluation workflow
def main():
    """Runs the evaluation workflow."""
    args = parse_arguments()
    set_random_seed(config.RANDOM_SEED)

    logger.info("=" * 70)
    logger.info(f"COMPREHENSIVE EVALUATION with ALL 21 METRICS")
    logger.info(f"Models: {', '.join([m.upper() for m in args.models_to_evaluate])}")
    logger.info("=" * 70)

    predictions_dir = Path(args.predictions_dir)
    data_dir = Path(args.data_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    logger.info(f"Predictions from: {predictions_dir}")
    logger.info(f"Data from: {data_dir}")
    logger.info(f"Results will be saved to: {output_dir}")

    # Load test data and training history once
    actuals_test_set_df, full_train_history_df = load_actuals_and_training_data(data_dir)

    if actuals_test_set_df is None or full_train_history_df is None:
        logger.error("Missing critical data for evaluation. Exiting.")
        sys.exit(1)

    all_model_results = {}
    combined_accuracy_metrics = []
    combined_uncertainty_metrics = []

    # Evaluate each model
    for model_type in args.models_to_evaluate:
        logger.info(f"\n{'='*50}")
        logger.info(f"EVALUATING MODEL: {model_type.upper()}")
        logger.info(f"{'='*50}")
        
        # Load predictions for this model
        point_predictions_df, quantile_predictions_df = load_prediction_files(predictions_dir, model_type)
        
        if point_predictions_df is None and quantile_predictions_df is None:
            logger.warning(f"No prediction files found for model {model_type}. Skipping.")
            continue

        # Calculate accuracy metrics (if point predictions available)
        accuracy_metrics_df = pd.DataFrame()
        if point_predictions_df is not None:
            accuracy_metrics_df = calculate_all_accuracy_metrics_per_series(
                point_predictions_df, actuals_test_set_df, full_train_history_df
            )
            if not accuracy_metrics_df.empty:
                accuracy_metrics_df['model'] = model_type
                combined_accuracy_metrics.append(accuracy_metrics_df)

        # Calculate uncertainty metrics (if quantile predictions available)
        uncertainty_metrics_df = pd.DataFrame()
        if quantile_predictions_df is not None:
            uncertainty_metrics_df = calculate_all_uncertainty_metrics_per_series(
                quantile_predictions_df, actuals_test_set_df, full_train_history_df
            )
            if not uncertainty_metrics_df.empty:
                uncertainty_metrics_df['model'] = model_type
                combined_uncertainty_metrics.append(uncertainty_metrics_df)

        # Merge results for this model
        model_comprehensive_df = pd.DataFrame()
        if not accuracy_metrics_df.empty and not uncertainty_metrics_df.empty:
            model_comprehensive_df = pd.merge(accuracy_metrics_df, uncertainty_metrics_df, on="id", how="outer")
            model_comprehensive_df['model'] = model_type
        elif not accuracy_metrics_df.empty:
            model_comprehensive_df = accuracy_metrics_df
        elif not uncertainty_metrics_df.empty:
            model_comprehensive_df = uncertainty_metrics_df

        # Calculate aggregated metrics for this model
        if not model_comprehensive_df.empty:
            numeric_columns = model_comprehensive_df.select_dtypes(include=[np.number]).columns
            aggregated_metrics = model_comprehensive_df[numeric_columns].mean().to_dict()
            all_model_results[model_type] = aggregated_metrics

            # Save individual model results
            model_comprehensive_df.to_csv(output_dir / f"{model_type}_comprehensive_metrics.csv", index=False)
            logger.info(f"Saved comprehensive metrics to: {model_type}_comprehensive_metrics.csv")

            # Log key results for this model
            rmsse_val = aggregated_metrics.get('rmsse', 'N/A')
            mae_val = aggregated_metrics.get('mae', 'N/A')
            pinball_val = aggregated_metrics.get('pinball_loss', 'N/A')
            
            logger.info(f"Results for {model_type.upper()}:")
            if isinstance(rmsse_val, (int, float)) and not np.isnan(rmsse_val):
                logger.info(f"  RMSSE: {rmsse_val:.4f}")
            else:
                logger.info(f"  RMSSE: N/A")
                
            if isinstance(mae_val, (int, float)) and not np.isnan(mae_val):
                logger.info(f"  MAE: {mae_val:.4f}")
            else:
                logger.info(f"  MAE: N/A")
                
            if isinstance(pinball_val, (int, float)) and not np.isnan(pinball_val):
                logger.info(f"  Pinball Loss: {pinball_val:.4f}")
            else:
                logger.info(f"  Pinball Loss: N/A")

    # Combine all results
    if combined_accuracy_metrics:
        all_accuracy_df = pd.concat(combined_accuracy_metrics, ignore_index=True)
        all_accuracy_df.to_csv(output_dir / "all_models_accuracy_metrics.csv", index=False)
        logger.info("Saved combined accuracy metrics to: all_models_accuracy_metrics.csv")
        
    if combined_uncertainty_metrics:
        all_uncertainty_df = pd.concat(combined_uncertainty_metrics, ignore_index=True)
        all_uncertainty_df.to_csv(output_dir / "all_models_uncertainty_metrics.csv", index=False)
        logger.info("Saved combined uncertainty metrics to: all_models_uncertainty_metrics.csv")

    # Save overall summary
    summary = {
        "evaluation_timestamp": datetime.now().isoformat(),
        "models_evaluated": args.models_to_evaluate,
        "num_models": len(args.models_to_evaluate),
        "aggregated_results": all_model_results,
        "available_accuracy_metrics": 13,
        "available_uncertainty_metrics": 8,
        "total_metrics": 21,
        "generate_plots": args.generate_plots,
        "statistical_comparison": args.statistical_comparison
    }
    
    with open(output_dir / "evaluation_summary.json", 'w') as f:
        json.dump(summary, f, indent=4, default=lambda x: float(x) if isinstance(x, (np.float32, np.float64, np.float_)) else x)

    logger.info("Saved evaluation summary to: evaluation_summary.json")
    
    # Final summary
    logger.info("\n" + "=" * 70)
    logger.info(f"EVALUATION COMPLETED")
    logger.info(f"Successfully evaluated {len(all_model_results)} models with 21 metrics each")
    logger.info(f"Results saved to: {output_dir}")
    logger.info("=" * 70)

if __name__ == "__main__":
    main()