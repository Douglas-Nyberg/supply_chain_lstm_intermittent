#////////////////////////////////////////////////////////////////////////////////#
# File:         4_predict_models.py                                              #
# Author:       Douglas Nyberg                                                   #
# Email:        douglas1.nyberg@gmail.com                                        #
# Date:         2025-04-15                                                       #
#////////////////////////////////////////////////////////////////////////////////#
#!/usr/bin/env python3
"""
Generate forecasts from trained models.

Loads models, prepares input data, generates predictions.
Saves in M5 format with metadata.
"""

# imports
import argparse
import json
import logging
import pickle
import sys
import time
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any
from datetime import datetime

import numpy as np
import pandas as pd
import torch  # For LSTM model and tensor operations
from tqdm import tqdm

# Set up project path
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent.parent
sys.path.append(str(PROJECT_ROOT))

# Import project-specific modules
from src.models.classical import ARIMAForecaster, CrostonSBA, TSB # For loading pickled models
from src.models.lstm import LSTMModel # For LSTM model instantiation
import config_exp1 as config
from src.utils import set_random_seed, sanitize_for_path # Ensure sanitize_for_path is in src/utils.py

# Setup logging
config.LOGS_DIR.mkdir(parents=True, exist_ok=True)
# Clear existing handlers
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

# Define command-line arguments
def parse_arguments() -> argparse.Namespace:
    """Parses command-line arguments for the prediction script."""
    parser = argparse.ArgumentParser(description="Generate predictions from trained models (Per-Series).")

    parser.add_argument(
        "--model-type", type=str, required=True,
        choices=config.AVAILABLE_MODELS["classical"] + config.AVAILABLE_MODELS["deep_learning"],
        help="Type of model to generate predictions for (e.g., 'croston', 'lstm')."
    )

    parser.add_argument(
        "--input-data-dir", type=str,
        default=config.DATA_CONFIG["preprocessed_output_dir"],
        help="Directory containing preprocessed data (train, val, test files)."
    )

    parser.add_argument(
        "--run-experiment-name", type=str, required=True,
        help="The specific name/tag of the experiment run for which to generate predictions."
    )

    parser.add_argument(
        "--limit-series", type=int, default=None, # Changed from limit-items
        help="Limit number of unique series ('id') for prediction (for quick tests)."
    )

    parser.add_argument(
        "--device", type=str,
        default=config.DEVICE_CONFIG["device_name_lstm"], choices=["cuda", "cpu"],
        help="Device for LSTM predictions (cuda or cpu)."
    )
    return parser.parse_args()

# Helper functions for data loading and prediction

def load_prediction_context_for_series(
    series_id_value: str,
    historical_val_df_global: pd.DataFrame,
    lstm_seq_len: int
) -> Optional[pd.DataFrame]:
    """
    Loads the most recent historical data (from the validation set) for an LSTM series
    to form the input sequence for predicting the test set.
    """
    series_val_history = historical_val_df_global[historical_val_df_global['id'] == series_id_value].copy()
    if series_val_history.empty:
        logger.warning(f"LSTM Context: No validation data found for series {series_id_value}.")
        return None

    series_val_history = series_val_history.sort_values('date')

    # We need the last `lstm_seq_len` days from this validation data as input context for test set predictions
    if len(series_val_history) < lstm_seq_len:
        logger.warning(f"LSTM Context: Series {series_id_value}: Not enough validation history ({len(series_val_history)} days) "
                       f"to form an input sequence of length {lstm_seq_len}. Using all available validation history.")
        return series_val_history
    
    return series_val_history.tail(lstm_seq_len)

def predict_with_classical_series_model(
    series_id_value: str, # Changed from item_id
    model_type: str,
    # Classical models are re-fit or use their stored state; actual sales series not strictly needed if model persists state.
    # However, if the model interface expects historical_train_sales, it should be provided.
    # For simplicity, assuming classical models loaded can predict `horizon` steps ahead directly.
    horizon: int,
    quantiles_to_predict: List[float]
) -> Optional[Dict[str, Any]]:
    """Generates point and quantile forecasts from a trained classical model for a single series."""
    
    # Construct model path
    # Classical models might be saved as series_id_value_modeltype.pkl or item/store subdirs
    # Assuming a flat structure for classical models within their type directory for now
    sanitized_id_for_filename = sanitize_for_path(series_id_value)
    model_storage_dir = config.get_model_storage_dir(model_type) # e.g., .../classical/croston/
    model_file = model_storage_dir / f"{sanitized_id_for_filename}_{model_type}_model.pkl"

    if not model_file.exists():
        logger.warning(f"Classical model for series {series_id_value} ({model_type}) not found at {model_file}.")
        return None
    
    try:
        # Load model
        with open(model_file, 'rb') as f:
            trained_model = pickle.load(f)
        
        # Generate point forecast
        point_forecast_values = trained_model.predict(h=horizon)
        
        # Generate quantile forecasts
        quantile_forecasts_dict = {}
        if hasattr(trained_model, 'predict_quantiles'):
            logger.debug(f"Model {model_type} for {series_id_value} has 'predict_quantiles'. Attempting to generate for: {quantiles_to_predict}")
            try:
                # Call predict_quantiles with all quantiles at once 
                quantile_results = trained_model.predict_quantiles(h=horizon, quantiles=quantiles_to_predict)
                
                # Convert to expected format (string keys for quantile levels)
                for q_level, q_values in quantile_results.items():
                    if len(q_values) == horizon:
                        quantile_forecasts_dict[f"{q_level:.3f}"] = q_values
                    else:
                        logger.warning(f"Length mismatch for {model_type} quantile {q_level} prediction: expected {horizon}, got {len(q_values)}. Skipping.")
                        
            except Exception as e_q:
                logger.warning(f"Failed to predict quantiles for {model_type} series {series_id_value}: {e_q}")
                # Fallback: use point forecast for all requested quantiles
                for q_level in quantiles_to_predict:
                    quantile_forecasts_dict[f"{q_level:.3f}"] = point_forecast_values
        elif hasattr(trained_model, 'predict_intervals'):
            logger.warning(f"Model {model_type} for {series_id_value} uses interval fallback. Approximating quantiles.")
            # Simplistic: use point forecast as median, and estimate others if possible or repeat point
            for q_level in quantiles_to_predict:
                quantile_forecasts_dict[f"{q_level:.3f}"] = point_forecast_values
        else:
            logger.warning(f"Model {model_type} for {series_id_value} lacks direct quantile/interval method. Using point forecast for all quantiles.")
            for q_level in quantiles_to_predict:
                quantile_forecasts_dict[f"{q_level:.3f}"] = point_forecast_values

        # Format results
        return {
            "id_value": series_id_value, "model_type": model_type, "status": "success",
            "point_forecast": point_forecast_values.tolist(),
            "quantile_forecasts": {k: v.tolist() for k, v in quantile_forecasts_dict.items()}
        }

    except Exception as e:
        logger.error(f"Error predicting with classical model {model_type} for series {series_id_value}: {e}", exc_info=True)
        return {"id_value": series_id_value, "model_type": model_type, "status": "error", "error_message": str(e)}


def predict_with_per_series_lstm(
    series_id_value: str, 
    series_history_for_input_df: pd.DataFrame, # Last `sequence_length` days of data with features
    global_embedding_specs: Dict[str, Tuple[int, int]],
    numerical_col_names: List[str],
    categorical_col_names: List[str],
    lstm_hyperparams: Dict,
    device_name: str,
    base_model_dir: Path # Base directory for all LSTM models 
) -> Optional[Dict[str, Any]]:
    """Generates forecasts from a trained per-series LSTM model."""
    
    # Construct model and scaler paths
    sanitized_id_for_path = sanitize_for_path(series_id_value)
    series_specific_model_dir = base_model_dir / sanitized_id_for_path # e.g., .../lstm/FOODS_1_001_CA_1_validation/
    model_file = series_specific_model_dir / "lstm_model.pt" # Filename used during training
    scaler_file = series_specific_model_dir / "scaler.pkl"  # Filename used during training
    
    model_exists = model_file.exists()
    scaler_exists_if_needed = True 
    if numerical_col_names: 
        scaler_exists_if_needed = scaler_file.exists()

    if not model_exists or not scaler_exists_if_needed:
        missing_info = []
        if not model_exists: missing_info.append("model file")
        if numerical_col_names and not scaler_exists_if_needed: missing_info.append("scaler file")
        logger.warning(f"LSTM {', '.join(missing_info)} for series {series_id_value} not found in {series_specific_model_dir}.")
        return None

    try:
        # Load series-specific scaler
        series_scaler = None
        if numerical_col_names:
            with open(scaler_file, 'rb') as f:
                series_scaler = pickle.load(f)

        # Prepare input sequence for LSTM
        # Check history has exactly sequence_length records
        if len(series_history_for_input_df) != lstm_hyperparams["sequence_length"]:
             logger.warning(f"Series {series_id_value}: Provided history length ({len(series_history_for_input_df)}) "
                            f"does not match LSTM sequence_length ({lstm_hyperparams['sequence_length']}). Cannot predict accurately.")
             return {"id_value": series_id_value, "model_type": "lstm", "status": "error", "error_message": "Incorrect history length for input sequence."}

        input_df_slice = series_history_for_input_df 

        # Extract categorical features (already integer encoded from preprocessing)
        X_cat_input_dict = {
            col: torch.tensor(input_df_slice[col].values, dtype=torch.long).unsqueeze(0).to(device_name) # Add batch dim [1, seq_len]
            for col in categorical_col_names
        }

        # Extract and scale numerical features
        if numerical_col_names and series_scaler:
            num_data_slice_scaled = series_scaler.transform(input_df_slice[numerical_col_names])
            X_num_input_tensor = torch.tensor(num_data_slice_scaled, dtype=torch.float32).unsqueeze(0).to(device_name) # Add batch dim [1, seq_len, num_num_feats]
        elif not numerical_col_names:
             # Create an empty tensor with correct batch and sequence dimensions if no numerical features
             X_num_input_tensor = torch.empty(1, lstm_hyperparams["sequence_length"], 0, dtype=torch.float32).to(device_name)
        else:
            raise FileNotFoundError(f"Scaler file expected but not found for series {series_id_value} with numerical features.")

        # Load LSTM model and make prediction
        num_numerical_features = len(numerical_col_names)
        
        # For backward compatibility with existing models, use defaults for target statistics
        # New models will have these properly initialized during training
        model_instance = LSTMModel(
            numerical_input_dim=num_numerical_features,
            embedding_specs=global_embedding_specs,
            hidden_dim=lstm_hyperparams["hidden_dim"],
            num_layers=lstm_hyperparams["num_layers"],
            output_dim=lstm_hyperparams["forecast_horizon"],
            dropout=lstm_hyperparams["dropout"], # Automatically off during model.eval()
            bidirectional=lstm_hyperparams["bidirectional"],
            use_attention=lstm_hyperparams["use_attention"],
            quantile_output=lstm_hyperparams["quantile_output"],
            quantile_levels=lstm_hyperparams["quantiles"] if lstm_hyperparams["quantile_output"] else None,
            target_mean=None,  # Will use defaults for existing models
            target_std=None    # Will use defaults for existing models
        ).to(device_name)
        
        # Load the trained state dictionary
        model_instance.load_state_dict(torch.load(model_file, map_location=device_name, weights_only=True))
        model_instance.eval() # Set to evaluation mode (crucial for dropout, batchnorm etc.)

        with torch.no_grad(): # Disable gradient calculations for inference
            predictions_tensor = model_instance(X_cat_input_dict, X_num_input_tensor)

        # Process predictions
        point_forecast_values_list: List[float]
        quantile_forecasts_dict_list: Dict[str, List[float]] = {}

        if lstm_hyperparams["quantile_output"]:
            # `predictions_tensor` is expected to be a dict: {quantile_float: tensor_of_shape_(batch, horizon)}
            # Ensure 0.5 is present for point forecast if using quantiles
            median_quantile_key = 0.5
            if median_quantile_key not in predictions_tensor: # Fallback if 0.5 not explicitly predicted
                logger.warning(f"Median (0.5) quantile not found in LSTM predictions for {series_id_value}. Using first available quantile as point forecast.")
                # Take the first available quantile as a fallback point forecast
                first_q_key = next(iter(predictions_tensor))
                point_forecast_values_list = predictions_tensor[first_q_key].squeeze().cpu().numpy().tolist()
            else:
                point_forecast_values_list = predictions_tensor[median_quantile_key].squeeze().cpu().numpy().tolist()
            
            quantile_forecasts_dict_list = {
                f"{q_level:.3f}": preds.squeeze().cpu().numpy().tolist()
                for q_level, preds in predictions_tensor.items()
            }
        else:
            # `predictions_tensor` is a tensor_of_shape_(batch, horizon)
            point_forecast_values_list = predictions_tensor.squeeze().cpu().numpy().tolist()
            # No quantiles were trained for, so this will be empty
            logger.debug(f"LSTM for series {series_id_value} not trained for quantiles. Quantile forecasts will be empty.")

        return {
            "id_value": series_id_value, "model_type": "lstm", "status": "success",
            "point_forecast": point_forecast_values_list,
            "quantile_forecasts": quantile_forecasts_dict_list
        }

    except Exception as e:
        logger.error(f"Error predicting with LSTM model for series {series_id_value}: {e}", exc_info=True)
        return {"id_value": series_id_value, "model_type": "lstm", "status": "error", "error_message": str(e)}

# main prediction function
def main():
    """Runs the entire prediction generation workflow."""
    # Parse arguments and initial setup
    args = parse_arguments()
    set_random_seed(config.RANDOM_SEED) # For any stochasticity if present in post-processing

    logger.info("=" * 70)
    logger.info(f"STARTING: MODEL PREDICTIONS for Experiment Run '{args.run_experiment_name}'")
    logger.info(f"Model Type: {args.model_type.upper()}")
    logger.info(f"Processing Level: Per-Series (Item-Store 'id')")
    logger.info("=" * 70)

    # Define paths and load shared data
    preprocessed_data_dir = Path(args.input_data_dir)
    predictions_output_dir = config.get_predictions_output_dir(args.run_experiment_name, args.model_type)
    predictions_output_dir.mkdir(parents=True, exist_ok=True)
    logger.info(f"Predictions will be saved to: {predictions_output_dir}")

    # Load metadata (for LSTM features and embedding specs)
    metadata_path = preprocessed_data_dir / config.DATA_CONFIG["metadata_file_name"]
    if not metadata_path.exists():
        logger.error(f"Metadata file not found in {preprocessed_data_dir}. Exiting.")
        sys.exit(1)
    with open(metadata_path, 'r') as f:
        preprocessing_metadata = json.load(f)
    
    # Load full training and validation data (used for context by different model types)
    train_data_path = preprocessed_data_dir / config.DATA_CONFIG["train_file_name"]
    val_data_path = preprocessed_data_dir / config.DATA_CONFIG["validation_file_name"]
    
    if not train_data_path.exists() or not val_data_path.exists():
        logger.error(f"Full train or validation data not found in {preprocessed_data_dir}. Needed for prediction context. Exiting.")
        sys.exit(1)
    
    full_train_df_global = pd.read_csv(train_data_path, parse_dates=['date'])
    full_val_df_global = pd.read_csv(val_data_path, parse_dates=['date'])
    logger.info(f"Loaded global train data ({full_train_df_global.shape}) and validation data ({full_val_df_global.shape}) for context.")


    # Identify series with trained models
    model_storage_base_dir = config.get_model_storage_dir(args.model_type) # e.g., .../deep_learning/lstm/ OR .../classical/croston/
    
    # Get all unique IDs from the training data and check which have trained models
    # This approach ensures we use the original IDs and properly handle sanitization
    all_ids_in_data = full_train_df_global['id'].unique()
    trained_series_ids = []
    
    for series_id_val in all_ids_in_data:
        sanitized_id_val = sanitize_for_path(series_id_val)
        if args.model_type == 'lstm':
            model_file = model_storage_base_dir / sanitized_id_val / "lstm_model.pt"
            if model_file.exists():
                trained_series_ids.append(series_id_val) # Use original ID
        else:
            model_file = model_storage_base_dir / f"{sanitized_id_val}_{args.model_type}_model.pkl"
            if model_file.exists():
                trained_series_ids.append(series_id_val) # Use original ID


    if args.limit_series:
        trained_series_ids = trained_series_ids[:args.limit_series]
    
    if not trained_series_ids:
        logger.warning(f"No trained models found for type '{args.model_type}' in {model_storage_base_dir} matching IDs in data. Exiting.")
        return
    logger.info(f"Found {len(trained_series_ids)} series with trained {args.model_type} models to predict for.")

    # Generate predictions for each series
    all_series_predictions_list = []
    num_failed_predictions = 0

    # LSTM-specific global info
    global_embedding_specs_for_lstm = None
    lstm_numerical_cols = []
    lstm_categorical_cols = []
    if args.model_type == 'lstm':
        # Embedding specs from metadata (vocab_size, embedding_dim)
        raw_embedding_specs = preprocessing_metadata['embedding_info']
        # Only include categorical features that are actually used in LSTM training
        categorical_features_used = preprocessing_metadata['feature_info']['categorical_features']
        global_embedding_specs_for_lstm = {
            name: (spec_dict['vocab_size'], spec_dict['embedding_dim'])
            for name, spec_dict in raw_embedding_specs.items()
            if name in categorical_features_used
        }
        lstm_numerical_cols = preprocessing_metadata['feature_info']['numerical_features']
        lstm_categorical_cols = preprocessing_metadata['feature_info']['categorical_features']

    for series_id_value in tqdm(trained_series_ids, desc=f"Predicting for {args.model_type.upper()}"):
        prediction_result_dict = None

        if args.model_type == 'lstm':
            # For LSTM, the context is the last `sequence_length` days of the validation data
            series_prediction_context_df = load_prediction_context_for_series(
                series_id_value=series_id_value,
                historical_val_df_global=full_val_df_global,
                lstm_seq_len=config.LSTM_CONFIG["sequence_length"]
            )
            if series_prediction_context_df is None or series_prediction_context_df.empty:
                logger.warning(f"No context data for LSTM prediction for series {series_id_value}. Skipping.")
                num_failed_predictions += 1
                continue
            
            prediction_result_dict = predict_with_per_series_lstm(
                series_id_value=series_id_value,
                series_history_for_input_df=series_prediction_context_df,
                global_embedding_specs=global_embedding_specs_for_lstm,
                numerical_col_names=lstm_numerical_cols,
                categorical_col_names=lstm_categorical_cols,
                lstm_hyperparams=config.LSTM_CONFIG,
                device_name=args.device,
                base_model_dir=model_storage_base_dir # Pass base LSTM model dir
            )
        elif args.model_type in config.AVAILABLE_MODELS["classical"]:
            # Classical models use their internal state; no specific input history DF needed 
            prediction_result_dict = predict_with_classical_series_model(
                series_id_value=series_id_value,
                model_type=args.model_type,
                horizon=config.FORECAST_CONFIG["horizon"],
                quantiles_to_predict=config.FORECAST_CONFIG["quantiles"]
            )
        
        if prediction_result_dict and prediction_result_dict.get("status") == "success":
            all_series_predictions_list.append(prediction_result_dict)
        else:
            num_failed_predictions += 1
            error_msg = "Unknown reason"
            if prediction_result_dict and prediction_result_dict.get('error_message'):
                error_msg = prediction_result_dict.get('error_message')
            logger.warning(f"Prediction failed for series {series_id_value}, model {args.model_type}. Reason: {error_msg}")

    # Combine and save all predictions
    if not all_series_predictions_list:
        logger.error(f"No predictions were successfully generated for model type {args.model_type}. Exiting.")
        sys.exit(1)

    logger.info(f"Combining {len(all_series_predictions_list)} series predictions into final output files...")
    
    point_forecast_records_list = []
    quantile_forecast_records_list = []

    for pred_dict_item in all_series_predictions_list:
        current_id_value = pred_dict_item["id_value"] # This is the original 'id', e.g., FOODS_1_001_CA_1
        
        # Point forecast record
        # For M5 format, the ID is typically ITEM_STORE_ID_validation or ITEM_STORE_ID_evaluation
        # We are predicting for the test period, which follows validation.
        # Using "_evaluation" suffix to align with typical M5 test set predictions.
        point_record_id_m5_format = f"{current_id_value}_evaluation"
        point_record = {"id": point_record_id_m5_format}
        
        actual_point_forecast_values = pred_dict_item["point_forecast"] # Already a list
        for i, forecast_val in enumerate(actual_point_forecast_values):
            point_record[f"F{i+1}"] = forecast_val
        point_forecast_records_list.append(point_record)

        # Quantile forecast records 
        if pred_dict_item.get("quantile_forecasts"):
            for q_str_key, q_values_list in pred_dict_item["quantile_forecasts"].items():
                quantile_float_val = float(q_str_key)
                quantile_record_id_m5_format = f"{current_id_value}_{quantile_float_val:.3f}_evaluation"
                q_record = {"id": quantile_record_id_m5_format}
                for i, forecast_val in enumerate(q_values_list):
                    q_record[f"F{i+1}"] = forecast_val
                quantile_forecast_records_list.append(q_record)
    
    # Create DataFrames from the collected records
    point_forecast_final_df = pd.DataFrame(point_forecast_records_list)
    point_output_file_path = predictions_output_dir / f"{args.model_type}_point_forecasts.csv"
    point_forecast_final_df.to_csv(point_output_file_path, index=False)
    logger.info(f"  Saved point forecasts to {point_output_file_path.name}")

    if quantile_forecast_records_list:
        quantile_forecast_final_df = pd.DataFrame(quantile_forecast_records_list)
        quantile_output_file_path = predictions_output_dir / f"{args.model_type}_quantile_forecasts.csv"
        quantile_forecast_final_df.to_csv(quantile_output_file_path, index=False)
        logger.info(f"  Saved quantile forecasts to {quantile_output_file_path.name}")
    else:
        logger.info("  No quantile forecasts were generated or to be saved.")
        
    # Save prediction run metadata
    prediction_run_metadata = {
        "run_timestamp": datetime.now().isoformat(),
        "run_experiment_name": args.run_experiment_name,
        "model_type_predicted": args.model_type,
        "processing_level": "per_series_item_store",
        "input_data_source_dir": str(preprocessed_data_dir),
        "output_predictions_dir": str(predictions_output_dir),
        "num_series_attempted_for_prediction": len(trained_series_ids),
        "num_series_successfully_predicted": len(all_series_predictions_list),
        "num_series_failed_prediction": num_failed_predictions,
        "forecast_horizon_predicted": config.FORECAST_CONFIG["horizon"],
        "quantiles_generated_if_any": config.FORECAST_CONFIG["quantiles"] if quantile_forecast_records_list else []
    }
    metadata_output_file_path = predictions_output_dir / f"{args.model_type}_prediction_run_metadata.json"
    with open(metadata_output_file_path, 'w') as f:
        json.dump(prediction_run_metadata, f, indent=4)
    logger.info(f"  Saved prediction run metadata to {metadata_output_file_path.name}")
    
    # Final log messages
    logger.info("=" * 70)
    logger.info(f"PREDICTIONS GENERATED for model {args.model_type.upper()}")
    logger.info(f"  Successfully generated predictions for {len(all_series_predictions_list)} series.")
    logger.info(f"  Failed to generate predictions for {num_failed_predictions} series.")
    logger.info(f"  Output files saved to: {predictions_output_dir}")
    logger.info("=" * 70)

# Execute main workflow
if __name__ == "__main__":
    main()