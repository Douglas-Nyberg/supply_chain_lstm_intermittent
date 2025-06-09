#////////////////////////////////////////////////////////////////////////////////#
# File:         3_train_lstm_models.py                                           #
# Author:       Douglas Nyberg                                                   #
# Email:        douglas1.nyberg@gmail.com                                        #
# Date:         2025-04-05                                                       #
#////////////////////////////////////////////////////////////////////////////////#
#!/usr/bin/env python3
"""
Trains one LSTM model per series (item-store 'id') for the unified experiment.

This script:
1.  Loads preprocessed and split train/validation data.
2.  Loads metadata including embedding specifications and feature lists.
3.  For each unique item-store series ('id'):
    a. Extracts its specific training and validation data.
    b. Initializes and fits a series-specific StandardScaler for numerical features.
    c. Creates sequences for LSTM input:
        - Training sequences from the training portion of the series.
        - Validation sequences where input (X) uses history up to the validation
          period, and the target (y) is within the validation period.
    d. Instantiates the LSTMModel (from src.models.lstm) using global embedding
       specs and the series' numerical feature dimension.
    e. Trains the per-series LSTM model, using its validation data for early stopping.
    f. Saves the trained series-specific model and its scaler.
4.  Uses parallel processing for training series if specified.
5.  Reports overall training statistics.

Adheres to the 9 key principles for readability.
"""

# imports
import argparse
import json
import logging
import pickle
import sys
import time
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Tuple, Optional, Any
from concurrent.futures import ProcessPoolExecutor, as_completed # For CPU parallelism

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler # For per-series numerical feature scaling
from tqdm import tqdm

# set up project path
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent.parent
sys.path.append(str(PROJECT_ROOT))

# import project modules
from src.models.lstm import LSTMModel, QuantileLoss 
import config_exp1 as config
from src.utils import set_random_seed, sanitize_for_path
from src.cv_utils import IndexBasedWalkForwardValidator, get_lstm_fold_data, aggregate_cv_metrics
from src.feature_engineering import create_lstm_sequences_for_series

# setup logging
config.LOGS_DIR.mkdir(parents=True, exist_ok=True)
# Clear existing handlers if any
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

# command line args
def parse_arguments() -> argparse.Namespace:
    """Parses command-line arguments for the LSTM training script."""
    parser = argparse.ArgumentParser(description="Train Per-Series (Item-Store) LSTM models for Unified Experiment 1")

    parser.add_argument(
        "--input-dir", type=str,
        default=config.DATA_CONFIG["preprocessed_output_dir"],
        help="Directory of preprocessed data (output of 1_preprocess_exp1.py)."
    )

    parser.add_argument(
        "--parallel-jobs", type=int,
        default=config.WORKFLOW_CONFIG["parallel_jobs"],
        help="Number of parallel jobs for training series (CPU-based)."
    )

    parser.add_argument(
        "--limit-series", type=int,
        default=config.DATA_CONFIG["limit_items"], 
        help="Limit number of unique series ('id') for training."
    )

    parser.add_argument(
        "--force", action="store_true",
        help="Force retraining even if series models exist."
    )

    parser.add_argument(
        "--device", type=str,
        default=config.DEVICE_CONFIG["device_name_lstm"],
        choices=["cuda", "cpu"],
        help="Device to use for training (cuda or cpu). Parallel jobs > 1 might force CPU."
    )
    return parser.parse_args()

# series validation (aligned with classical models)
def validate_series_for_lstm_training(
    series_data: pd.DataFrame, 
    series_id: str,
    min_periods: int = 20,
    min_non_zero: int = 2
) -> bool:
    """
    Validate if a series has sufficient data for LSTM training.
    
    Uses same criteria as classical models to ensure fair comparison:
    - Minimum number of historical periods
    - Some non-zero sales (for intermittent demand models)
    - Some variability in the data
    
    Args:
        series_data: DataFrame for a single series
        series_id: ID of the series
        min_periods: Minimum required periods
        min_non_zero: Minimum required non-zero periods
        
    Returns:
        True if series is valid for training, False otherwise
    """
    sales_values = series_data['sales'].values
    
    if len(sales_values) < min_periods:
        logger.debug(
            f"Series '{series_id}': Too short ({len(sales_values)} < {min_periods})"
        )
        return False
    
    non_zero_count = np.sum(sales_values > 0)
    if non_zero_count < min_non_zero:
        logger.debug(
            f"Series '{series_id}': Too few non-zero values ({non_zero_count} < {min_non_zero})"
        )
        return False
    
    unique_values = len(np.unique(sales_values))
    if unique_values < 2:
        logger.debug(
            f"Series '{series_id}': No variability (all values constant)"
        )
        return False
    
    return True

# data loading
def load_split_data_and_metadata(input_dir: Path) -> Tuple[pd.DataFrame, Dict, Dict]:
    """
    Loads train_val data and relevant metadata for LSTM training.
    CV splits will be created during training if enabled.
    """
    logger.info(f"Loading train_val data and metadata for LSTM training from: {input_dir}")

    train_val_path = input_dir / config.DATA_CONFIG["train_val_file_name"]
    meta_json_path = input_dir / config.DATA_CONFIG["metadata_file_name"]
    meta_pickle_path = input_dir / config.DATA_CONFIG["embedding_info_pickle_name"]

    # Check for existence of all required files
    paths_to_check = [train_val_path, meta_json_path, meta_pickle_path]
    missing_files = [p for p in paths_to_check if not p.exists()]
    if missing_files:
        error_message = f"Required files not found for LSTM training: {missing_files}. Run 1_preprocess_exp1.py."
        logger.error(error_message)
        raise FileNotFoundError(error_message)

    # Load train_val dataframe
    train_val_df = pd.read_csv(train_val_path, parse_dates=['date'])
    logger.info(f"Loaded train_val data: {train_val_df.shape}")

    # Load metadata
    with open(meta_json_path, 'r') as f:
        metadata_json = json.load(f)
    logger.info(f"Loaded {meta_json_path.name}.")

    with open(meta_pickle_path, 'rb') as f:
        _ = pickle.load(f) # embedding_info_full_with_encoders (currently unused)
    logger.info(f"Loaded {meta_pickle_path.name}.")

    # Extract embedding specifications (vocab size, embed dim) needed for model instantiation
    # These are global, derived from all data during preprocessing
    # IMPORTANT: Only include categorical features that are actually used in the model
    categorical_features_used = metadata_json['feature_info']['categorical_features']
    embedding_specs_for_model = {
        name: (info['vocab_size'], info['embedding_dim'])
        for name, info in metadata_json['embedding_info'].items()
        if name in categorical_features_used
    }
    logger.info(f"Extracted embedding specifications for {len(embedding_specs_for_model)} categorical features used in LSTM model.")

    return train_val_df, metadata_json, embedding_specs_for_model

# per-series data prep
class SeriesLSTMDataset(Dataset): # Renamed from ItemLSTMDataset
    """Custom PyTorch Dataset for a single series' sequences."""
    def __init__(self, X_cat_series: Dict[str, np.ndarray], X_num_series: np.ndarray, y_series: np.ndarray):
        self.X_cat_series = {key: torch.tensor(data, dtype=torch.long) for key, data in X_cat_series.items()}
        self.X_num_series = torch.tensor(X_num_series, dtype=torch.float32)
        self.y_series = torch.tensor(y_series, dtype=torch.float32)
        self.categorical_feature_keys = list(X_cat_series.keys())

    def __len__(self) -> int:
        return len(self.y_series)

    def __getitem__(self, index: int) -> Tuple[Dict[str, torch.Tensor], torch.Tensor, torch.Tensor]:
        categorical_sample = {key: self.X_cat_series[key][index] for key in self.categorical_feature_keys}
        numerical_sample = self.X_num_series[index]
        target_sample = self.y_series[index]
        return categorical_sample, numerical_sample, target_sample

def create_sequences_for_single_series(
    series_df_chronological: pd.DataFrame,
    categorical_col_names: List[str],
    numerical_col_names: List[str],
    target_col_name: str,
    sequence_length: int,
    forecast_horizon: int
) -> Optional[Tuple[Dict[str, np.ndarray], np.ndarray, np.ndarray]]:
    """
    Creates feature sequences (X_cat, X_num) and target sequences (y) for one series.
    
    This is a wrapper around the shared utility function for backward compatibility.
    Assumes series_df_chronological is sorted by date.
    """
    # Use the shared utility function
    return create_lstm_sequences_for_series(
        series_data=series_df_chronological,
        sequence_length=sequence_length,
        forecast_horizon=forecast_horizon,
        numerical_cols=numerical_col_names,
        categorical_cols=categorical_col_names,
        target_col=target_col_name
    )


# CV training helper
def train_lstm_with_cv(
    series_id: str,
    series_train_val_df: pd.DataFrame,
    full_train_val_df: pd.DataFrame,
    global_embedding_specs: Dict[str, Tuple[int, int]],
    numerical_col_names: List[str],
    categorical_col_names: List[str],
    lstm_hyperparams: Dict,
    device_name: str
) -> Tuple[Any, Any, float, Optional[Dict]]:
    """
    Train LSTM using cross-validation and return best model.
    Returns: (best_model, scaler, best_val_loss, cv_results)
    """
    # Initialize CV validator
    validator = IndexBasedWalkForwardValidator(
        initial_train_size=config.CV_CONFIG["initial_train_size"],
        step_size=config.CV_CONFIG["step_size"],
        max_splits=config.CV_CONFIG["max_splits"],
        gap=config.CV_CONFIG.get("gap", 0)
    )
    
    # Get CV splits
    data_length = len(series_train_val_df)
    cv_splits = validator.get_split_indices(data_length)
    
    if not cv_splits:
        return None, None, float('inf'), None
    
    # Train on each CV fold
    cv_fold_results = []
    cv_models = []
    best_fold_val_loss = float('inf')
    best_fold_idx = -1
    
    for fold_idx, (train_idx, val_idx) in enumerate(cv_splits):
        logger.debug(f"  CV Fold {fold_idx + 1}/{len(cv_splits)} for series '{series_id}'")
        
        try:
            # Get fold data using cv_utils
            fold_data = get_lstm_fold_data(
                full_train_val_df,
                series_id,
                train_idx,
                val_idx,
                lstm_hyperparams["sequence_length"],
                numerical_col_names,
                categorical_col_names
            )
            
            # Check if we have enough sequences
            if fold_data['train_samples'] == 0 or fold_data['val_samples'] == 0:
                logger.warning(f"Insufficient sequences for fold {fold_idx + 1}, skipping")
                continue
            
            # Create datasets from fold data
            train_dataset = SeriesLSTMDataset(
                fold_data['train']['X_cat'],
                fold_data['train']['X_num'],
                fold_data['train']['y']
            )
            
            val_dataset = SeriesLSTMDataset(
                fold_data['val']['X_cat'],
                fold_data['val']['X_num'],
                fold_data['val']['y']
            )
            
            # Calculate target statistics for data-driven scaling initialization
            # This provides principled initialization based on actual training data characteristics
            train_targets = fold_data['train']['y']  # Shape: (n_samples, forecast_horizon)
            target_mean = float(np.mean(train_targets))
            target_std = float(np.std(train_targets))
            
            # Log the calculated statistics for debugging and verification
            logger.info(f"Fold {fold_idx + 1} target statistics: mean={target_mean:.4f}, std={target_std:.4f}")
            
            # Initialize model for this fold
            num_numerical_features = fold_data['train']['X_num'].shape[-1] if fold_data['train']['X_num'] is not None else 0
            fold_model = LSTMModel(
                numerical_input_dim=num_numerical_features,
                embedding_specs=global_embedding_specs,
                hidden_dim=lstm_hyperparams["hidden_dim"],
                num_layers=lstm_hyperparams["num_layers"],
                output_dim=lstm_hyperparams["forecast_horizon"],
                dropout=lstm_hyperparams["dropout"],
                bidirectional=lstm_hyperparams["bidirectional"],
                use_attention=lstm_hyperparams["use_attention"],
                quantile_output=lstm_hyperparams["quantile_output"],
                quantile_levels=lstm_hyperparams["quantiles"] if lstm_hyperparams["quantile_output"] else None,
                target_mean=target_mean,
                target_std=target_std
            ).to(device_name)
            
            # Set up training
            train_loader = DataLoader(train_dataset, batch_size=lstm_hyperparams["batch_size"], shuffle=True)
            val_loader = DataLoader(val_dataset, batch_size=lstm_hyperparams["batch_size"], shuffle=False)
            
            if lstm_hyperparams["quantile_output"]:
                criterion = QuantileLoss(lstm_hyperparams["quantiles"])
            else:
                criterion = nn.MSELoss()
            
            optimizer = optim.Adam(
                fold_model.parameters(),
                lr=lstm_hyperparams["learning_rate"],
                weight_decay=lstm_hyperparams["weight_decay"]
            )
            
            # Simple training loop for fold
            fold_model.train()
            for epoch in range(min(10, lstm_hyperparams["epochs"])):  # Limit epochs for CV
                for batch_cat, batch_num, batch_y in train_loader:
                    batch_cat = {k: v.to(device_name) for k, v in batch_cat.items()}
                    batch_num = batch_num.to(device_name)
                    batch_y = batch_y.to(device_name)
                    
                    optimizer.zero_grad()
                    
                    # debug: check input tensors
                    logger.debug(f"Batch input shapes: cat keys={list(batch_cat.keys())}, num={batch_num.shape}, target={batch_y.shape}")
                    logger.debug(f"Input ranges: num=[{batch_num.min():.3f}, {batch_num.max():.3f}], target=[{batch_y.min():.3f}, {batch_y.max():.3f}]")
                    logger.debug(f"Input has NaN/inf: num_nan={torch.isnan(batch_num).any()}, num_inf={torch.isinf(batch_num).any()}")
                    
                    # debug: check categorical indicies
                    for cat_name, cat_tensor in batch_cat.items():
                        cat_min, cat_max = cat_tensor.min().item(), cat_tensor.max().item() 
                        logger.debug(f"Cat {cat_name}: range=[{cat_min}, {cat_max}], has_nan={torch.isnan(cat_tensor).any()}")
                    
                    predictions = fold_model(batch_cat, batch_num)
                    
                    # debug: check prediction tensors
                    if lstm_hyperparams["quantile_output"]:
                        first_quantile = list(predictions.values())[0]
                        logger.debug(f"Prediction shapes: {first_quantile.shape} vs targets {batch_y.shape}")
                        logger.debug(f"Prediction ranges: [{first_quantile.min():.3f}, {first_quantile.max():.3f}]")
                        logger.debug(f"Prediction has inf/nan: inf={torch.isinf(first_quantile).any()}, nan={torch.isnan(first_quantile).any()}")
                    
                    loss = criterion(predictions, batch_y)
                    logger.debug(f"Loss value: {loss.item():.6f}, has inf/nan: inf={torch.isinf(loss)}, nan={torch.isnan(loss)}")
                    loss.backward()
                    optimizer.step()
            
            # Evaluate fold
            fold_model.eval()
            val_loss = 0.0
            with torch.no_grad():
                for batch_cat, batch_num, batch_y in val_loader:
                    batch_cat = {k: v.to(device_name) for k, v in batch_cat.items()}
                    batch_num = batch_num.to(device_name)
                    batch_y = batch_y.to(device_name)
                    
                    predictions = fold_model(batch_cat, batch_num)
                    loss = criterion(predictions, batch_y)
                    val_loss += loss.item()
            
            avg_val_loss = val_loss / len(val_loader)
            
            # Record results
            fold_metrics = {'val_loss': float(avg_val_loss), 'fold': fold_idx}
            cv_fold_results.append(fold_metrics)
            cv_models.append(fold_model)
            
            if avg_val_loss < best_fold_val_loss:
                best_fold_val_loss = avg_val_loss
                best_fold_idx = fold_idx
                
        except Exception as e:
            logger.warning(f"Error in CV fold {fold_idx + 1}: {e}")
            continue
    
    if not cv_fold_results:
        return None, None, float('inf'), None, None
    
    # Aggregate CV results
    cv_results = aggregate_cv_metrics(cv_fold_results)
    
    # Select best model
    best_model = cv_models[best_fold_idx]
    
    # Fit scaler on all train_val data
    scaler = None
    if numerical_col_names:
        scaler = StandardScaler()
        scaler.fit(series_train_val_df[numerical_col_names])
    
    logger.debug(f"CV complete: selected fold {best_fold_idx + 1} with val_loss={best_fold_val_loss:.4f}")
    
    return best_model, scaler, best_fold_val_loss, best_fold_idx, cv_results


# per-series LSTM training function
def train_lstm_for_single_series(
    id_value: str,
    series_train_val_df: pd.DataFrame,  # Combined train+val data for this series
    full_train_val_df: pd.DataFrame,    # Full dataset needed for CV
    global_embedding_specs: Dict[str, Tuple[int, int]],
    numerical_col_names: List[str],
    categorical_col_names: List[str],
    lstm_hyperparams: Dict,
    series_model_dir: Path,
    device_name: str,
    force_retrain: bool
) -> Dict:
    """
    Trains and saves an LSTM model for a single item-store series ('id').
    Supports cross-validation when enabled in config.
    """
    # Imports needed if this function is run in a separate process by ProcessPoolExecutor
    from datetime import datetime # For timestamping metadata
    import torch # For model saving and operations
    from sklearn.preprocessing import StandardScaler # For scaling
    import pickle # For saving scaler
    from pathlib import Path # For path handling

    set_random_seed(config.RANDOM_SEED) # Ensure reproducibility for each series' training if run in parallel


    # Define file paths for this series' model, scaler, and metadata
    # Sanitize id_value for use in file/directory names if it contains special characters
    sanitized_id_value_for_path = sanitize_for_path(id_value)
    # Ensure series_model_dir is a Path object
    series_model_dir = Path(series_model_dir)
    model_output_subdir = series_model_dir / sanitized_id_value_for_path
    model_output_subdir.mkdir(parents=True, exist_ok=True)
    logger.debug(f"Created model directory: {model_output_subdir}")

    series_model_file = model_output_subdir / "lstm_model.pt"
    series_scaler_file = model_output_subdir / "scaler.pkl"
    series_meta_file = model_output_subdir / "training_meta.json"
    logger.debug(f"Model will be saved to: {series_model_file}")


    # Check if model already exists (if not forcing retrain) ---
    if series_model_file.exists() and series_scaler_file.exists() and series_meta_file.exists() and not force_retrain:
        logger.debug(f"LSTM model for series {id_value} already exists. Skipping (--force not set).")
        return {"id_value": id_value, "status": "skipped_exists", "model_file": str(series_model_file)}

    logger.debug(f"Starting LSTM training for series {id_value} on device '{device_name}'.")

    try:
        # Ensure data is sorted by date ---
        series_train_val_sorted = series_train_val_df.sort_values('date').reset_index(drop=True)
        
        # Cross-Validation Logic ---
        if config.CV_CONFIG.get("use_cv", False):
            logger.debug(f"Using cross-validation for LSTM on series '{id_value}'")
            
            # Initialize CV validator
            validator = IndexBasedWalkForwardValidator(
                initial_train_size=config.CV_CONFIG["initial_train_size"],
                step_size=config.CV_CONFIG["step_size"],
                max_splits=config.CV_CONFIG["max_splits"],
                gap=config.CV_CONFIG.get("gap", 0)
            )
            
            # Get CV splits
            data_length = len(series_train_val_sorted)
            cv_splits = validator.get_split_indices(data_length)
            
            # Use CV training
            best_model, series_scaler, best_val_loss, best_fold_idx, cv_results = train_lstm_with_cv(
                id_value, series_train_val_sorted, full_train_val_df,
                global_embedding_specs, numerical_col_names, categorical_col_names,
                lstm_hyperparams, device_name
            )
            
            if best_model is None:
                logger.warning(f"CV training failed for series {id_value}, falling back to simple split")
                # Fall back to simple split
                val_size = min(28, data_length // 5)
                series_train_df = series_train_val_sorted.iloc[:-val_size]
                series_val_df = series_train_val_sorted.iloc[-val_size:]
                cv_results = None
                # Continue with existing non-CV logic below
            else:
                # CV training successful - save model and return
                logger.info(f"Saving LSTM model for {id_value} to: {series_model_file}")
                torch.save(best_model.state_dict(), series_model_file)
                logger.info(f"Model saved successfully to: {series_model_file}")
                
                if series_scaler:
                    logger.debug(f"Saving scaler to: {series_scaler_file}")
                    with open(series_scaler_file, 'wb') as f_scaler:
                        pickle.dump(series_scaler, f_scaler)
                
                final_training_metadata = {
                    "id_value": id_value,
                    "status": "success",
                    "best_val_loss": float(best_val_loss),
                    "model_file": series_model_file.name,
                    "scaler_file": series_scaler_file.name if series_scaler else None,
                    "model_output_subdir": str(model_output_subdir.relative_to(series_model_dir.parent)),
                    "numerical_features_scaled": numerical_col_names,
                    "categorical_features_used": categorical_col_names,
                    "training_timestamp": datetime.now().isoformat(),
                    "cv_enabled": True,
                    "cv_best_fold_selected": best_fold_idx,
                    "cv_statistics": cv_results  # Complete CV stats
                }
                
                with open(series_meta_file, 'w') as f_meta:
                    json.dump(final_training_metadata, f_meta, indent=4)
                
                logger.info(f"Series {id_value}: CV training successful. Best val_loss: {best_val_loss:.4f}")
                return final_training_metadata
        
        else:
            # Non-CV training - use simple train/val split
            val_size = min(28, len(series_train_val_sorted) // 5)
            series_train_df = series_train_val_sorted.iloc[:-val_size]
            series_val_df = series_train_val_sorted.iloc[-val_size:]
            cv_results = None

        # Scale Numerical Features (Series-Specific Scaler) ---
        series_scaler = None # Initialize scaler
        if not numerical_col_names:
            logger.debug(f"Series {id_value}: No numerical features to scale.")
            series_train_num_scaled_df = pd.DataFrame(index=series_train_df.index)
            series_val_num_scaled_df = pd.DataFrame(index=series_val_df.index) # For consistent structure
        else:
            series_scaler = StandardScaler()
            # Fit scaler on this series' training numerical data
            train_num_np_scaled = series_scaler.fit_transform(series_train_df[numerical_col_names])
            series_train_num_scaled_df = pd.DataFrame(train_num_np_scaled, columns=numerical_col_names, index=series_train_df.index)

            # Transform validation data using the fitted scaler
            # Handle case where validation set might be empty or too short for some numerical features
            if not series_val_df.empty and not series_val_df[numerical_col_names].empty:
                val_num_np_scaled = series_scaler.transform(series_val_df[numerical_col_names])
                series_val_num_scaled_df = pd.DataFrame(val_num_np_scaled, columns=numerical_col_names, index=series_val_df.index)
            else:
                series_val_num_scaled_df = pd.DataFrame(columns=numerical_col_names, index=series_val_df.index)


        # Combine scaled numerical with categorical and target for sequence creation ---
        # Training data for sequences
        series_train_df_for_seq = pd.concat([
            series_train_df[categorical_col_names + ['sales', 'date']],
            series_train_num_scaled_df
        ], axis=1)

        # Validation data for sequences (will be combined with end of train for context)
        series_val_df_for_seq = pd.concat([
            series_val_df[categorical_col_names + ['sales', 'date']],
            series_val_num_scaled_df
        ], axis=1)

        # Create Training Sequences ---
        logger.debug(f"Series {id_value}: Creating training sequences. Train data length: {len(series_train_df_for_seq)}")
        seq_result_train = create_sequences_for_single_series(
            series_train_df_for_seq, categorical_col_names, numerical_col_names, 'sales',
            lstm_hyperparams["sequence_length"], lstm_hyperparams["forecast_horizon"]
        )

        # Create Validation Sequences (Corrected Strategy) ---
        # Validation sequences need input features (X) from history leading up to the validation period,
        # and targets (y) from the validation period itself.
        # Here we combine the tail of training data (for X context) with the validation data (for X context and y targets).
        
        history_for_val_X = series_train_df_for_seq.tail(lstm_hyperparams["sequence_length"]) # Add full sequence_length for context
        
        # Concatenate the training history with the actual validation data
        # This combined DataFrame will be used to generate X,y pairs where y falls in validation
        df_for_val_sequences = pd.concat([history_for_val_X, series_val_df_for_seq]).sort_values('date').reset_index(drop=True)
        
        logger.debug(f"Series {id_value}: Creating validation sequences. Combined data length for val: {len(df_for_val_sequences)}")
        seq_result_val = create_sequences_for_single_series(
            df_for_val_sequences, categorical_col_names, numerical_col_names, 'sales',
            lstm_hyperparams["sequence_length"], lstm_hyperparams["forecast_horizon"]
        )
        
        # Check if sequences were created and unpack ---
        logger.debug(f"Series {id_value}: Training sequences created: {'Yes' if seq_result_train else 'No'}. Validation sequences created: {'Yes' if seq_result_val else 'No'}")

        if not seq_result_train or not seq_result_val:
            error_msg_detail = ""
            if not seq_result_train: error_msg_detail += "No training sequences. "
            if not seq_result_val: error_msg_detail += "No validation sequences. "
            logger.warning(f"Series {id_value}: Not enough data to create train/val sequences. {error_msg_detail}Skipping.")
            return {"id_value": id_value, "status": "error_insufficient_data_for_sequences", "error_message": f"Insufficient data for sequences. {error_msg_detail}"}

        X_cat_train_series, X_num_train_series, y_train_series = seq_result_train
        X_cat_val_series, X_num_val_series, y_val_series = seq_result_val

        if y_train_series.shape[0] == 0 or y_val_series.shape[0] == 0:
            logger.warning(f"Series {id_value}: Zero sequences created for train or val after attempting. Skipping.")
            return {"id_value": id_value, "status": "error_zero_sequences", "error_message": "Zero sequences created."}

        # Create PyTorch DataLoaders ---
        train_dataset_series = SeriesLSTMDataset(X_cat_train_series, X_num_train_series, y_train_series)
        val_dataset_series = SeriesLSTMDataset(X_cat_val_series, X_num_val_series, y_val_series)

        train_loader_series = DataLoader(train_dataset_series, batch_size=lstm_hyperparams["batch_size"], shuffle=True, num_workers=0)
        val_loader_series = DataLoader(val_dataset_series, batch_size=lstm_hyperparams["batch_size"], shuffle=False, num_workers=0)

        # Calculate target statistics for data-driven scaling initialization
        # This provides principled initialization based on actual training data characteristics
        target_mean = float(np.mean(y_train_series))
        target_std = float(np.std(y_train_series))
        
        # Log the calculated statistics for debugging and verification
        logger.info(f"Series {id_value} target statistics: mean={target_mean:.4f}, std={target_std:.4f}")
        
        # Initialize LSTM Model ---
        num_numerical_features_for_series = len(numerical_col_names)
        model_series = LSTMModel(
            numerical_input_dim=num_numerical_features_for_series,
            embedding_specs=global_embedding_specs, # Global specs from all data
            hidden_dim=lstm_hyperparams["hidden_dim"],
            num_layers=lstm_hyperparams["num_layers"],
            output_dim=lstm_hyperparams["forecast_horizon"],
            dropout=lstm_hyperparams["dropout"],
            bidirectional=lstm_hyperparams["bidirectional"],
            use_attention=lstm_hyperparams["use_attention"],
            quantile_output=lstm_hyperparams["quantile_output"],
            quantile_levels=lstm_hyperparams["quantiles"] if lstm_hyperparams["quantile_output"] else None,
            target_mean=target_mean,
            target_std=target_std
        ).to(device_name)

        # Define Optimizer, Loss Function, and Scheduler ---
        optimizer = optim.Adam(model_series.parameters(), lr=lstm_hyperparams["learning_rate"], weight_decay=lstm_hyperparams["weight_decay"])
        criterion = QuantileLoss(lstm_hyperparams["quantiles"]) if lstm_hyperparams["quantile_output"] else nn.MSELoss()
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=lstm_hyperparams["patience"] // 2, factor=0.5, verbose=False)

        # Training Loop for this Series ---
        best_val_loss_series = float('inf')
        patience_counter_series = 0
        epochs_trained_series = 0

        for epoch_num in range(1, lstm_hyperparams["epochs"] + 1):
            model_series.train() # Set model to training mode
            current_epoch_train_loss = 0.0
            for batch_cat, batch_num, batch_y in train_loader_series:
                # Move batch tensors to the designated device
                batch_cat_device = {key: val.to(device_name) for key, val in batch_cat.items()}
                batch_num_device = batch_num.to(device_name)
                batch_y_device = batch_y.to(device_name)

                optimizer.zero_grad() # Clear previous gradients
                predictions = model_series(batch_cat_device, batch_num_device) # Forward pass
                loss = criterion(predictions, batch_y_device) # Calculate loss
                loss.backward() # Backward pass (compute gradients)
                torch.nn.utils.clip_grad_norm_(model_series.parameters(), lstm_hyperparams["gradient_clip"]) # Gradient clipping
                optimizer.step() # Update model parameters
                current_epoch_train_loss += loss.item()

            avg_epoch_train_loss = current_epoch_train_loss / len(train_loader_series)

            # Validation phase
            model_series.eval() # Set model to evaluation mode
            current_epoch_val_loss = 0.0
            with torch.no_grad(): # Disable gradient calculations for validation
                for batch_cat_val, batch_num_val, batch_y_val in val_loader_series:
                    batch_cat_val_device = {key: val.to(device_name) for key, val in batch_cat_val.items()}
                    batch_num_val_device = batch_num_val.to(device_name)
                    batch_y_val_device = batch_y_val.to(device_name)

                    predictions_val = model_series(batch_cat_val_device, batch_num_val_device)
                    loss_val = criterion(predictions_val, batch_y_val_device)
                    current_epoch_val_loss += loss_val.item()

            avg_epoch_val_loss = current_epoch_val_loss / len(val_loader_series)
            scheduler.step(avg_epoch_val_loss) # Adjust learning rate based on val loss
            epochs_trained_series = epoch_num

            # Early stopping and model saving logic
            if avg_epoch_val_loss < best_val_loss_series:
                best_val_loss_series = avg_epoch_val_loss
                patience_counter_series = 0
                torch.save(model_series.state_dict(), series_model_file) # Save best model
                logger.debug(f"Series {id_value}, Epoch {epoch_num}: New best val_loss: {best_val_loss_series:.4f}. Model saved.")
            else:
                patience_counter_series += 1

            if patience_counter_series >= lstm_hyperparams["patience"]:
                logger.info(f"Series {id_value}: Early stopping at epoch {epoch_num}. Best val_loss: {best_val_loss_series:.4f}")
                break
        
        # Save Scaler and Training Metadata ---
        if series_scaler: # Save only if scaler was created (i.e., if numerical features existed)
            with open(series_scaler_file, 'wb') as f_scaler:
                pickle.dump(series_scaler, f_scaler)

        final_training_metadata = {
            "id_value": id_value, "status": "success",
            "epochs_trained": epochs_trained_series, "best_val_loss": best_val_loss_series,
            "model_file": series_model_file.name,
            "scaler_file": series_scaler_file.name if series_scaler else None,
            "model_output_subdir": str(model_output_subdir.relative_to(series_model_dir.parent)), # Relative path
            "numerical_features_scaled": numerical_col_names,
            "categorical_features_used": categorical_col_names,
            "training_timestamp": datetime.now().isoformat(),
            "cv_enabled": config.CV_CONFIG.get("use_cv", False),
            "cv_results": cv_results if 'cv_results' in locals() else None
        }
        with open(series_meta_file, 'w') as f_meta:
            json.dump(final_training_metadata, f_meta, indent=4)

        logger.info(f"Series {id_value}: Training successful. Final val_loss: {best_val_loss_series:.4f} after {epochs_trained_series} epochs.")
        return final_training_metadata

    except Exception as e:
        logger.error(f"Series {id_value}: LSTM training failed. Error: {e}", exc_info=True) # exc_info=True for traceback in logs
        return {"id_value": id_value, "status": "error_exception", "error_message": str(e)}

# main workflow
def main():
    """Runs the entire LSTM model training workflow."""

    # parse arguments
    args = parse_arguments()
    set_random_seed(config.RANDOM_SEED) # Global seed for main process

    logger.info("=" * 70)
    logger.info("STARTING: PER-SERIES (ITEM-STORE) LSTM MODEL TRAINING (Experiment 1)")
    logger.info(f"Input Preprocessed Data Directory: {args.input_dir}")
    logger.info(f"Number of Parallel Jobs: {args.parallel_jobs}")
    logger.info(f"Target Device for Training: {args.device}")
    if args.limit_series:
        logger.info(f"Limiting to {args.limit_series} series.")
    logger.info("=" * 70)

    input_dir_path = Path(args.input_dir)
    # Models are saved under a base "lstm" dir, then per-series subdirectories
    lstm_base_output_dir = config.get_model_storage_dir("lstm") # e.g., .../trained_models/exp1_tag/deep_learning/lstm/
    logger.info(f"LSTM models will be saved to base directory: {lstm_base_output_dir}")
    lstm_base_output_dir.mkdir(parents=True, exist_ok=True)

    # load preprocessed data
    train_val_df_global, metadata_json, global_embedding_specs = \
        load_split_data_and_metadata(input_dir_path)

    numerical_feature_names = metadata_json['feature_info']['numerical_features']
    categorical_feature_names = metadata_json['feature_info']['categorical_features']

    # prepare tasks for parallel processing
    all_unique_series_ids = train_val_df_global['id'].unique() # Using 'id' column
    if args.limit_series:
        all_unique_series_ids = all_unique_series_ids[:args.limit_series]
    
    logger.info(f"Total unique series before validation: {len(all_unique_series_ids)}")

    # Validate series using same criteria as classical models to ensure fair comparison
    valid_series_ids = []
    skipped_series = []
    
    for id_val in all_unique_series_ids:
        # Extract data for the current unique series ('id')
        series_train_val_data = train_val_df_global[train_val_df_global['id'] == id_val].copy()
        
        # Use same validation criteria as classical models (from 2_train_classical_models.py)
        min_periods = config.DATA_CONFIG.get('min_train_periods_classical', 20)
        min_non_zero = config.DATA_CONFIG.get('min_non_zero_periods', 2)
        
        if validate_series_for_lstm_training(series_train_val_data, id_val, min_periods, min_non_zero):
            valid_series_ids.append(id_val)
        else:
            skipped_series.append(id_val)
            logger.debug(f"Skipping series '{id_val}' due to insufficient data for training")

    logger.info(f"Valid series for LSTM training: {len(valid_series_ids)}")
    logger.info(f"Skipped series: {len(skipped_series)}")
    if skipped_series:
        logger.info(f"Skipped series list: {skipped_series}")

    training_tasks = []
    for id_val in valid_series_ids:
        # Extract data for the current unique series ('id')
        series_train_val_data = train_val_df_global[train_val_df_global['id'] == id_val].copy()

        # Pass the base LSTM directory and full df for CV
        training_tasks.append(
            (id_val, series_train_val_data, train_val_df_global,
             global_embedding_specs, numerical_feature_names, categorical_feature_names,
             config.LSTM_CONFIG, lstm_base_output_dir, args.device, args.force)
        )

    # execute training tasks
    overall_series_results = {}
    training_start_overall_time = time.time()

    # Determine effective parallelism based on device and requested jobs
    effective_parallel_jobs = args.parallel_jobs
    if args.device == "cuda" and args.parallel_jobs > 1:
        logger.warning(
            "Device set to 'cuda' but parallel_jobs > 1. "
            "Running LSTM training sequentially on GPU to avoid conflicts. "
            "For true parallelism with PyTorch on GPU, consider using DistributedDataParallel or one GPU per process."
        )
        effective_parallel_jobs = 1

    if effective_parallel_jobs == 1 or not training_tasks:
        logger.info("Running LSTM training sequentially for all series.")
        for task_args_tuple in tqdm(training_tasks, desc="Training Per-Series LSTMs (Sequential)"):
            series_result = train_lstm_for_single_series(*task_args_tuple)
            overall_series_results[series_result["id_value"]] = series_result
    elif training_tasks:
        # Prepare tasks for CPU-based parallelism if effective_parallel_jobs > 1
        # Ensure 'cpu' is passed as device_name to avoid CUDA issues in subprocesses
        logger.info(f"Running LSTM training in parallel with {effective_parallel_jobs} CPU workers.")
        cpu_targeted_tasks = []
        for task_args_list_original in training_tasks:
            task_args_list_modified = list(task_args_list_original)
            task_args_list_modified[8] = "cpu" # Index 8 is device_name in the tuple (index 7 is series_model_dir!)
            cpu_targeted_tasks.append(tuple(task_args_list_modified))

        with ProcessPoolExecutor(max_workers=effective_parallel_jobs) as executor:
            # Submit all tasks to the pool
            future_to_id = {
                executor.submit(train_lstm_for_single_series, *task_args): task_args[0] # task_args[0] is id_value
                for task_args in cpu_targeted_tasks
            }
            # Process completed futures
            for future in tqdm(as_completed(future_to_id), total=len(future_to_id), desc="Training Per-Series LSTMs (Parallel)"):
                id_value_of_task = future_to_id[future]
                try:
                    series_result = future.result()
                    overall_series_results[series_result["id_value"]] = series_result
                except Exception as e:
                    logger.error(f"Parallel LSTM training task for series {id_value_of_task} raised an unexpected exception: {e}", exc_info=True)
                    overall_series_results[id_value_of_task] = {
                        "id_value": id_value_of_task, "status": "error_future_exception", "error_message": str(e)
                    }
    
    total_script_run_time = time.time() - training_start_overall_time
    logger.info(f"Overall per-series LSTM training process completed in {total_script_run_time:.2f} seconds.")

    # save training summary
    summary_output_dir = config.RESULTS_BASE_DIR / "lstm_training_summary"
    summary_output_dir.mkdir(parents=True, exist_ok=True)
    current_time_str = datetime.now().strftime('%Y%m%d_%H%M%S')
    summary_file_path = summary_output_dir / f"lstm_per_series_training_run_summary_{current_time_str}.json"

    # Aggregate statistics from results
    num_successful_series = sum(1 for res in overall_series_results.values() if res.get("status") == "success")
    num_error_series = sum(1 for res in overall_series_results.values() if res.get("status", "").startswith("error"))
    num_skipped_series = sum(1 for res in overall_series_results.values() if res.get("status", "") == "skipped_exists")

    run_summary_content = {
        "run_timestamp": datetime.now().isoformat(),
        "models_trained_type": "per_series_lstm (item-store)",
        "total_series_considered": len(all_unique_series_ids),
        "series_limited_to": args.limit_series if args.limit_series else "all",
        "force_retrain_flag": args.force,
        "device_requested_for_script": args.device,
        "parallel_jobs_requested": args.parallel_jobs,
        "effective_parallel_jobs_used": effective_parallel_jobs,
        "results_per_series_summary_stats": {
            "successful_series_trained": num_successful_series,
            "error_series_training": num_error_series,
            "skipped_series_existing": num_skipped_series,
        },
        "detailed_results_per_series": overall_series_results,
        "total_script_runtime_seconds": total_script_run_time,
        "lstm_hyperparameters_used": config.LSTM_CONFIG,
        "global_embedding_specs_used": global_embedding_specs
    }

    with open(summary_file_path, 'w') as f:
        json.dump(run_summary_content, f, indent=4)
    logger.info(f"Saved LSTM training run summary to: {summary_file_path}")

    # final summary
    logger.info("=" * 70)
    logger.info("PER-SERIES (ITEM-STORE) LSTM MODEL TRAINING COMPLETED!")
    logger.info(f"  Successfully trained models for {num_successful_series} series.")
    logger.info(f"  Errors during training for {num_error_series} series.")
    logger.info(f"  Skipped training for {num_skipped_series} series (already exist).")
    logger.info("=" * 70)

# run main
if __name__ == "__main__":
    main()