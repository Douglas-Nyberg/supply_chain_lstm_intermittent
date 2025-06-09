#////////////////////////////////////////////////////////////////////////////////#
# File:         utils.py                                                         #
# Author:       Douglas Nyberg                                                   #
# Email:        douglas1.nyberg@gmail.com                                        #
# Date:         2025-03-18                                                       #
#////////////////////////////////////////////////////////////////////////////////#





"""
Utility functions for the supply chain forecasting project.
"""
import json
import os
import re
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Union, Any

import pandas as pd
import numpy as np
import torch

from src import config


def create_directory(directory: Union[str, Path]) -> None:
    """create directory if it doesnt exsit"""
    Path(directory).mkdir(parents=True, exist_ok=True)


def save_json(data: Dict, filepath: Union[str, Path]) -> None:
    """Save dict to JSON."""
    with open(filepath, 'w') as f:
        json.dump(data, f, indent=4)


def load_json(filepath: Union[str, Path]) -> Dict:
    """load dict from json file"""
    with open(filepath, 'r') as f:
        data = json.load(f)
    return data


def setup_torch_device() -> torch.device:
    """setup torch device (cpu or cuda)"""
    if torch.cuda.is_available():
        device = torch.device("cuda")
        print(f"Using GPU: {torch.cuda.get_device_name(0)}")
    else:
        device = torch.device("cpu")
        print("Using CPU")  # no gpu availabe
    return device


def set_random_seed(seed: int = None) -> None:
    """set random seed for reproducability"""
    if seed is None:
        seed = config.RANDOM_SEED
    
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def format_time(seconds: float) -> str:
    """format seconds into readable string"""
    hours, remainder = divmod(seconds, 3600)
    minutes, seconds = divmod(remainder, 60)
    return f"{int(hours)}h {int(minutes)}m {int(seconds)}s"



def sanitize_for_path(name: str) -> str:
    """
    Replaces characters problematic for file or directory names with underscores.
    Also removes leading/trailing whitespace and reduces multiple underscores.
    
    Args:
        name: String to sanitize for use in file/directory paths
        
    Returns:
        Sanitized string safe for use in file paths
    """
    if not isinstance(name, str):
        name = str(name) 

    # Remove leading/trailing whitespace
    sanitized_name = name.strip()

    # Replace common problematic characters with a single underscore
    sanitized_name = re.sub(r'[\\/:*?"<>|\s]', '_', sanitized_name)

    # Replace multiple consecutive underscores with a single underscore
    sanitized_name = re.sub(r'_+', '_', sanitized_name)

    # Ensure it's not empty after sanitization, provide a default if it is
    if not sanitized_name:
        return "sanitized_empty_name"
        
    return sanitized_name


def extract_base_id_from_prediction_id(prediction_id_str: str, is_quantile_prediction: bool) -> str:
    """
    Extracts the base series 'id' (e.g., 'FOODS_1_001_CA_1') from M5 formatted prediction IDs.
    Point forecast ID format: 'SERIES_ID_evaluation'
    Quantile forecast ID format: 'SERIES_ID_QLEVEL_evaluation'
    
    Args:
        prediction_id_str: The full prediction ID string to parse
        is_quantile_prediction: Whether this is a quantile prediction (affects parsing logic)
        
    Returns:
        The base series ID extracted from the prediction ID
    """
    parts = prediction_id_str.split('_')
    if is_quantile_prediction:
        if len(parts) >= 3 and '.' in parts[-2] and parts[-1] == "evaluation":
            base_id = "_".join(parts[:-2])
            return base_id
        else:
            logger.debug(f"Unexpected quantile ID format for base ID extraction: '{prediction_id_str}'. Using heuristic.")
            return "_".join(parts[:-2]) if len(parts) >= 2 else prediction_id_str
    else: # Point forecast ID
        if len(parts) >= 2 and parts[-1] == "evaluation":
            base_id = "_".join(parts[:-1])
            return base_id
        else:
            logger.debug(f"Unexpected point ID format for base ID extraction: '{prediction_id_str}'. Using heuristic.")
            return "_".join(parts[:-1]) if len(parts) >= 1 else prediction_id_str


def prepare_model_inputs(
    df: pd.DataFrame,
    target_col: str = 'sales',
    date_col: str = 'date',
    id_col: str = 'id',
    feature_cols: Optional[List[str]] = None,
    sequence_length: Optional[int] = None,
    forecast_horizon: Optional[int] = None,
    include_last_observed: bool = False
) -> Dict[str, np.ndarray]:
    """
    Prepare input tensors for the LSTM model.
    
    This function handles data from multiple time series (as identified by `id_col`),
    creates sequences, and prepares the necessary mappings for ID embedding in grouped
    or global LSTM models. It implicitly includes historical values of the target variable
    in the input features, which is crucial for the model to learn seasonality and
    autoregressive patterns.
    
    Args:
        df: DataFrame with features and target
        target_col: Column name for the target variable
        date_col: Column name for time/date
        id_col: Column name for the series identifier
        feature_cols: List of feature column names to include
        sequence_length: Length of input sequences
        forecast_horizon: Length of forecast horizon
        include_last_observed: Whether to include the last observed value in the output sequence
        
    Returns:
        Dictionary with prepared data arrays:
            - 'X': Input feature sequences for the LSTM, shape (n_sequences, sequence_length, n_features)
                   Note: These sequences include historical values of the target itself as features
            - 'y': Target values to predict, shape (n_sequences, forecast_horizon)
            - 'ids': Integer-mapped series IDs, shape (n_sequences,)
            - 'id_mapping': Dictionary mapping original series IDs to integers (used for embedding)
    """
    # Set defaults from config if not provided
    if sequence_length is None:
        sequence_length = config.DEFAULT_SEQUENCE_LENGTH
    if forecast_horizon is None:
        forecast_horizon = config.M5_HORIZON
    if feature_cols is None:
        # Use all columns except ID, date, and target
        feature_cols = [col for col in df.columns if col not in [id_col, date_col, target_col]]
    
    # Ensure data is sorted by ID and date
    df = df.sort_values([id_col, date_col])
    
    # Create a mapping of series IDs to integers
    unique_ids = df[id_col].unique()
    id_to_int = {id_val: i for i, id_val in enumerate(unique_ids)}
    
    # Group by series ID
    grouped = df.groupby(id_col)
    
    all_X = []
    all_y = []
    all_ids = []
    
    for series_id, group in grouped:
        group = group.sort_values(date_col)
        features = group[feature_cols].values
        target = group[target_col].values
        
        # Create sequences
        for i in range(len(group) - sequence_length - forecast_horizon + 1):
            # Feature sequence
            X_seq = features[i:i+sequence_length]
            
            # Target sequence
            if include_last_observed:
                y_seq = np.concatenate([
                    [target[i+sequence_length-1]],  # Last observed value
                    target[i+sequence_length:i+sequence_length+forecast_horizon]  # Future values
                ])
            else:
                y_seq = target[i+sequence_length:i+sequence_length+forecast_horizon]
            
            all_X.append(X_seq)
            all_y.append(y_seq)
            all_ids.append(id_to_int[series_id])
    
    # Convert to numpy arrays
    X_array = np.array(all_X)
    y_array = np.array(all_y)
    ids_array = np.array(all_ids)
    
    return {
        'X': X_array,
        'y': y_array,
        'ids': ids_array,
        'id_mapping': id_to_int
    }


def convert_to_torch_tensors(
    data_dict: Dict[str, np.ndarray],
    device: Optional[torch.device] = None
) -> Dict[str, torch.Tensor]:
    """
    Convert numpy arrays to PyTorch tensors.
    
    Args:
        data_dict: Dictionary with numpy arrays
        device: PyTorch device to move tensors to
        
    Returns:
        Dictionary with PyTorch tensors
    """
    tensor_dict = {}
    
    for key, value in data_dict.items():
        if isinstance(value, np.ndarray):
            tensor = torch.tensor(value, dtype=torch.float32)
            if key == 'ids':
                tensor = tensor.long()  # ids should be long tensors for embedding, wide for cv
            if device is not None:
                tensor = tensor.to(device)
            tensor_dict[key] = tensor
        else:
            tensor_dict[key] = value
    
    return tensor_dict