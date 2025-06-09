#////////////////////////////////////////////////////////////////////////////////#
# File:         data_loader.py                                                   #
# Author:       Douglas Nyberg                                                   #
# Email:        douglas1.nyberg@gmail.com                                        #
# Date:         2025-03-12                                                       #
# Description:  Data loading functions for M5 competition data with embedding    #
#               support.                                                         #
# Affiliation:  Physics Department, Purdue University                            #
#////////////////////////////////////////////////////////////////////////////////#
"""
Data loading functions for M5 competition data with embedding-aware feature engineering.
"""
import logging
import os
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Union, Any

import pandas as pd
import numpy as np
import torch

from sklearn.preprocessing import StandardScaler, MinMaxScaler

import src.config as config
from src.feature_engineering import preprocess_m5_features_for_embeddings, create_embedding_layer_specs

logger = logging.getLogger(__name__)

# Constants for memory-efficient processing
DEFAULT_CHUNK_SIZE = 50000  # Process data in 50k row chunks for memory efficiency
WEEKLY_STEP_SIZE = 7  # Weekly alignment for sequence creation


def validate_m5_data_for_experiment(
    splice_file_path: str,
    sequence_length: int = 28,
    forecast_horizon: int = 28,
    validation_days: int = 28,
    min_valid_items: int = 10,
    limit_items: Optional[int] = None
) -> Dict[str, Any]:
    """
    validate that m5 data has sufficient sequences for training before starting experiment.
    returns dict with validation results.
    """
    logger.info(f"validating m5 data from {splice_file_path}")
    
    # load and split data using memory-efficient loading
    data = load_m5_splice(splice_file_path, limit_items=limit_items)
    
    # calculate split point
    unique_dates = sorted(data['date'].unique())
    split_date = unique_dates[-validation_days]
    
    train_data = data[data['date'] < split_date]
    val_data = data[data['date'] >= split_date]
    
    # check each item
    valid_items = 0
    total_items = 0
    issues = []
    
    for item_id in data['id'].unique():
        total_items += 1
        
        # Check training data
        item_train = train_data[train_data['id'] == item_id]
        train_min_points = sequence_length + forecast_horizon
        
        if len(item_train) < train_min_points:
            issues.append(f"Item {item_id}: insufficient training data ({len(item_train)} < {train_min_points})")
            continue
            
        # Check validation data
        item_val = val_data[val_data['id'] == item_id]
        val_min_points = sequence_length
        
        if len(item_val) < val_min_points:
            issues.append(f"Item {item_id}: insufficient validation data ({len(item_val)} < {val_min_points})")
            continue
            
        valid_items += 1
    
    # Prepare results
    is_valid = valid_items >= min_valid_items
    results = {
        'valid': is_valid,
        'valid_items': valid_items,
        'total_items': total_items,
        'issues': issues[:10],  # Only show first 10 issues
        'validation_rate': valid_items / total_items if total_items > 0 else 0.0
    }
    
    logger.info(f"Data validation: {valid_items}/{total_items} items valid ({results['validation_rate']:.1%})")
    
    if not is_valid:
        error_msg = f"Data validation failed: only {valid_items}/{total_items} items have sufficient data (need >= {min_valid_items})"
        if issues:
            error_msg += f"\nFirst few issues: {issues[:3]}"
        raise ValueError(error_msg)
    
    return results


def create_sequences_with_embeddings(
    categorical_data: Dict[str, np.ndarray],
    numerical_data: np.ndarray,
    targets: np.ndarray,
    sequence_length: int,
    forecast_horizon: int = 28,
    step: int = 1,
    is_training: bool = True,
    max_sequences_per_item: Optional[int] = None
) -> Tuple[Dict[str, np.ndarray], np.ndarray, np.ndarray]:
    """
    create sequences for lstm input with categorical and numerical features.
    returns tuple of (categorical_seqs, numerical_seqs, target_seqs)
    """
    n_samples = len(targets)
    
    # check if we have enough data - validation only needs seq_length since we dont predict
    if is_training:
        min_required = sequence_length + forecast_horizon
    else:
        min_required = sequence_length
    
    if n_samples < min_required:
        raise ValueError(
            f"Data length {n_samples} is less than required {min_required} "
            f"(sequence_length={sequence_length}, forecast_horizon={forecast_horizon}, is_training={is_training})"
        )
    
    cat_seqs = {feature_name: [] for feature_name in categorical_data.keys()}
    num_seqs = []
    target_seqs = []
    
    if is_training:
        # training: create sequences with weekly step size
        step_size = WEEKLY_STEP_SIZE  # weekly alignment regardles of seq_length
        max_start_position = n_samples - sequence_length - forecast_horizon + 1
        
        for start_idx in range(0, max_start_position, step_size):
            input_end_idx = start_idx + sequence_length
            target_start_idx = input_end_idx
            target_end_idx = target_start_idx + forecast_horizon
            
            # get categorical sequences
            for feature_name, feature_data in categorical_data.items():
                input_seq = feature_data[start_idx:input_end_idx]
                cat_seqs[feature_name].append(input_seq)
            
            # Get numerical sequnece
            numerical_seq = numerical_data[start_idx:input_end_idx]
            num_seqs.append(numerical_seq)
            
            # Get target sequence  
            target_seq = targets[target_start_idx:target_end_idx]
            target_seqs.append(target_seq)
            
    else:
        # Validation: single sequence from end of data
        for feature_name, feature_data in categorical_data.items():
            seq = feature_data[-sequence_length:]
            cat_seqs[feature_name].append(seq)
        
        num_seq = numerical_data[-sequence_length:]
        num_seqs.append(num_seq)
        
        target_seq = targets[-sequence_length:]
        target_seqs.append(target_seq)
    
    # Convert to numpy arrays
    for feature_name in cat_seqs:
        cat_seqs[feature_name] = np.array(cat_seqs[feature_name])
    
    num_seqs = np.array(num_seqs)
    target_seqs = np.array(target_seqs)
    
    return cat_seqs, num_seqs, target_seqs


def scale_numerical_data(
    data: np.ndarray,
    scaler: Optional[Union[StandardScaler, MinMaxScaler]] = None,
    fit_scaler: bool = True
) -> Tuple[np.ndarray, Union[StandardScaler, MinMaxScaler]]:
    """
    Scale numerical data for LSTM input.
    
    Args:
        data: Numerical data to scale
        scaler: Pre-fitted scaler (if None, a new MinMaxScaler will be created)
        fit_scaler: Whether to fit the scaler on this data
        
    Returns:
        scaled_data: Scaled numerical data
        scaler: Fitted scaler for future use
    """
    if scaler is None:
        scaler = MinMaxScaler()
    
    if len(data.shape) == 3:
        # handle 3d data (sequences)
        original_shape = data.shape
        # reshape from (n_sequences, sequence_length, n_features) to (n_samples, n_features)
        # the -1 means "infer this dimension" - it becomes n_sequences * sequence_length
        reshaped_data = data.reshape(-1, data.shape[-1])
        
        if fit_scaler:
            scaler.fit(reshaped_data)
        
        scaled_data = scaler.transform(reshaped_data)
        # restore original 3d shape after scaling
        scaled_data = scaled_data.reshape(original_shape)
    else:
        # handle 2d data
        if fit_scaler:
            scaler.fit(data)
        scaled_data = scaler.transform(data)
    
    # handle nan values
    if np.isnan(scaled_data).any():
        logger.warning("nan values detected after scaling. replacing with zeros")
        scaled_data = np.nan_to_num(scaled_data)
    
    return scaled_data, scaler


def load_m5_splice(splice_file_path: str, limit_items: Optional[int] = None) -> pd.DataFrame:
    """
    load m5 data splice with features, optionally limiting items for memory efficency.
    returns dataframe with m5 data and features.
    """
    if not os.path.exists(splice_file_path):
        raise FileNotFoundError(f"Splice file not found: {splice_file_path}")
    
    try:
        if limit_items is not None:
            logger.info(f"Memory-efficient loading: limiting to {limit_items} items during CSV read")
            
            # First, read just the ID column to get unique items
            id_sample = pd.read_csv(splice_file_path, usecols=['id'], dtype={'id': 'str'})
            unique_ids = id_sample['id'].unique()[:limit_items]
            target_ids = set(unique_ids)
            logger.info(f"Target items: {len(target_ids)} of {len(id_sample['id'].unique())} total")
            
            # Now read the full data but filter during read using chunking
            chunks = []
            chunk_size = DEFAULT_CHUNK_SIZE  # Process in chunks for memory efficiency
            
            for chunk in pd.read_csv(splice_file_path, chunksize=chunk_size, dtype={'id': 'str'}):
                # filter chunk to only include target items
                filtered_chunk = chunk[chunk['id'].isin(target_ids)]
                if len(filtered_chunk) > 0:
                    chunks.append(filtered_chunk)
            
            if not chunks:
                raise ValueError(f"no data found for the first {limit_items} items")
            
            # combine filtered chunks
            data = pd.concat(chunks, ignore_index=True)
            logger.info(f"memory-efficient loading complete. filtered shape: {data.shape}")
            
        else:
            # load normally if no limit
            data = pd.read_csv(splice_file_path, dtype={'id': 'str'})
        
        # convert date column to datetime if it exsits
        if 'date' in data.columns:
            data['date'] = pd.to_datetime(data['date'])
            logger.info(f"converted 'date' column to datetime")
        
        logger.info(f"loaded m5 splice from {splice_file_path}")
        logger.info(f"final data shape: {data.shape}")
        logger.info(f"columns: {list(data.columns)}")
        
        return data
        
    except pd.errors.EmptyDataError:
        logger.error(f"file is empty: {splice_file_path}")
        raise ValueError(f"Empty file: {splice_file_path}")
    except Exception as e:
        logger.error(f"error loading data from {splice_file_path}: {str(e)}")
        raise


def load_m5_splice_metadata(splice_file_path: str) -> Optional[Dict]:
    """
    load metadata for m5 splice if available.
    returns dictionary with metadata or none if not found.
    """
    # generate metadata file path
    metadata_path = splice_file_path.replace('.csv', '_metadata.json')
    
    if os.path.exists(metadata_path):
        metadata = pd.read_json(metadata_path, typ='series').to_dict()
        logger.info(f"Loaded metadata from {metadata_path}")
        return metadata
    else:
        logger.warning(f"No metadata file found at {metadata_path}")
        return None


def prepare_m5_data_for_lstm(
    splice_file_path: str,
    sequence_length: int = 28,
    forecast_horizon: int = 28,
    validation_days: int = None,
    target_col: str = 'sales',
    step: int = 1,
    limit_items: Optional[int] = None
) -> Tuple[Dict[str, torch.Tensor], torch.Tensor, torch.Tensor, 
           Dict[str, torch.Tensor], torch.Tensor, torch.Tensor,
           Dict[str, Dict], Dict[str, List[str]], object]:
    """
    prepare m5 data for lstm training with temporal spliting.
    returns tuple of training and validation data.
    """
    logger.info(f"preparing m5 data from {splice_file_path}")
    
    # Load data with memory-efficient filtering  
    raw_data = load_m5_splice(splice_file_path, limit_items=limit_items)
    
    # Apply embedding-aware preprocessing
    processed_data_full, feature_info, embedding_info = preprocess_m5_features_for_embeddings(raw_data)
    
    # Split data temporally to avoid data leakage
    processed_train_df, processed_val_df = split_data_temporal(
        processed_data_full,
        validation_days=validation_days
    )
    
    # Helper function to create sequences from a dataframe
    def create_sequences_from_df(df: pd.DataFrame, is_training: bool = True):
        """
        Create sequences from a DataFrame for LSTM training or validation.
        
        Args:
            df: DataFrame containing time series data with columns including 'id', 'date', 
                target_col, and all features specified in feature_info
            is_training: If True, creates training sequences with forecast horizon targets.
                        If False, creates validation sequences with sequence_length targets.
        
        Returns:
            Tuple containing:
            - categorical_features: Dict mapping feature names to arrays of sequences
            - numerical_features: Array of numerical feature sequences
            - targets: Array of target sequences
        """
        categorical_sequences = {feature: [] for feature in feature_info['categorical']}
        numerical_sequences = []
        target_sequences = []
        
        # Group by item ID
        grouped = df.groupby('id')
        
        for item_id, item_data in grouped:
            # Sort by date
            item_data = item_data.sort_values('date').reset_index(drop=True)
            
            # Check if we have enough data - validation only needs sequence_length since we don't predict
            minimum_data_points = sequence_length + forecast_horizon if is_training else sequence_length
            if len(item_data) < minimum_data_points:
                logger.warning(f"Item {item_id} has insufficient data ({len(item_data)} < {minimum_data_points} rows), skipping")
                continue
            
            # Extract categorical data
            categorical_data = {}
            for feature in feature_info['categorical']:
                categorical_data[feature] = item_data[feature].values
            
            # Extract numerical data
            numerical_data = item_data[feature_info['numerical']].values
            
            # Explicitly impute NaNs in numerical data with 0
            # This handles lag features and rolling windows where NaN indicates insufficient prior data
            numerical_data = np.where(np.isnan(numerical_data), 0.0, numerical_data)
            
            # Extract targets
            targets = item_data[target_col].values
            
            # Create sequences
            try:
                cat_sequences, num_sequences, target_seqs = create_sequences_with_embeddings(
                    categorical_data=categorical_data,
                    numerical_data=numerical_data,
                    targets=targets,
                    sequence_length=sequence_length,
                    forecast_horizon=forecast_horizon,
                    step=step,
                    is_training=is_training
                )
                
                # Append to all sequences
                for feature in feature_info['categorical']:
                    categorical_sequences[feature].append(cat_sequences[feature])
                
                numerical_sequences.append(num_sequences)
                target_sequences.append(target_seqs)
                
            except ValueError as e:
                logger.warning(f"Error creating sequences for item {item_id}: {e}")
                continue
        
        if not numerical_sequences:
            raise ValueError("No valid sequences could be created from the data")
        
        # Concatenate all sequences from different items into single arrays
        categorical_features = {}
        for feature in feature_info['categorical']:
            # Concatenate along axis=0 (first dimension) to stack sequences vertically
            categorical_features[feature] = np.concatenate(categorical_sequences[feature], axis=0)
        
        numerical_features = np.concatenate(numerical_sequences, axis=0)
        targets = np.concatenate(target_sequences, axis=0)
        
        logger.info(f"Created {len(targets)} sequences from {'training' if is_training else 'validation'} data")
        
        return categorical_features, numerical_features, targets
    
    # Create training sequences
    train_categorical_features, train_numerical_features, train_targets = create_sequences_from_df(
        processed_train_df, is_training=True
    )
    
    # Create validation sequences
    val_categorical_features, val_numerical_features, val_targets = create_sequences_from_df(
        processed_val_df, is_training=False
    )
    
    # Fit scaler only on training data
    train_numerical_scaled, scaler = scale_numerical_data(
        train_numerical_features, 
        fit_scaler=True
    )
    
    # Apply scaler to validation data
    val_numerical_scaled, _ = scale_numerical_data(
        val_numerical_features,
        scaler=scaler,
        fit_scaler=False
    )
    
    # Convert to tensors
    train_categorical = {}
    for feature in feature_info['categorical']:
        train_categorical[feature] = torch.tensor(train_categorical_features[feature], dtype=torch.long)
    
    train_numerical = torch.tensor(train_numerical_scaled, dtype=torch.float32)
    train_targets_tensor = torch.tensor(train_targets, dtype=torch.float32)
    
    val_categorical = {}
    for feature in feature_info['categorical']:
        val_categorical[feature] = torch.tensor(val_categorical_features[feature], dtype=torch.long)
    
    val_numerical = torch.tensor(val_numerical_scaled, dtype=torch.float32)
    val_targets_tensor = torch.tensor(val_targets, dtype=torch.float32)
    
    logger.info(f"Final data shapes:")
    logger.info(f"  Train: {len(train_targets_tensor)} sequences, numerical {train_numerical.shape}")
    logger.info(f"  Validation: {len(val_targets_tensor)} sequences, numerical {val_numerical.shape}")
    
    return (train_categorical, train_numerical, train_targets_tensor,
            val_categorical, val_numerical, val_targets_tensor,
            embedding_info, feature_info, scaler)


def prepare_m5_data_for_prediction(
    splice_file_path: str,
    embedding_info: Dict[str, Dict],
    feature_info: Dict[str, List[str]],
    scaler: object,
    sequence_length: int = 28,
    limit_items: Optional[int] = None
) -> Tuple[Dict[str, torch.Tensor], torch.Tensor, List[str]]:
    """
    Prepare M5 data for prediction (inference).
    
    Args:
        splice_file_path: Path to M5 splice file
        sequence_length: Length of input sequences
        embedding_info: Dictionary with embedding specifications
        feature_info: Dictionary with feature lists
        scaler: Fitted scaler from training
        limit_items: Optional limit on number of items
        
    Returns:
        Tuple containing:
        - categorical_features: Dict of categorical features
        - numerical_features: Numerical features
        - item_ids: List of item IDs corresponding to sequences
    """
    logger.info(f"Preparing M5 data for prediction from {splice_file_path}")
    
    # Load data with memory-efficient filtering
    raw_data = load_m5_splice(splice_file_path, limit_items=limit_items)
    
    # Apply embedding-aware preprocessing
    processed_data, _, _ = preprocess_m5_features_for_embeddings(raw_data)
    
    # Prepare prediction data
    categorical_sequences = {feature: [] for feature in feature_info['categorical']}
    numerical_sequences = []
    item_ids = []
    
    # Group by item ID
    grouped = processed_data.groupby('id')
    
    for item_id, item_data in grouped:
        # Sort by date and take last sequence_length points
        item_data = item_data.sort_values('date').tail(sequence_length).reset_index(drop=True)
        
        if len(item_data) < sequence_length:
            logger.warning(f"Item {item_id} has insufficient data for prediction ({len(item_data)} < {sequence_length})")
            continue
        
        # Extract categorical data
        for feature in feature_info['categorical']:
            categorical_sequences[feature].append(item_data[feature].values)
        
        # Extract numerical data
        numerical_data = item_data[feature_info['numerical']].values
        numerical_sequences.append(numerical_data)
        
        item_ids.append(item_id)
    
    if not numerical_sequences:
        raise ValueError("No valid prediction sequences could be created")
    
    # Convert to arrays
    categorical_features = {}
    for feature in feature_info['categorical']:
        categorical_features[feature] = np.array(categorical_sequences[feature])
    
    numerical_features = np.array(numerical_sequences)
    
    # Scale numerical features using existing scaler
    numerical_features_scaled, _ = scale_numerical_data(
        numerical_features,
        scaler=scaler,
        fit_scaler=False
    )
    
    # Convert to tensors
    categorical_tensors = {}
    for feature in feature_info['categorical']:
        categorical_tensors[feature] = torch.tensor(categorical_features[feature], dtype=torch.long)
    
    numerical_tensor = torch.tensor(numerical_features_scaled, dtype=torch.float32)
    
    logger.info(f"Prepared prediction data for {len(item_ids)} items")
    
    return categorical_tensors, numerical_tensor, item_ids


def split_data_temporal(
    df: pd.DataFrame,
    time_col: str = 'date',
    validation_days: int = None,
    id_col: str = 'id'
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Split data into training and validation sets temporally.
    
    Args:
        df: DataFrame with time series data
        time_col: Column name for time/date
        validation_days: Number of days to use for validation
        id_col: Column name for the series identifier
        
    Returns:
        train_df, val_df: Training and validation DataFrames
    """
    if validation_days is None:
        validation_days = config.VALIDATION_DAYS
    
    # Sort by ID and date (not in-place to avoid modifying input)
    df_sorted = df.sort_values([id_col, time_col])
    
    # Get unique dates
    dates = sorted(df_sorted[time_col].unique())
    
    # Split into train and validation
    split_date = dates[-validation_days]
    
    train_df = df_sorted[df_sorted[time_col] < split_date].copy()
    val_df = df_sorted[df_sorted[time_col] >= split_date].copy()
    
    logger.info(f"Split data: {len(train_df)} training rows, {len(val_df)} validation rows")
    logger.info(f"Split date: {split_date}")
    
    return train_df, val_df


def validate_m5_data(df: pd.DataFrame) -> bool:
    """
    Validate M5 data format and content.
    
    Args:
        df: DataFrame to validate
        
    Returns:
        True if data is valid
        
    Raises:
        ValueError: If data format is invalid
    """
    required_columns = ['id', 'date', 'sales']
    
    # Check required columns
    missing_cols = []
    for col in required_columns:
        if col not in df.columns:
            missing_cols.append(col)
    
    if missing_cols:
        raise ValueError(f"Missing required columns: {missing_cols}")
    
    # Check for null values in critical columns
    for col in required_columns:
        if df[col].isnull().any():
            raise ValueError(f"Null values found in required column: {col}")
    
    # Check date format
    try:
        pd.to_datetime(df['date'])
    except Exception as e:
        raise ValueError(f"Invalid date format: {e}")
    
    # Check sales values
    if (df['sales'] < 0).any():
        logger.warning("Negative sales values found in data")
    
    # Check for duplicate entries
    duplicates = df.duplicated(subset=['id', 'date']).sum()
    if duplicates > 0:
        raise ValueError(f"Found {duplicates} duplicate id-date combinations")
    
    logger.info("Data validation passed")
    return True


def get_available_splices(splice_dir: str = "data/m5_splices") -> List[str]:
    """
    Get list of available M5 splice files.
    
    Args:
        splice_dir: Directory containing splice files
        
    Returns:
        List of available splice file paths
    """
    splice_path = Path(splice_dir)
    
    if not splice_path.exists():
        logger.warning(f"Splice directory not found: {splice_dir}")
        return []
    
    # Find all CSV files in splice directory
    csv_files = list(splice_path.glob("*.csv"))
    
    # Filter out metadata files
    splice_files = []
    for f in csv_files:
        if not f.name.endswith('_metadata.csv'):
            splice_files.append(str(f))
    
    logger.info(f"Found {len(splice_files)} splice files in {splice_dir}")
    
    return sorted(splice_files)


def load_splice_by_name(
    splice_name: str,
    splice_dir: str = "data/m5_splices"
) -> pd.DataFrame:
    """
    Load a specific M5 splice by name.
    
    Args:
        splice_name: Name of the splice (with or without .csv extension)
        splice_dir: Directory containing splice files
        
    Returns:
        DataFrame with splice data
    """
    # Ensure .csv extension
    if not splice_name.endswith('.csv'):
        splice_name += '.csv'
    
    splice_path = Path(splice_dir) / splice_name
    
    if not splice_path.exists():
        available_splices = get_available_splices(splice_dir)
        available_names = []
        for s in available_splices:
            available_names.append(Path(s).name)
        raise FileNotFoundError(
            f"Splice '{splice_name}' not found in {splice_dir}.\n"
            f"Available splices: {available_names}"
        )
    
    return load_m5_splice(str(splice_path))


def get_embedding_specs_from_data(embedding_info: Dict[str, Dict]) -> Dict[str, Tuple[int, int]]:
    """
    Extract embedding specifications for PyTorch model creation.
    
    Args:
        embedding_info: Embedding information from preprocessing
        
    Returns:
        Dictionary mapping feature names to (vocab_size, embedding_dim) tuples
    """
    return create_embedding_layer_specs(embedding_info)




