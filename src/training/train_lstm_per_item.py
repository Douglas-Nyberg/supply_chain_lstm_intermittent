#////////////////////////////////////////////////////////////////////////////////#
# File:         train_lstm_per_item.py                                           #
# Author:       Douglas Nyberg                                                   #
# Email:        douglas1.nyberg@gmail.com                                        #
# Date:         2025-05-31                                                       #
# Description:  Training script for LSTM models per item with proper feature     #
#               separation between numerical and categorical inputs              #
# Affiliation:  Physics Department, Purdue University                            #
#////////////////////////////////////////////////////////////////////////////////#
"""
Training script for LSTM models per item using optimized hyperparameters.

This script properly handles the separation of numerical and categorical features
as required by the LSTMModel's forward method. It preprocesses data using the
embedding-aware pipeline and creates appropriate data loaders for training.
"""

import argparse
import json
import logging
import os
import pickle
import sys
import time
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error

# Add project root to path for imports
PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.append(str(PROJECT_ROOT))

from src.models.lstm import LSTMModel, QuantileLoss
from src.feature_engineering import (
    add_time_features, 
    preprocess_m5_features_for_embeddings, 
    create_embedding_layer_specs,
    add_lag_features,
    add_rolling_features,
    create_lstm_sequences_with_separation
)
from src import config

# Set up logging with more informative formatting
logging.basicConfig(
    level=logging.INFO, 
    format='%(asctime)s | %(name)s | %(levelname)s | %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)


def parse_arguments():
    """
    Parse command line arguments.
    
    Args:
        None
        
    Returns:
        argparse.Namespace object containing all parsed command line arguments
    """
    parser = argparse.ArgumentParser(
        description="Train LSTM models per item with optimized hyperparameters",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    # Data arguments
    parser.add_argument(
        "--splice-file-path",
        type=str,
        required=True,
        help="Path to the data splice file (CSV with features)"
    )
    
    parser.add_argument(
        "--output-dir",
        type=str,
        required=True,
        help="Output directory for trained models"
    )
    
    # Model hyperparameters
    parser.add_argument("--hidden-dim", type=int, default=64, help="Hidden dimension")
    parser.add_argument("--num-layers", type=int, default=2, help="Number of LSTM layers")
    parser.add_argument("--dropout", type=float, default=0.2, help="Dropout rate")
    parser.add_argument("--sequence-length", type=int, default=28, help="Input sequence length")
    parser.add_argument("--batch-size", type=int, default=32, help="Training batch size")
    parser.add_argument("--learning-rate", type=float, default=0.001, help="Learning rate")
    parser.add_argument("--weight-decay", type=float, default=1e-5, help="Weight decay")
    parser.add_argument("--epochs", type=int, default=50, help="Training epochs")
    parser.add_argument("--patience", type=int, default=10, help="Early stopping patience")
    
    # Model options
    parser.add_argument("--bidirectional", action="store_true", help="Use bidirectional LSTM")
    parser.add_argument("--use-attention", action="store_true", help="Use attention mechanism")
    parser.add_argument("--quantile-forecast", action="store_true", help="Train for quantile forecasting")
    
    # Feature options
    parser.add_argument("--use-price-features", action="store_true", help="Use price features")
    parser.add_argument("--use-lag-features", action="store_true", help="Use lag features")
    parser.add_argument("--use-rolling-features", action="store_true", help="Use rolling features")
    parser.add_argument("--use-calendar-features", action="store_true", help="Use calendar features")
    parser.add_argument("--use-event-features", action="store_true", help="Use event features")
    
    # Other options
    parser.add_argument("--limit-items", type=int, default=None, help="Limit number of items")
    parser.add_argument("--random-seed", type=int, default=42, help="Random seed")
    parser.add_argument("--verbose", action="store_true", help="Verbose output")
    
    return parser.parse_args()


class M5TimeSeriesDataset(Dataset):
    """
    Custom PyTorch Dataset for M5 time series with separate numerical and categorical features.
    
    This dataset properly handles the separation of features as required by the LSTM model's
    forward method, which expects categorical features as a dictionary and numerical features
    as a tensor.
    """
    
    def __init__(
        self, 
        sequences_numerical: np.ndarray,
        sequences_categorical: Dict[str, np.ndarray],
        targets: np.ndarray
    ):
        """
        Initialize the dataset.
        
        Args:
            sequences_numerical: Array of shape (n_samples, sequence_length, n_numerical_features)
            sequences_categorical: Dict mapping feature names to arrays of shape (n_samples, sequence_length)
            targets: Array of shape (n_samples, forecast_horizon)
        """
        self.sequences_numerical = torch.FloatTensor(sequences_numerical)
        self.sequences_categorical = {
            feature_name: torch.LongTensor(feature_data)
            for feature_name, feature_data in sequences_categorical.items()
        }
        self.targets = torch.FloatTensor(targets)
        
        # Validate shapes
        n_samples = len(self.sequences_numerical)
        assert len(self.targets) == n_samples, "Number of targets must match number of sequences"
        for feature_name, feature_data in self.sequences_categorical.items():
            assert len(feature_data) == n_samples, f"Categorical feature {feature_name} has wrong number of samples"
    
    def __len__(self):
        """Return the number of samples in the dataset."""
        return len(self.sequences_numerical)
    
    def __getitem__(self, idx):
        """
        Get a single sample from the dataset.
        
        Returns:
            Tuple of (categorical_features_dict, numerical_features, target)
        """
        # Extract categorical features for this sample
        categorical_sample = {
            feature_name: feature_data[idx]
            for feature_name, feature_data in self.sequences_categorical.items()
        }
        
        # Extract numerical features and target
        numerical_sample = self.sequences_numerical[idx]
        target_sample = self.targets[idx]
        
        return categorical_sample, numerical_sample, target_sample


def load_and_validate_data(splice_file_path: str, limit_items: Optional[int] = None) -> pd.DataFrame:
    """
    Load data from splice file and perform initial validation.
    
    Args:
        splice_file_path: Path to the CSV file containing feature-engineered data
        limit_items: Optional limit on number of unique items to process
        
    Returns:
        Loaded and validated DataFrame
        
    Raises:
        FileNotFoundError: If splice file doesn't exist
        ValueError: If data doesn't meet minimum requirements
    """
    # validate file exists
    splice_path = Path(splice_file_path)
    if not splice_path.exists():
        logger.error(f"Splice file not found: {splice_path}")
        raise FileNotFoundError(f"Splice file not found: {splice_path}")
    
    # load data
    logger.info("=" * 60)
    logger.info("LOADING DATA")
    logger.info("=" * 60)
    logger.info(f"Reading from: {splice_path}")
    
    data = pd.read_csv(splice_file_path)
    
    # report initial stats
    initial_shape = data.shape
    initial_items = data['item_id'].nunique()
    logger.info(f"Initial data shape: {initial_shape}")
    logger.info(f"Unique items found: {initial_items}")
    
    # apply item limit if requested
    if limit_items:
        unique_items = data['item_id'].unique()[:limit_items]
        data = data[data['item_id'].isin(unique_items)]
        logger.info(f"Limited to {limit_items} items (was {initial_items})")
    
    # validate min data requirements
    min_required_rows = 100  # Minimum rows to create meaningful sequences
    if len(data) < min_required_rows:
        raise ValueError(f"Insufficient data: {len(data)} rows, need at least {min_required_rows}")
    
    # check for required columns
    required_columns = ['item_id', 'date', 'sales']
    missing_columns = set(required_columns) - set(data.columns)
    if missing_columns:
        raise ValueError(f"Missing required columns: {missing_columns}")
    
    # report final stats
    logger.info(f"Final data shape: {data.shape}")
    logger.info(f"Date range: {data['date'].min()} to {data['date'].max()}")
    logger.info(f"Sales range: [{data['sales'].min():.0f}, {data['sales'].max():.0f}]")
    
    return data


def prepare_features(data: pd.DataFrame, args: argparse.Namespace) -> pd.DataFrame:
    """
    Prepare features based on command line arguments.
    
    This function applies various feature engineering steps based on the flags
    provided. Each step is logged for transparency.
    
    Args:
        data: Raw DataFrame with M5 data
        args: Command line arguments specifying which features to add
        
    Returns:
        DataFrame with added features
    """
    logger.info("=" * 60)
    logger.info("PREPARING FEATURES")
    logger.info("=" * 60)
    
    # Start with a copy to avoid modifying original data
    features_df = data.copy()
    initial_columns = len(features_df.columns)
    
    # add calendar features
    if args.use_calendar_features:
        logger.info("Adding calendar features...")
        features_df = add_time_features(features_df)
        calendar_columns_added = len(features_df.columns) - initial_columns
        logger.info(f"  Added {calendar_columns_added} calendar features")
        initial_columns = len(features_df.columns)
    
    # add lag features
    if args.use_lag_features:
        logger.info("Adding lag features...")
        # Use default lags from config
        lag_days = [1, 2, 3, 7, 14, 28]
        features_df = add_lag_features(features_df, lag_days=lag_days)
        lag_columns_added = len(features_df.columns) - initial_columns
        logger.info(f"  Added {lag_columns_added} lag features")
        initial_columns = len(features_df.columns)
    
    # add rolling features
    if args.use_rolling_features:
        logger.info("Adding rolling features...")
        windows = [7, 14, 28]
        features_df = add_rolling_features(features_df, windows=windows)
        rolling_columns_added = len(features_df.columns) - initial_columns
        logger.info(f"  Added {rolling_columns_added} rolling features")
        initial_columns = len(features_df.columns)
    
    # report final feature count
    logger.info(f"Total features after preparation: {len(features_df.columns)}")
    
    return features_df


def create_sequences_with_separation(
    data: pd.DataFrame, 
    sequence_length: int,
    forecast_horizon: int = 28,
    target_col: str = 'sales'
) -> Tuple[np.ndarray, Dict[str, np.ndarray], np.ndarray, Dict[str, Tuple[int, int]], Dict[str, Any]]:
    """
    Create sequences for LSTM training with proper separation of numerical and categorical features.
    
    This is a wrapper around the shared utility function that adds preprocessing
    and embedding spec creation specific to this training script.
    
    Args:
        data: DataFrame with time series data
        sequence_length: Number of time steps in input sequences
        forecast_horizon: Number of steps to forecast (default: 28 for M5)
        target_col: Name of the target column
        
    Returns:
        Tuple containing:
        - sequences_numerical: Array of shape (n_samples, seq_len, n_numerical_features)
        - sequences_categorical: Dict mapping feature names to arrays of shape (n_samples, seq_len)
        - targets: Array of shape (n_samples, forecast_horizon)
        - embedding_specs: Dict with embedding layer specifications
        - preprocessing_info: Dict with preprocessing metadata
    """
    logger.info("=" * 60)
    logger.info("CREATING SEQUENCES WITH FEATURE SEPARATION")
    logger.info("=" * 60)
    
    # preprocess data with embedding support
    logger.info("Preprocessing data for embeddings...")
    # For cross-item training, we need ID columns as features
    processed_data, feature_info, embedding_info = preprocess_m5_features_for_embeddings(data, exclude_id_columns=False)
    embedding_specs = create_embedding_layer_specs(embedding_info)
    
    # use shared utility to create sequences
    sequences_numerical, sequences_categorical, targets, sequence_info = create_lstm_sequences_with_separation(
        data=processed_data,
        sequence_length=sequence_length,
        forecast_horizon=forecast_horizon,
        feature_info=feature_info,
        target_col=target_col,
        item_col='item_id',
        date_col='date'
    )
    
    # enhance preprocessing info with embedding details
    preprocessing_info = {
        **sequence_info,  # Include all info from shared utility
        'embedding_info': embedding_info,
        'embedding_specs': embedding_specs
    }
    
    # report final stats
    logger.info("=" * 60)
    logger.info("SEQUENCE CREATION COMPLETE")
    logger.info("=" * 60)
    logger.info(f"Total sequences created: {sequence_info['total_sequences']}")
    logger.info(f"Items processed: {sequence_info['items_processed']}")
    if sequence_info['skipped_items']:
        logger.info(f"Items skipped (insufficient data): {len(sequence_info['skipped_items'])}")
    logger.info(f"Numerical features shape: {sequences_numerical.shape}")
    logger.info(f"Target shape: {targets.shape}")
    
    return sequences_numerical, sequences_categorical, targets, embedding_specs, preprocessing_info


def collate_fn_with_separation(batch):
    """
    Custom collate function for DataLoader that handles separated features.
    
    The batch is a list of tuples: [(cat_dict1, num1, target1), (cat_dict2, num2, target2), ...]
    We need to batch them properly for the model.
    
    Args:
        batch: List of samples from M5TimeSeriesDataset
        
    Returns:
        Tuple of (batched_categorical_dict, batched_numerical, batched_targets)
    """
    # Separate the components
    categorical_dicts = [sample[0] for sample in batch]
    numerical_tensors = [sample[1] for sample in batch]
    target_tensors = [sample[2] for sample in batch]
    
    # Stack numerical features and targets
    batched_numerical = torch.stack(numerical_tensors, dim=0)
    batched_targets = torch.stack(target_tensors, dim=0)
    
    # Batch categorical features by stacking each feature separately
    batched_categorical = {}
    if categorical_dicts:  # Check if we have categorical features
        feature_names = list(categorical_dicts[0].keys())
        for feature_name in feature_names:
            feature_tensors = [cat_dict[feature_name] for cat_dict in categorical_dicts]
            batched_categorical[feature_name] = torch.stack(feature_tensors, dim=0)
    
    return batched_categorical, batched_numerical, batched_targets


def train_model_with_separation(
    model: nn.Module, 
    train_loader: DataLoader, 
    val_loader: DataLoader,
    optimizer: torch.optim.Optimizer, 
    criterion: nn.Module,
    epochs: int, 
    patience: int, 
    device: torch.device
) -> Dict[str, Any]:
    """
    Train the LSTM model with proper handling of separated features.
    
    This training function correctly passes categorical and numerical features
    separately to the model's forward method, as expected by the LSTMModel class.
    
    Args:
        model: LSTM model instance
        train_loader: DataLoader for training data
        val_loader: DataLoader for validation data
        optimizer: PyTorch optimizer
        criterion: Loss function
        epochs: Maximum number of training epochs
        patience: Early stopping patience
        device: CUDA or CPU device
        
    Returns:
        Dictionary containing training history and best model state
    """
    # --- Initialize tracking variables ---
    train_losses = []
    val_losses = []
    best_val_loss = float('inf')
    best_model_state = None
    patience_counter = 0
    
    logger.info("=" * 60)
    logger.info("STARTING TRAINING")
    logger.info("=" * 60)
    logger.info(f"Training for up to {epochs} epochs with patience {patience}")
    logger.info(f"Device: {device}")
    
    # --- Training loop ---
    for epoch in range(epochs):
        epoch_start_time = time.time()
        
        # Training phase
        model.train()
        train_loss = 0.0
        train_batches = 0
        
        for batch_idx, (batch_categorical, batch_numerical, batch_targets) in enumerate(train_loader):
            # Move data to device
            batch_numerical = batch_numerical.to(device)
            batch_targets = batch_targets.to(device)
            
            # Move categorical features to device
            batch_categorical_device = {
                feat_name: feat_tensor.to(device)
                for feat_name, feat_tensor in batch_categorical.items()
            }
            
            # Zero gradients
            optimizer.zero_grad()
            
            # Forward pass with separated features
            outputs = model(batch_categorical_device, batch_numerical)
            
            # Compute loss
            loss = criterion(outputs, batch_targets)
            
            # Add regularization if model supports it
            if hasattr(model, 'get_regularization_loss'):
                reg_loss = model.get_regularization_loss()
                loss = loss + reg_loss
            
            # Backward pass
            loss.backward()
            
            # Gradient clipping to prevent exploding gradients
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            # Update weights
            optimizer.step()
            
            # Track loss
            train_loss += loss.item()
            train_batches += 1
            
            # Log progress every 50 batches
            if batch_idx % 50 == 0 and batch_idx > 0:
                logger.debug(f"  Epoch {epoch+1}, Batch {batch_idx}/{len(train_loader)}, "
                           f"Current Loss: {loss.item():.6f}")
        
        # Calculate average training loss
        avg_train_loss = train_loss / train_batches
        train_losses.append(avg_train_loss)
        
        # Validation phase
        model.eval()
        val_loss = 0.0
        val_batches = 0
        
        with torch.no_grad():
            for batch_categorical, batch_numerical, batch_targets in val_loader:
                # Move data to device
                batch_numerical = batch_numerical.to(device)
                batch_targets = batch_targets.to(device)
                
                # Move categorical features to device
                batch_categorical_device = {
                    feat_name: feat_tensor.to(device)
                    for feat_name, feat_tensor in batch_categorical.items()
                }
                
                # Forward pass
                outputs = model(batch_categorical_device, batch_numerical)
                
                # Compute loss
                loss = criterion(outputs, batch_targets)
                
                # Track loss
                val_loss += loss.item()
                val_batches += 1
        
        # Calculate average validation loss
        avg_val_loss = val_loss / val_batches
        val_losses.append(avg_val_loss)
        
        # Early stopping check
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            best_model_state = model.state_dict().copy()
            patience_counter = 0
            improvement_flag = " *"
        else:
            patience_counter += 1
            improvement_flag = ""
        
        # Calculate epoch time
        epoch_time = time.time() - epoch_start_time
        
        # Log epoch results
        logger.info(f"Epoch {epoch+1:3d}/{epochs} | "
                   f"Train Loss: {avg_train_loss:.6f} | "
                   f"Val Loss: {avg_val_loss:.6f}{improvement_flag} | "
                   f"Time: {epoch_time:.1f}s")
        
        # Check early stopping
        if patience_counter >= patience:
            logger.info(f"Early stopping triggered at epoch {epoch+1}")
            break
    
    # --- Restore best model ---
    if best_model_state is not None:
        model.load_state_dict(best_model_state)
        logger.info(f"Restored best model with validation loss: {best_val_loss:.6f}")
    
    # --- Create training summary ---
    training_summary = {
        'train_losses': train_losses,
        'val_losses': val_losses,
        'best_val_loss': best_val_loss,
        'epochs_trained': len(train_losses),
        'early_stopped': patience_counter >= patience,
        'final_train_loss': train_losses[-1],
        'final_val_loss': val_losses[-1]
    }
    
    logger.info("=" * 60)
    logger.info("TRAINING COMPLETE")
    logger.info("=" * 60)
    logger.info(f"Total epochs: {len(train_losses)}")
    logger.info(f"Best validation loss: {best_val_loss:.6f}")
    logger.info(f"Final train/val loss: {train_losses[-1]:.6f} / {val_losses[-1]:.6f}")
    
    return training_summary


def main():
    """
    Main training function that runs the entire training pipeline.
    
    This function:
    1. Parses command line arguments
    2. Loads and prepares data with proper feature engineering
    3. Creates sequences with separated numerical/categorical features
    4. Sets up the LSTM model with embedding layers
    5. Trains the model with early stopping
    6. Saves the trained model and metadata
    """
    # --- Parse arguments ---
    args = parse_arguments()
    
    # --- Set random seeds for reproducibility ---
    torch.manual_seed(args.random_seed)
    np.random.seed(args.random_seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    
    # --- Set device ---
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"Using device: {device}")
    if device.type == 'cuda':
        logger.info(f"GPU: {torch.cuda.get_device_name(0)}")
        logger.info(f"Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
    
    # --- Create output directory ---
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    logger.info(f"Output directory: {output_dir}")
    
    # Load and prepare data
    
    # Load and validate data
    raw_data = load_and_validate_data(args.splice_file_path, args.limit_items)
    
    # Apply feature engineering
    features_data = prepare_features(raw_data, args)
    
    # Create sequences with proper separation
    sequences_numerical, sequences_categorical, targets, embedding_specs, preprocessing_info = \
        create_sequences_with_separation(
            features_data, 
            sequence_length=args.sequence_length,
            forecast_horizon=config.M5_HORIZON
        )
    
    # Split data
    
    logger.info("=" * 60)
    logger.info("SPLITTING DATA")
    logger.info("=" * 60)
    
    # Calculate split index (80/20 split)
    n_samples = len(sequences_numerical)
    split_idx = int(0.8 * n_samples)
    
    # Create train/validation splits
    train_sequences_num = sequences_numerical[:split_idx]
    val_sequences_num = sequences_numerical[split_idx:]
    
    train_sequences_cat = {
        feat: sequences[:split_idx]
        for feat, sequences in sequences_categorical.items()
    }
    val_sequences_cat = {
        feat: sequences[split_idx:]
        for feat, sequences in sequences_categorical.items()
    }
    
    train_targets = targets[:split_idx]
    val_targets = targets[split_idx:]
    
    logger.info(f"Training samples: {len(train_sequences_num)}")
    logger.info(f"Validation samples: {len(val_sequences_num)}")
    
    # Create datasets and dataloaders
    
    # Create custom datasets
    train_dataset = M5TimeSeriesDataset(
        train_sequences_num, 
        train_sequences_cat, 
        train_targets
    )
    val_dataset = M5TimeSeriesDataset(
        val_sequences_num, 
        val_sequences_cat, 
        val_targets
    )
    
    # Create data loaders with custom collate function
    train_loader = DataLoader(
        train_dataset, 
        batch_size=args.batch_size, 
        shuffle=True,
        collate_fn=collate_fn_with_separation,
        num_workers=2,
        pin_memory=True if device.type == 'cuda' else False
    )
    val_loader = DataLoader(
        val_dataset, 
        batch_size=args.batch_size, 
        shuffle=False,
        collate_fn=collate_fn_with_separation,
        num_workers=2,
        pin_memory=True if device.type == 'cuda' else False
    )
    
    # Create model
    
    logger.info("=" * 60)
    logger.info("CREATING MODEL")
    logger.info("=" * 60)
    
    # Get input dimensions
    numerical_input_dim = sequences_numerical.shape[2]
    output_dim = config.M5_HORIZON
    
    # Create model with proper architecture
    model = LSTMModel(
        numerical_input_dim=numerical_input_dim,
        embedding_specs=embedding_specs,
        hidden_dim=args.hidden_dim,
        num_layers=args.num_layers,
        output_dim=output_dim,
        dropout=args.dropout,
        bidirectional=args.bidirectional,
        use_attention=args.use_attention,
        quantile_output=args.quantile_forecast,
        quantile_levels=config.M5_QUANTILE_LEVELS if args.quantile_forecast else None
    ).to(device)
    
    # Report model architecture
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"Model parameters: {total_params:,} total, {trainable_params:,} trainable")
    logger.info(f"Numerical input dimension: {numerical_input_dim}")
    logger.info(f"Embedding specifications: {len(embedding_specs)} categorical features")
    logger.info(f"Output dimension: {output_dim} (28-day forecast)")
    
    # Setup optimizer and loss
    
    # Create optimizer
    optimizer = optim.Adam(
        model.parameters(), 
        lr=args.learning_rate, 
        weight_decay=args.weight_decay
    )
    
    # Create loss function
    if args.quantile_forecast:
        criterion = QuantileLoss(config.M5_QUANTILE_LEVELS)
    else:
        criterion = nn.MSELoss()
    
    # Train model
    
    # Train the model
    training_summary = train_model_with_separation(
        model, train_loader, val_loader, optimizer, criterion,
        args.epochs, args.patience, device
    )
    
    # Save model and metadata
    
    logger.info("=" * 60)
    logger.info("SAVING MODEL AND METADATA")
    logger.info("=" * 60)
    
    # Save model checkpoint
    model_path = output_dir / "lstm_model.pth"
    checkpoint = {
        'model_state_dict': model.state_dict(),
        'model_config': {
            'numerical_input_dim': numerical_input_dim,
            'embedding_specs': embedding_specs,
            'hidden_dim': args.hidden_dim,
            'num_layers': args.num_layers,
            'output_dim': output_dim,
            'dropout': args.dropout,
            'bidirectional': args.bidirectional,
            'use_attention': args.use_attention,
            'quantile_output': args.quantile_forecast,
            'quantile_levels': config.M5_QUANTILE_LEVELS if args.quantile_forecast else None
        },
        'preprocessing_info': preprocessing_info,
        'training_args': vars(args),
        'training_summary': training_summary
    }
    torch.save(checkpoint, model_path)
    logger.info(f"Model checkpoint saved to: {model_path}")
    
    # Save embedding info separately for easy access
    embedding_info_path = output_dir / "embedding_info.pkl"
    with open(embedding_info_path, 'wb') as f:
        pickle.dump(preprocessing_info['embedding_info'], f)
    logger.info(f"Embedding info saved to: {embedding_info_path}")
    
    # Save training info as JSON
    training_info = {
        'training_completed': True,
        'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
        'model_path': str(model_path),
        'embedding_info_path': str(embedding_info_path),
        'hyperparameters': vars(args),
        'training_summary': {
            'best_val_loss': float(training_summary['best_val_loss']),
            'epochs_trained': training_summary['epochs_trained'],
            'early_stopped': training_summary['early_stopped'],
            'final_train_loss': float(training_summary['final_train_loss']),
            'final_val_loss': float(training_summary['final_val_loss'])
        },
        'data_info': {
            'splice_file': args.splice_file_path,
            'total_sequences': preprocessing_info['total_sequences'],
            'items_processed': preprocessing_info['items_processed'],
            'train_sequences': len(train_sequences_num),
            'val_sequences': len(val_sequences_num),
            'numerical_features': preprocessing_info['n_numerical_features'],
            'categorical_features': preprocessing_info['n_categorical_features'],
            'sequence_length': preprocessing_info['sequence_length'],
            'forecast_horizon': preprocessing_info['forecast_horizon']
        },
        'model_info': {
            'total_parameters': total_params,
            'trainable_parameters': trainable_params,
            'architecture': {
                'hidden_dim': args.hidden_dim,
                'num_layers': args.num_layers,
                'bidirectional': args.bidirectional,
                'use_attention': args.use_attention,
                'quantile_forecast': args.quantile_forecast
            }
        }
    }
    
    info_path = output_dir / "training_info.json"
    with open(info_path, 'w') as f:
        json.dump(training_info, f, indent=2)
    logger.info(f"Training info saved to: {info_path}")
    
    # Final summary
    
    logger.info("=" * 60)
    logger.info("TRAINING PIPELINE COMPLETE")
    logger.info("=" * 60)
    logger.info(f"✓ Model saved to: {model_path}")
    logger.info(f"✓ Embedding info saved to: {embedding_info_path}")
    logger.info(f"✓ Training info saved to: {info_path}")
    logger.info(f"✓ Best validation loss: {training_summary['best_val_loss']:.6f}")
    logger.info(f"✓ Total training time: {time.strftime('%H:%M:%S', time.gmtime(time.time()))}")


if __name__ == "__main__":
    main()