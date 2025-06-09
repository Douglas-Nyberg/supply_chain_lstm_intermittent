#////////////////////////////////////////////////////////////////////////////////#
# File:         lstm.py                                                          #
# Author:       Douglas Nyberg                                                   #
# Email:        douglas1.nyberg@gmail.com                                        #
# Date:         2025-04-02                                                       #
#////////////////////////////////////////////////////////////////////////////////#


"""
LSTM model implementation for time series forecasting with categorical embedding support and learned output scaling.           
"""


import os
import pickle
from typing import Dict, List, Tuple, Optional, Union, Callable

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, MinMaxScaler


class LSTMModel(nn.Module):
    """
    LSTM Model for time series forecasting with categorical embedding support.
    """
    def __init__(
        self,
        numerical_input_dim: int,
        embedding_specs: Dict[str, Tuple[int, int]],
        hidden_dim: int,
        num_layers: int,
        output_dim: int,
        dropout: float = 0.2,
        bidirectional: bool = False,
        quantile_output: bool = False,
        quantile_levels: List[float] = None,
        use_attention: bool = False,
        l1_reg: float = 0.0,
        l2_reg: float = 0.0,
        target_mean: float = None,
        target_std: float = None
    ):
        """
        Initialize the LSTM model with categorical embedding support.
        
        Args:
            numerical_input_dim: Number of numerical input features
            embedding_specs: Dict mapping feature names to (vocab_size, embedding_dim) tuples
            hidden_dim: Number of hidden units in LSTM layers
            num_layers: Number of LSTM layers
            output_dim: Number of output units per forecast step
            dropout: Dropout rate (applied between LSTM layers)
            bidirectional: Whether to use bidirectional LSTM
            quantile_output: Whether to generate quantile forecasts (vs point forecasts)
            quantile_levels: List of quantile levels to forecast (required if quantile_output=True)
            use_attention: Whether to use attention mechanism
            l1_reg: L1 regularization weight (0.0 = no L1 regularization)
            l2_reg: L2 regularization weight (0.0 = no L2 regularization)
            target_mean: Mean of training targets for data-driven scaling initialization
            target_std: Standard deviation of training targets for data-driven scaling initialization
        """
        super(LSTMModel, self).__init__()
        
        # Store all architecture parameters for save/load functionality
        self.numerical_input_dim = numerical_input_dim
        self.embedding_specs = embedding_specs
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.output_dim = output_dim  # store output dim
        self.dropout_rate = dropout   # store dropout rate for save/load
        self.bidirectional = bidirectional
        self.quantile_output = quantile_output
        self.use_attention = use_attention
        self.l1_reg = l1_reg
        self.l2_reg = l2_reg
        
        # Store target statistics for data-driven scaling initialization
        # These values inform proper initialization of scaling parameters
        self.target_mean = target_mean
        self.target_std = target_std
        
        # embedding layers for categorical features
        # convert categorical indices to dense vectors
        self.embeddings = nn.ModuleDict()
        total_embedding_dim = 0
        
        for feature_name, (vocab_size, embedding_dim) in embedding_specs.items():
            # embedding lookup table
            # vocab_size = num categories, embedding_dim = vector size
            self.embeddings[feature_name] = nn.Embedding(vocab_size, embedding_dim)
            total_embedding_dim += embedding_dim
        
        # Calculate total input dimension for LSTM
        lstm_input_dim = numerical_input_dim + total_embedding_dim
        
        # LSTM layers
        self.lstm = nn.LSTM(
            input_size=lstm_input_dim,    # Total features: numerical + embedded categorical
            hidden_size=hidden_dim,       # Size of hidden state (memory capacity)
            num_layers=num_layers,        # Number of stacked LSTM layers
            batch_first=True,             # Input shape: (batch, sequence, features)
            dropout=dropout if num_layers > 1 else 0,  # Dropout only if multiple layers
            bidirectional=bidirectional   # If True, processes sequence in both directions
        )
        
        # Calculate factor for bidirectional
        # Bidirectional LSTM outputs twice the hidden_dim (forward + backward)
        direction_factor = 2 if bidirectional else 1
        
        # Attention mechanism (optional)
        if use_attention:
            # Attention learns which time steps are most important for prediction
            # Architecture: hidden_features → tanh activation → single score
            self.attention = nn.Sequential(
                nn.Linear(hidden_dim * direction_factor, hidden_dim),  # Compress features
                nn.Tanh(),                                            # Non-linear activation
                nn.Linear(hidden_dim, 1)                            # Output attention score
            )
        
        # Dropout for regularization
        # Note: BatchNorm removed because it normalizes outputs to different scale than training targets
        # During training, targets (sales) are in original units but BatchNorm forces predictions to normalized scale
        # This creates a scale mismatch that requires post prediction denormalization
        self.dropout_layer = nn.Dropout(dropout)
        
        # Output layers
        if not quantile_output:
            # Single output layer for point forecasts (mean prediction)
            self.fc = nn.Linear(hidden_dim * direction_factor, output_dim)
            
            # Learned scaling parameters for point forecasts
            # These parameters allow the model to learn the optimal scaling during training
            # to bridge the gap between normalized input features and raw sales targets
            
            # Data-driven initialization using training target statistics
            # This approach provides a principled starting point based on actual data characteristics
            if target_std is not None and target_mean is not None:
                # Initialize scale to target standard deviation (captures typical output magnitude)
                scale_init_value = float(target_std)
                # Initialize bias to target mean (centers predictions appropriately)
                bias_init_value = float(target_mean)
            else:
                # Fallback to conservative defaults if statistics not provided
                # Only log warning if we expect this to be a fresh training (not model loading for prediction)
                import logging
                logger = logging.getLogger(__name__)
                # Use debug level instead of warning for backward compatibility scenarios
                logger.debug("Target statistics not provided for LSTM scaling initialization. Using defaults (scale=1.0, bias=0.0).")
                scale_init_value = 1.0
                bias_init_value = 0.0
            
            # Initialize with FloatTensor to ensure consistent behavior across devices
            self.output_scale = nn.Parameter(torch.FloatTensor([scale_init_value]))
            self.output_bias = nn.Parameter(torch.FloatTensor([bias_init_value]))
        else:
            # Multiple output layers for quantile forecasts (uncertainty estimation)
            if quantile_levels is None:
                raise ValueError("quantile_levels must be provided when quantile_output=True")
            
            self.quantile_levels = quantile_levels
            # Create separate output layer for each quantile (e.g., 0.1, 0.5, 0.9)
            # This allows model to predict confidence intervals
            self.quantile_outputs = nn.ModuleDict()
            
            # Learned scaling parameters for each quantile
            # Separate scaling for each quantile allows different confidence levels
            # to have different scaling characteristics
            self.quantile_scales = nn.ParameterDict()
            self.quantile_biases = nn.ParameterDict()
            
            # Data-driven initialization for quantile scaling parameters
            if target_std is not None and target_mean is not None:
                base_scale = float(target_std)
                base_bias = float(target_mean)
            else:
                # Fallback to defaults if statistics not provided
                import logging
                logger = logging.getLogger(__name__)
                # Use debug level instead of warning for backward compatibility scenarios
                logger.debug("Target statistics not provided for LSTM quantile scaling initialization. Using defaults.")
                base_scale = 1.0
                base_bias = 0.0
            
            for q in quantile_levels:
                q_key = f"q{str(q).replace('.', '')}"  # Convert 0.5 → "q05"
                self.quantile_outputs[q_key] = nn.Linear(hidden_dim * direction_factor, output_dim)
                
                # Quantile-specific initialization strategy
                # Extreme quantiles (further from median) may need different scaling
                # Adjust scale based on distance from median (0.5)
                quantile_distance_from_median = abs(q - 0.5)
                # Higher variance for extreme quantiles
                quantile_scale_adjustment = 1.0 + (quantile_distance_from_median * 0.5)
                adjusted_scale = base_scale * quantile_scale_adjustment
                
                # All quantiles center around the same mean initially
                self.quantile_scales[q_key] = nn.Parameter(torch.FloatTensor([adjusted_scale]))
                self.quantile_biases[q_key] = nn.Parameter(torch.FloatTensor([base_bias]))
    
    def forward(self, categorical_features: Dict[str, torch.Tensor], numerical_features: torch.Tensor) -> Union[torch.Tensor, Dict[float, torch.Tensor]]:
        """
        Forward pass through the LSTM model.
        
        Args:
            categorical_features: Dict mapping feature names to tensors of shape (batch_size, sequence_length)
            numerical_features: Tensor of shape (batch_size, sequence_length, numerical_input_dim)
            
        Returns:
            For point forecasts: Tensor of shape (batch_size, output_dim)
            For quantile forecasts: Dictionary mapping quantile levels to tensors of shape (batch_size, output_dim)
        """
        batch_size, sequence_length = numerical_features.shape[:2]
        
        # Process categorical features through embeddings
        embedded_features = []
        
        for feature_name, feature_tensor in categorical_features.items():
            if feature_name in self.embeddings:
                # Apply embedding lookup: convert indices to dense vectors
                # Input: (batch_size, sequence_length) with integer indices
                # Output: (batch_size, sequence_length, embedding_dim) with real values
                embedded = self.embeddings[feature_name](feature_tensor)
                embedded_features.append(embedded)
        
        # Concatenate all embedded features
        if embedded_features:
            # torch.cat along dim=2 (feature dimension) combines embeddings
            # E.g., weekday_emb + month_emb → combined categorical features
            embedded_categorical = torch.cat(embedded_features, dim=2)  # Shape: (batch, seq, total_embed_dim)
            
            # Combine numerical and categorical features along feature dimension
            # Final shape: (batch, seq, numerical_dim + total_embed_dim)
            x = torch.cat([numerical_features, embedded_categorical], dim=2)
        else:
            # Only numerical features (no categorical data available)
            x = numerical_features
        
        # Initialize hidden state and cell state for LSTM
        # h0 = hidden state (short-term memory), c0 = cell state (long-term memory)
        # Both start as zeros and will be updated during LSTM processing
        if self.bidirectional:
            # Bidirectional: need twice as many states (forward + backward directions)
            h0 = torch.zeros(self.num_layers * 2, batch_size, self.hidden_dim).to(x.device)
            c0 = torch.zeros(self.num_layers * 2, batch_size, self.hidden_dim).to(x.device)
        else:
            # Unidirectional: one set of states per layer
            h0 = torch.zeros(self.num_layers, batch_size, self.hidden_dim).to(x.device)
            c0 = torch.zeros(self.num_layers, batch_size, self.hidden_dim).to(x.device)
        
        # LSTM forward pass
        lstm_out, _ = self.lstm(x, (h0, c0))
        
        # Apply attention if enabled
        if self.use_attention:
            # Calculate attention weights
            # Start with raw attention scores
            raw_attention_scores = self.attention(lstm_out)  # Shape: (batch, seq, 1)
            # Remove the last dimension to get (batch, seq)
            attention_weights = raw_attention_scores.squeeze(-1)
            # Normalize weights so they sum to 1.0 across the sequence dimension
            attention_weights = torch.softmax(attention_weights, dim=1)
            
            # Apply attention weights
            attn_weights = attention_weights.unsqueeze(1)
            weighted_out = torch.bmm(attn_weights, lstm_out)
            context_vector = weighted_out.squeeze(1)
            lstm_out = context_vector
        else:
            # Use last time step output
            lstm_out = lstm_out[:, -1, :]
        
        # Apply dropout
        # Note: BatchNorm application removed because predictions now in same scale as training targets 
        lstm_out = self.dropout_layer(lstm_out)
        
        # Output layer(s) with learned scaling
        if not self.quantile_output:
            # Point forecast with learned scaling
            out = self.fc(lstm_out)
            # Apply learned scaling to bridge normalized predictions to sales units
            scaled_out = out * self.output_scale + self.output_bias
            return scaled_out
        else:
            # Quantile forecasts with separate learned scaling for each quantile
            quantile_forecasts = {}
            
            for i, quantile_level in enumerate(self.quantile_levels):
                quantile_key = f"q{str(quantile_level).replace('.', '')}"
                quantile_output_layer = self.quantile_outputs[quantile_key]
                quantile_forecast = quantile_output_layer(lstm_out)
                
                # Apply learned scaling specific to this quantile
                scale = self.quantile_scales[quantile_key]
                bias = self.quantile_biases[quantile_key]
                scaled_quantile_forecast = quantile_forecast * scale + bias
                quantile_forecasts[quantile_level] = scaled_quantile_forecast
            
            return quantile_forecasts
    
    def get_regularization_loss(self) -> torch.Tensor:
        """
        Compute L1 and L2 regularization loss.
        
        Returns:
            Combined regularization loss tensor
        """
        # Initialize loss accumulators on the same device as model parameters
        device = next(self.parameters()).device
        l1_loss = torch.tensor(0.0, device=device)
        l2_loss = torch.tensor(0.0, device=device)
        
        # Only compute regularization if weights are non-zero (for efficiency)
        if self.l1_reg > 0.0 or self.l2_reg > 0.0:
            # Loop through all model parameters (weights and biases)
            for parameter in self.parameters():
                if self.l1_reg > 0.0:
                    # L1: sum of absolute values (promotes sparsity)
                    absolute_values = torch.abs(parameter)
                    l1_loss += torch.sum(absolute_values)
                    
                if self.l2_reg > 0.0:
                    # L2: sum of squared values (promotes small weights)
                    squared_values = parameter ** 2
                    l2_loss += torch.sum(squared_values)
        
        # Combine losses with their respective weights
        total_regularization_loss = (self.l1_reg * l1_loss) + (self.l2_reg * l2_loss)
        return total_regularization_loss
    
    def get_scaling_info(self) -> Dict[str, float]:
        """
        Return current learned scaling parameters for logging and analysis.
        
        Returns:
            Dictionary containing scaling parameters (scale and bias values)
        """
        scaling_info = {}
        
        if not self.quantile_output:
            # Point forecast scaling parameters
            scaling_info["output_scale"] = self.output_scale.item()
            scaling_info["output_bias"] = self.output_bias.item()
        else:
            # Quantile forecast scaling parameters
            for q_key in self.quantile_scales.keys():
                scaling_info[f"{q_key}_scale"] = self.quantile_scales[q_key].item()
                scaling_info[f"{q_key}_bias"] = self.quantile_biases[q_key].item()
        
        return scaling_info


def create_lstm_model(
    numerical_input_dim: int,
    embedding_specs: Dict[str, Tuple[int, int]],
    hidden_dim: int = 64,
    num_layers: int = 2,
    output_dim: int = 28,  # M5 horizon
    dropout: float = 0.2,
    bidirectional: bool = False,
    quantile_output: bool = False,
    quantile_levels: List[float] = None,
    use_attention: bool = False,
    l1_reg: float = 0.0,
    l2_reg: float = 0.0
) -> LSTMModel:
    """
    Factory function to create an LSTM model with categorical embedding support.
    
    Args:
        numerical_input_dim: Number of numerical input features
        embedding_specs: Dict mapping feature names to (vocab_size, embedding_dim) tuples
        hidden_dim: Number of hidden units in LSTM layers
        num_layers: Number of LSTM layers
        output_dim: Number of output units per forecast step
        dropout: Dropout rate (applied between LSTM layers)
        bidirectional: Whether to use bidirectional LSTM
        quantile_output: Whether to generate quantile forecasts (vs point forecasts)
        quantile_levels: List of quantile levels to forecast (required if quantile_output=True)
        use_attention: Whether to use attention mechanism
        l1_reg: L1 regularization weight
        l2_reg: L2 regularization weight
        
    Returns:
        Configured LSTM model
    """
    model = LSTMModel(
        numerical_input_dim=numerical_input_dim,
        embedding_specs=embedding_specs,
        hidden_dim=hidden_dim,
        num_layers=num_layers,
        output_dim=output_dim,
        dropout=dropout,
        bidirectional=bidirectional,
        quantile_output=quantile_output,
        quantile_levels=quantile_levels,
        use_attention=use_attention,
        l1_reg=l1_reg,
        l2_reg=l2_reg
    )
    return model




def train_lstm_model(
    model: LSTMModel,
    train_data: Tuple[Dict[str, torch.Tensor], torch.Tensor, torch.Tensor],
    val_data: Optional[Tuple[Dict[str, torch.Tensor], torch.Tensor, torch.Tensor]] = None,
    epochs: int = 100,
    batch_size: int = 32,
    learning_rate: float = 0.001,
    weight_decay: float = 1e-5,
    patience: int = 10,
    min_delta: float = 0.001,
    quantile_levels: List[float] = None,
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
) -> Dict[str, List[float]]:
    """
    Train the LSTM model with embedding-aware inputs.
    
    Args:
        model: LSTM model to train
        train_data: Tuple of (categorical_features_dict, numerical_features, targets)
        val_data: Optional validation data in same format
        epochs: Number of training epochs
        batch_size: Batch size for training
        learning_rate: Learning rate for optimizer
        weight_decay: Weight decay for optimizer
        patience: Number of epochs to wait for improvement before early stopping
        min_delta: Minimum change in validation loss to qualify as improvement
        quantile_levels: List of quantile levels (for quantile forecasting)
        device: Device to use for training ("cuda" or "cpu")
        
    Returns:
        Dictionary with training history (loss, val_loss)
    """
    categorical_train, numerical_train, targets_train = train_data
    
    # Move model to device
    model = model.to(device)
    
    # Move data to device
    categorical_train = {k: v.to(device) for k, v in categorical_train.items()}
    numerical_train = numerical_train.to(device)
    targets_train = targets_train.to(device)
    
    if val_data is not None:
        categorical_val, numerical_val, targets_val = val_data
        categorical_val = {k: v.to(device) for k, v in categorical_val.items()}
        numerical_val = numerical_val.to(device)
        targets_val = targets_val.to(device)
    
    # Define optimizer
    optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    
    # Define loss function
    if model.quantile_output:
        if quantile_levels is None:
            quantile_levels = model.quantile_levels
        criterion = QuantileLoss(quantile_levels)
    else:
        criterion = nn.MSELoss()
    
    # Initialize training history
    history = {"loss": [], "val_loss": []}
    
    # Early stopping variables
    best_val_loss = float("inf")
    no_improvement_count = 0
    
    # Create datasets for batching
    n_samples = numerical_train.shape[0]
    indices = torch.arange(n_samples)  # Create sequential indices [0, 1, 2, ..., n_samples-1]
    
    # Training loop
    for epoch in range(epochs):
        model.train()  # Set model to training mode (enables dropout, batch norm updates)
        total_loss = 0
        
        # Shuffle indices for each epoch to randomize training order
        # torch.randperm(n) creates random permutation [0, 1, ..., n-1]
        shuffled_indices = indices[torch.randperm(n_samples)]
        # Calculate number of batches needed (round up using ceiling division)
        n_batches = (n_samples + batch_size - 1) // batch_size
        
        for batch_idx in range(n_batches):
            # Calculate batch boundaries
            start_idx = batch_idx * batch_size
            end_idx = min(start_idx + batch_size, n_samples)  # handle last batch
            batch_indices = shuffled_indices[start_idx:end_idx]
            
            # extract batch data
            categorical_batch = {}
            for feature_name, feature_data in categorical_train.items():
                categorical_batch[feature_name] = feature_data[batch_indices]
            # select rows for numerical and targets
            numerical_batch = numerical_train[batch_indices]
            targets_batch = targets_train[batch_indices]
            
            # clear gradients
            optimizer.zero_grad()
            
            # forward pass
            y_pred = model(categorical_batch, numerical_batch)
            
            # Check for NaNs/Infs
            if torch.isnan(numerical_batch).any():
                print(f"WARNING: NaNs detected in numerical_batch at epoch {epoch + 1}")
            if torch.isnan(targets_batch).any():
                print(f"WARNING: NaNs detected in targets_batch at epoch {epoch + 1}")
            
            # Compute loss
            if model.quantile_output:
                loss = criterion(y_pred, targets_batch)
            else:
                loss = criterion(y_pred, targets_batch)
            
            # Add regularization
            reg_loss = model.get_regularization_loss()
            loss = loss + reg_loss
            
            # Check for NaN/Inf in loss
            if torch.isnan(loss) or torch.isinf(loss):
                print(f"WARNING: NaN/Inf detected in loss at epoch {epoch + 1}")
                continue
            
            # Backward pass: compute gradients via backpropagation
            loss.backward()
            
            # Apply gradient clipping to prevent exploding gradients
            # Rescales gradients if their norm exceeds max_norm=1.0
            # This stabilizes training for RNNs/LSTMs which are prone to gradient explosion
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            # Update model parameters using computed gradients
            optimizer.step()
            
            total_loss += loss.item()
        
        # Compute average training loss
        avg_train_loss = total_loss / n_batches
        history["loss"].append(avg_train_loss)
        
        # Validation phase
        if val_data is not None:
            model.eval()  # Set model to evaluation mode (disables dropout, fixes batch norm)
            # torch.no_grad() disables gradient computation for faster inference
            with torch.no_grad():
                y_pred = model(categorical_val, numerical_val)
                
                if model.quantile_output:
                    val_loss = criterion(y_pred, targets_val)
                else:
                    val_loss = criterion(y_pred, targets_val)
                
                history["val_loss"].append(val_loss.item())
                
                # Early stopping mechanism: stop training if validation loss plateaus
                # Check if validation loss improved by at least min_delta
                if val_loss.item() < best_val_loss - min_delta:
                    best_val_loss = val_loss.item()
                    no_improvement_count = 0  # Reset counter
                else:
                    no_improvement_count += 1  # Increment counter
                
                # Stop training if no improvement for 'patience' epochs
                if no_improvement_count >= patience:
                    print(f"Early stopping after {epoch + 1} epochs")
                    break
        
        print(f"Epoch {epoch + 1}/{epochs}, "
              f"Loss: {avg_train_loss:.4f}"
              + (f", Val Loss: {val_loss.item():.4f}" if val_data is not None else ""))
    
    return history


def predict_with_lstm(
    model: LSTMModel,
    categorical_features: Dict[str, torch.Tensor],
    numerical_features: torch.Tensor,
    scaler: Optional[Union[StandardScaler, MinMaxScaler]] = None,
    device: str = "cuda" if torch.cuda.is_available() else "cpu",
    batch_size: int = 64
) -> Union[np.ndarray, Dict[float, np.ndarray]]:
    """
    Generate predictions using the trained LSTM model.
    
    Args:
        model: Trained LSTM model
        categorical_features: Dict mapping feature names to tensors
        numerical_features: Numerical input tensor
        scaler: Fitted scaler for inverse transforming predictions
        device: Device to use for prediction ("cuda" or "cpu")
        batch_size: Batch size for predictions
        
    Returns:
        For point forecasts: Array of shape (batch_size, output_dim)
        For quantile forecasts: Dictionary mapping quantiles to arrays
    """
    # Move model and data to specified device (GPU/CPU)
    model = model.to(device)
    # Move all categorical tensors to device using dictionary comprehension
    categorical_features = {k: v.to(device) for k, v in categorical_features.items()}
    numerical_features = numerical_features.to(device)
    
    # Set model to evaluation mode (disables dropout, fixes batch norm)
    model.eval()
    
    n_samples = numerical_features.shape[0]
    predictions_list = []
    
    # Generate predictions in batches to manage memory usage
    # torch.no_grad() disables gradient computation for faster inference
    with torch.no_grad():
        # Process data in chunks of batch_size using range(start, stop, step)
        for batch_idx in range(0, n_samples, batch_size):
            end_idx = min(batch_idx + batch_size, n_samples)  # Handle last batch
            
            # Extract batch data using slice indexing
            # Create categorical batch by slicing each feature
            categorical_batch = {}
            for feature_name, feature_data in categorical_features.items():
                categorical_batch[feature_name] = feature_data[batch_idx:end_idx]
            numerical_batch = numerical_features[batch_idx:end_idx]
            
            # Forward pass: generate predictions for current batch
            batch_pred = model(categorical_batch, numerical_batch)
            
            if model.quantile_output:
                # Convert quantile tensors to numpy arrays
                # .cpu() moves tensor from GPU to CPU, .numpy() converts to numpy
                batch_pred_np = {q: pred.cpu().numpy() for q, pred in batch_pred.items()}
                predictions_list.append(batch_pred_np)
            else:
                # Convert point forecast tensor to numpy array
                predictions_list.append(batch_pred.cpu().numpy())
    
    # Concatenate all batch predictions along batch dimension (axis=0)
    if model.quantile_output:
        # Combine quantile predictions: each quantile level gets its own array
        predictions = {}
        for q in predictions_list[0].keys():  # Iterate through quantile levels
            # Concatenate arrays for this quantile across all batches
            predictions[q] = np.concatenate([batch_pred[q] for batch_pred in predictions_list], axis=0)
    else:
        # Concatenate point forecast arrays from all batches
        predictions = np.concatenate(predictions_list, axis=0)
    
    # Inverse transform predictions back to original scale if scaler was used during training
    if scaler is not None:
        predictions = inverse_scale_predictions(predictions, scaler)
    
    return predictions


def inverse_scale_predictions(
    predictions: Union[np.ndarray, Dict[float, np.ndarray]],
    scaler: Union[StandardScaler, MinMaxScaler],
    target_column_idx: int = 0
) -> Union[np.ndarray, Dict[float, np.ndarray]]:
    """
    Inverse transform scaled predictions back to original scale.
    
    Args:
        predictions: Scaled predictions
        scaler: Fitted scaler used for scaling
        target_column_idx: Index of target column in the original data
        
    Returns:
        Predictions in original scale
    """
    if isinstance(predictions, np.ndarray):
        # Single array case: handle point forecasts
        if len(predictions.shape) == 1:
            # 1D predictions: create dummy array with zeros for all features
            # scaler.scale_.shape[0] gives number of original features
            dummy = np.zeros((len(predictions), scaler.scale_.shape[0]))
            dummy[:, target_column_idx] = predictions  # Place predictions in target column
            
            # Apply inverse transformation using the fitted scaler
            inverse_transformed = scaler.inverse_transform(dummy)
            result = inverse_transformed[:, target_column_idx]  # Extract target column
            
            # Ensure non-negative values for sales data (sales cannot be negative)
            result = np.maximum(result, 0)
            
            return result
        else:
            # 2D predictions (batch_size, forecast_horizon) - process each time step separately
            batch_size, horizon = predictions.shape
            result = np.zeros_like(predictions)
            
            # Process each time step individually to preserve temporal structure
            for t in range(horizon):
                # Create dummy array for this time step
                dummy = np.zeros((batch_size, scaler.scale_.shape[0]))
                dummy[:, target_column_idx] = predictions[:, t]
                
                # Apply inverse transformation
                inverse_transformed = scaler.inverse_transform(dummy)
                result[:, t] = inverse_transformed[:, target_column_idx]
            
            # Ensure non-negative values for sales data
            result = np.maximum(result, 0)
            
            return result
    
    elif isinstance(predictions, dict):
        # Dictionary of quantiles case: handle quantile forecasts
        inverse_predictions = {}
        
        # Process each quantile level separately
        for quantile, quantile_preds in predictions.items():
            if len(quantile_preds.shape) == 1:
                # 1D predictions: create dummy array for inverse transformation
                dummy = np.zeros((len(quantile_preds), scaler.scale_.shape[0]))
                dummy[:, target_column_idx] = quantile_preds
                
                # Apply inverse transformation for this quantile
                inverse_transformed = scaler.inverse_transform(dummy)
                result = inverse_transformed[:, target_column_idx]
                
                # Ensure non-negative values (important for sales forecasting)
                result = np.maximum(result, 0)
                
                inverse_predictions[quantile] = result
            else:
                # 2D predictions (batch_size, forecast_horizon) - process each time step separately
                batch_size, horizon = quantile_preds.shape
                result = np.zeros_like(quantile_preds)
                
                # Process each time step individually to preserve temporal structure
                for t in range(horizon):
                    # Create dummy array for this time step
                    dummy = np.zeros((batch_size, scaler.scale_.shape[0]))
                    dummy[:, target_column_idx] = quantile_preds[:, t]
                    
                    # Apply inverse transformation
                    inverse_transformed = scaler.inverse_transform(dummy)
                    result[:, t] = inverse_transformed[:, target_column_idx]
                
                # Ensure non-negative values for sales data
                result = np.maximum(result, 0)
                
                inverse_predictions[quantile] = result
        
        return inverse_predictions
    
    else:
        raise ValueError("Predictions must be either a numpy array or a dictionary")




def pinball_loss(predictions: torch.Tensor, targets: torch.Tensor, quantile: float) -> torch.Tensor:
    """
    Compute the pinball loss (quantile loss) for quantile regression.
    
    The pinball loss is asymmetric: it penalizes underestimation more heavily
    for high quantiles and overestimation more heavily for low quantiles.
    
    Args:
        predictions: Predicted values
        targets: Target values
        quantile: Quantile level (between 0 and 1)
        
    Returns:
        Pinball loss value
    """
    # calculate prediction errors
    errors = targets - predictions
    
    # apply asymmetric penalty based on quantile level
    # underestimation (error > 0): penalty = quantile * error
    # overestimation (error < 0): penalty = (quantile - 1) * error
    underestimation_penalty = quantile * errors
    overestimation_penalty = (quantile - 1) * errors
    
    # take max of the two penalties
    pinball_losses = torch.max(underestimation_penalty, overestimation_penalty)
    
    # return avg loss
    return torch.mean(pinball_losses)


class QuantileLoss(nn.Module):
    """
    PyTorch module for pinball loss with multiple quantiles.
    """
    def __init__(self, quantiles: List[float], weight_higher_quantiles: bool = False):
        """
        Initialize the quantile loss.
        
        Args:
            quantiles: List of quantile levels
            weight_higher_quantiles: Whether to give more weight to higher quantiles
        """
        super(QuantileLoss, self).__init__()
        self.quantiles = quantiles
        self.weight_higher_quantiles = weight_higher_quantiles
        
        # Create weights for quantiles if needed
        if weight_higher_quantiles:
            # Give higher quantiles more importance (e.g., 0.9 gets more weight than 0.1)
            self.weights = {}
            for q in quantiles:
                # Weight formula: 0.5 + (q - 0.5) gives range [0.0 to 1.0]
                # Example: q=0.1 → weight=0.1, q=0.5 → weight=0.5, q=0.9 → weight=0.9
                raw_weight = 0.5 + (q - 0.5)
                self.weights[q] = raw_weight
            
            # Normalize weights so they sum to 1.0
            weight_sum = sum(self.weights.values())
            for q in self.weights:
                self.weights[q] = self.weights[q] / weight_sum
        else:
            # Equal weights for all quantiles (uniform importance)
            self.weights = {}
            equal_weight = 1.0 / len(quantiles)
            for q in quantiles:
                self.weights[q] = equal_weight
    
    def forward(self, predictions: Dict[float, torch.Tensor], targets: torch.Tensor) -> torch.Tensor:
        """
        Compute the quantile loss.
        
        Args:
            predictions: Dictionary mapping quantile levels to predictions
            targets: Target values
            
        Returns:
            Weighted average pinball loss across all quantiles
        """
        # Collect losses and weights for each quantile
        losses = []
        weights = []
        
        # Calculate loss for each quantile level
        for quantile_level, quantile_predictions in predictions.items():
            # Compute pinball loss for this quantile
            quantile_loss = pinball_loss(quantile_predictions, targets, quantile_level)
            losses.append(quantile_loss)
            
            # Get the weight for this quantile
            quantile_weight = self.weights[quantile_level]
            weights.append(quantile_weight)
        
        # Convert lists to tensors for computation
        stacked_losses = torch.stack(losses)  # Shape: (n_quantiles,)
        weights_tensor = torch.tensor(weights, device=stacked_losses.device)
        
        # Compute weighted average of all quantile losses
        weighted_losses = stacked_losses * weights_tensor
        total_loss = torch.sum(weighted_losses)
        
        return total_loss


def save_lstm(
    model: LSTMModel,
    scaler: Optional[Union[StandardScaler, MinMaxScaler]],
    embedding_info: Dict[str, Dict],
    feature_info: Dict[str, List[str]],
    path: str,
    model_name: str = "lstm_model"
) -> None:
    """
    Save the trained LSTM model with embedding metadata.
    
    Args:
        model: Trained LSTM model
        scaler: Fitted scaler
        embedding_info: Dictionary with embedding specifications
        feature_info: Dictionary with categorical and numerical feature lists
        path: Directory path to save the model
        model_name: Name for the saved model files
    """
    # Ensure directory exists
    os.makedirs(path, exist_ok=True)
    
    # Save model
    torch.save(model.state_dict(), os.path.join(path, f"{model_name}.pt"))
    
    # Save model architecture information
    model_info = {
        "numerical_input_dim": model.numerical_input_dim,
        "embedding_specs": model.embedding_specs,
        "hidden_dim": model.hidden_dim,
        "num_layers": model.num_layers,
        "bidirectional": model.bidirectional,
        "quantile_output": model.quantile_output,
        "use_attention": model.use_attention,
        "quantile_levels": getattr(model, "quantile_levels", None),
        "output_dim": model.output_dim,  # CRITICAL: Save output dimension
        "dropout": model.dropout_rate,   # CRITICAL: Save dropout rate (float)
        "l1_reg": model.l1_reg,          # CRITICAL: Save L1 regularization parameter
        "l2_reg": model.l2_reg,          # CRITICAL: Save L2 regularization parameter
        "embedding_info": embedding_info,
        "feature_info": feature_info
    }
    
    with open(os.path.join(path, f"{model_name}_info.pkl"), "wb") as f:
        pickle.dump(model_info, f)
    
    # Save scaler if provided
    if scaler is not None:
        with open(os.path.join(path, f"{model_name}_scaler.pkl"), "wb") as f:
            pickle.dump(scaler, f)


def load_lstm(
    path: str,
    model_name: str = "lstm_model",
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
) -> Tuple[LSTMModel, Optional[Union[StandardScaler, MinMaxScaler]], Dict[str, Dict], Dict[str, List[str]]]:
    """
    Load a saved LSTM model with embedding metadata.
    
    Args:
        path: Directory path where the model is saved
        model_name: Name of the saved model files
        device: Device to load the model to ("cuda" or "cpu")
        
    Returns:
        Tuple of (loaded model, loaded scaler, embedding_info, feature_info)
    """
    # Load model architecture information
    with open(os.path.join(path, f"{model_name}_info.pkl"), "rb") as f:
        model_info = pickle.load(f)
    
    # Create model with the saved architecture
    model = create_lstm_model(
        numerical_input_dim=model_info["numerical_input_dim"],
        embedding_specs=model_info["embedding_specs"],
        hidden_dim=model_info["hidden_dim"],
        num_layers=model_info["num_layers"],
        output_dim=model_info.get("output_dim", 28),      # CRITICAL: Restore output dimension
        dropout=model_info.get("dropout", 0.2),           # CRITICAL: Restore dropout rate (float)
        bidirectional=model_info["bidirectional"],
        quantile_output=model_info["quantile_output"],
        quantile_levels=model_info["quantile_levels"],
        use_attention=model_info.get("use_attention", False),
        l1_reg=model_info.get("l1_reg", 0.0),             # CRITICAL: Restore L1 regularization parameter
        l2_reg=model_info.get("l2_reg", 0.0)              # CRITICAL: Restore L2 regularization parameter
    )
    
    # Load model weights
    model.load_state_dict(torch.load(os.path.join(path, f"{model_name}.pt"), map_location=device))
    model = model.to(device)
    
    # Load scaler if available
    scaler = None
    scaler_path = os.path.join(path, f"{model_name}_scaler.pkl")
    if os.path.exists(scaler_path):
        with open(scaler_path, "rb") as f:
            scaler = pickle.load(f)
    
    return model, scaler, model_info["embedding_info"], model_info["feature_info"]


# Define M5 competition specific quantile levels
M5_QUANTILE_LEVELS = [0.005, 0.025, 0.165, 0.250, 0.500, 0.750, 0.835, 0.975, 0.995]


def prepare_m5_forecasts(
    predictions: Dict[float, np.ndarray],
    series_ids: List[str],
    forecast_horizon: int = 28,
    quantile_levels: Optional[List[float]] = None
) -> pd.DataFrame:
    """
    Prepare forecasts in M5 competition submission format.
    
    Args:
        predictions: Dictionary mapping quantile levels to arrays of shape (n_series, forecast_horizon)
        series_ids: List of series IDs corresponding to the predictions
        forecast_horizon: Length of forecast horizon
        quantile_levels: Optional list of quantile levels to include, defaults to M5 quantiles
        
    Returns:
        DataFrame in M5 submission format with (series_id, quantile) MultiIndex
    """
    # Use provided quantile levels or default to M5 quantiles
    if quantile_levels is None:
        quantile_levels = M5_QUANTILE_LEVELS
    
    # Create MultiIndex rows for DataFrame
    # Each series will have predictions for each quantile level
    rows = []
    for series_id in series_ids:
        for quantile in quantile_levels:
            # Create tuple (series_id, quantile) for MultiIndex
            row_tuple = (series_id, quantile)
            rows.append(row_tuple)
    
    # Create columns for forecast horizon (F1, F2, ..., F28 for M5)
    columns = []
    for day in range(forecast_horizon):
        column_name = f"F{day + 1}"  # F1 for day 0, F2 for day 1, etc.
        columns.append(column_name)
    
    # Create empty DataFrame
    forecast_df = pd.DataFrame(index=pd.MultiIndex.from_tuples(rows, names=["id", "quantile"]),
                              columns=columns)
    
    # Fill DataFrame with predictions
    for i, series_id in enumerate(series_ids):
        for q in quantile_levels:
            # Skip if this quantile was not predicted
            if q not in predictions:
                continue
                
            # Validate prediction array dimensions
            pred_array = predictions[q]
            
            # Check array compatibility before indexing
            if len(pred_array.shape) == 1:
                # For 1D arrays, validate against both possible interpretations
                if not (pred_array.shape[0] == len(series_ids) or 
                       (len(series_ids) == 1 and pred_array.shape[0] == forecast_horizon)):
                    continue  # Skip incompatible arrays
                # For 1D case where shape[0] = n_series, check if current series index is valid
                if pred_array.shape[0] == len(series_ids) and i >= pred_array.shape[0]:
                    continue
            elif len(pred_array.shape) == 2:
                # For 2D arrays, check both dimensions
                if pred_array.shape[0] <= i or pred_array.shape[1] < forecast_horizon:
                    continue
            else:
                continue  # Skip arrays with unsupported dimensions
                
            # Handle different prediction shapes
            if len(pred_array.shape) == 1:
                # 1D array case: determine if it's (n_series,) or (forecast_horizon,)
                if len(series_ids) == 1 and pred_array.shape[0] == forecast_horizon:
                    # Single series with multiple time steps: shape (forecast_horizon,)
                    forecast_df.loc[(series_id, q), :] = pred_array[:forecast_horizon]
                elif len(series_ids) > 1 and pred_array.shape[0] == len(series_ids):
                    # Multiple series each with their own forecast value
                    # For multi-step forecasting, this is an error condition
                    raise ValueError(f"1D predictions with shape {pred_array.shape} cannot be "
                                   f"broadcast to {len(series_ids)} series and {forecast_horizon} time steps. "
                                   f"For M5 multi-step forecasting, predictions should be 2D or 3D.")
                else:
                    # Ambiguous case: 1D array doesn't match expected dimensions
                    raise ValueError(f"1D predictions with shape {pred_array.shape} is incompatible with "
                                   f"{len(series_ids)} series and {forecast_horizon} time steps. "
                                   f"Expected shape: ({len(series_ids)}, {forecast_horizon})")
            elif len(pred_array.shape) == 2:
                # Multiple series, horizon predictions: shape (n_series, forecast_horizon)
                forecast_df.loc[(series_id, q), :] = pred_array[i, :forecast_horizon]
            else:
                raise ValueError(f"Predictions for quantile {q} have unsupported shape: {pred_array.shape}. "
                               f"Expected 1D or 2D array.")
    
    return forecast_df