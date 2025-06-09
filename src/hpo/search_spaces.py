#////////////////////////////////////////////////////////////////////////////////#
# File:         search_spaces.py                                                 #
# Author:       Douglas Nyberg                                                   #
# Email:        douglas1.nyberg@gmail.com                                        #
# Date:         2025-05-02                                                       #
#////////////////////////////////////////////////////////////////////////////////#




"""
Hyperparameter search spaces for models.
"""
from typing import Dict, Tuple, List, Any, Union


def get_lstm_param_space() -> Dict[str, Union[Tuple, List]]:
    """
    Define search space for LSTM hyperparams.
    
    Returns:
        Dict mapping param names to search spaces.
    """
    return {
        # Architecture parameters
        'hidden_dim': (32, 256, 32),  # Range from 32 to 256 in steps of 32
        'num_layers': (1, 5, 1),  # 1 to 5 layers
        'dropout': (0.0, 0.5, 0.05),  # Dropout from 0 to 0.5 in steps of 0.05
        'bidirectional': [True, False],  # Whether to use bidirectional LSTM
        'use_attention': [True, False],  # Whether to use attention mechanism
        'sequence_length': [28, 42, 56],  # Discrete weekly-aligned choices (4, 6, 8 weeks)
        
        # Training parameters
        'batch_size': (16, 128, 16),  # Batch size from 16 to 128 in steps of 16
        'learning_rate': (1e-5, 1e-2, 'log'),  # Learning rate on log scale
        'weight_decay': (1e-6, 1e-3, 'log'),  # Weight decay (Adam optimizer L2 regularization) on log scale
        'epochs': (30, 100, 10),  # Number of epochs from 30 to 100 in steps of 10
        'patience': (5, 20, 5),  # Early stopping patience from 5 to 20 in steps of 5
        
        # Regularization parameters
        'l1_reg': (1e-6, 1e-3, 'log'),  # L1 regularization weight on log scale
        'l2_reg': (1e-6, 1e-3, 'log'),  # L2 regularization weight on log scale (additional to weight_decay)
        
        # Feature engineering parameters
        'use_price_features': [True, False],  # Whether to use price-related features
        'use_lag_features': [True, False],  # Whether to use lag features
        'use_rolling_features': [True, False],  # Whether to use rolling window features
        'use_calendar_features': [True, False],  # Whether to use calendar features
        'use_event_features': [True, False],  # Whether to use event features
    }


def get_lstm_param_space_limited() -> Dict[str, Union[Tuple, List]]:
    """
    Define a more limited search space for LSTM hyperparameters.
    Useful for quicker exploration or when computational resources are limited.
    
    Returns:
        Dictionary mapping parameter names to their search spaces.
    """
    return {
        'hidden_dim': (32, 128, 32),  # Range from 32 to 128 in steps of 32
        'num_layers': (1, 3, 1),  # 1 to 3 layers
        'dropout': (0.0, 0.3, 0.1),  # Dropout from 0 to 0.3 in steps of 0.1
        'bidirectional': [False, True],  # Whether to use bidirectional LSTM
        'use_attention': [True, False],  # Whether to use attention mechanism
        'sequence_length': [28, 42],  # Discrete weekly-aligned choices (4, 6 weeks)
        
        'batch_size': (32, 64, 32),  # Batch size from 32 to 64 in steps of 32
        'learning_rate': (1e-4, 1e-3, 'log'),  # Limited learning rate range
        'weight_decay': (1e-5, 1e-4, 'log'),  # Limited weight decay range
        'epochs': (30, 50, 10),  # Number of epochs from 30 to 50 in steps of 10
        'patience': (5, 10, 5),  # Early stopping patience from 5 to 10 in steps of 5
        
        # Limited regularization parameters
        'l1_reg': (1e-5, 1e-4, 'log'),  # L1 regularization weight
        'l2_reg': (1e-5, 1e-4, 'log'),  # L2 regularization weight
    }


def get_lstm_param_space_focused() -> Dict[str, Union[Tuple, List]]:
    """
    Define a focused search space for LSTM hyperparameters.
    This is designed for intermittent demand series where model
    architecture and training stability are particularly important.
    
    Returns:
        Dictionary mapping parameter names to their search spaces.
    """
    return {
        # Architecture parameters - focused on stable, deeper networks
        'hidden_dim': (64, 256, 64),  # Larger hidden dimensions
        'num_layers': (2, 4, 1),  # Multiple layers for capturing complex patterns
        'dropout': (0.1, 0.4, 0.1),  # Higher dropout for preventing overfitting
        'bidirectional': [True],  # Using bidirectional for better context
        'use_attention': [True],  # Using attention for temporal importance
        'sequence_length': [28, 42, 56],  # Discrete weekly-aligned choices (4, 6, 8 weeks)
        
        # Training parameters - focused on stable convergence
        'batch_size': (32, 64, 32),  # Moderate batch size for stable gradients
        'learning_rate': (5e-5, 5e-4, 'log'),  # Lower learning rates for stability
        'weight_decay': (1e-5, 1e-4, 'log'),  # Moderate regularization
        'epochs': (50, 100, 25),  # More epochs for convergence
        'patience': (10, 20, 5),  # More patience for finding optimum
        
        # Focused regularization parameters
        'l1_reg': (1e-6, 1e-4, 'log'),  # L1 regularization weight
        'l2_reg': (1e-6, 1e-4, 'log'),  # L2 regularization weight
    }


def get_dual_output_lstm_space() -> Dict[str, Union[Tuple, List]]:
    """
    Define a search space for dual-output LSTM model hyperparameters.
    This is designed for the Croston-inspired architecture with separate 
    outputs for demand occurrence probability and demand size.
    
    Returns:
        Dictionary mapping parameter names to their search spaces.
    """
    return {
        # Shared architecture parameters
        'hidden_dim': (64, 256, 64),  # Hidden dimensions for both networks
        'num_layers': (2, 4, 1),  # Number of layers for both networks
        'dropout': (0.1, 0.4, 0.1),  # Dropout for regularization
        'sequence_length': (28, 56, 7),  # Input sequence length
        
        # Demand probability network specific parameters
        'prob_hidden_layers': (1, 3, 1),  # Hidden layers in probability network
        'prob_hidden_dim': (32, 128, 32),  # Hidden dim in probability network
        
        # Demand size network specific parameters
        'size_hidden_layers': (1, 3, 1),  # Hidden layers in size network
        'size_hidden_dim': (32, 128, 32),  # Hidden dim in size network
        
        # Training parameters
        'batch_size': (32, 64, 32),  # Batch size
        'learning_rate': (5e-5, 5e-4, 'log'),  # Learning rate
        'weight_decay': (1e-5, 1e-4, 'log'),  # L2 regularization
        'epochs': (50, 100, 25),  # Training epochs
        'patience': (10, 20, 5),  # Early stopping patience
        
        # Loss weighting - how much to weight each part of the loss
        'demand_prob_weight': (0.3, 0.7, 0.1),  # Weight for demand occurrence loss
    }


def get_custom_loss_lstm_space() -> Dict[str, Union[Tuple, List]]:
    """
    Define a search space for LSTM with custom loss function parameters.
    Focused on loss function weights for intermittent demand.
    
    Returns:
        Dictionary mapping parameter names to their search spaces.
    """
    return {
        # Standard LSTM parameters
        'hidden_dim': (64, 256, 64),
        'num_layers': (2, 4, 1),
        'dropout': (0.1, 0.4, 0.1),
        'bidirectional': [True, False],
        'use_attention': [True, False],
        'sequence_length': (28, 56, 7),
        
        # Training parameters
        'batch_size': (32, 64, 32),
        'learning_rate': (5e-5, 5e-4, 'log'),
        'weight_decay': (1e-5, 1e-4, 'log'),
        'epochs': (50, 100, 25),
        'patience': (10, 20, 5),
        
        # Custom loss parameters
        'loss_type': ['custom_intermittent', 'mse', 'mae', 'huber', 'quantile'],
        'zero_penalty_weight': (0.5, 5.0, 0.5),  # Penalty for incorrectly predicting zeros
        'adi_factor': (0.1, 1.0, 0.1),  # Factor to scale loss by Average Demand Interval
    }


def get_unified_param_space() -> Dict[str, Union[Tuple, List]]:
    """
    Define a unified search space for LSTM hyperparameters that works well with
    all optimization methods (GA, RL, and Bayesian). This space is carefully
    designed to balance exploration and exploitation, with reasonable bounds
    and appropriate parameter types for each optimization method.
    
    The unified space focuses on parameters that are most likely to impact
    performance on intermittent demand time series forecasting.
    
    Returns:
        Dictionary mapping parameter names to their search spaces.
    """
    return {
        # Architecture parameters - carefully bounded for all optimization methods
        'hidden_dim': (32, 256, 32),  # Powers of 2 work well for GPU optimization
        'num_layers': (1, 4, 1),  # More layers can capture complex patterns but risk overfitting
        'dropout': (0.0, 0.5, 0.1),  # Standard range for dropout regularization
        'bidirectional': [True, False],  # Bidirectional can capture better context
        'use_attention': [True, False],  # Attention helps focus on relevant timesteps
        
        # Sequence parameters - critical for time series
        'sequence_length': (14, 56, 7),  # Weekly multiples make sense for retail data
        
        # Training parameters - bounded to avoid divergence/instability
        'batch_size': (16, 128, 16),  # Larger batches for smoother gradients
        'learning_rate': (1e-5, 1e-2, 'log'),  # Log scale for learning rate is essential
        'weight_decay': (1e-6, 1e-3, 'log'),  # Log scale for regularization strength
        'epochs': (30, 100, 10),  # Allow enough epochs for convergence
        'patience': (5, 20, 5),  # Early stopping patience for efficient training
        
        # Feature selection parameters - binary choices
        'use_price_features': [True, False],
        'use_lag_features': [True, False],
        'use_rolling_features': [True, False],
        'use_calendar_features': [True, False],
        'use_event_features': [True, False],
    }