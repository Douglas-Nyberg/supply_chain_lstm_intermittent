#////////////////////////////////////////////////////////////////////////////////#
# File:         __init__.py                                                      #
# Author:       Douglas Nyberg                                                   #
# Email:        douglas1.nyberg@gmail.com                                        #
# Date:         2025-03-05                                                       #
# Description:  Models package initialization for forecasting algorithms.       #
#////////////////////////////////////////////////////////////////////////////////#

"""
Models package for supply chain demand forecasting.

Exports LSTM and classical forecasting models.
"""

# Export all forecasting models
from .classical import (
    ARIMAForecaster,
    CrostonsMethod,
    CrostonSBA,
    TSBForecaster,
    MovingAverage,
    SimpleExponentialSmoothing,
    HoltWintersForecaster,
    ETS,
    ADIDAForecaster,
    IMAPAForecaster,
    # Aliases for backward compatibility
    TSB,
    HoltWinters,
    ADIDA,
    IMAPA
)

from .lstm import (
    LSTMModel,
    create_lstm_model,
    train_lstm_model,
    predict_with_lstm,
    save_lstm,
    load_lstm,
    QuantileLoss
)

__all__ = [
    # Classical methods
    'ARIMAForecaster',
    'CrostonsMethod', 
    'CrostonSBA',
    'TSBForecaster',
    'MovingAverage',
    'SimpleExponentialSmoothing',
    'HoltWintersForecaster',
    'ETS',
    'ADIDAForecaster',
    'IMAPAForecaster',
    # Aliases
    'TSB',
    'HoltWinters',
    'ADIDA',
    'IMAPA',
    # LSTM methods
    'LSTMModel',
    'create_lstm_model',
    'train_lstm_model',
    'predict_with_lstm',
    'save_lstm',
    'load_lstm',
    'QuantileLoss'
]