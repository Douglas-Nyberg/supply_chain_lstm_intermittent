#////////////////////////////////////////////////////////////////////////////////#
# File:         classical.py                                                     #
# Author:       Douglas Nyberg                                                   #
# Email:        douglas1.nyberg@gmail.com                                        #
# Date:         2025-03-20                                                       #
# Description:  Classical time series forecasting models using statsforecast. Pro#
#               vides a unified interface for all traditional forecasting      #
#               methods.                                                       #
# Affiliation:  Physics Department, Purdue University                            #
#////////////////////////////////////////////////////////////////////////////////#
"""wrapper for statsforecast classical models"""
import warnings
from typing import Dict, List, Tuple, Optional, Union, Any

import numpy as np
import pandas as pd

from statsforecast import StatsForecast
from statsforecast.models import (
    ARIMA as SF_ARIMA,
    CrostonClassic as SF_CrostonClassic,
    CrostonOptimized as SF_CrostonOptimized,
    CrostonSBA as SF_CrostonSBA,
    TSB as SF_TSB,
    ADIDA as SF_ADIDA,
    IMAPA as SF_IMAPA,
    HoltWinters as SF_HoltWinters,
    AutoETS as SF_AutoETS,
    SimpleExponentialSmoothing as SF_SES,
    SimpleExponentialSmoothingOptimized as SF_SES_Optimized,
    WindowAverage as SF_WindowAverage,
    SeasonalNaive as SF_SeasonalNaive,
    Naive as SF_Naive
)

# Suppress warnings
warnings.filterwarnings('ignore')


class BaseForecaster:
    """base class for all forecasters with common interface"""
    
    def __init__(self, model_instance):
        # init with model instance
        self.model = model_instance
        self.sf = None
        self.fitted = False
        self._last_train_data = None
        self._freq = 'D'
        
    def fit(self, y: np.ndarray, freq: str = 'D') -> 'BaseForecaster':
        # fit model to data
        # need to convert to statsforecast format
        df = pd.DataFrame({
            'unique_id': 1,
            'ds': pd.date_range(start='2020-01-01', periods=len(y), freq=freq),
            'y': y
        })
        
        # Store training data and frequency for later use
        self._last_train_data = df.copy()
        self._freq = freq
        
        # Create StatsForecast object
        self.sf = StatsForecast(
            models=[self.model],
            freq=freq,
            n_jobs=1
        )
        
        # Fit the model
        self.sf.fit(df)
        self.fitted = True
        
        return self
        
    def predict(self, h: int = 28) -> np.ndarray:
        # get point forecasts
        if not self.fitted:
            raise ValueError("model must be fitted before predicton")
            
        # generate forecasts
        forecasts_df = self.sf.forecast(df=self._last_train_data, h=h)
        
        # extract forecast values - column name is model alias
        forecast_cols = [col for col in forecasts_df.columns if col not in ['unique_id', 'ds']]
        if forecast_cols:
            return forecasts_df[forecast_cols[0]].values
        else:
            raise ValueError("no forecast column found")
        
    def predict_quantiles(self, h: int = 28, quantiles: List[float] = None) -> Dict[float, np.ndarray]:
        # quantile forecasts - kinda hacky but works
        if not self.fitted:
            raise ValueError("model must be fitted before predicton")
            
        if quantiles is None:
            quantiles = [0.05, 0.1, 0.25, 0.5, 0.75, 0.9, 0.95]
            
        # Get point forecasts
        point_forecasts = self.predict(h)
        
        # For models without built-in prediction intervals, 
        # we'll use a simple residual-based approach
        try:
            # Try to get fitted values
            fitted_df = self.sf.fitted_[0]
            y_train = self._last_train_data['y'].values
            
            # Get the fitted values column
            fitted_cols = [col for col in fitted_df.columns if col not in ['unique_id', 'ds', 'y']]
            if fitted_cols:
                fitted_values = fitted_df[fitted_cols[0]].values
                
                # calculate residuals
                residuals = y_train - fitted_values
                
                # remove nan residuals
                residuals = residuals[~np.isnan(residuals)]
                
                if len(residuals) > 0:
                    # calculate standard deviation of residuals
                    std_residuals = np.std(residuals)
                else:
                    # fallback to 10% of mean forecast
                    std_residuals = 0.1 * np.mean(np.abs(point_forecasts))
            else:
                # fallback if no fitted values
                std_residuals = 0.1 * np.mean(np.abs(point_forecasts))
        except:
            # fallback for any errors
            std_residuals = 0.1 * np.mean(np.abs(point_forecasts))
            
        # generate quantile forecasts
        quantile_forecasts = {}
        for q in quantiles:
            if q == 0.5:
                quantile_forecasts[q] = point_forecasts
            else:
                # use normal distribution assumtion
                from scipy.stats import norm
                z_score = norm.ppf(q)
                quantile_forecasts[q] = point_forecasts + z_score * std_residuals
                
        return quantile_forecasts


# Specific model implementations

class ARIMAForecaster(BaseForecaster):
    """arima wrapper"""
    
    def __init__(self, order: Tuple[int, int, int] = (1, 1, 1), 
                 seasonal_order: Tuple[int, int, int, int] = (0, 0, 0, 0)):
        # arima in statsforecast uses different parameter format
        model = SF_ARIMA(
            order=order,
            season_length=seasonal_order[3] if seasonal_order[3] > 0 else 1,
            seasonal_order=seasonal_order[:3] if seasonal_order[3] > 0 else (0, 0, 0)
        )
        super().__init__(model)


class CrostonsMethod(BaseForecaster):
    """crostons for intermitent demand"""
    
    def __init__(self, alpha: float = 0.1):
        # note: statsforecasts crostonclassic doesnt take alpha parameter
        # it optimizes it internally
        model = SF_CrostonClassic()
        super().__init__(model)


class CrostonSBA(BaseForecaster):
    """crostons sba variant"""
    
    def __init__(self, alpha: float = 0.1):
        # alpha not used - statsforecast optimizes internaly
        # statsforecast version optimizes params internally
        model = SF_CrostonSBA()
        super().__init__(model)


class TSBForecaster(BaseForecaster):
    """tsb method"""
    
    def __init__(self, alpha: float = 0.1, beta: float = 0.1):
        # tsb requires alpha_d and alpha_p params
        model = SF_TSB(alpha_d=alpha, alpha_p=beta)
        super().__init__(model)


class MovingAverage(BaseForecaster):
    
    def __init__(self, window_size: int = 7):
        model = SF_WindowAverage(window_size=window_size)
        super().__init__(model)


class SimpleExponentialSmoothing(BaseForecaster):
    """ses forecaster"""
    
    def __init__(self, alpha: float = None):
        if alpha is None:
            # use optimized version
            model = SF_SES_Optimized()
        else:
            # Use fixed alpha
            model = SF_SES(alpha=alpha)
        super().__init__(model)


class HoltWintersForecaster(BaseForecaster):
    
    def __init__(self, season_length: int = 7, error_type: str = 'add', 
                 trend_type: str = 'add', season_type: str = 'add'):
        # holtwinters params
        # map string types to statsforecast format
        trend = trend_type if trend_type != 'None' else None
        seasonal = season_type if season_type != 'None' else None
        
        # holtwinters in statsforecast only takes season_length and error_type
        model = SF_HoltWinters(
            season_length=season_length,
            error_type=error_type[0].upper() if error_type else 'A'  # 'A' or 'M'
        )
        super().__init__(model)


class ETS(BaseForecaster):
    """ets model"""
    
    def __init__(self, season_length: int = 7, model: str = 'ZZZ'):
        # autoets automatically selects the best ets model
        model_instance = SF_AutoETS(season_length=season_length)
        super().__init__(model_instance)


class ADIDAForecaster(BaseForecaster):
    """adida for intermittent"""
    
    def __init__(self):
        model = SF_ADIDA()
        super().__init__(model)


class IMAPAForecaster(BaseForecaster):
    
    def __init__(self):
        model = SF_IMAPA()
        super().__init__(model)


# Aliases for backward compatibility
TSB = TSBForecaster
HoltWinters = HoltWintersForecaster
ADIDA = ADIDAForecaster
IMAPA = IMAPAForecaster