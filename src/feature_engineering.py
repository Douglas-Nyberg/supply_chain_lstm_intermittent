#////////////////////////////////////////////////////////////////////////////////#
# File:         feature_engineering.py                                           #
# Author:       Douglas Nyberg                                                   #
# Email:        douglas1.nyberg@gmail.com                                        #
# Date:         2025-03-28                                                       #
# Description:  Feature engineering for time series forecasting with embeddings.#
#////////////////////////////////////////////////////////////////////////////////#

"""
Feature engineering for time series forecasting with embedding support.
"""
import logging
from typing import Dict, List, Tuple, Optional, Union, Any
from datetime import datetime, timedelta

import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder

from . import config


def validate_config_availability() -> None:
    """
    Ensure required config variables exist. 
    """
    required_vars = [
        'MAX_LAGS', 'ROLLING_WINDOWS', 'USE_LAG_FEATURES', 
        'USE_ROLLING_FEATURES', 'USE_CALENDAR_FEATURES', 
        'USE_PRICE_FEATURES', 'USE_EVENT_FEATURES', 'USE_INTERMITTENCY_FEATURES',
        'PROMOTION_THRESHOLD'
    ]
    missing_vars = []
    
    try:
        for var_name in required_vars:
            if not hasattr(config, var_name):
                missing_vars.append(var_name)
        
        if missing_vars:
            logger = logging.getLogger(__name__)
            error_message = (
                f"Error: Missing required config variables: {missing_vars}. "
                f"Please add these to your config file. "
                f"Expected config variables: {required_vars}"
            )
            logger.error(error_message)
            raise ValueError(error_message)
    except Exception as e:
        logger = logging.getLogger(__name__)
        logger.error(f"Error validating config: {str(e)}")
        raise


def get_categorical_feature_specs(data: pd.DataFrame = None) -> Dict[str, Dict]:
    """
    Define specifications for categorical features that will use embeddings.
    
    Args:
        data: Optional DataFrame to calculate actual vocab sizes from data
    
    Returns:
        Dictionary mapping feature names to their embedding specifications
    """
    # Base specifications with embedding dimensions and fallback vocab sizes
    base_specs = {
        'weekday': {
            'vocab_size': 7,  # Fallback: Monday=0, Tuesday=1, ..., Sunday=6
            'embedding_dim': 3,  # Small embedding for days of week
            'description': 'Day of week (0=Monday, 6=Sunday)'
        },
        'event_name_1': {
            'vocab_size': 12,  # Fallback: 11 unique events + 1 for 'None'
            'embedding_dim': 5,  # Moderate embedding for event names
            'description': 'Primary event name (SuperBowl, ValentinesDay, etc.)'
        },
        'event_type_1': {
            'vocab_size': 5,   # Fallback: 4 unique types + 1 for 'None' 
            'embedding_dim': 3,  # Small embedding for event types
            'description': 'Primary event type (Sporting, Cultural, National, Religious)'
        },
        'event_name_2': {
            'vocab_size': 2,   # Fallback: 1 unique event + 1 for 'None'
            'embedding_dim': 2,  # Very small embedding
            'description': 'Secondary event name (mostly None)'
        },
        'event_type_2': {
            'vocab_size': 2,   # Fallback: 1 unique type + 1 for 'None'
            'embedding_dim': 2,  # Very small embedding  
            'description': 'Secondary event type (mostly None)'
        },
        'month': {
            'vocab_size': 12,  # Fallback: January=0, February=1, ..., December=11
            'embedding_dim': 4,  # Moderate embedding for months
            'description': 'Month of year (0=January, 11=December)'
        },
        'quarter': {
            'vocab_size': 4,   # Fallback: Q1=0, Q2=1, Q3=2, Q4=3
            'embedding_dim': 2,  # Small embedding for quarters
            'description': 'Quarter of year (0=Q1, 3=Q4)'
        },
        # ID embeddings - useful for hierarchical models (store-level, all-stores, etc.)
        # NOT useful for per-item-per-store models where these are constants
        'store_id': {
            'vocab_size': 10,  # Fallback: CA_1, CA_2, CA_3, CA_4, TX_1, TX_2, TX_3, WI_1, WI_2, WI_3
            'embedding_dim': 4,  # Moderate size for store-specific patterns
            'description': 'Store identifier (e.g., CA_1, TX_2) - useful for multi-store models'
        },
        'dept_id': {
            'vocab_size': 7,   # Fallback: FOODS_1, FOODS_2, FOODS_3, HOBBIES_1, HOBBIES_2, HOUSEHOLD_1, HOUSEHOLD_2
            'embedding_dim': 3,  # Small-medium for department patterns
            'description': 'Department identifier - useful for multi-item models'
        },
        'cat_id': {
            'vocab_size': 3,   # Fallback: FOODS, HOBBIES, HOUSEHOLD
            'embedding_dim': 2,  # Small for just 3 main categories
            'description': 'Product category - useful for multi-category models'
        },
        'state_id': {
            'vocab_size': 3,   # Fallback: CA, TX, WI
            'embedding_dim': 2,  # Small for just 3 states
            'description': 'State identifier - useful for multi-state models'
        },
        'item_id': {
            'vocab_size': 3049,  # Fallback: ~3049 unique items in M5
            'embedding_dim': 16,  # Larger embedding for item-specific patterns
            'description': 'Item identifier - useful for store-level models with multiple items'
        }
    }
    
    # If data is provided, calculate actual vocab sizes
    if data is not None:
        logger = logging.getLogger(__name__)
        
        for feature in base_specs:
            if feature in data.columns:
                # Calculate actual unique values (including NaN as 'None')
                actual_vocab_size = data[feature].fillna('None').nunique()
                fallback_vocab_size = base_specs[feature]['vocab_size']
                
                if actual_vocab_size != fallback_vocab_size:
                    logger.info(f"Updated {feature} vocab_size from {fallback_vocab_size} to {actual_vocab_size}")
                    base_specs[feature]['vocab_size'] = actual_vocab_size
    
    return base_specs


def prepare_categorical_features_for_embeddings(data: pd.DataFrame) -> Tuple[pd.DataFrame, Dict[str, Dict]]:
    """
    Prepare categorical features for embedding layers instead of label encoding.
    
    Args:
        data: DataFrame with categorical features
        
    Returns:
        Tuple of (processed_data, embedding_info)
        - processed_data: DataFrame with categorical features converted to integers
        - embedding_info: Dictionary with actual vocabulary sizes and mappings
    """
    logger = logging.getLogger(__name__)
    data_processed = data.copy()
    
    # Get categorical feature specifications
    categorical_specifications = get_categorical_feature_specs()
    
    # Track actual embedding info (vocab sizes may differ from specs)
    embedding_info = {}
    
    # Process each categorical feature
    for column_name in categorical_specifications.keys():
        if column_name in data_processed.columns:
            # Handle NaN values by treating them as a separate category
            data_processed[column_name] = data_processed[column_name].fillna('None')
            
            # Use label encoding to convert text to integers
            label_encoder = LabelEncoder()
            encoded_values = label_encoder.fit_transform(data_processed[column_name])
            
            # Store the encoded values (keep original column name for embeddings)
            data_processed[column_name] = encoded_values
            
            # Store embedding information
            actual_vocab_size = len(label_encoder.classes_)
            column_spec = categorical_specifications[column_name]
            
            embedding_info[column_name] = {
                'vocab_size': actual_vocab_size,
                'embedding_dim': column_spec['embedding_dim'],
                'label_encoder': label_encoder,
                'classes': label_encoder.classes_.tolist(),
                'description': column_spec['description']
            }
            
            logger.info(f"Prepared {column_name} for embedding: {actual_vocab_size} categories, {column_spec['embedding_dim']}d embedding")
    
    return data_processed, embedding_info


def separate_feature_types(data: pd.DataFrame, exclude_id_columns: bool = True) -> Tuple[List[str], List[str]]:
    """
    Separate features into categorical (for embeddings) and numerical (for direct input).
    
    Args:
        data: DataFrame with all columns
        exclude_id_columns: If True, exclude ID columns (for per-series models).
                           If False, include them (for cross-item models).
        
    Returns:
        Tuple of (categorical_features, numerical_features)
    """
    # Base columns to always exclude
    columns_to_exclude = ['id', 'd', 'date', 'sales']
    
    # ID columns that may be excluded based on model type
    id_columns = ['item_id', 'dept_id', 'cat_id', 'store_id', 'state_id']
    
    if exclude_id_columns:
        # for per-item-per-store models these are constants
        columns_to_exclude.extend(id_columns)
    # else: for hierarchical/cross-item models, these are meaningfull features
    
    # Get categorical feature specifications
    categorical_specifications = get_categorical_feature_specs()
    categorical_features = []
    numerical_features = []
    
    # track columns that dont fit any category
    unclassified_columns = []
    
    # Check each column in the data
    for column_name in data.columns:
        # Skip columns we dont want as features
        if column_name in columns_to_exclude:
            continue
            
        # Check if column is a categorical feature for embeddings
        if column_name in categorical_specifications:
            categorical_features.append(column_name)
        else:
            # check if column has numeric data type
            column_dtype = data[column_name].dtype
            is_numeric_type = column_dtype in ['int64', 'float64', 'int32', 'float32']
            
            if is_numeric_type:
                numerical_features.append(column_name)
            else:
                # Column is neither excluded, categorical, nor numeric 
                unclassified_columns.append(column_name)
    
    # Warn about unclassified columns - (check data quality)
    if unclassified_columns:
        logger = logging.getLogger(__name__)
        logger.warning(
            f"Found {len(unclassified_columns)} unclassified columns (not excluded, categorical, or numeric): "
            f"{unclassified_columns}. These columns have dtypes: "
            f"{[(col, str(data[col].dtype)) for col in unclassified_columns]}. "
            f"Consider adding them to exclusions or converting to appropriate types."
        )
    
    return categorical_features, numerical_features


def preprocess_m5_features_for_embeddings(raw_data: pd.DataFrame, exclude_id_columns: bool = True) -> Tuple[pd.DataFrame, Dict[str, List[str]], Dict[str, Dict]]:
    """
    M5 preprocessing with embedding support.
    
    Args:
        raw_data: Raw M5 data with categorical features
        exclude_id_columns: exclude ID columns for per-series models
        
    Returns:
        Tuple of (processed_data, feature_info, embedding_info)
    """
    logger = logging.getLogger(__name__)
    
    logger.info(f"Starting embedding-aware preprocessing with {len(raw_data.columns)} columns")
    
    # prepare categorical features for embeddings
    processed_data, embedding_info = prepare_categorical_features_for_embeddings(raw_data)
    logger.info(f"Prepared {len(embedding_info)} categorical features for embeddings")
    
    # separate categorical and numerical features
    categorical_features, numerical_features = separate_feature_types(processed_data, exclude_id_columns)
    
    feature_info = {
        'categorical': categorical_features,
        'numerical': numerical_features
    }
    
    logger.info(f"Feature separation: {len(categorical_features)} categorical, {len(numerical_features)} numerical")
    logger.info(f"Categorical features: {categorical_features}")
    logger.info(f"Numerical features: {numerical_features[:5]}..." if len(numerical_features) > 5 else f"Numerical features: {numerical_features}")
    
    return processed_data, feature_info, embedding_info


def calculate_lstm_input_dimension(feature_info: Dict[str, List[str]], embedding_info: Dict[str, Dict]) -> int:
    """
    Calculate the total input dimension for LSTM after embedding concatenation.
    
    Args:
        feature_info: Dictionary with 'categorical' and 'numerical' feature lists
        embedding_info: Dictionary with embedding specifications
        
    Returns:
        Total input dimension for LSTM
    """
    # Sum up embedding dimensions
    embedding_dim = sum(info['embedding_dim'] for info in embedding_info.values())
    
    # Add numerical feature dimensions
    numerical_dim = len(feature_info['numerical'])
    
    total_dim = embedding_dim + numerical_dim
    
    return total_dim


def create_embedding_layer_specs(embedding_info: Dict[str, Dict]) -> Dict[str, Tuple[int, int]]:
    """
    Create embedding layer specifications for PyTorch model.
    
    Args:
        embedding_info: Dictionary with embedding specifications
        
    Returns:
        Dictionary mapping feature names to (vocab_size, embedding_dim) tuples
    """
    embedding_layer_specs = {}
    
    for feature_name, feature_info in embedding_info.items():
        vocab_size = feature_info['vocab_size']
        embedding_dim = feature_info['embedding_dim']
        embedding_layer_specs[feature_name] = (vocab_size, embedding_dim)
    
    return embedding_layer_specs


def add_time_features(df: pd.DataFrame, date_col: str = 'date') -> pd.DataFrame:
    """
    Add calendar-based time features, avoiding duplicates with existing columns.
    
    Args:
        df: DataFrame with time series data
        date_col: Column name with date information
        
    Returns:
        DataFrame with added time features
    """
    logger = logging.getLogger(__name__)
    
    # Ensure date column is datetime
    if df[date_col].dtype != 'datetime64[ns]':
        df[date_col] = pd.to_datetime(df[date_col])
    
    # Work inplace to save memory 
    result_df = df
    
    added_features = []
    skipped_features = []
    
    # Extract basic date components (only if not already present)
    if 'day_of_week' not in result_df.columns:
        result_df['day_of_week'] = df[date_col].dt.dayofweek
        added_features.append('day_of_week')
    else:
        skipped_features.append('day_of_week (already exists)')
    
    if 'day_of_month' not in result_df.columns:
        result_df['day_of_month'] = df[date_col].dt.day
        added_features.append('day_of_month')
    
    if 'day_of_year' not in result_df.columns:
        result_df['day_of_year'] = df[date_col].dt.dayofyear
        added_features.append('day_of_year')
    
    # Only add month/year if not already present 
    if 'month' not in result_df.columns:
        result_df['month_numeric'] = df[date_col].dt.month  
        added_features.append('month_numeric')
    else:
        skipped_features.append('month (already exists)')
    
    if 'year' not in result_df.columns:
        result_df['year'] = df[date_col].dt.year
        added_features.append('year')
    else:
        skipped_features.append('year (already exists)')
    
    if 'is_weekend' not in result_df.columns:
        result_df['is_weekend'] = df[date_col].dt.dayofweek.isin([5, 6]).astype(int)
        added_features.append('is_weekend')
    else:
        skipped_features.append('is_weekend (already exists)')
    
    # Add week number and quarter
    if 'week_of_year' not in result_df.columns:
        result_df['week_of_year'] = df[date_col].dt.isocalendar().week
        added_features.append('week_of_year')
    else:
        skipped_features.append('week_of_year (already exists)')
    
    if 'quarter' not in result_df.columns:
        result_df['quarter_numeric'] = df[date_col].dt.quarter  
        added_features.append('quarter_numeric')
    else:
        skipped_features.append('quarter (already exists)')
    
    # TEMPORARY COMPATIBILITY: Add quarter_numeric for existing trained models
    # TODO: Remove this when retraining models with categorical quarter embeddings
    if 'quarter_numeric' not in result_df.columns and 'quarter' in result_df.columns:
        result_df['quarter_numeric'] = df[date_col].dt.quarter
        added_features.append('quarter_numeric (temporary compatibility for existing LSTM models)')
        logger = logging.getLogger(__name__)
        logger.info("Added quarter_numeric for compatibility with existing trained LSTM models (will be removed in future versions)")
    

    # cyclical features for periodic patterns
    # WHY: linear encoding (0-6 for days) makes day 6 seem far from day 0
    # SOLUTION: map to unit circle so distance preserves cyclical proximity
    # MATH: for period P, map value v to angle θ = 2π * (v/P)
    # then project to 2D: x = cos(θ), y = sin(θ)
    # key property: v=0 and v=P-1 are adjacent on circle
    # Day of week: period = 7 days
    result_df['day_of_week_sin'] = np.sin(2 * np.pi * df[date_col].dt.dayofweek / 7)
    result_df['day_of_week_cos'] = np.cos(2 * np.pi * df[date_col].dt.dayofweek / 7)
    
    # Month: period = 12 months (subtract 1 for 0-based)
    result_df['month_sin'] = np.sin(2 * np.pi * (df[date_col].dt.month - 1) / 12)
    result_df['month_cos'] = np.cos(2 * np.pi * (df[date_col].dt.month - 1) / 12)
    added_features.extend(['day_of_week_sin', 'day_of_week_cos', 'month_sin', 'month_cos'])
    
    logger.info(f"Time features - Added: {len(added_features)}, Skipped: {len(skipped_features)}")
    if skipped_features:
        logger.debug(f"Skipped time features: {skipped_features}")
    
    return result_df


def add_lag_features(
    df: pd.DataFrame, 
    target_col: str = 'sales', 
    id_col: str = 'id',
    date_col: str = 'date',
    lag_days: List[int] = None
) -> pd.DataFrame:
    """
    Add lagged values of target variable.
    
    Args:
        df: DataFrame with time series data
        target_col: target column name
        id_col: series identifier
        date_col: date column
        lag_days: list of lag days
        
    Returns:
        DataFrame with lag features
    """
    logger = logging.getLogger(__name__)
    
    if lag_days is None:
        lag_days = config.MAX_LAGS
    
    # Work in-place to save memory 
    result_df = df
    
    # Ensure data is sorted by id and date
    result_df.sort_values([id_col, date_col], inplace=True)
    
    # Create lag features for each lag day
    for lag in lag_days:
        result_df[f'lag_{lag}'] = result_df.groupby(id_col)[target_col].shift(lag)
        
        # Log missing data statistics
        nan_count = result_df[f'lag_{lag}'].isna().sum()
        total_count = len(result_df)
        nan_percentage = (nan_count / total_count) * 100
        
        logger.info(f"Lag {lag}: Created {nan_count:,} NaN values ({nan_percentage:.1f}% of data)")
        
        # Warn if too many NaNs created
        if nan_percentage > 20:
            logger.warning(f"Lag {lag} created {nan_percentage:.1f}% NaN values - consider shorter lags for this dataset")
    
    return result_df


def add_rolling_features(
    df: pd.DataFrame, 
    target_col: str = 'sales', 
    id_col: str = 'id',
    date_col: str = 'date',
    windows: List[int] = None,
    functions: Dict[str, callable] = None
) -> pd.DataFrame:
    """
    Add rolling window features (moving averages, etc.).
    
    Args:
        df: DataFrame with time series data
        target_col: Column name for the target variable
        id_col: Column name for the series identifier
        date_col: Column name for time/date
        windows: List of window sizes
        functions: Dictionary mapping function names to functions
        
    Returns:
        DataFrame with added rolling window features
    """
    if windows is None:
        windows = config.ROLLING_WINDOWS
    
    if functions is None:
        functions = {
            'mean': np.mean,
            'std': np.std,
            'max': np.max,
            'min': np.min
        }
    
    # work inplace to save memory
    result_df = df
    
    # ensure data is sorted by id and date
    result_df.sort_values([id_col, date_col], inplace=True)
    
    # Create rolling window features
    for window_size in windows:
        for function_name, aggregation_function in functions.items():
            # Calculate rolling statistics for each series separately
            grouped_data = result_df.groupby(id_col)[target_col]
            rolling_data = grouped_data.rolling(window=window_size, min_periods=1)
            aggregated_values = rolling_data.agg(aggregation_function)
            
            # Reset index to match original dataframe
            feature_values = aggregated_values.reset_index(level=0, drop=True)
            
            # add to result dataframe with descripitive column name
            column_name = f'rolling_{window_size}_{function_name}'
            result_df[column_name] = feature_values
    
    return result_df


def add_price_features(
    df: pd.DataFrame,
    price_col: str = 'sell_price',
    id_col: str = 'id',
    date_col: str = 'date'
) -> pd.DataFrame:
    """
    Add price-related features.
    
    Args:
        df: DataFrame with time series data
        price_col: Column name for the price variable
        id_col: Column name for the series identifier
        date_col: Column name for time/date
        
    Returns:
        DataFrame with added price features
    """
    # Check if price column exists
    if price_col not in df.columns:
        return df
    
    # Work inplace to save memory 
    result_df = df
    
    # Ensure data is sorted by id and date
    result_df.sort_values([id_col, date_col], inplace=True)
    
    # Calculate price change from previous day
    price_grouped_by_id = result_df.groupby(id_col)[price_col]
    result_df['price_change'] = price_grouped_by_id.pct_change()
    
    # Calculate price relative to item's average price
    average_price_per_item = price_grouped_by_id.transform('mean')
    result_df['price_relative_to_avg'] = result_df[price_col] / average_price_per_item
    
    # Calculate rolling average price over last 7 days
    rolling_price_7_days = price_grouped_by_id.rolling(window=7, min_periods=1)
    price_7d_average = rolling_price_7_days.mean()
    result_df['price_7d_avg'] = price_7d_average.reset_index(level=0, drop=True)
    
    # Calculate if price is on promotion (configurable threshold)
    promotion_threshold = getattr(config, 'PROMOTION_THRESHOLD', 0.9)
    is_below_threshold = result_df['price_relative_to_avg'] < promotion_threshold
    result_df['is_promotion'] = is_below_threshold.astype(int)
    
    return result_df


def add_event_features(
    df: pd.DataFrame,
    event_columns: List[str] = None
) -> pd.DataFrame:
    """
    Add event-related features from the calendar.
    
    Args:
        df: DataFrame with time series data
        event_columns: List of columns containing event information
        
    Returns:
        DataFrame with added event features
    """
    if event_columns is None:
        event_columns = ['event_name_1', 'event_type_1', 'event_name_2', 'event_type_2']
    
    # Check if event columns exist
    missing_event_columns = []
    for column_name in event_columns:
        if column_name not in df.columns:
            missing_event_columns.append(column_name)
    
    if missing_event_columns:
        return df
    
    # Work inplace to save memory 
    result_df = df
    
    # Create binary event indicators
    for event_column in event_columns:
        # Check if the event column has non-null values
        has_event_data = ~result_df[event_column].isna()
        
        # Create binary indicator column
        indicator_column_name = f'has_{event_column}'
        result_df[indicator_column_name] = has_event_data.astype(int)
    
    # Days until/since specific events (if they are infrequent)
    # This would require more complex logic specific to each event type
    
    return result_df


def add_intermittency_features(
    df: pd.DataFrame,
    target_col: str = 'sales',
    id_col: str = 'id',
    date_col: str = 'date'
) -> pd.DataFrame:
    """
    Add features specific to intermittent demand patterns using vectorized operations.
    
    Args:
        df: DataFrame with time series data
        target_col: Column name for the target variable
        id_col: Column name for the series identifier
        date_col: Column name for time/date
        
    Returns:
        DataFrame with added intermittency features
    """
    # Work in-place to save memory 
    result_df = df
    
    # Ensure data is sorted by id and date
    result_df.sort_values([id_col, date_col], inplace=True)
    
    # Vectorized due to runtime concerns
    def calculate_days_since_last_sale_vectorized(series):
        """Calculate days since last non-zero sale."""
        # Create boolean mask for non-zero sales
        has_sale = (series > 0)
        
        # forward fill to propagate index of last sale
        sale_indices = pd.Series(np.arange(len(series)), index=series.index)
        # keep indices where sales>0, ffill NaN values
        sale_indices = sale_indices.where(has_sale).ffill()
        
        # Handle the case where no sale has occurred yet (beginning of series)
        sale_indices = sale_indices.fillna(-1)
        
        # days since last sale: current index - last sale index
        days_since = np.arange(len(series)) - sale_indices.values
        
        # if current position has sale, days since = 0
        days_since = np.where(has_sale, 0, days_since)
        
        # Handle case where no sale has occurred yet (days since start)
        no_previous_sale = (sale_indices == -1)
        # For positions with no previous sale, use position + 1 as days since start
        days_since = np.where(no_previous_sale, np.arange(len(series)) + 1, days_since)
        
        return pd.Series(days_since.astype(int), index=series.index)
    
    result_df['days_since_last_sale'] = result_df.groupby(id_col)[target_col].apply(calculate_days_since_last_sale_vectorized).values
    
    
    def calculate_zero_streak_vectorized(series):
        """Calculate current consecutive zero-sales streak."""
        # Create boolean mask for zero sales
        is_zero = (series == 0)
        
        # cumsum trick to identify streak groups
        # ~is_zero creates True where sales > 0, False where sales = 0
        # cumsum increments counter each time we see non-zero sale
        streak_groups = (~is_zero).cumsum()
        
        # within each group of consecutive zeros, calculate position
        zero_positions = is_zero.groupby(streak_groups).cumsum()
        
        # Zero positions only where we actually have zeros
        # This ensures non-zero sales show streak_length = 0
        streak_lengths = np.where(is_zero, zero_positions, 0)
        
        return pd.Series(streak_lengths, index=series.index)
    
    result_df['zero_streak_length'] = result_df.groupby(id_col)[target_col].apply(calculate_zero_streak_vectorized).values
    
    # Historical intermittency rate (rolling average of zero-sales days)
    def calculate_intermittency_rate_vectorized(sales_series):
        """Calculate expanding mean of zero-sales days."""
        # convert boolean to int for math operations
        is_zero_sales = (sales_series == 0).astype(int)
        # expanding mean calculates cumulative avg
        intermittency_rate = is_zero_sales.expanding().mean()
        return intermittency_rate
    
    result_df['intermittency_rate'] = result_df.groupby(id_col)[target_col].transform(calculate_intermittency_rate_vectorized)
    
    # Average demand interval (average days between sales)
    def calculate_avg_demand_interval_vectorized(series):
        """Calculate average interval between non-zero sales."""
        # get positions of non-zero sales
        non_zero_positions = np.where(series > 0)[0]
        
        if len(non_zero_positions) < 2:
            avg_interval = np.nan  # cant calculate interval with < 2 sales
        else:
            # calculate differences between consecutive sale positions
            intervals = np.diff(non_zero_positions)
            avg_interval = np.mean(intervals)
        
        # Return constant value for all positions in this series
        return pd.Series([avg_interval] * len(series), index=series.index)
    
    result_df['avg_demand_interval'] = result_df.groupby(id_col)[target_col].apply(calculate_avg_demand_interval_vectorized).values
    
    # Vectorized: Demand coefficient of variation (volatility when there is demand)
    def calculate_demand_coefficient_of_variation_vectorized(sales_series):
        """Calculate coefficient of variation for non-zero sales."""
        # Filter to non-zero sales
        non_zero_sales = sales_series[sales_series > 0]
        
        if len(non_zero_sales) <= 1:
            coefficient_of_variation = 0.0
        else:
            mean_sales = non_zero_sales.mean()
            std_sales = non_zero_sales.std()
            
            # small epsilon to prevent division by zero
            coefficient_of_variation = std_sales / (mean_sales + 1e-8)
        
        return coefficient_of_variation
    
    result_df['demand_cv'] = result_df.groupby(id_col)[target_col].transform(calculate_demand_coefficient_of_variation_vectorized)
    
    return result_df


def create_feature_set(
    df: pd.DataFrame,
    target_col: str = 'sales',
    id_col: str = 'id',
    date_col: str = 'date',
    price_col: str = 'sell_price'
) -> pd.DataFrame:
    """
    Create feature set for time series forecasting.
    
    Args:
        df: DataFrame with time series data
        target_col: target variable column name
        id_col: series identifier column  
        date_col: date column name
        price_col: price variable column
        
    Returns:
        DataFrame with all features added
    """
    logger = logging.getLogger(__name__)
    
    # Validate config availability first
    validate_config_availability()
    
    # Make a copy ONCE at the top level to save memory
    result_df = df.copy()
    
    logger.info(f"Starting feature engineering with {len(result_df.columns)} initial columns")
    logger.info("Note: Working in-place after initial copy to optimize memory usage")
    
    # Add calendar/time features
    if config.USE_CALENDAR_FEATURES:
        result_df = add_time_features(result_df, date_col)
        logger.info("Added calendar/time features")
    
    # Add lag features
    if config.USE_LAG_FEATURES:
        result_df = add_lag_features(
            result_df, 
            target_col=target_col, 
            id_col=id_col, 
            date_col=date_col, 
            lag_days=config.MAX_LAGS
        )
        logger.info("Added lag features")
    
    # Add rolling window features
    if config.USE_ROLLING_FEATURES:
        result_df = add_rolling_features(
            result_df, 
            target_col=target_col, 
            id_col=id_col, 
            date_col=date_col, 
            windows=config.ROLLING_WINDOWS
        )
        logger.info("Added rolling window features")
    
    # Add price features if price column exists
    if config.USE_PRICE_FEATURES and price_col in result_df.columns:
        result_df = add_price_features(
            result_df,
            price_col=price_col,
            id_col=id_col,
            date_col=date_col
        )
        logger.info("Added price features")
    
    # Add event features if relevant columns exist
    if config.USE_EVENT_FEATURES:
        result_df = add_event_features(result_df)
        logger.info("Added event features")
    
    # Add intermittency features for supply chain forecasting
    if config.USE_INTERMITTENCY_FEATURES:
        result_df = add_intermittency_features(
            result_df,
            target_col=target_col,
            id_col=id_col,
            date_col=date_col
        )
        logger.info("Added intermittency features")
    
    logger.info(f"Feature engineering complete: {len(result_df.columns)} total columns")
    
    return result_df





#////////////////////////////////////////////////////////////////////////////////#
#                        LSTM SEQUENCE CREATION UTILITIES                        #
#////////////////////////////////////////////////////////////////////////////////#

def create_lstm_sequences_with_separation(
    data: pd.DataFrame,
    sequence_length: int,
    forecast_horizon: int,
    feature_info: Dict[str, List[str]],
    target_col: str = 'sales',
    item_col: str = 'item_id',
    date_col: str = 'date'
) -> Tuple[np.ndarray, Dict[str, np.ndarray], np.ndarray, Dict[str, Any]]:
    """
    Create sequences for LSTM training.
    
    Args:
        data: preprocessed DataFrame with embeddings
        sequence_length: time steps in input sequences
        forecast_horizon: steps to forecast 
        feature_info: dict with 'categorical' and 'numerical' feature lists
        target_col: target column name
        item_col: item/series identifier
        date_col: date column name
        
    Returns:
        Tuple of (sequences_numerical, sequences_categorical, targets, sequence_info)
    """
    logger = logging.getLogger(__name__)
    
    # Extract feature lists
    categorical_features = feature_info.get('categorical', [])
    numerical_features = feature_info.get('numerical', [])
    
    # Initialize sequence storage
    sequences_numerical = []
    sequences_categorical = {feat: [] for feat in categorical_features}
    targets = []
    
    # Track statistics
    total_sequences = 0
    items_processed = 0
    skipped_items = []
    
    # Process each item/series separately
    for item_id, item_data in data.groupby(item_col):
        # Sort by date to ensure temporal order
        item_data = item_data.sort_values(date_col).reset_index(drop=True)
        
        # Check if we have enough data
        min_required_length = sequence_length + forecast_horizon
        if len(item_data) < min_required_length:
            skipped_items.append((item_id, len(item_data)))
            continue
        
        # Extract features for this item
        if numerical_features:
            numerical_data = item_data[numerical_features].values.astype(np.float32)
        else:
            # dummy numerical data if no numerical features
            numerical_data = np.zeros((len(item_data), 1), dtype=np.float32)
        
        # Extract categorical features
        categorical_data = {
            feat: item_data[feat].values.astype(np.int64)
            for feat in categorical_features
        }
        
        # Extract target values
        target_data = item_data[target_col].values.astype(np.float32)
        
        # Create sequences with sliding window
        sequences_for_item = 0
        for start_idx in range(len(item_data) - sequence_length - forecast_horizon + 1):
            # input and target indices
            seq_end_idx = start_idx + sequence_length
            target_start_idx = seq_end_idx
            target_end_idx = target_start_idx + forecast_horizon
            
            # extract sequences
            num_seq = numerical_data[start_idx:seq_end_idx]
            sequences_numerical.append(num_seq)
            
            for feat in categorical_features:
                cat_seq = categorical_data[feat][start_idx:seq_end_idx]
                sequences_categorical[feat].append(cat_seq)
            
            target_seq = target_data[target_start_idx:target_end_idx]
            targets.append(target_seq)
            
            sequences_for_item += 1
        
        total_sequences += sequences_for_item
        items_processed += 1
    
    # Convert to numpy arrays
    sequences_numerical = np.array(sequences_numerical, dtype=np.float32)
    sequences_categorical = {
        feat: np.array(seqs, dtype=np.int64)
        for feat, seqs in sequences_categorical.items()
    }
    targets = np.array(targets, dtype=np.float32)
    
    # Handle NaN values in numerical features
    nan_count = np.isnan(sequences_numerical).sum()
    if nan_count > 0:
        logger.warning(f"Found {nan_count} NaN values in numerical features, replacing with 0.0")
        sequences_numerical = np.nan_to_num(sequences_numerical, nan=0.0)
    
    # Create sequence info
    sequence_info = {
        'total_sequences': total_sequences,
        'items_processed': items_processed,
        'skipped_items': skipped_items,
        'sequence_length': sequence_length,
        'forecast_horizon': forecast_horizon,
        'n_numerical_features': sequences_numerical.shape[2] if len(sequences_numerical) > 0 else 0,
        'n_categorical_features': len(categorical_features),
        'categorical_features': categorical_features,
        'numerical_features': numerical_features
    }
    
    return sequences_numerical, sequences_categorical, targets, sequence_info


def create_lstm_sequences_for_series(
    series_data: pd.DataFrame,
    sequence_length: int,
    forecast_horizon: int,
    numerical_cols: List[str],
    categorical_cols: List[str],
    target_col: str = 'sales'
) -> Tuple[Dict[str, np.ndarray], np.ndarray, np.ndarray]:
    """
    Create sequences for a single series (used by 3_train_lstm_models.py).
    
    This function is specifically for creating sequences from a single time series,
    as needed when training per-series models.
    
    Args:
        series_data: DataFrame for a single series (already sorted by date)
        sequence_length: Number of time steps in input sequences
        forecast_horizon: Number of steps to forecast
        numerical_cols: List of numerical feature column names
        categorical_cols: List of categorical feature column names
        target_col: Name of the target column
        
    Returns:
        Tuple containing:
        - X_categorical: Dict mapping feature names to arrays of shape (n_samples, seq_len)
        - X_numerical: Array of shape (n_samples, seq_len, n_numerical_features)
        - y: Array of shape (n_samples, forecast_horizon)
        
        Returns (None, None, None) if not enough data for sequences
    """
    # Check if we have enough data
    if len(series_data) < sequence_length + forecast_horizon:
        return None, None, None
    
    # Extract feature arrays
    if numerical_cols:
        numerical_data = series_data[numerical_cols].values.astype(np.float32)
    else:
        numerical_data = np.zeros((len(series_data), 1), dtype=np.float32)
    
    categorical_data = series_data[categorical_cols].values.astype(np.int64)
    target_data = series_data[target_col].values.astype(np.float32)
    
    # Create sequences
    X_numerical = []
    X_categorical = []
    y = []
    
    num_sequences = len(series_data) - sequence_length - forecast_horizon + 1
    
    for i in range(num_sequences):
        # Input sequences
        X_numerical.append(numerical_data[i:i + sequence_length])
        X_categorical.append(categorical_data[i:i + sequence_length])
        
        # Target sequence
        y.append(target_data[i + sequence_length:i + sequence_length + forecast_horizon])
    
    # Convert to arrays
    X_numerical = np.array(X_numerical, dtype=np.float32)
    X_categorical = np.array(X_categorical, dtype=np.int64)
    y = np.array(y, dtype=np.float32)
    
    # Restructure categorical data into dictionary format
    X_categorical_dict = {
        col: X_categorical[:, :, idx]
        for idx, col in enumerate(categorical_cols)
    }
    
    # Handle NaN values
    if np.isnan(X_numerical).any():
        X_numerical = np.nan_to_num(X_numerical, nan=0.0)
    
    return X_categorical_dict, X_numerical, y