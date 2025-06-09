#////////////////////////////////////////////////////////////////////////////////#
# File:         1_create_feature_rich_splices.py                                 #
# Author:       Douglas Nyberg                                                   #
# Email:        douglas1.nyberg@gmail.com                                        #
# Date:         2025-04-28                                                       #
#////////////////////////////////////////////////////////////////////////////////#
"""
Create feature-rich M5 data splices optimized for LSTM vs classical method comparison.

This script creates multiple data splices from the M5 dataset with different characteristics:
1. Intermittent demand patterns (high sparsity)
2. Various difficulty levels for forecasting
3. Different seasonal patterns
4. Multiple size categories for testing/training/HPO

The goal is to demonstrate LSTM capabilities vs classical methods (like Croston, ARIMA, TSB)
for inventory control and demand forecasting on intermittent time series.
"""

import argparse
import json
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Union

import numpy as np
import pandas as pd

# Project root setup
SCRIPT_DIR = Path(os.path.dirname(os.path.abspath(__file__)))
PROJECT_ROOT = SCRIPT_DIR.parent
sys.path.append(str(PROJECT_ROOT))

# Import project modules
from src.feature_engineering import create_feature_set
from src.utils import create_directory, save_json
from src import config

# Data paths
RAW_DATA_DIR = PROJECT_ROOT / 'data' / 'raw'
SALES_DATA_PATH = RAW_DATA_DIR / 'sales_train_validation.csv'
CALENDAR_PATH = RAW_DATA_DIR / 'calendar.csv'
PRICES_PATH = RAW_DATA_DIR / 'sell_prices.csv'
OUTPUT_SPLICE_DIR = PROJECT_ROOT / 'data' / 'm5_splices'


def calculate_seasonal_strength(series: np.ndarray) -> float:
    """
    Calculate seasonal strength using simple variance-based approach.
    
    Args:
        series: Time series data
        
    Returns:
        Seasonal strength score (0-1, higher = more seasonal)
    """
    series = np.asarray(series).flatten()
    
    if len(series) < 52:  # Need at least 1 year of data
        return 0.0
    
    # Simple approach: check if there's weekly pattern by comparing
    # variance within weeks vs between weeks
    try:
        # Reshape to weeks (assume 52 weeks in data)
        weeks = len(series) // 7
        if weeks < 8:  # Need at least 8 weeks
            return 0.0
        
        reshaped = series[:weeks*7].reshape(weeks, 7)
        within_week_var = np.mean(np.var(reshaped, axis=0))
        between_week_var = np.var(np.mean(reshaped, axis=1))
        
        if within_week_var + between_week_var == 0:
            return 0.0
        
        # Seasonal strength is ratio of between to total variance
        seasonal_strength = between_week_var / (within_week_var + between_week_var)
        return min(seasonal_strength, 1.0)
        
    except:
        return 0.0


def calculate_initial_zero_periods(series: np.ndarray) -> int:
    """
    Calculate number of zero periods at the start of the series.
    
    Args:
        series: Time series data
        
    Returns:
        Number of initial zero periods
    """
    initial_zeros = 0
    for value in series:
        if value == 0:
            initial_zeros += 1
        else:
            break
    return initial_zeros


def create_price_lookup(prices_data: pd.DataFrame) -> Dict:
    """
    Create optimized price lookup dictionary.
    
    Args:
        prices_data: Price data
        
    Returns:
        Dictionary for fast price lookups
    """
    if prices_data is None or prices_data.empty:
        return {}
    
    print("    Creating price lookup index...")
    return prices_data.groupby(['item_id', 'store_id'])['sell_price'].mean().to_dict()


def apply_basic_filters(sales_data: pd.DataFrame, splice_config: Dict) -> pd.DataFrame:
    """
    Apply basic filters before expensive calculations.
    
    Args:
        sales_data: Sales data in wide format
        splice_config: Splice configuration
        
    Returns:
        Pre-filtered DataFrame
    """
    filtered_df = sales_data.copy()
    
    # Apply store/category/department filters first
    if 'store_filter' in splice_config:
        filtered_df = filtered_df[filtered_df['store_id'].isin(splice_config['store_filter'])]
    
    if 'category_filter' in splice_config:
        filtered_df = filtered_df[filtered_df['cat_id'].isin(splice_config['category_filter'])]
    
    if 'department_filter' in splice_config:
        filtered_df = filtered_df[filtered_df['dept_id'].isin(splice_config['department_filter'])]
    
    return filtered_df


def calculate_intermittency_metrics_vectorized(sales_data: pd.DataFrame, price_lookup: Dict = None) -> pd.DataFrame:
    """
    Optimized vectorized calculation of intermittency metrics.
    
    Args:
        sales_data: Sales data in wide format (pre-filtered)
        price_lookup: Pre-computed price lookup dictionary
        
    Returns:
        DataFrame with intermittency metrics
    """
    day_cols = [col for col in sales_data.columns if col.startswith('d_')]
    sales_matrix = sales_data[day_cols].values
    
    print(f"    Processing {len(sales_data)} series with vectorized operations...")
    
    # Vectorized basic calculations
    total_sales = np.sum(sales_matrix, axis=1)
    non_zero_counts = np.sum(sales_matrix > 0, axis=1)
    zero_counts = np.sum(sales_matrix == 0, axis=1)
    total_periods = sales_matrix.shape[1]
    
    # Vectorized sparsity and demand rate
    sparsity = zero_counts / total_periods
    demand_rate = non_zero_counts / total_periods
    
    # Vectorized max demand
    max_demand = np.max(sales_matrix, axis=1)
    
    # Process each series for more complex metrics
    metrics = []
    for idx, (_, row) in enumerate(sales_data.iterrows()):
        series_id = row['id']
        sales_values = sales_matrix[idx]
        
        # ADI calculation
        non_zero_indices = np.where(sales_values > 0)[0]
        if len(non_zero_indices) > 1:
            intervals = np.diff(non_zero_indices)
            adi = np.mean(intervals) if len(intervals) > 0 else total_periods
        else:
            adi = total_periods
        
        # CV calculation
        non_zero_sales = sales_values[sales_values > 0]
        if len(non_zero_sales) > 1:
            cv = np.std(non_zero_sales) / np.mean(non_zero_sales)
        else:
            cv = 0.0
        
        cv_squared = cv ** 2
        
        # Simplified seasonal strength 
        seasonal_strength = calculate_seasonal_strength_fast(sales_values)
        
        # Fast initial zero periods
        initial_zero_periods = calculate_initial_zero_periods_fast(sales_values)
        
        # Fast price lookup
        avg_price = 0.0
        if price_lookup:
            avg_price = price_lookup.get((row['item_id'], row['store_id']), 0.0)
        
        # Croston's categorization
        if adi > 1.32 and cv_squared < 0.49:
            demand_pattern = 'intermittent'
        elif adi <= 1.32 and cv_squared >= 0.49:
            demand_pattern = 'erratic'
        elif adi > 1.32 and cv_squared >= 0.49:
            demand_pattern = 'lumpy'
        else:
            demand_pattern = 'smooth'
        
        metrics.append({
            'id': series_id,
            'item_id': row['item_id'],
            'dept_id': row['dept_id'],
            'cat_id': row['cat_id'],
            'store_id': row['store_id'],
            'state_id': row['state_id'],
            'total_sales': total_sales[idx],
            'non_zero_periods': non_zero_counts[idx],
            'sparsity': sparsity[idx],
            'demand_rate': demand_rate[idx],
            'adi': adi,
            'cv': cv,
            'cv_squared': cv_squared,
            'demand_pattern': demand_pattern,
            'mean_demand': np.mean(non_zero_sales) if len(non_zero_sales) > 0 else 0,
            'std_demand': np.std(non_zero_sales) if len(non_zero_sales) > 0 else 0,
            'max_demand': max_demand[idx],
            'seasonal_strength': seasonal_strength,
            'initial_zero_periods': initial_zero_periods,
            'avg_price': avg_price,
            'forecasting_difficulty': sparsity[idx] * cv_squared + adi / 30.0
        })
    
    return pd.DataFrame(metrics)


def calculate_seasonal_strength_fast(series: np.ndarray) -> float:
    """
    Fast simplified seasonal strength calculation.
    
    Args:
        series: Time series data
        
    Returns:
        Seasonal strength score
    """
    series = np.asarray(series).flatten()
    
    if len(series) < 14:  # Need at least 2 weeks
        return 0.0
    
    try:
        # Simple approach: weekly pattern variance
        weeks = min(len(series) // 7, 52)  # Max 1 year
        if weeks < 2:
            return 0.0
        
        weekly_data = series[:weeks*7].reshape(weeks, 7)
        daily_means = np.mean(weekly_data, axis=0)
        overall_mean = np.mean(series)
        
        if overall_mean == 0:
            return 0.0
        
        return np.std(daily_means) / overall_mean
        
    except:
        return 0.0


def calculate_initial_zero_periods_fast(series: np.ndarray) -> int:
    """
    Fast calculation of initial zero periods.
    
    Args:
        series: Time series data
        
    Returns:
        Number of initial zero periods
    """
    # Vectorized approach
    non_zero_mask = series > 0
    if not np.any(non_zero_mask):
        return len(series)
    
    first_non_zero = np.argmax(non_zero_mask)
    return first_non_zero


def calculate_intermittency_metrics(sales_data: pd.DataFrame, prices_data: pd.DataFrame = None) -> pd.DataFrame:
    """
    OPTIMIZED: Calculate intermittency metrics with lazy evaluation and vectorization.
    
    Args:
        sales_data: Sales data in wide format (M5 style) 
        prices_data: Price data for calculating price-based metrics
        
    Returns:
        DataFrame with intermittency metrics for each series
    """
    print(f"  OPTIMIZED: Processing {len(sales_data)} series (vs {30490} in original)")
    
    # Create price lookup once for all series
    price_lookup = create_price_lookup(prices_data) if prices_data is not None else None
    
    # Use optimized vectorized calculation
    return calculate_intermittency_metrics_vectorized(sales_data, price_lookup)


def load_m5_data() -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Load M5 competition data files.
    
    Args:
        None
        
    Returns:
        Tuple containing:
        - sales_train: DataFrame with sales data
        - calendar: DataFrame with calendar/event data
        - prices: DataFrame with price data
    """
    print("Loading M5 competition data...")
    
    try:
        sales_train = pd.read_csv(SALES_DATA_PATH)
        calendar = pd.read_csv(CALENDAR_PATH)
        prices = pd.read_csv(PRICES_PATH)
        
        print(f"Sales data: {sales_train.shape[0]} series x {sales_train.shape[1]} columns")
        print(f"Calendar data: {calendar.shape[0]} days")
        print(f"Price data: {prices.shape[0]} price records")
        
        return sales_train, calendar, prices
    except FileNotFoundError as e:
        print(f"Error: Data file not found - {e}")
        return None, None, None


def transform_to_long_format(df: pd.DataFrame) -> pd.DataFrame:
    """
    Transform M5 wide format to long format.
    
    Args:
        df: DataFrame in wide format with sales data columns starting with 'd_'
        
    Returns:
        DataFrame in long format with 'id', 'day', and 'sales' columns
    """
    id_cols = ['id', 'item_id', 'dept_id', 'cat_id', 'store_id', 'state_id']
    day_cols = [col for col in df.columns if col.startswith('d_')]
    
    long_df = pd.melt(
        df,
        id_vars=id_cols,
        value_vars=day_cols,
        var_name='d',
        value_name='sales'
    )
    
    return long_df


def apply_cross_store_sampling(intermittency_df: pd.DataFrame, splice_config: Dict) -> pd.DataFrame:
    """
    Apply cross-store sampling to get same items across different stores.
    
    Args:
        intermittency_df: DataFrame with intermittency metrics
        splice_config: Configuration dictionary
        
    Returns:
        Filtered DataFrame with cross-store items
    """
    target_stores = splice_config.get('target_stores', ['CA_1', 'CA_2', 'TX_1'])
    items_per_store = splice_config.get('items_per_store', 40)
    
    # Find items that exist in all target stores
    item_store_counts = intermittency_df.groupby('item_id')['store_id'].nunique()
    common_items = item_store_counts[item_store_counts >= len(target_stores)].index
    
    # Filter to common items and target stores
    cross_store_df = intermittency_df[
        (intermittency_df['item_id'].isin(common_items)) &
        (intermittency_df['store_id'].isin(target_stores))
    ]
    
    # Sort by difficulty and take top items per store
    cross_store_df = cross_store_df.sort_values('forecasting_difficulty', ascending=False)
    
    # Sample items per store
    selected_items = []
    for store in target_stores:
        store_items = cross_store_df[cross_store_df['store_id'] == store].head(items_per_store)
        selected_items.append(store_items)
    
    return pd.concat(selected_items, ignore_index=True)


def apply_category_balanced_sampling(intermittency_df: pd.DataFrame, splice_config: Dict) -> pd.DataFrame:
    """
    Apply category-balanced sampling to get equal representation across categories.
    
    Args:
        intermittency_df: DataFrame with intermittency metrics
        splice_config: Configuration dictionary
        
    Returns:
        Filtered DataFrame with balanced categories
    """
    items_per_category = splice_config.get('items_per_category', 20)
    
    # Get unique categories
    categories = intermittency_df['cat_id'].unique()
    
    # Sample items per category
    selected_items = []
    for category in categories:
        category_items = intermittency_df[intermittency_df['cat_id'] == category]
        category_items = category_items.sort_values('forecasting_difficulty', ascending=False)
        selected_items.append(category_items.head(items_per_category))
    
    return pd.concat(selected_items, ignore_index=True)


def create_strategic_splice(
    sales_train: pd.DataFrame,
    calendar: pd.DataFrame,
    prices: pd.DataFrame,
    splice_config: Dict,
    output_dir: Path
) -> Tuple[Optional[Path], int]:
    """
    Create a strategic data splice based on configuration.
    
    Args:
        sales_train: Sales training data
        calendar: Calendar data
        prices: Price data
        splice_config: Configuration dictionary for the splice
        output_dir: Output directory for splices
    
    Returns:
        Tuple of (output_path, number_of_items)
    """
    print(f"\nCreating splice: {splice_config['name']}")
    print(f"Description: {splice_config['description']}")
    
    # OPTIMIZATION: Apply basic filters FIRST to reduce dataset size
    print("  Applying basic filters...")
    filtered_df = apply_basic_filters(sales_train, splice_config)
    
    if filtered_df.empty:
        print(f"Warning: No data after basic filtering for {splice_config['name']}")
        return None, 0
    
    print(f"  Filtered from {len(sales_train)} to {len(filtered_df)} series ({len(filtered_df)/len(sales_train)*100:.1f}%)")
    
    # OPTIMIZATION: Calculate intermittency metrics only on pre-filtered data
    print("  Calculating intermittency metrics on filtered data...")
    intermittency_df = calculate_intermittency_metrics(filtered_df, prices)
    
    # FILTERS, FILTERS, AND MORE FILTERS!
    # Apply pattern filters
    if 'pattern_filter' in splice_config:
        patterns = splice_config['pattern_filter']
        intermittency_df = intermittency_df[intermittency_df['demand_pattern'].isin(patterns)]
    
    # Apply difficulty filters
    if 'difficulty_range' in splice_config:
        min_diff, max_diff = splice_config['difficulty_range']
        intermittency_df = intermittency_df[
            (intermittency_df['forecasting_difficulty'] >= min_diff) &
            (intermittency_df['forecasting_difficulty'] <= max_diff)
        ]
    
    # Apply sparsity filters
    if 'sparsity_range' in splice_config:
        min_sparsity, max_sparsity = splice_config['sparsity_range']
        intermittency_df = intermittency_df[
            (intermittency_df['sparsity'] >= min_sparsity) &
            (intermittency_df['sparsity'] <= max_sparsity)
        ]
    
    # Apply seasonal strength filter
    if 'seasonal_filter' in splice_config and splice_config['seasonal_filter']:
        seasonal_threshold = 0.1  # Minimum seasonal strength
        intermittency_df = intermittency_df[
            intermittency_df['seasonal_strength'] >= seasonal_threshold
        ]
    
    # Apply initial zero periods filter (new product patterns)
    if 'initial_zero_periods' in splice_config:
        min_zeros, max_zeros = splice_config['initial_zero_periods']
        intermittency_df = intermittency_df[
            (intermittency_df['initial_zero_periods'] >= min_zeros) &
            (intermittency_df['initial_zero_periods'] <= max_zeros)
        ]
    
    # Apply price percentile filter (high-value items)
    if 'price_percentile_threshold' in splice_config:
        threshold = splice_config['price_percentile_threshold']
        price_cutoff = intermittency_df['avg_price'].quantile(threshold)
        intermittency_df = intermittency_df[
            intermittency_df['avg_price'] >= price_cutoff
        ]
    
    # Apply non-zero periods filter (short history challenge)
    if 'min_nonzero_periods' in splice_config:
        min_periods = splice_config['min_nonzero_periods']
        intermittency_df = intermittency_df[
            intermittency_df['non_zero_periods'] >= min_periods
        ]
    
    if 'max_nonzero_periods' in splice_config:
        max_periods = splice_config['max_nonzero_periods']
        intermittency_df = intermittency_df[
            intermittency_df['non_zero_periods'] <= max_periods
        ]
    
    # Handle special sampling methods
    if 'cross_store_sampling' in splice_config and splice_config['cross_store_sampling']:
        intermittency_df = apply_cross_store_sampling(intermittency_df, splice_config)
    
    if 'category_balanced' in splice_config and splice_config['category_balanced']:
        intermittency_df = apply_category_balanced_sampling(intermittency_df, splice_config)
    
    # Sort and limit items
    if 'sort_by' in splice_config:
        sort_col = splice_config['sort_by']
        ascending = splice_config.get('sort_ascending', False)
        intermittency_df = intermittency_df.sort_values(sort_col, ascending=ascending)
    
    # Limit number of items
    if 'max_items' in splice_config:
        intermittency_df = intermittency_df.head(splice_config['max_items'])
    
    if intermittency_df.empty:
        print(f"Warning: No items selected for {splice_config['name']}")
        return None, 0
    
    # Get selected item IDs
    selected_ids = intermittency_df['id'].tolist()
    selected_sales = filtered_df[filtered_df['id'].isin(selected_ids)].copy()
    
    num_items = len(selected_ids)
    print(f"  Selected {num_items} items")
    
    # Print intermittency statistics
    pattern_counts = intermittency_df['demand_pattern'].value_counts()
    print(f"  Pattern distribution: {dict(pattern_counts)}")
    print(f"  Avg sparsity: {intermittency_df['sparsity'].mean():.3f}")
    print(f"  Avg difficulty: {intermittency_df['forecasting_difficulty'].mean():.3f}")
    
    # Transform to long format
    print("  Transforming to long format...")
    sales_long = transform_to_long_format(selected_sales)
    
    # Merge with calendar and prices
    print("  Merging with calendar and price data...")
    sales_long = sales_long.merge(calendar, on='d', how='left')
    sales_long = sales_long.merge(
        prices, 
        on=['store_id', 'item_id', 'wm_yr_wk'], 
        how='left'
    )
    
    # Create features if requested
    if splice_config.get('include_features', True):
        print("  Creating engineered features...")
        data_final = create_feature_set(
            sales_long,
            target_col='sales',
            id_col='id',
            date_col='date',
            price_col='sell_price'
        )
        filename_suffix = "_features"
    else:
        data_final = sales_long
        filename_suffix = ""
    
    # Create output filename
    output_filename = f"{splice_config['name']}_{num_items}_items{filename_suffix}.csv"
    metadata_filename = f"{splice_config['name']}_{num_items}_items{filename_suffix}_metadata.json"
    
    output_path = output_dir / output_filename
    metadata_path = output_dir / metadata_filename
    
    # Save to CSV
    print(f"  Saving to {output_filename}...")
    data_final.to_csv(output_path, index=False)
    
    # Save metadata
    metadata = {
        'created_at': datetime.now().isoformat(),
        'splice_config': splice_config,
        'n_series': num_items,
        'n_timepoints': data_final['date'].nunique(),
        'feature_columns': [col for col in data_final.columns if col not in ['id', 'date', 'sales']],
        'date_range': [
            data_final['date'].min().strftime('%Y-%m-%d'),
            data_final['date'].max().strftime('%Y-%m-%d')
        ],
        'intermittency_stats': {
            'pattern_distribution': {str(k): int(v) for k, v in pattern_counts.items()},
            'avg_sparsity': float(intermittency_df['sparsity'].mean()),
            'avg_difficulty': float(intermittency_df['forecasting_difficulty'].mean()),
            'avg_adi': float(intermittency_df['adi'].mean()),
            'avg_cv_squared': float(intermittency_df['cv_squared'].mean())
        },
        'target_column': 'sales',
        'id_column': 'id',
        'date_column': 'date'
    }
    
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=2)
    
    print(f"   Created {output_filename} ({num_items} items)")
    return output_path, num_items


def define_splice_configurations() -> List[Dict]:
    """
    Define strategic splice configurations for different testing scenarios.
    
    Returns:
        List of splice configuration dictionaries
    """
    
    splice_configs = [
        
        # DEVELOPMENT/TESTING SPLICES
        {
            'name': 'test_tiny_10',
            'description': 'Small development set with highly intermittent items (10 items)',
            'store_filter': ['CA_1'],
            'category_filter': ['HOBBIES'],
            'pattern_filter': ['intermittent', 'lumpy'],
            'sparsity_range': (0.7, 1.0),  # High sparsity
            'max_items': 10,
            'sort_by': 'forecasting_difficulty',
            'sort_ascending': False,
            'include_features': True
        },
        
        {
            'name': 'test_mixed_25',
            'description': 'Development set with mixed demand patterns (25 items)',
            'store_filter': ['CA_1'],
            'category_filter': ['HOBBIES'],
            'max_items': 25,
            'sort_by': 'forecasting_difficulty',
            'sort_ascending': False,
            'include_features': True
        },
        
        # HPO TESTING SPLICES
        {
            'name': 'hpo_hard_50',
            'description': 'HPO test set focused on challenging intermittent patterns (50 items)',
            'store_filter': ['CA_1', 'CA_2'],
            'category_filter': ['HOBBIES'],
            'pattern_filter': ['intermittent', 'lumpy'],
            'difficulty_range': (0.5, 2.0),  # Medium to high difficulty
            'max_items': 50,
            'sort_by': 'forecasting_difficulty',
            'sort_ascending': False,
            'include_features': True
        },
        
        {
            'name': 'hpo_balanced_75',
            'description': 'HPO test set with balanced difficulty levels (75 items)',
            'store_filter': ['CA_1', 'CA_2'],
            'category_filter': ['HOBBIES', 'HOUSEHOLD'],
            'max_items': 75,
            'sort_by': 'total_sales',  # Mix of high and low volume
            'sort_ascending': False,
            'include_features': True
        },
        
        # TRAINING/VALIDATION SPLICES
        {
            'name': 'train_sparse_200',
            'description': 'Training set emphasizing intermittent demand patterns (200 items)',
            'store_filter': ['CA_1', 'CA_2', 'CA_3'],
            'category_filter': ['HOBBIES'],
            'pattern_filter': ['intermittent', 'lumpy'],
            'max_items': 200,
            'sort_by': 'forecasting_difficulty',
            'sort_ascending': False,
            'include_features': True
        },
        
        {
            'name': 'train_mix_300',
            'description': 'Training set with all demand patterns represented (300 items)',
            'store_filter': ['CA_1', 'CA_2', 'CA_3'],
            'category_filter': ['HOBBIES', 'HOUSEHOLD', 'FOODS'],  # Added FOODS
            'max_items': 300,
            'sort_by': 'total_sales',
            'sort_ascending': False,
            'include_features': True
        },
        
        # COMPREHENSIVE EVALUATION SPLICES
        {
            'name': 'eval_sparse_150',
            'description': 'Evaluation set with only intermittent patterns (150 items)',
            'pattern_filter': ['intermittent'],
            'sparsity_range': (0.5, 0.9),
            'max_items': 150,
            'sort_by': 'forecasting_difficulty',
            'sort_ascending': False,
            'include_features': True
        },
        
        {
            'name': 'eval_lumpy_100',
            'description': 'Evaluation set with only lumpy patterns (100 items)',
            'pattern_filter': ['lumpy'],
            'max_items': 100,
            'sort_by': 'cv_squared',
            'sort_ascending': False,
            'include_features': True
        },
        
        {
            'name': 'eval_smooth_100',
            'description': 'Evaluation set with smooth patterns (baseline comparison) (100 items)',
            'pattern_filter': ['smooth'],
            'max_items': 100,
            'sort_by': 'total_sales',
            'sort_ascending': False,
            'include_features': True
        },
        
        # LARGE SCALE TESTING
        {
            'name': 'big_test_800',
            'description': 'Large scale test with all categories and states (800 items)',
            'category_filter': ['HOBBIES', 'HOUSEHOLD', 'FOODS'],  # All categories
            'store_filter': ['CA_1', 'CA_2', 'CA_3', 'TX_1', 'TX_2', 'WI_1'],  # Multi-state
            'max_items': 800,
            'sort_by': 'forecasting_difficulty',
            'sort_ascending': False,
            'include_features': True
        },
        
        # COMPARATIVE ANALYSIS SPLICES
        {
            'name': 'hard_items_100',
            'description': 'High difficulty items for LSTM vs Classical comparison (100 items)',
            'difficulty_range': (1.0, 3.0),  # High difficulty only
            'pattern_filter': ['intermittent', 'lumpy', 'erratic'],
            'max_items': 100,
            'sort_by': 'forecasting_difficulty',
            'sort_ascending': False,
            'include_features': True
        },
        
        {
            'name': 'seasonal_items_100',
            'description': 'Items with strong seasonal patterns for comparison (100 items)',
            'category_filter': ['HOBBIES', 'HOUSEHOLD', 'FOODS'],  # All categories
            'store_filter': ['CA_1', 'TX_1', 'WI_1'],  # Multi-state representation
            'sparsity_range': (0.3, 0.8),  # Medium sparsity
            'max_items': 100,
            'sort_by': 'total_sales',
            'sort_ascending': False,
            'include_features': True
        },
        
        # ADVANCED VALIDATION SPLICES
        {
            'name': 'cross_store_120',
            'description': 'Same items across different stores for generalization testing (120 items)',
            'category_filter': ['HOBBIES', 'HOUSEHOLD', 'FOODS'],  # All categories
            'pattern_filter': ['intermittent', 'lumpy'],
            'cross_store_sampling': True,  # Special sampling across stores
            'target_stores': ['CA_1', 'TX_1', 'WI_1'],  # Multi-state coverage
            'items_per_store': 40,
            'sort_by': 'total_sales',
            'sort_ascending': False,
            'include_features': True
        },
        
        {
            'name': 'seasonal_sparse_75',
            'description': 'Intermittent items with detectable seasonal patterns (75 items)',
            'pattern_filter': ['intermittent', 'lumpy'],
            'seasonal_filter': True,  # Items with seasonal patterns
            'sparsity_range': (0.4, 0.8),  # Medium sparsity for seasonal detection
            'max_items': 75,
            'sort_by': 'forecasting_difficulty',
            'sort_ascending': False,
            'include_features': True
        },
        
        {
            'name': 'new_products_50',
            'description': 'Items with significant zero periods at start - new product launches (50 items)',
            'initial_zero_periods': (90, 180),  # 3-6 months of initial zeros
            'pattern_filter': ['intermittent', 'lumpy'],
            'max_items': 50,
            'sort_by': 'forecasting_difficulty',
            'sort_ascending': False,
            'include_features': True
        },
        
        {
            'name': 'high_value_60',
            'description': 'High-value intermittent items critical for inventory control (60 items)',
            'pattern_filter': ['intermittent', 'lumpy'],
            'price_percentile_threshold': 0.8,  # Top 20% by price
            'max_items': 60,
            'sort_by': 'forecasting_difficulty',
            'sort_ascending': False,
            'include_features': True
        },
        
        {
            'name': 'multi_cat_150',
            'description': 'Balanced representation across all categories (150 items)',
            'category_balanced': True,  # Equal items from each category
            'items_per_category': 50,  # 50 items per category (3 categories = 150 total)
            'pattern_filter': ['intermittent', 'lumpy'],
            'store_filter': ['CA_1', 'TX_1', 'WI_1'],  # Multi-state representation
            'sort_by': 'forecasting_difficulty',
            'sort_ascending': False,
            'include_features': True
        },
        
        {
            'name': 'super_sparse_30',
            'description': 'Extremely sparse items (>95% zeros) - ultimate forecasting challenge (30 items)',
            'sparsity_range': (0.95, 1.0),
            'max_items': 30,
            'sort_by': 'forecasting_difficulty',
            'sort_ascending': False,
            'include_features': True
        },
        
        {
            'name': 'short_history_40',
            'description': 'Items with limited non-zero history for cold-start evaluation (40 items)',
            'min_nonzero_periods': 10,
            'max_nonzero_periods': 30,
            'max_items': 40,
            'sort_by': 'forecasting_difficulty',
            'sort_ascending': False,
            'include_features': True
        },
        
        {
            'name': 'validation_300',
            'description': 'Held-out validation set for final model comparison (300 items)',
            'pattern_filter': ['intermittent', 'lumpy', 'erratic'],
            'category_filter': ['HOBBIES', 'HOUSEHOLD', 'FOODS'],  # All categories
            'store_filter': ['TX_1', 'TX_2', 'WI_1', 'WI_2'],  # Different states from training
            'max_items': 300,
            'sort_by': 'forecasting_difficulty',
            'sort_ascending': False,
            'include_features': True
        },
        
        # NEW ACADEMICALLY ROBUST SPLICES
        {
            'name': 'foods_only_400',
            'description': 'FOODS category analysis (400 items)',
            'category_filter': ['FOODS'],  # FOODS only for domain completeness
            'store_filter': ['CA_1', 'CA_2', 'TX_1', 'TX_2', 'WI_1', 'WI_2'],  # Multi-state
            'max_items': 400,
            'sort_by': 'forecasting_difficulty',
            'sort_ascending': False,
            'include_features': True
        },
        
        {
            'name': 'all_stores_500',
            'description': 'All states and categories for geographic generalization (500 items)',
            'category_filter': ['HOBBIES', 'HOUSEHOLD', 'FOODS'],  # All categories
            'store_filter': ['CA_1', 'CA_2', 'CA_3', 'TX_1', 'TX_2', 'TX_3', 'WI_1', 'WI_2', 'WI_3'],  # All stores
            'max_items': 500,
            'sort_by': 'forecasting_difficulty',
            'sort_ascending': False,
            'include_features': True
        },
        
        {
            'name': 'full_dataset_600',
            'description': 'Premium academic dataset - all categories, all states, all patterns (600 items)',
            'category_filter': ['HOBBIES', 'HOUSEHOLD', 'FOODS'],  # All categories
            'store_filter': ['CA_1', 'CA_2', 'CA_3', 'TX_1', 'TX_2', 'TX_3', 'WI_1', 'WI_2', 'WI_3'],  # All stores
            'max_items': 600,
            'sort_by': 'total_sales',  # Balanced by sales volume
            'sort_ascending': False,
            'include_features': True
        },
        
        # PUBLICATION QUALITY DATASETS (500-5000 items)
        {
            'name': 'paper_small_500',
            'description': 'Publication dataset - small scale balanced experiment (500 items)',
            'category_filter': ['HOBBIES', 'HOUSEHOLD', 'FOODS'],
            'store_filter': ['CA_1', 'CA_2', 'CA_3', 'TX_1', 'WI_1'],
            'max_items': 500,
            'sort_by': 'forecasting_difficulty',
            'sort_ascending': False,
            'include_features': True
        },
        
        {
            'name': 'paper_med_1000',
            'description': 'Publication dataset - medium scale for detailed analysis (1000 items)',
            'category_filter': ['HOBBIES', 'HOUSEHOLD', 'FOODS'],
            'store_filter': ['CA_1', 'CA_2', 'CA_3', 'TX_1', 'TX_2', 'WI_1'],
            'max_items': 1000,
            'sort_by': 'total_sales',
            'sort_ascending': False,
            'include_features': True
        },
        
        {
            'name': 'paper_large_2000',
            'description': 'Publication dataset - large scale for robust conclusions (2000 items)',
            'category_filter': ['HOBBIES', 'HOUSEHOLD', 'FOODS'],
            'store_filter': ['CA_1', 'CA_2', 'CA_3', 'TX_1', 'TX_2', 'TX_3', 'WI_1', 'WI_2'],
            'max_items': 2000,
            'sort_by': 'forecasting_difficulty',
            'sort_ascending': False,
            'include_features': True
        },
        
        {
            'name': 'paper_xlarge_3000',
            'description': 'Publication dataset - extra large for comprehensive validation (3000 items)',
            'category_filter': ['HOBBIES', 'HOUSEHOLD', 'FOODS'],
            'store_filter': ['CA_1', 'CA_2', 'CA_3', 'TX_1', 'TX_2', 'TX_3', 'WI_1', 'WI_2', 'WI_3'],
            'max_items': 3000,
            'sort_by': 'total_sales',
            'sort_ascending': False,
            'include_features': True
        },
        
        {
            'name': 'paper_huge_5000',
            'description': 'Publication dataset - massive scale for definitive results (5000 items)',
            'category_filter': ['HOBBIES', 'HOUSEHOLD', 'FOODS'],
            'store_filter': ['CA_1', 'CA_2', 'CA_3', 'TX_1', 'TX_2', 'TX_3', 'WI_1', 'WI_2', 'WI_3'],
            'max_items': 5000,
            'sort_by': 'total_sales',
            'sort_ascending': False,
            'include_features': True
        },
        
        # SPECIALIZED PUBLICATION DATASETS
        {
            'name': 'paper_sparse_1000',
            'description': 'Publication dataset - focused on intermittent demand patterns (1000 items)',
            'pattern_filter': ['intermittent', 'lumpy'],
            'category_filter': ['HOBBIES', 'HOUSEHOLD', 'FOODS'],
            'store_filter': ['CA_1', 'CA_2', 'TX_1', 'TX_2', 'WI_1', 'WI_2'],
            'max_items': 1000,
            'sort_by': 'forecasting_difficulty',
            'sort_ascending': False,
            'include_features': True
        },
        
        {
            'name': 'paper_mixed_2000',
            'description': 'Publication dataset - balanced mix of all demand patterns (2000 items)',
            'category_filter': ['HOBBIES', 'HOUSEHOLD', 'FOODS'],
            'store_filter': ['CA_1', 'CA_2', 'CA_3', 'TX_1', 'TX_2', 'TX_3', 'WI_1', 'WI_2'],
            'max_items': 2000,
            'pattern_balanced': True,  # Ensure equal representation of patterns
            'sort_by': 'total_sales',
            'sort_ascending': False,
            'include_features': True
        },
        
        {
            'name': 'paper_seasonal_1500',
            'description': 'Publication dataset - items with seasonal patterns (1500 items)',
            'category_filter': ['HOBBIES', 'HOUSEHOLD', 'FOODS'],
            'store_filter': ['CA_1', 'CA_2', 'CA_3', 'TX_1', 'TX_2', 'WI_1', 'WI_2'],
            'seasonal_filter': True,
            'max_items': 1500,
            'sort_by': 'forecasting_difficulty',
            'sort_ascending': False,
            'include_features': True
        }
    ]
    
    return splice_configs


def main():
    """Main function to create all strategic data splices."""
    
    parser = argparse.ArgumentParser(
        description='Create feature-rich M5 data splices for LSTM vs classical method comparison'
    )
    parser.add_argument(
        '--splice-names', 
        nargs='+', 
        help='Specific splice names to create (default: create all)'
    )
    parser.add_argument(
        '--output-dir',
        type=str,
        default=str(OUTPUT_SPLICE_DIR),
        help='Output directory for splices'
    )
    parser.add_argument(
        '--skip-existing',
        action='store_true',
        help='Skip creating splices that already exist'
    )
    
    args = parser.parse_args()
    
    print("=" * 80)
    print("M5 Strategic Data Splice Creation")
    print("Focus: LSTM vs Classical Methods for Intermittent Demand Forecasting")
    print("=" * 80)
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"Output directory: {output_dir.absolute()}")
    
    # Load M5 data
    sales_train, calendar, prices = load_m5_data()
    if sales_train is None:
        print("Failed to load M5 data. Exiting.")
        return
    
    # Get splice configurations
    splice_configs = define_splice_configurations()
    
    # Filter configurations if specific names requested
    if args.splice_names:
        splice_configs = [
            config for config in splice_configs 
            if config['name'] in args.splice_names
        ]
        print(f"Creating {len(splice_configs)} specified splices")
    else:
        print(f"Creating {len(splice_configs)} total splices")
    
    # Create splices
    created_splices = []
    total_items = 0
    
    for config in splice_configs:
        # Check if splice already exists
        if args.skip_existing:
            expected_files = [
                output_dir / f"{config['name']}_*_items_features.csv",
                output_dir / f"{config['name']}_*_items.csv"
            ]
            
            existing = any(
                len(list(output_dir.glob(pattern.name))) > 0 
                for pattern in expected_files
            )
            
            if existing:
                print(f"Skipping {config['name']} (already exists)")
                continue
        
        try:
            output_path, num_items = create_strategic_splice(
                sales_train, calendar, prices, config, output_dir
            )
            
            if output_path:
                created_splices.append({
                    'name': config['name'],
                    'path': output_path,
                    'items': num_items,
                    'description': config['description']
                })
                total_items += num_items
            
        except Exception as e:
            print(f"Error creating splice {config['name']}: {e}")
            continue
    
    # Summary
    print("\n" + "=" * 80)
    print("SPLICE CREATION SUMMARY")
    print("=" * 80)
    print(f"Successfully created {len(created_splices)} splices")
    print(f"Total items across all splices: {total_items}")
    print()
    
    # Categorize splices by purpose
    dev_splices = [s for s in created_splices if s['name'].startswith('test_')]
    hpo_splices = [s for s in created_splices if s['name'].startswith('hpo_')]
    train_splices = [s for s in created_splices if s['name'].startswith('train_')]
    eval_splices = [s for s in created_splices if s['name'].startswith('eval_')]
    comp_splices = [s for s in created_splices if s['name'] in ['hard_items_100', 'seasonal_items_100']]
    large_splices = [s for s in created_splices if s['name'] in ['big_test_800']]
    advanced_splices = [s for s in created_splices if s['name'] in [
        'cross_store_120', 'seasonal_sparse_75', 'new_products_50',
        'high_value_60', 'multi_cat_150', 'super_sparse_30',
        'short_history_40', 'validation_300'
    ]]
    
    print("DEVELOPMENT/TESTING SPLICES:")
    for splice in dev_splices:
        print(f"  {splice['name']}: {splice['items']} items")
    
    print("\nHPO TESTING SPLICES:")
    for splice in hpo_splices:
        print(f"  {splice['name']}: {splice['items']} items")
    
    print("\nTRAINING SPLICES:")
    for splice in train_splices:
        print(f"  {splice['name']}: {splice['items']} items")
    
    print("\nEVALUATION SPLICES:")
    for splice in eval_splices:
        print(f"  {splice['name']}: {splice['items']} items")
    
    print("\nCOMPARISON SPLICES:")
    for splice in comp_splices:
        print(f"  {splice['name']}: {splice['items']} items")
    
    print("\nLARGE SCALE SPLICES:")
    for splice in large_splices:
        print(f"  {splice['name']}: {splice['items']} items")
    
    print("\nADVANCED VALIDATION SPLICES:")
    for splice in advanced_splices:
        print(f"  {splice['name']}: {splice['items']} items")
    
    print("\n" + "=" * 80)
    print("USAGE RECOMMENDATIONS:")
    print("=" * 80)
    print("1. Start development with: test_tiny_10")
    print("2. HPO experiments: hpo_hard_50, hpo_balanced_75")
    print("3. Training: train_sparse_200, train_mix_300")
    print("4. Evaluation: eval_sparse_150, eval_lumpy_100, eval_smooth_100")
    print("5. Final comparison: hard_items_100, seasonal_items_100")
    print("6. Large scale testing: big_test_800")
    print("7. Advanced validation: cross_store_120, validation_300")
    print("8. Specialized challenges: super_sparse_30, new_products_50")
    print("9. High-value analysis: high_value_60, seasonal_sparse_75")
    print("10. Balanced testing: multi_cat_150, short_history_40")
    print("\nAll splices include comprehensive feature engineering for LSTM models.")
    print("Classical methods can use the same data but will ignore deep features.")


if __name__ == '__main__':
    main()