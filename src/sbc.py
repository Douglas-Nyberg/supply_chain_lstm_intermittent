#////////////////////////////////////////////////////////////////////////////////#
# File:         sbc.py                                                           #
# Author:       Douglas Nyberg                                                   #
# Email:        douglas1.nyberg@gmail.com                                        #
# Date:         2025-04-10                                                       #
#////////////////////////////////////////////////////////////////////////////////#

"""
supply chain benchmarking (sbc) utilities for analyzing intermittent demand patterns
in the m5 dataset and supporting lstm vs classical forecasting method comparisons.

provides utilities for intermittent demand pattern classification, forecasting
difficulty assessment, and benchmarking support for model comparison.
"""

import warnings
from typing import Dict, List, Tuple, Optional, Union

import numpy as np
import pandas as pd
from scipy import stats

# Suppress pandas warnings for cleaner output
warnings.filterwarnings('ignore', category=pd.errors.SettingWithCopyWarning)


class IntermittentDemandAnalyzer:
    """
    analyzer for intermittent demand patterns following crostons taxonomy
    and extended metrics for supply chain forecasting evaluation.
    """
    
    def __init__(self):
        """initialize analyzer with standard thresholds"""
        # standard croston classification thresholds
        self.adi_threshold = 1.32  # Average Demand Interval threshold
        self.cv_squared_threshold = 0.49  # Coefficient of Variation squared threshold
        
    def calculate_adi(self, demand_series: np.ndarray) -> float:
        """
        calculate average demand interval (adi).
        
        args:
            demand_series: Array of demand values
            
        returns:
            ADI value
        """
        non_zero_indices = np.where(demand_series > 0)[0]
        
        if len(non_zero_indices) <= 1:
            return len(demand_series)  # if 0 or 1 demand periods, return total length
        
        intervals = np.diff(non_zero_indices)
        return np.mean(intervals) if len(intervals) > 0 else len(demand_series)
    
    def calculate_cv_squared(self, demand_series: np.ndarray) -> float:
        """
        calculate squared coefficient of variation for non-zero demands.
        
        args:
            demand_series: Array of demand values
            
        returns:
            CV� value
        """
        non_zero_demands = demand_series[demand_series > 0]
        
        if len(non_zero_demands) <= 1:
            return 0.0
        
        mean_demand = np.mean(non_zero_demands)
        std_demand = np.std(non_zero_demands)
        
        if mean_demand == 0:
            return 0.0
        
        cv = std_demand / mean_demand
        return cv ** 2
    
    def classify_demand_pattern(self, demand_series: np.ndarray) -> str:
        """
        classify demand pattern using crostons taxonomy.
        
        args:
            demand_series: Array of demand values
            
        returns:
            Pattern classification: 'smooth', 'intermittent', 'erratic', or 'lumpy'
        """
        adi = self.calculate_adi(demand_series)
        cv_squared = self.calculate_cv_squared(demand_series)
        
        if adi <= self.adi_threshold and cv_squared < self.cv_squared_threshold:
            return 'smooth'
        elif adi > self.adi_threshold and cv_squared < self.cv_squared_threshold:
            return 'intermittent'
        elif adi <= self.adi_threshold and cv_squared >= self.cv_squared_threshold:
            return 'erratic'
        else:  # adi > threshold and cv_squared >= threshold
            return 'lumpy'
    
    def calculate_sparsity(self, demand_series: np.ndarray) -> float:
        """
        calculate sparsity (proportion of zero demand periods).
        
        args:
            demand_series: Array of demand values
            
        returns:
            Sparsity value (0.0 = no zeros, 1.0 = all zeros)
        """
        return np.sum(demand_series == 0) / len(demand_series)
    
    def calculate_demand_rate(self, demand_series: np.ndarray) -> float:
        """
        calculate demand rate (proportion of non-zero demand periods).
        
        args:
            demand_series: Array of demand values
            
        returns:
            Demand rate value (0.0 = no demand, 1.0 = constant demand)
        """
        return np.sum(demand_series > 0) / len(demand_series)
    
    def calculate_forecasting_difficulty(self, demand_series: np.ndarray) -> float:
        """
        calculate composite forecasting difficulty score.
        
        args:
            demand_series: Array of demand values
            
        returns:
            Difficulty score (higher = more difficult to forecast)
        """
        sparsity = self.calculate_sparsity(demand_series)
        cv_squared = self.calculate_cv_squared(demand_series)
        adi = self.calculate_adi(demand_series)
        
        # composite score combining sparsity, variability, and irregularity
        difficulty = sparsity * cv_squared + (adi / 30.0)
        
        return difficulty
    
    def analyze_series(self, demand_series: np.ndarray, series_id: str = None) -> Dict:
        """
        perform comprehensive analysis of demand series.
        
        args:
            demand_series: Array of demand values
            series_id: Optional identifier for the series
            
        returns:
            Dictionary with all analysis metrics
        """
        non_zero_demands = demand_series[demand_series > 0]
        
        analysis = {
            'series_id': series_id,
            'total_periods': len(demand_series),
            'total_demand': np.sum(demand_series),
            'non_zero_periods': len(non_zero_demands),
            'zero_periods': np.sum(demand_series == 0),
            'sparsity': self.calculate_sparsity(demand_series),
            'demand_rate': self.calculate_demand_rate(demand_series),
            'adi': self.calculate_adi(demand_series),
            'cv_squared': self.calculate_cv_squared(demand_series),
            'pattern': self.classify_demand_pattern(demand_series),
            'forecasting_difficulty': self.calculate_forecasting_difficulty(demand_series),
            'mean_demand': np.mean(non_zero_demands) if len(non_zero_demands) > 0 else 0,
            'std_demand': np.std(non_zero_demands) if len(non_zero_demands) > 0 else 0,
            'median_demand': np.median(non_zero_demands) if len(non_zero_demands) > 0 else 0,
            'max_demand': np.max(demand_series),
            'min_demand': np.min(demand_series),
            'demand_range': np.max(demand_series) - np.min(demand_series)
        }
        
        return analysis


class M5DatasetAnalyzer:
    """
    Specialized analyzer for M5 competition dataset structure and patterns.
    """
    
    def __init__(self):
        """initialize m5 dataset analyzer"""
        self.demand_analyzer = IntermittentDemandAnalyzer()
        
    def analyze_m5_sales_data(self, sales_df: pd.DataFrame) -> pd.DataFrame:
        """
        analyze m5 sales data for intermittent patterns.
        
        args:
            sales_df: M5 sales dataframe in wide format
            
        returns:
            DataFrame with analysis results for each series
        """
        day_cols = [col for col in sales_df.columns if col.startswith('d_')]
        
        results = []
        for _, row in sales_df.iterrows():
            series_id = row['id']
            demand_series = row[day_cols].values
            
            # Perform analysis
            analysis = self.demand_analyzer.analyze_series(
                demand_series, 
                series_id=series_id
            )
            
            # Add M5-specific metadata
            analysis.update({
                'item_id': row['item_id'],
                'dept_id': row['dept_id'],
                'cat_id': row['cat_id'],
                'store_id': row['store_id'],
                'state_id': row['state_id']
            })
            
            results.append(analysis)
        
        return pd.DataFrame(results)
    
    def generate_pattern_summary(self, analysis_df: pd.DataFrame) -> Dict:
        """
        generate summary statistics for demand patterns.
        
        args:
            analysis_df: DataFrame from analyze_m5_sales_data
            
        returns:
            Dictionary with pattern summary statistics
        """
        pattern_counts = analysis_df['pattern'].value_counts()
        
        summary = {
            'total_series': len(analysis_df),
            'pattern_distribution': dict(pattern_counts),
            'pattern_percentages': dict(pattern_counts / len(analysis_df) * 100),
            'avg_sparsity': analysis_df['sparsity'].mean(),
            'avg_adi': analysis_df['adi'].mean(),
            'avg_cv_squared': analysis_df['cv_squared'].mean(),
            'avg_difficulty': analysis_df['forecasting_difficulty'].mean(),
            'sparsity_quartiles': analysis_df['sparsity'].quantile([0.25, 0.5, 0.75]).to_dict(),
            'difficulty_quartiles': analysis_df['forecasting_difficulty'].quantile([0.25, 0.5, 0.75]).to_dict()
        }
        
        # Pattern-specific statistics
        for pattern in pattern_counts.index:
            pattern_data = analysis_df[analysis_df['pattern'] == pattern]
            summary[f'{pattern}_stats'] = {
                'count': len(pattern_data),
                'avg_sparsity': pattern_data['sparsity'].mean(),
                'avg_adi': pattern_data['adi'].mean(),
                'avg_cv_squared': pattern_data['cv_squared'].mean(),
                'avg_difficulty': pattern_data['forecasting_difficulty'].mean()
            }
        
        return summary
    
    def select_items_by_criteria(
        self,
        analysis_df: pd.DataFrame,
        patterns: List[str] = None,
        sparsity_range: Tuple[float, float] = None,
        difficulty_range: Tuple[float, float] = None,
        max_items: int = None,
        sort_by: str = 'forecasting_difficulty',
        ascending: bool = False
    ) -> pd.DataFrame:
        """
        select items based on specific criteria for testing/training.
        
        args:
            analysis_df: DataFrame from analyze_m5_sales_data
            patterns: List of patterns to include
            sparsity_range: Tuple of (min_sparsity, max_sparsity)
            difficulty_range: Tuple of (min_difficulty, max_difficulty)
            max_items: Maximum number of items to select
            sort_by: Column to sort by
            ascending: Sort order
            
        returns:
            Filtered and sorted DataFrame
        """
        filtered_df = analysis_df.copy()
        
        # apply pattern filter
        if patterns:
            filtered_df = filtered_df[filtered_df['pattern'].isin(patterns)]
        
        # apply sparsity filter
        if sparsity_range:
            min_sparsity, max_sparsity = sparsity_range
            filtered_df = filtered_df[
                (filtered_df['sparsity'] >= min_sparsity) &
                (filtered_df['sparsity'] <= max_sparsity)
            ]
        
        # apply difficulty filter
        if difficulty_range:
            min_difficulty, max_difficulty = difficulty_range
            filtered_df = filtered_df[
                (filtered_df['forecasting_difficulty'] >= min_difficulty) &
                (filtered_df['forecasting_difficulty'] <= max_difficulty)
            ]
        
        # Sort
        if sort_by in filtered_df.columns:
            filtered_df = filtered_df.sort_values(sort_by, ascending=ascending)
        
        # Limit
        if max_items:
            filtered_df = filtered_df.head(max_items)
        
        return filtered_df


def analyze_intermittent_patterns(sales_data: pd.DataFrame) -> Tuple[pd.DataFrame, Dict]:
    """
    convenience function to analyze intermittent patterns in m5 sales data.
    
    args:
        sales_data: M5 sales dataframe in wide format
        
    returns:
        Tuple of (analysis_dataframe, summary_statistics)
    """
    analyzer = M5DatasetAnalyzer()
    analysis_df = analyzer.analyze_m5_sales_data(sales_data)
    summary = analyzer.generate_pattern_summary(analysis_df)
    
    return analysis_df, summary


def select_challenging_items(
    sales_data: pd.DataFrame,
    target_patterns: List[str] = ['intermittent', 'lumpy'],
    max_items: int = 100,
    min_difficulty: float = 0.5
) -> List[str]:
    """
    select challenging items for lstm vs classical method comparision.
    
    args:
        sales_data: M5 sales dataframe in wide format
        target_patterns: Patterns to focus on
        max_items: Maximum number of items to select
        min_difficulty: Minimum difficulty threshold
        
    returns:
        List of selected item IDs
    """
    analyzer = M5DatasetAnalyzer()
    analysis_df = analyzer.analyze_m5_sales_data(sales_data)
    
    selected = analyzer.select_items_by_criteria(
        analysis_df,
        patterns=target_patterns,
        difficulty_range=(min_difficulty, float('inf')),
        max_items=max_items,
        sort_by='forecasting_difficulty',
        ascending=False
    )
    
    return selected['series_id'].tolist()


def get_benchmark_statistics(analysis_df: pd.DataFrame) -> Dict:
    """
    get benchmark statistics for model comparison evaluation.
    
    args:
        analysis_df: DataFrame from analyze_m5_sales_data
        
    returns:
        Dictionary with benchmark statistics
    """
    stats = {
        'dataset_characteristics': {
            'total_series': len(analysis_df),
            'intermittent_ratio': (analysis_df['pattern'] == 'intermittent').mean(),
            'lumpy_ratio': (analysis_df['pattern'] == 'lumpy').mean(),
            'avg_sparsity': analysis_df['sparsity'].mean(),
            'high_difficulty_ratio': (analysis_df['forecasting_difficulty'] > 1.0).mean()
        },
        'forecasting_challenges': {
            'high_sparsity_series': (analysis_df['sparsity'] > 0.7).sum(),
            'high_variability_series': (analysis_df['cv_squared'] > 1.0).sum(),
            'irregular_series': (analysis_df['adi'] > 5.0).sum(),
            'compound_difficulty_series': (analysis_df['forecasting_difficulty'] > 2.0).sum()
        },
        'expected_method_performance': {
            'classical_advantage': (analysis_df['pattern'] == 'smooth').sum(),
            'lstm_advantage': (
                (analysis_df['pattern'].isin(['intermittent', 'lumpy'])) & 
                (analysis_df['forecasting_difficulty'] > 1.0)
            ).sum(),
            'competitive_cases': (
                (analysis_df['pattern'] == 'erratic') |
                ((analysis_df['sparsity'] > 0.3) & (analysis_df['sparsity'] < 0.7))
            ).sum()
        }
    }
    
    return stats