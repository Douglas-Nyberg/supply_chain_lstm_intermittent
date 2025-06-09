#////////////////////////////////////////////////////////////////////////////////#
# File:         lstm_fitness.py                                                  #
# Author:       Douglas Nyberg                                                   #
# Email:        douglas1.nyberg@gmail.com                                        #
# Date:         2025-04-25                                                       #
#////////////////////////////////////////////////////////////////////////////////#





"""
lstm fitness evaluation for hyperparameter optimization.

for each hyperparam set from optimizer:
- loads hpo dataset (feature-rich m5 splice)
- selects subset of series if limit_series set
- for each series: prepares data, splits train/val, scales, creates sequences
- trains lstm with given hyperparams, evaluates on validation
- aggregates validation losses for fitness score
"""

# imports
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from sklearn.preprocessing import StandardScaler
import logging
from typing import Dict, Any, Tuple, List, Optional
from pathlib import Path
import sys
import random
import argparse

# project path setup
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.append(str(PROJECT_ROOT))
if str(PROJECT_ROOT / "src") not in sys.path:
    sys.path.append(str(PROJECT_ROOT / "src"))

# project modules
from src.models.lstm import LSTMModel, QuantileLoss # Import QuantileLoss for optional use
from src.data_loader import load_m5_splice
from src.feature_engineering import preprocess_m5_features_for_embeddings, get_categorical_feature_specs
# from src.evaluators.accuracy import calculate_wrmsse # Potentially too slow for HPO fitness
# from src.cross_validation.expanding_window_cv import WalkForwardValidator # CV logic can be added later
from src.utils import set_random_seed, sanitize_for_path
# We need a sequence creation function. Let's adapt the one from 3_train_lstm_models.py
# and SeriesLSTMDataset

# hpo dataset and sequence creation
class HPOSeriesLSTMDataset(Dataset):
    """pytorch dataset for single series sequences during hpo"""
    def __init__(self, X_cat_series: Dict[str, np.ndarray], X_num_series: np.ndarray, y_series: np.ndarray):
        self.X_cat_series = {key: torch.tensor(data, dtype=torch.long) for key, data in X_cat_series.items()}
        self.X_num_series = torch.tensor(X_num_series, dtype=torch.float32)
        self.y_series = torch.tensor(y_series, dtype=torch.float32)
        self.categorical_feature_keys = list(X_cat_series.keys())

    def __len__(self) -> int:
        return len(self.y_series)

    def __getitem__(self, index: int) -> Tuple[Dict[str, torch.Tensor], torch.Tensor, torch.Tensor]:
        categorical_sample = {key: self.X_cat_series[key][index] for key in self.categorical_feature_keys}
        numerical_sample = self.X_num_series[index]
        target_sample = self.y_series[index]
        return categorical_sample, numerical_sample, target_sample

def hpo_create_sequences_for_series( # renamed for clarity
    series_df_chronological: pd.DataFrame,
    categorical_col_names: List[str],
    numerical_col_names: List[str],
    target_col_name: str,
    sequence_length: int,
    forecast_horizon: int
) -> Optional[Tuple[Dict[str, np.ndarray], np.ndarray, np.ndarray]]:
    """creates feature sequences for one series"""
    # validate input data length
    if series_df_chronological.empty or len(series_df_chronological) < sequence_length + forecast_horizon:
        return None

    # extract data columns as numpy arrays
    # Handle cases where categorical or numerical columns might be empty
    if categorical_col_names:
        categorical_data_series = series_df_chronological[categorical_col_names].values
    else:
        # create empty array with correct shape for consistancy
        categorical_data_series = np.empty((len(series_df_chronological), 0), dtype=np.int32)
    
    if numerical_col_names:
        numerical_data_series = series_df_chronological[numerical_col_names].values
    else:
        # create empty array with correct shape for consistancy
        numerical_data_series = np.empty((len(series_df_chronological), 0), dtype=np.float32)
    
    target_values_series = series_df_chronological[target_col_name].values

    # create overlapping sequences
    list_X_categorical, list_X_numerical, list_y_target = [], [], []
    
    # calculate number of possible sequences
    # (length of data) - (length of X sequence) - (length of Y sequence) + 1 = num_samples
    num_possible_sequences = len(series_df_chronological) - sequence_length - forecast_horizon + 1

    for i in range(num_possible_sequences):
        # Input features sequence (X)
        categorical_sequence_slice = categorical_data_series[i : i + sequence_length]
        numerical_sequence_slice = numerical_data_series[i : i + sequence_length]

        # Target sequence (y)
        target_sequence_slice = target_values_series[i + sequence_length : i + sequence_length + forecast_horizon]

        list_X_categorical.append(categorical_sequence_slice)
        list_X_numerical.append(numerical_sequence_slice)
        list_y_target.append(target_sequence_slice)

    # check if sequences created
    # if no sequences could be created (eg series too short)
    if not list_X_categorical:
        return None

    # convert sequence lists to numpy arrays
    # shape: (num_sequences, sequence_length, num_features_of_type)
    X_cat_stacked_np = np.array(list_X_categorical, dtype=np.int32)
    X_num_stacked_np = np.array(list_X_numerical, dtype=np.float32)
    y_stacked_np = np.array(list_y_target, dtype=np.float32) # shape: (num_sequences, forecast_horizon)

    # restructure X_cat to dict
    # {feature_name: np.array_of_sequences_for_that_feature}
    X_cat_dict_final = {}
    if categorical_col_names:
        X_cat_dict_final = {
            col_name: X_cat_stacked_np[:, :, idx]
            for idx, col_name in enumerate(categorical_col_names)
        }
    
    # final validation
    # Handle edge case where no features exist but we have targets
    if not X_cat_dict_final and X_num_stacked_np.shape[-1] == 0:
        if y_stacked_np.shape[0] > 0:
            logger.warning("No categorical or numerical features resulted in sequences, but targets exsit. This configuration might be problematic for model training.")
    
    return X_cat_dict_final, X_num_stacked_np, y_stacked_np


# logging setup
logger = logging.getLogger(__name__) # Inherits config from HPO runner script

# main evaluator class
class LSTMFitnessEvaluator:
    """
    evaluates lstm hyperparams by training and validating models on subset of series.
    """
    def __init__(self,
                 splice_file_path: str,
                 forecast_horizon: int,
                 validation_days: int,
                 hpo_config: Dict[str, Any],
                 global_project_config: Any):
        
        self.splice_file_path = Path(splice_file_path)
        self.forecast_horizon = forecast_horizon
        self.validation_days = validation_days
        self.hpo_config = hpo_config
        self.config = global_project_config

        self.device = self.hpo_config.get('device', 'cuda' if torch.cuda.is_available() else 'cpu')
        logger.info(f"lstm fitness evaluator initialized. device: {self.device}, hpo data: {self.splice_file_path}")
       

        # load base hpo data - detect and handle both M5 and Monte Carlo data
        logger.info(f"loading hpo base data from: {self.splice_file_path}")
        base_hpo_df_raw = self._load_data_with_auto_detection()
        if base_hpo_df_raw.empty:
            error_msg = "failed to load hpo base data. file might be empty or path incorrect."
            logger.error(error_msg)
            raise ValueError(error_msg)
        base_hpo_df_raw['date'] = pd.to_datetime(base_hpo_df_raw['date'])

        # global feature definition and encoding
        logger.info("performing initial global feature definition and encoding for hpo dataset...")
        # this step determines the superset of all available features,
        # and it returns the dataframe with categorical features integer-encoded.
        # store this globally processed dataframe to be used for slicing.
        self.globally_processed_and_encoded_hpo_df, \
        self.global_embedding_info_dict, \
        self.global_superset_numerical_cols, \
        self.global_superset_categorical_cols = \
            self._perform_initial_global_feature_definition(base_hpo_df_raw.copy()) # pass the raw df to be processed
        
        logger.info(f"global feature definition and encoding complete: "
                    f"{len(self.global_superset_numerical_cols)} numerical, "
                    f"{len(self.global_superset_categorical_cols)} categorical features. "
                    f"processed hpo dataframe shape: {self.globally_processed_and_encoded_hpo_df.shape}")

        # select subset of series for hpo
        # use the globally_processed_and_encoded_hpo_df to get unique ids
        all_unique_ids_in_hpo_data = sorted(self.globally_processed_and_encoded_hpo_df['id'].unique())
        num_series_for_hpo_eval = self.hpo_config.get('limit_items', len(all_unique_ids_in_hpo_data))
        
        # handle none or invalid values
        if num_series_for_hpo_eval is None or num_series_for_hpo_eval <= 0: 
            num_series_for_hpo_eval = len(all_unique_ids_in_hpo_data)

        if num_series_for_hpo_eval < len(all_unique_ids_in_hpo_data):
            logger.info(f"limiting hpo evaluation to {num_series_for_hpo_eval} randomly selected series from the hpo splice.")
            hpo_subset_seed = self.hpo_config.get('random_seed', 42) 
            random.Random(hpo_subset_seed).shuffle(all_unique_ids_in_hpo_data)
            self.hpo_evaluation_series_ids = all_unique_ids_in_hpo_data[:num_series_for_hpo_eval]
        else:
            self.hpo_evaluation_series_ids = all_unique_ids_in_hpo_data
        logger.info(f"HPO will evaluate hyperparameters on {len(self.hpo_evaluation_series_ids)} series.")

    def _detect_data_type(self, df: pd.DataFrame) -> str:
        """
        Detect whether data is M5 format or Monte Carlo format based on column structure.
        
        Returns:
            'M5' for M5 competition data
            'MC' for Monte Carlo synthetic data
        """
        # M5 data has specific hierarchical columns
        m5_columns = {'item_id', 'dept_id', 'cat_id', 'store_id', 'state_id'}
        mc_indicators = {'synthetic', 'monte_carlo', 'mc_'}
        
        # Check for M5-specific columns
        if m5_columns.issubset(set(df.columns)):
            return 'M5'
        
        # Check for MC indicators in any column names or 'id' format
        if any(indicator in str(df.columns).lower() for indicator in mc_indicators):
            return 'MC'
        
        # Check for simple 'id' column (typical of MC data) vs M5's hierarchical structure
        if 'id' in df.columns and not any(col.endswith('_id') for col in df.columns if col != 'id'):
            return 'MC'
        
        # Default to M5 if unclear
        logger.warning("Could not definitively detect data type, defaulting to M5 format")
        return 'M5'

    def _load_data_with_auto_detection(self) -> pd.DataFrame:
        """
        Load data with automatic detection of M5 vs Monte Carlo format.
        """
        # First try loading as M5 splice
        try:
            df = load_m5_splice(str(self.splice_file_path))
        except Exception as e:
            # If that fails, try loading as regular CSV
            try:
                df = pd.read_csv(str(self.splice_file_path))
            except Exception as e2:
                raise ValueError(f"Failed to load data as M5 splice ({e}) or regular CSV ({e2})")
        
        # Detect data type
        self.data_type = self._detect_data_type(df)
        logger.info(f"Detected data type: {self.data_type}")
        
        return df

    def _load_monte_carlo_data_from_directory(self, data_dir: str) -> pd.DataFrame:
        """
        Load Monte Carlo data from preprocessed directory structure.
        Looks for train_data.csv, validation_data.csv, etc. and combines them.
        """
        data_path = Path(data_dir)
        
        # Look for preprocessed files
        train_file = data_path / "train_data.csv"
        val_file = data_path / "validation_data.csv"
        
        if train_file.exists():
            df = pd.read_csv(train_file)
            if val_file.exists():
                val_df = pd.read_csv(val_file)
                df = pd.concat([df, val_df], ignore_index=True)
            return df
        else:
            raise FileNotFoundError(f"No preprocessed Monte Carlo data found in {data_dir}")

    def _preprocess_monte_carlo_features(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, Dict]:
        """
        Process Monte Carlo synthetic data features without M5-specific dependencies.
        
        Returns:
            processed_df: DataFrame with engineered features
            embedding_info: Dictionary with embedding specifications
        """
        logger.info("Processing Monte Carlo features with simplified feature set")
        
        df_processed = df.copy()
        
        # Ensure required columns exist
        if 'date' not in df_processed.columns:
            raise ValueError("Monte Carlo data must have 'date' column")
        if 'id' not in df_processed.columns:
            raise ValueError("Monte Carlo data must have 'id' column")
            
        # Convert date column
        df_processed['date'] = pd.to_datetime(df_processed['date'])
        
        # Add basic calendar features (simplified version)
        df_processed['year'] = df_processed['date'].dt.year
        df_processed['month'] = df_processed['date'].dt.month
        df_processed['day'] = df_processed['date'].dt.day
        df_processed['weekday'] = df_processed['date'].dt.weekday
        df_processed['quarter'] = df_processed['date'].dt.quarter
        
        # Add cyclical encoding for temporal features
        df_processed['month_sin'] = np.sin(2 * np.pi * df_processed['month'] / 12)
        df_processed['month_cos'] = np.cos(2 * np.pi * df_processed['month'] / 12)
        df_processed['weekday_sin'] = np.sin(2 * np.pi * df_processed['weekday'] / 7)
        df_processed['weekday_cos'] = np.cos(2 * np.pi * df_processed['weekday'] / 7)
        
        # Find and standardize the target column to 'sales' (expected by fitness function)
        demand_col = None
        for col in df_processed.columns:
            if 'demand' in col.lower() or col.startswith('d_') or 'sales' in col.lower():
                demand_col = col
                break
        
        if demand_col is None:
            # Look for columns that look like daily sales data
            numeric_cols = df_processed.select_dtypes(include=[np.number]).columns
            potential_demand_cols = [col for col in numeric_cols if col not in 
                                   ['year', 'month', 'day', 'weekday', 'quarter']]
            if potential_demand_cols:
                demand_col = potential_demand_cols[0]
                logger.info(f"Using '{demand_col}' as demand column")
        
        if demand_col:
            # Rename to 'sales' for compatibility with fitness function
            if demand_col != 'sales':
                df_processed['sales'] = df_processed[demand_col]
                logger.info(f"Renamed '{demand_col}' to 'sales' for fitness function compatibility")
            
            # Add lag features
            df_processed = df_processed.sort_values(['id', 'date'])
            for lag in [1, 7, 14, 28]:
                df_processed[f'sales_lag_{lag}'] = df_processed.groupby('id')['sales'].shift(lag)
            
            # Add rolling features
            for window in [7, 14, 28]:
                df_processed[f'sales_rolling_mean_{window}'] = \
                    df_processed.groupby('id')['sales'].rolling(window).mean().reset_index(0, drop=True)
                df_processed[f'sales_rolling_std_{window}'] = \
                    df_processed.groupby('id')['sales'].rolling(window).std().reset_index(0, drop=True)
        
        # Create embedding specifications for categorical features
        embedding_info = {}
        
        # ID embedding (for item identity)
        unique_ids = df_processed['id'].nunique()
        embedding_info['id'] = {
            'vocab_size': unique_ids,
            'embedding_dim': min(50, (unique_ids + 1) // 2)
        }
        
        # Encode categorical features
        from sklearn.preprocessing import LabelEncoder
        le_id = LabelEncoder()
        df_processed['id'] = le_id.fit_transform(df_processed['id'])
        
        # Fill any NaN values created by lag/rolling features
        df_processed = df_processed.fillna(0)
        
        logger.info(f"Monte Carlo feature processing complete. Shape: {df_processed.shape}")
        logger.info(f"Features: {list(df_processed.columns)}")
        
        return df_processed, embedding_info

    def _get_monte_carlo_categorical_specs(self) -> Dict:
        """
        Get categorical feature specifications for Monte Carlo data.
        """
        return {
            'id': {
                'embedding_dim': 'auto',  # Will be calculated based on vocab size
                'description': 'Synthetic item identifier'
            }
        }

    def _perform_initial_global_feature_definition(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, Dict, List[str], List[str]]:
        """
        performs initial global feature processing to define available features and embedding structures.
        processes hpo splice data to establish superset of features and categorical embeddings.
        """
        # prepare data copy and handle dates
        df_copy_for_init = df.copy()
        if 'date' in df_copy_for_init.columns and not pd.api.types.is_datetime64_any_dtype(df_copy_for_init['date']):
            df_copy_for_init['date'] = pd.to_datetime(df_copy_for_init['date'])
        
        logger.debug(f"processing {len(df_copy_for_init)} rows across {df_copy_for_init['id'].nunique()} unique series for global feature definition")

        # process features using appropriate pipeline based on data type
        if hasattr(self, 'data_type') and self.data_type == 'MC':
            # Use Monte Carlo specific feature processing
            try:
                processed_df_for_definitions, embedding_info_full_dict = \
                    self._preprocess_monte_carlo_features(df_copy_for_init)
                logger.debug("successfully processed features using Monte Carlo feature processing")
            except Exception as e:
                logger.error(f"error during Monte Carlo feature preprocessing: {e}")
                raise ValueError(f"failed to process Monte Carlo features for hpo: {e}")
            
            # Get MC-specific categorical specifications
            categorical_specs_from_config = self._get_monte_carlo_categorical_specs()
            categorical_cols_superset = list(categorical_specs_from_config.keys())
        else:
            # Use standard M5 feature processing
            try:
                processed_df_for_definitions, _, embedding_info_full_dict = \
                    preprocess_m5_features_for_embeddings(df_copy_for_init)
                logger.debug("successfully processed features using preprocess_m5_features_for_embeddings")
            except Exception as e:
                logger.error(f"error during M5 feature preprocessing: {e}")
                raise ValueError(f"failed to process M5 features for hpo: {e}")
            
            # Get M5 categorical feature specifications
            try:
                categorical_specs_from_config = get_categorical_feature_specs()
                categorical_cols_superset = list(categorical_specs_from_config.keys())
                
                # Ensure these columns actually exist after preprocessing
                categorical_cols_superset = [
                    col for col in categorical_cols_superset 
                    if col in processed_df_for_definitions.columns
                ]
                logger.debug(f"Found {len(categorical_cols_superset)} M5 categorical columns: {categorical_cols_superset}")
                
            except Exception as e:
                logger.warning(f"Error getting M5 categorical feature specs: {e}. Using fallback detection.")
                # Fallback: detect categorical columns from embedding info
                categorical_cols_superset = list(embedding_info_full_dict.keys()) if embedding_info_full_dict else []
        
        # Ensure MC categorical columns exist after preprocessing
        if hasattr(self, 'data_type') and self.data_type == 'MC':
            categorical_cols_superset = [
                col for col in categorical_cols_superset 
                if col in processed_df_for_definitions.columns
            ]
            logger.debug(f"Found {len(categorical_cols_superset)} MC categorical columns: {categorical_cols_superset}")

        # determine numerical columns
        # Define columns to exclude from numerical features
        columns_to_exclude_from_numerical = set([
            'id', 'item_id', 'dept_id', 'cat_id', 'store_id', 'state_id',
            'd', 'date', 'sales', 'wm_yr_wk', 'evaluation'  # Core identifiers and target
        ] + categorical_cols_superset)
        
        # Filter to get numerical features
        numerical_cols_superset = []
        for col in processed_df_for_definitions.columns:
            if col not in columns_to_exclude_from_numerical:
                if pd.api.types.is_numeric_dtype(processed_df_for_definitions[col]):
                    numerical_cols_superset.append(col)
        
        logger.debug(f"Found {len(numerical_cols_superset)} numerical columns")
        
        # validate results
        if not numerical_cols_superset and not categorical_cols_superset:
            raise ValueError("No features detected after processing! Check your splice data format.")
        
        if not embedding_info_full_dict:
            logger.warning("No embedding info detected - categorical features may not work properly")
            
        # Log summary of detected features
        logger.info(f"Global feature definition complete: {len(numerical_cols_superset)} numerical, {len(categorical_cols_superset)} categorical features")
        
        return processed_df_for_definitions, embedding_info_full_dict, numerical_cols_superset, categorical_cols_superset


    def _select_active_features(self, hyperparams: Dict) -> Tuple[List[str], List[str]]:
        """
        Determines active numerical and categorical features based on hyperparameters.
        Uses comprehensive pattern matching to map hyperparameter flags to actual column names.
        """
        # start with global feature supersets
        active_numerical_cols = list(self.global_superset_numerical_cols)
        active_categorical_cols = list(self.global_superset_categorical_cols)

        # filter numerical features based on hyperparams
        
        # Price features: sell_price, price_related calculations
        if not hyperparams.get('use_price_features', True):
            active_numerical_cols = [
                col for col in active_numerical_cols 
                if not any(pattern in col.lower() for pattern in ['price', 'sell_price', 'revenue'])
            ]
            logger.debug(f"Removed price features. Remaining numerical: {len(active_numerical_cols)}")
        
        # Lag features: previous sales values
        if not hyperparams.get('use_lag_features', True):
            active_numerical_cols = [
                col for col in active_numerical_cols 
                if not col.startswith('lag_')
            ]
            logger.debug(f"Removed lag features. Remaining numerical: {len(active_numerical_cols)}")
        
        # Rolling features: moving averages, statistics
        if not hyperparams.get('use_rolling_features', True):
            active_numerical_cols = [
                col for col in active_numerical_cols 
                if not any(pattern in col for pattern in ['rolling_', 'ma_', 'ema_', 'std_', 'mean_'])
            ]
            logger.debug(f"Removed rolling features. Remaining numerical: {len(active_numerical_cols)}")
        
        # filter categorical features based on hyperparams
        
        # Calendar features: day of week, month, etc.
        if not hyperparams.get('use_calendar_features', True):
            calendar_patterns = ['wday', 'month', 'year', 'week', 'quarter', 'day']
            active_categorical_cols = [
                col for col in active_categorical_cols 
                if not any(pattern in col.lower() for pattern in calendar_patterns)
            ]
            logger.debug(f"Removed calendar features. Remaining categorical: {len(active_categorical_cols)}")
        
        # Event features: holidays, special events
        if not hyperparams.get('use_event_features', True):
            event_patterns = ['event', 'snap', 'holiday']
            active_categorical_cols = [
                col for col in active_categorical_cols 
                if not any(pattern in col.lower() for pattern in event_patterns)
            ]
            logger.debug(f"Removed event features. Remaining categorical: {len(active_categorical_cols)}")
        
        # final validation
        # Ensure we have at least some features to work with
        if not active_numerical_cols and not active_categorical_cols:
            logger.warning("All features were filtered out by hyperparameters! Adding basic features back.")
            # Add back essential features to prevent empty feature sets
            # Look for basic sales-related features
            for col in self.global_superset_numerical_cols:
                if 'sales' in col.lower() and 'lag_1' in col:
                    active_numerical_cols.append(col)
                    break
            
            # If still no features, add the first available numerical feature
            if not active_numerical_cols and self.global_superset_numerical_cols:
                active_numerical_cols.append(self.global_superset_numerical_cols[0])
        
        logger.debug(f"Feature selection complete: {len(active_numerical_cols)} numerical, {len(active_categorical_cols)} categorical")
        
        return active_numerical_cols, active_categorical_cols

    def evaluate(self, hyperparams: Dict) -> float:
        """
        Main evaluation function called by BayesianOptimizer for a set of hyperparameters.
        Returns a single fitness score (e.g., average validation loss, to be minimized).
        """
        # identify trial and set seed
        # The BayesianOptimizer class might pass a _trial_id_ if it's instrumented to do so
        current_trial_id_str = str(hyperparams.get('_trial_id_', "N/A"))
        log_friendly_hyperparams = {k: v for k,v in hyperparams.items() if k != '_trial_id_'}
        logger.info(f"--- HPO Trial {current_trial_id_str}: Evaluating Hyperparameters ---")
        logger.info(f"Current Hyperparameters: {log_friendly_hyperparams}")
        
        # Set seed for this trial for consistent data handling if any randomness involved
        # Add trial_id to base seed to ensure different trials have different seeds if base is same
        trial_seed_offset = 0
        if current_trial_id_str.replace('.', '', 1).isdigit(): # Check if it's a number
             trial_seed_offset = int(float(current_trial_id_str))
        current_trial_seed = self.hpo_config.get('random_seed', 42) + trial_seed_offset
        set_random_seed(current_trial_seed)

        # determine active features
        active_numerical_cols, active_categorical_cols = self._select_active_features(hyperparams)
        logger.debug(f"Trial {current_trial_id_str}: Active numerical features: {len(active_numerical_cols)}")
        logger.debug(f"Trial {current_trial_id_str}: Active categorical features: {len(active_categorical_cols)}")


        # perform evaluation
        if self.hpo_config.get('use_cv', False):
            logger.info(f"Trial {current_trial_id_str}: Using cross-validation with {self.hpo_config.get('cv_splits', 3)} folds")
            cv_scores = self._perform_cross_validation_evaluation(
                hyperparams,
                active_numerical_cols, 
                active_categorical_cols,
                current_trial_id_str
            )
            fitness_score = np.mean(cv_scores) if cv_scores else 1e8
            logger.debug(f"Trial {current_trial_id_str}: CV scores: {cv_scores}, Mean: {fitness_score:.6f}")
        else:
            # Perform evaluation using a single split of the HPO data for each series
            fitness_score = self._process_one_hpo_evaluation_split(
                hyperparams,
                active_numerical_cols,
                active_categorical_cols,
                current_trial_id_str # Pass trial ID for logging
            )
        
        logger.info(f"--- HPO Trial {current_trial_id_str}: Completed. Fitness = {fitness_score:.6f} ---")
        return fitness_score

    def _perform_cross_validation_evaluation(
        self,
        hyperparams: Dict,
        active_numerical_cols: List[str],
        active_categorical_cols: List[str],
        trial_id_log_prefix: str
    ) -> List[float]:
        """
        Performs time series cross-validation evaluation for more robust hyperparameter assessment.
        Uses expanding window strategy where each fold has more training data than the previous.
        """
        cv_splits = self.hpo_config.get('cv_splits', 3)
        cv_initial_train_size = self.hpo_config.get('cv_initial_train_size', 365)  # days
        
        cv_scores = []
        
        # For each series, perform CV and collect scores
        for series_id_value in self.hpo_evaluation_series_ids:
            series_full_data = self.globally_processed_and_encoded_hpo_df[
                    self.globally_processed_and_encoded_hpo_df['id'] == series_id_value].copy()
            if series_full_data.empty:
                continue
            
            series_full_data = series_full_data.sort_values('date').reset_index(drop=True)
            
            # Generate CV splits for this series
            series_cv_scores = []
            
            for fold_idx in range(cv_splits):
                # Calculate split points for expanding window
                # Each fold uses more training data than the previous
                total_days = len(series_full_data)
                val_size = self.validation_days
                
                # For expanding window: each fold starts with cv_initial_train_size + fold_idx * step_size days
                step_size = max(30, (total_days - cv_initial_train_size - val_size) // cv_splits)
                train_end_idx = min(cv_initial_train_size + fold_idx * step_size, total_days - val_size)
                
                if train_end_idx < cv_initial_train_size // 2:  # Minimum training size check
                    logger.debug(f"{trial_id_log_prefix} - Series {series_id_value}, Fold {fold_idx}: Insufficient data. Skipping.")
                    continue
                
                val_start_idx = train_end_idx
                val_end_idx = min(val_start_idx + val_size, total_days)
                
                # Create train/val splits for this fold
                fold_train_df = series_full_data.iloc[:train_end_idx].copy()
                fold_val_df = series_full_data.iloc[val_start_idx:val_end_idx].copy()
                
                if len(fold_train_df) < hyperparams.get('sequence_length', 28) + self.forecast_horizon or len(fold_val_df) < self.forecast_horizon:
                    logger.debug(f"{trial_id_log_prefix} - Series {series_id_value}, Fold {fold_idx}: Insufficient fold data. Skipping.")
                    continue
                
                # Train and evaluate on this fold
                try:
                    fold_score = self._train_and_eval_one_series_lstm_for_hpo(
                        series_id_value=f"{series_id_value}_fold{fold_idx}",
                        series_train_raw_df=fold_train_df,
                        series_val_raw_df=fold_val_df,
                        hyperparams_trial=hyperparams,
                        active_numerical_cols=active_numerical_cols,
                        active_categorical_cols=active_categorical_cols,
                        trial_id_log_prefix=f"{trial_id_log_prefix}_CV"
                    )
                    
                    if fold_score < 1000:  # Only use non-penalty scores
                        series_cv_scores.append(fold_score)
                        
                except Exception as e:
                    logger.warning(f"{trial_id_log_prefix} - Series {series_id_value}, Fold {fold_idx}: CV fold failed: {e}")
                    continue
            
            # Add mean score for this series if we have valid scores
            if series_cv_scores:
                cv_scores.append(np.mean(series_cv_scores))
            else:
                cv_scores.append(1e6)  # Penalty for series with no valid CV scores
        
        return cv_scores

    def _process_one_hpo_evaluation_split(
        self,
        hyperparams: Dict,
        active_numerical_cols: List[str],
        active_categorical_cols: List[str],
        trial_id_log_prefix: str
    ) -> float:
        """evaluate hyperparams on train/val split by iterating through series"""
        logger.debug(f"{trial_id_log_prefix}: Starting evaluation for single HPO split.")
        
        per_series_validation_losses: List[float] = []

        # iterate through series
        for series_id_value in self.hpo_evaluation_series_ids:
            logger.debug(f"{trial_id_log_prefix} - Series '{series_id_value}': Preparing data for HPO evaluation split.")
            
            # extract series data
            
            series_full_data_globally_processed_df = self.globally_processed_and_encoded_hpo_df[
                self.globally_processed_and_encoded_hpo_df['id'] == series_id_value
            ].copy()

            if series_full_data_globally_processed_df.empty:
                logger.warning(f"{trial_id_log_prefix} - Series '{series_id_value}': No data found in the globally processed HPO splice. Skipping this series for this trial.")
                # Optionally append a high penalty if this series was expected
                # per_series_validation_losses.append(1e7) 
                continue
            
            # Ensure data is sorted by date (it should be if globally_processed_and_encoded_hpo_df was sorted)
            # but an explicit sort here per series is safer.
            series_full_data_globally_processed_df = series_full_data_globally_processed_df.sort_values('date').reset_index(drop=True)
            
            # create hpo train/val split
            # This split is within the HPO dataset portion for *this specific series*.
            # The `self.validation_days` refers to the duration of the HPO validation period.
            max_date_for_this_series_in_hpo_splice = series_full_data_globally_processed_df['date'].max()
            
            # Calculate the start date for the HPO validation set for this series
            hpo_validation_start_date_for_this_series = max_date_for_this_series_in_hpo_splice - pd.Timedelta(days=self.validation_days - 1)
            
            # Create the HPO training DataFrame for this series
            hpo_train_df_for_this_series = series_full_data_globally_processed_df[
                series_full_data_globally_processed_df['date'] < hpo_validation_start_date_for_this_series
            ].copy()
            
            # Create the HPO validation DataFrame for this series
            hpo_val_df_for_this_series = series_full_data_globally_processed_df[
                series_full_data_globally_processed_df['date'] >= hpo_validation_start_date_for_this_series
            ].copy()

            # check sufficient data
            # Determine minimum training length needed based on sequence length and forecast horizon
            current_sequence_length_hp = hyperparams.get('sequence_length', self.config.LSTM_CONFIG['sequence_length'])
            minimum_hpo_train_length_needed = current_sequence_length_hp + self.forecast_horizon
            
            if hpo_train_df_for_this_series.empty or \
               len(hpo_train_df_for_this_series) < minimum_hpo_train_length_needed or \
               hpo_val_df_for_this_series.empty or \
               len(hpo_val_df_for_this_series) < self.forecast_horizon:
                
                logger.debug(f"{trial_id_log_prefix} - Series '{series_id_value}': Insufficient data for HPO train/val split. "
                             f"Train length: {len(hpo_train_df_for_this_series)} (need {minimum_hpo_train_length_needed}), "
                             f"Val length: {len(hpo_val_df_for_this_series)} (need {self.forecast_horizon}). Assigning penalty.")
                per_series_validation_losses.append(1e6) # Assign a high penalty score
                continue

            # train and evaluate lstm
            try:
                series_validation_loss = self._train_and_eval_one_series_lstm_for_hpo(
                    series_id_value=series_id_value,
                    series_train_raw_df=hpo_train_df_for_this_series, 
                    series_val_raw_df=hpo_val_df_for_this_series,   
                    hyperparams_trial=hyperparams,
                    active_numerical_cols=active_numerical_cols,
                    active_categorical_cols=active_categorical_cols,
                    trial_id_log_prefix=trial_id_log_prefix # Pass prefix for detailed logging
                )
                per_series_validation_losses.append(series_validation_loss)
                logger.debug(f"{trial_id_log_prefix} - Series '{series_id_value}': Validation loss = {series_validation_loss:.6f}")

            except Exception as e:
                logger.error(f"{trial_id_log_prefix} - Series '{series_id_value}': Unhandled error during HPO training/evaluation: {e}", exc_info=True)
                per_series_validation_losses.append(1e7) # Assign an even higher penalty for unexpected crashes

        # aggregate fitness score
        if not per_series_validation_losses:
            logger.warning(f"{trial_id_log_prefix}: No series were successfully evaluated for this hyperparameter set. Returning maximum penalty.")
            return 1000 # Return a very high penalty if no series could be evaluated
        
        # Calculate the mean of valid (non-None, non-NaN) losses
        valid_losses = [loss for loss in per_series_validation_losses if loss is not None and not np.isnan(loss)]
        if not valid_losses:
            logger.warning(f"{trial_id_log_prefix}: All series evaluations resulted in None/NaN losses. Returning maximum penalty.")
            return 1000

        average_fitness_across_series = np.mean(valid_losses)
        
        # Handle case where mean might still be NaN (e.g., if all losses were NaN, though filtered above)
        return average_fitness_across_series if not np.isnan(average_fitness_across_series) else 1e8


    def _train_and_eval_one_series_lstm_for_hpo(
        self, series_id_value: str,
        series_train_raw_df: pd.DataFrame, # Raw train data for this series for this HPO (CV) fold
        series_val_raw_df: pd.DataFrame,   # Raw val data for this series for this HPO (CV) fold
        hyperparams_trial: Dict,
        active_numerical_cols: List[str], 
        active_categorical_cols: List[str],
        trial_id_log_prefix: str
    ) -> float:
        """
        Trains a single LSTM for one series with given HPs and evaluates its validation loss.
        Adapts logic from 3_train_lstm_models.py's train_lstm_for_single_series.
        """
        logger.debug(f"{trial_id_log_prefix} - Series {series_id_value}: Starting HPO model training...")

        # prep data for serie
        # Apply scaling based on *this HPO training split*
        series_scaler = StandardScaler()
        series_train_num_scaled_df = pd.DataFrame(index=series_train_raw_df.index, columns=active_numerical_cols)
        series_val_num_scaled_df = pd.DataFrame(index=series_val_raw_df.index, columns=active_numerical_cols)

        if active_numerical_cols: # Only scale if numerical features are active
            # Impute NaNs with 0 before scaling for robustness in HPO
            series_train_numerical_data = series_train_raw_df[active_numerical_cols].fillna(0)
            series_val_numerical_data = series_val_raw_df[active_numerical_cols].fillna(0)

            if not series_train_numerical_data.empty:
                series_train_num_scaled_df[active_numerical_cols] = series_scaler.fit_transform(series_train_numerical_data)
            if not series_val_numerical_data.empty:
                 series_val_num_scaled_df[active_numerical_cols] = series_scaler.transform(series_val_numerical_data)
        
        # Get sequence_length from current hyperparameters
        current_sequence_length = hyperparams_trial.get('sequence_length', self.config.LSTM_CONFIG['sequence_length'])

        # Combine features for sequence creation
        series_train_df_for_seq_creation = pd.concat([
            series_train_raw_df[active_categorical_cols + ['sales', 'date']], 
            series_train_num_scaled_df
        ], axis=1)
        
        series_val_df_for_target_sequences = pd.concat([ # Used to form the combined df for val sequences
            series_val_raw_df[active_categorical_cols + ['sales', 'date']],
            series_val_num_scaled_df
        ], axis=1)

        # Create training sequences
        train_sequences_tuple = hpo_create_sequences_for_series(
            series_train_df_for_seq_creation, active_categorical_cols, active_numerical_cols, 'sales',
            current_sequence_length, self.forecast_horizon
        )
        
        # Create validation sequences using the corrected strategy
        history_for_validation_input_X = series_train_df_for_seq_creation.tail(current_sequence_length) # Context from train
        # If history_for_validation_input_X is shorter than current_sequence_length, it means train set is too short after all.
        if len(history_for_validation_input_X) < current_sequence_length :
            logger.debug(f"{trial_id_log_prefix} - Series {series_id_value}: Training history too short ({len(history_for_validation_input_X)}) for sequence length {current_sequence_length} to create validation input context. Penalizing.")
            return 1001.0 # Penalty

        df_for_validation_sequences = pd.concat([
            history_for_validation_input_X, 
            series_val_df_for_target_sequences
        ]).sort_values('date').reset_index(drop=True)
        
        validation_sequences_tuple = hpo_create_sequences_for_series(
            df_for_validation_sequences, active_categorical_cols, active_numerical_cols, 'sales',
            current_sequence_length, self.forecast_horizon
        )

        # Check if sequences are valid
        if not train_sequences_tuple or not validation_sequences_tuple or \
           train_sequences_tuple[2].shape[0] == 0 or validation_sequences_tuple[2].shape[0] == 0:
            logger.debug(f"{trial_id_log_prefix} - Series {series_id_value}: Insufficient data to create train/val sequences with seq_len {current_sequence_length}. Penalizing.")
            return 1002.0 # High penalty

        X_cat_train, X_num_train, y_train = train_sequences_tuple
        X_cat_val, X_num_val, y_val = validation_sequences_tuple
        
        # create dataloaders
        train_dataset = HPOSeriesLSTMDataset(X_cat_train, X_num_train, y_train)
        val_dataset = HPOSeriesLSTMDataset(X_cat_val, X_num_val, y_val)
        
        hpo_batch_size = hyperparams_trial.get('batch_size', self.config.LSTM_CONFIG['batch_size'])
        train_loader = DataLoader(train_dataset, batch_size=hpo_batch_size, shuffle=True, num_workers=0)
        val_loader = DataLoader(val_dataset, batch_size=hpo_batch_size, shuffle=False, num_workers=0)

        # create lstm with hyperparams
        # Use globally derived embedding vocabulary sizes, but embedding_dim can be tuned
        current_trial_embedding_specs = {
            feat_name: (
                self.global_embedding_info_dict[feat_name]['vocab_size'], # Use global vocab_size
                hyperparams_trial.get(f'{feat_name}_embedding_dim', self.global_embedding_info_dict[feat_name]['embedding_dim']) # HP for dim
            ) for feat_name in active_categorical_cols if feat_name in self.global_embedding_info_dict
        }

        model = LSTMModel(
            numerical_input_dim=len(active_numerical_cols),
            embedding_specs=current_trial_embedding_specs,
            hidden_dim=hyperparams_trial.get('hidden_dim', self.config.LSTM_CONFIG['hidden_dim']),
            num_layers=hyperparams_trial.get('num_layers', self.config.LSTM_CONFIG['num_layers']),
            output_dim=self.forecast_horizon, # This is fixed for the problem
            dropout=hyperparams_trial.get('dropout', self.config.LSTM_CONFIG['dropout']),
            bidirectional=hyperparams_trial.get('bidirectional', self.config.LSTM_CONFIG['bidirectional']),
            use_attention=hyperparams_trial.get('use_attention', self.config.LSTM_CONFIG['use_attention']),
            quantile_output=False, # HPO usually optimizes for point forecast accuracy (e.g., MSE)
            quantile_levels=None
        ).to(self.device)

        # define optimizer and loss
        optimizer = optim.Adam(
            model.parameters(), 
            lr=hyperparams_trial.get('learning_rate', self.config.LSTM_CONFIG['learning_rate']),
            weight_decay=hyperparams_trial.get('weight_decay', self.config.LSTM_CONFIG['weight_decay'])
        )
        criterion = nn.MSELoss() # Using MSE as the HPO objective (validation loss)
        
        # Add learning rate scheduler for better convergence
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', 
            patience=hyperparams_trial.get('patience', self.config.LSTM_CONFIG['patience']) // 2, 
            factor=0.5, verbose=False
        )
        
        # Get gradient clipping value from hyperparams or config
        gradient_clip_value = hyperparams_trial.get('gradient_clip', self.config.LSTM_CONFIG.get('gradient_clip', 1.0))

        # training loop
        max_hpo_epochs = hyperparams_trial.get('epochs', self.config.LSTM_CONFIG['epochs'])
        hpo_patience = hyperparams_trial.get('patience', self.config.LSTM_CONFIG['patience'])
        
        best_series_val_loss = float('inf')
        epochs_without_improvement = 0

        for epoch_num in range(1, max_hpo_epochs + 1):
            model.train()
            current_epoch_train_loss = 0.0
            
            # Training phase for one epoch
            for cat_batch_train, num_batch_train, target_batch_train in train_loader:
                cat_batch_train = {name: tensor.to(self.device) for name, tensor in cat_batch_train.items()}
                num_batch_train = num_batch_train.to(self.device)
                target_batch_train = target_batch_train.to(self.device)
                
                optimizer.zero_grad()
                predictions_train = model(cat_batch_train, num_batch_train)
                loss_train = criterion(predictions_train, target_batch_train)
                loss_train.backward()
                
                # Apply gradient clipping to prevent exploding gradients
                torch.nn.utils.clip_grad_norm_(model.parameters(), gradient_clip_value)
                
                optimizer.step()
                current_epoch_train_loss += loss_train.item()
            
            # Validation phase for one epoch
            model.eval()
            current_epoch_val_loss_sum = 0.0
            with torch.no_grad():
                for cat_batch_val, num_batch_val, target_batch_val in val_loader:
                    cat_batch_val = {name: tensor.to(self.device) for name, tensor in cat_batch_val.items()}
                    num_batch_val = num_batch_val.to(self.device)
                    target_batch_val = target_batch_val.to(self.device)
                    
                    predictions_val = model(cat_batch_val, num_batch_val)
                    loss_val_batch = criterion(predictions_val, target_batch_val)
                    current_epoch_val_loss_sum += loss_val_batch.item()
            
            average_epoch_val_loss = current_epoch_val_loss_sum / len(val_loader) if len(val_loader) > 0 else float('inf')
            average_epoch_train_loss = current_epoch_train_loss / len(train_loader) if len(train_loader) > 0 else float('inf')
            
            # Update learning rate based on validation loss
            scheduler.step(average_epoch_val_loss)

            # Early stopping logic
            if average_epoch_val_loss < best_series_val_loss:
                best_series_val_loss = average_epoch_val_loss
                epochs_without_improvement = 0
                # In HPO, we don't save the model, just track the best loss
            else:
                epochs_without_improvement += 1
            
            if epochs_without_improvement >= hpo_patience:
                logger.debug(f"{trial_id_log_prefix} - Series {series_id_value}: Early stopping at HPO epoch {epoch_num}. Best val loss: {best_series_val_loss:.4f}")
                break
        
        logger.debug(f"{trial_id_log_prefix} - Series {series_id_value}: HPO training finished. Best val_loss for this series & HPs: {best_series_val_loss:.4f}")
        return best_series_val_loss if best_series_val_loss != float('inf') else 1003.0 # Return high penalty if something went wrong

# factory function for hpo runner
def create_lstm_fitness_function(
    splice_file_path: str,
    forecast_horizon: int,
    validation_days: int,
    hpo_runner_args: argparse.Namespace, # Args from run_hpo_lstm_bo.py
    global_project_config: Any          # The config_exp1 module
) -> callable:
    """
    Factory function to create and return the fitness evaluation method for LSTM HPO.
    """
    # Extract HPO-specific configuration settings from the runner's arguments
    hpo_evaluator_config = {
        'limit_items': hpo_runner_args.limit_items,
        'use_cv': hpo_runner_args.use_cv,
        'cv_splits': hpo_runner_args.cv_splits,
        'cv_initial_train_size': hpo_runner_args.cv_initial_train_size,
        'device': hpo_runner_args.device,
        'random_seed': hpo_runner_args.random_seed
    }

    # Instantiate the evaluator
    lstm_fitness_evaluator_instance = LSTMFitnessEvaluator(
        splice_file_path=splice_file_path,
        forecast_horizon=forecast_horizon,
        validation_days=validation_days,
        hpo_config=hpo_evaluator_config,
        global_project_config=global_project_config
    )
    
    # Return the 'evaluate' method of the instance
    return lstm_fitness_evaluator_instance.evaluate