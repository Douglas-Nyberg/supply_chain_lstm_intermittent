#////////////////////////////////////////////////////////////////////////////////#
# File:         1_preprocess_exp_mc.py                                           #
# Author:       Douglas Nyberg                                                   #
# Email:        douglas1.nyberg@gmail.com                                        #
# Date:         2025-06-04                                                       # 
# Description:  Preprocessing for synthetic monte carlo data experiment.         #
# Affiliation:  Physics Department, Purdue University                            #
#////////////////////////////////////////////////////////////////////////////////#
#!/usr/bin/env python3
"""
Preprocessing script for monte carlo synthetic data.

This is basically the same as the exp1 preprocessing but for synthetic data.
Should be simpler since we dont have all the M5 complexities.
"""

import argparse
import json
import logging
import pickle
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple, Any, Optional

import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from tqdm import tqdm

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent.parent
sys.path.append(str(PROJECT_ROOT))

import config_mc_master as config

# setup logging - avoid duplicate logs
config.LOGS_DIR.mkdir(parents=True, exist_ok=True)
for handler in logging.root.handlers[:]:
    logging.root.removeHandler(handler)
    
logging.basicConfig(
    level=getattr(logging, config.LOGGING_CONFIG["level"]),
    format=config.LOGGING_CONFIG["format"],
    handlers=[
        logging.FileHandler(str(config.LOGGING_CONFIG["log_file"]), mode='a'),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

def parse_arguments() -> argparse.Namespace:
    """parse command line arguments"""
    parser = argparse.ArgumentParser(
        description="preprocessing for monte carlo synthetic data experiment",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument("--data-dir", type=str, default=config.DATA_CONFIG["raw_data_dir"], 
                       help="directory with synthetic m5 csv files")
    parser.add_argument("--output-dir", type=str, default=config.DATA_CONFIG["preprocessed_output_dir"], 
                       help="where to save preprocessed data")
    parser.add_argument("--limit-items", type=int, default=config.DATA_CONFIG.get("limit_items"), 
                       help="limit processing to N items (for testing)")
    parser.add_argument("--min-periods", type=int, default=config.DATA_CONFIG.get("min_non_zero_periods", 3), 
                       help="min non-zero demand periods per item")
    parser.add_argument("--force", action="store_true", help="force reprocessing even if files exist")
    return parser.parse_args()

def load_synthetic_m5_data(data_dir: str) -> Tuple[Optional[pd.DataFrame], Optional[pd.DataFrame], Optional[pd.DataFrame]]:
    """load synthetic m5-format data files"""
    data_path = Path(data_dir)
    logger.info(f"loading synthetic m5 data from {data_path}")
    
    # setup file paths
    sales_path = data_path / "sales_train_validation.csv"
    calendar_path = data_path / "calendar.csv"
    prices_path = data_path / "sell_prices.csv" 

    # check if required files exsit
    if not sales_path.exists() or not calendar_path.exists():
        logger.error(f"missing required files - sales exists: {sales_path.exists()}, calendar exists: {calendar_path.exists()}")
        return None, None, None
    
    # load sales and calendar data
    try:
        sales_df = pd.read_csv(sales_path)
        calendar_df = pd.read_csv(calendar_path)
        calendar_df['date'] = pd.to_datetime(calendar_df['date'])  # convert dates
        logger.info(f"loaded sales data: {sales_df.shape}, calendar data: {calendar_df.shape}")
    except Exception as e:
        logger.error(f"error loading csv files: {e}", exc_info=True)
        return None, None, None
        
    # try to load price data if needed
    prices_df = None
    if prices_path.exists() and config.LSTM_CONFIG.get("use_price_features", False):
        try:
            prices_df = pd.read_csv(prices_path)
            logger.info(f"loaded price data: {prices_df.shape}")
        except Exception as e:
            logger.warning(f"couldnt load price data from {prices_path}: {e}. continuing without prices.")
    else:
        logger.info("skipping price data (use_price_features=False or file missing)")
    
    return sales_df, calendar_df, prices_df

def create_synthetic_long_format(sales_df: pd.DataFrame, calendar_df: pd.DataFrame) -> Optional[pd.DataFrame]:
    """convert wide format sales data to long format"""
    logger.info("converting data to long format")
    
    # get all the day columns (d_1, d_2, etc)
    day_cols = [col for col in sales_df.columns if col.startswith('d_')]
    id_vars_synthetic = ['id', 'item_id', 'dept_id', 'cat_id', 'store_id', 'state_id']
    
    # check if we have all required id columns
    missing_id_vars = [v for v in id_vars_synthetic if v not in sales_df.columns]
    if missing_id_vars:
        logger.error(f"missing id columns for melting: {missing_id_vars}")
        return None

    try:
        # melt from wide to long format
        df_long = sales_df.melt(id_vars=id_vars_synthetic, value_vars=day_cols, var_name='d', value_name='sales')
        
        # add calendar dates
        df_long = df_long.merge(calendar_df[['d', 'date']].copy(), on='d', how='left') 
        df_long['date'] = pd.to_datetime(df_long['date'])  # make sure dates are datetime
        df_long = df_long.sort_values(['id', 'date']).reset_index(drop=True)
        
        logger.info(f"long format data shape: {df_long.shape}")
        return df_long
    except Exception as e:
        logger.error(f"error during long format conversion: {e}", exc_info=True)
        return None

def apply_simplified_feature_engineering(df: pd.DataFrame, full_calendar_df: pd.DataFrame) -> Optional[pd.DataFrame]:
    logger.info("Applying simplified feature engineering")
    if df.empty: 
        logger.warning("Input DataFrame for feature engineering is empty.")
        return pd.DataFrame()

    calendar_features_to_add = ['d'] 
    if 'wm_yr_wk' in full_calendar_df.columns: calendar_features_to_add.append('wm_yr_wk')
    calendar_subset = full_calendar_df[calendar_features_to_add].drop_duplicates(subset=['d']).copy()
        
    df_merged = df.merge(calendar_subset, on='d', how='left', suffixes=('', '_cal'))
    if 'date_cal' in df_merged.columns : df_merged.drop(columns=['date_cal'], inplace=True, errors='ignore')

    processed_groups = []
    for _, group_df in tqdm(df_merged.groupby('id'), desc="Engineering features per series"):
        item_df = group_df.copy().sort_values('date').reset_index(drop=True)
        
        if config.LSTM_CONFIG.get("use_calendar_features", True):
            item_df['weekday'] = item_df['date'].dt.weekday.astype('int32')
            item_df['month'] = item_df['date'].dt.month.astype('int32')
            item_df['quarter'] = item_df['date'].dt.quarter.astype('int32')
            item_df['day_of_year'] = item_df['date'].dt.dayofyear.astype('int32')
            item_df['week_of_year'] = item_df['date'].dt.isocalendar().week.astype('int32')
            item_df['year'] = item_df['date'].dt.year.astype('int32')

            item_df['weekday_sin'] = np.sin(2 * np.pi * item_df['weekday'] / 6.0) 
            item_df['weekday_cos'] = np.cos(2 * np.pi * item_df['weekday'] / 6.0)
            item_df['month_sin'] = np.sin(2 * np.pi * item_df['month'] / 12.0) 
            item_df['month_cos'] = np.cos(2 * np.pi * item_df['month'] / 12.0)
        
        if config.LSTM_CONFIG.get("use_lag_features", True):
            lags = config.LSTM_CONFIG.get("lag_periods", [7, 14, 28, 35, 42])
            for lag in lags:
                item_df[f'sales_lag_{lag}'] = item_df['sales'].shift(lag)
        
        if config.LSTM_CONFIG.get("use_rolling_features", True):
            windows = config.LSTM_CONFIG.get("rolling_windows", [7, 14, 28])
            for window in windows:
                item_df[f'sales_rolling_mean_{window}'] = item_df['sales'].rolling(window=window, min_periods=1).mean()
                item_df[f'sales_rolling_std_{window}'] = item_df['sales'].rolling(window=window, min_periods=1).std()
        processed_groups.append(item_df)
    
    if not processed_groups: return pd.DataFrame()
    df_final = pd.concat(processed_groups, ignore_index=True).fillna(0) 
    logger.info(f"Feature engineering completed. Final shape: {df_final.shape}")
    return df_final

def encode_categorical_features(df: pd.DataFrame) -> Tuple[pd.DataFrame, Dict[str, LabelEncoder]]:
    df_encoded = df.copy()
    encoders = {}
    cols_to_label_encode = ['item_id', 'dept_id', 'cat_id', 'store_id', 'state_id']
    
    actual_cols_to_encode = []
    for col in cols_to_label_encode:
        if col in df_encoded.columns and df_encoded[col].dtype == 'object':
            actual_cols_to_encode.append(col)
        elif col in df_encoded.columns:
            logger.debug(f"Column '{col}' not 'object' (is {df_encoded[col].dtype}), skipping LabelEncoding.")
            
    if not actual_cols_to_encode:
        logger.warning("No string categorical columns for LabelEncoding. Original DataFrame returned for this step.")
        return df_encoded, encoders

    logger.info(f"Applying Label Encoding to: {actual_cols_to_encode}")
    for col in actual_cols_to_encode:
        logger.debug(f"LabelEncoding: '{col}' (dtype: {df_encoded[col].dtype})")
        sample_before = df_encoded[col].astype(str).unique()[:3]
        encoder = LabelEncoder()
        df_encoded[col] = encoder.fit_transform(df_encoded[col].astype(str)).astype('int32')
        encoders[col] = encoder
        sample_after = df_encoded[col].unique()[:3]
        logger.debug(f"  Encoded '{col}': {len(encoder.classes_)} uniques. Sample: {sample_before} -> {sample_after}. New dtype: {df_encoded[col].dtype}")
            
    logger.info("Verification of dtypes after string label encoding:")
    for col in actual_cols_to_encode:
        if col in df_encoded.columns and col in encoders:
            logger.info(f"  Post-LabelEncoding '{col}': {df_encoded[col].dtype}, sample: {df_encoded[col].unique()[:3]}")
            if not pd.api.types.is_integer_dtype(df_encoded[col].dtype):
                 logger.error(f"CRITICAL: Column '{col}' is NOT int32 after LabelEncoding! Dtype: {df_encoded[col].dtype}")
    return df_encoded, encoders

def get_categorical_feature_specs_for_embedding(
    df: pd.DataFrame, string_label_encoded_cols: List[str], 
    other_integer_categorical_cols: List[str]
) -> Tuple[List[str], Dict[str, Dict[str, Any]]]:
    feature_specs = {}
    final_categorical_list_for_model = [] 

    all_potential_categoricals = list(set(string_label_encoded_cols + other_integer_categorical_cols))
    logger.info(f"Generating embedding specs. Candidates: {all_potential_categoricals}")

    for col in all_potential_categoricals:
        if col in df.columns and pd.api.types.is_integer_dtype(df[col].dtype):
            final_categorical_list_for_model.append(col)
        elif col in df.columns:
             logger.warning(f"Column '{col}' intended for embedding spec is not integer ({df[col].dtype}). Skipping spec.")
        else:
            logger.warning(f"Column '{col}' for spec not found in DataFrame. Skipping spec.")
    
    logger.info(f"Final list for embedding specs: {final_categorical_list_for_model}")
    for col in final_categorical_list_for_model: 
        if df[col].notna().any():
            max_val = df[col].max()
            vocab_size = int(max_val) + 1 
        else:
            logger.warning(f"Categorical column '{col}' all NaN. Vocab size 1 (placeholder).")
            vocab_size = 1 
        embedding_dim = min(config.LSTM_CONFIG.get("max_embedding_dim", 50), (vocab_size + 2) // 2)
        embedding_dim = max(1, embedding_dim) 
        feature_specs[col] = {'vocab_size': vocab_size, 'embedding_dim': embedding_dim, 'description': f'Cat feature: {col}'}
        logger.debug(f"  Spec for '{col}': vocab_size={vocab_size}, embedding_dim={embedding_dim}")
    return final_categorical_list_for_model, feature_specs

def create_train_test_splits(df: pd.DataFrame, test_days: int) -> Optional[Tuple[pd.DataFrame, pd.DataFrame]]:
    logger.info(f"Creating train/test splits with {test_days} test days")
    if 'date' not in df.columns or df['date'].isnull().all():
        logger.error("Date column missing or all null for splitting."); return None
    max_date = df['date'].max()
    if pd.isna(max_date): logger.error("Max date is NaT for splitting."); return None
        
    split_date = max_date - pd.Timedelta(days=test_days - 1)
    train_df = df[df['date'] < split_date].copy()
    test_df = df[df['date'] >= split_date].copy()
    
    min_train_date, max_train_date = (train_df['date'].min(), train_df['date'].max()) if not train_df.empty else ('N/A', 'N/A')
    min_test_date, max_test_date = (test_df['date'].min(), test_df['date'].max()) if not test_df.empty else ('N/A', 'N/A')

    logger.info(f"Train: {train_df.shape} (Dates: {min_train_date} to {max_train_date})")
    logger.info(f"Test: {test_df.shape} (Dates: {min_test_date} to {max_test_date})")
    
    if train_df.empty and not df.empty: logger.warning("Train df is empty after split. Test period might be too large or data too short.")
    if test_df.empty and not df.empty: logger.warning("Test df is empty after split. This is unusual if train_df is not empty.")

    return train_df, test_df

def filter_items_by_demand(df: pd.DataFrame, min_periods: int) -> pd.DataFrame:
    logger.info(f"Filtering items by min {min_periods} non-zero demand periods")
    if df.empty: 
        logger.warning("Input df to filter_items_by_demand is empty.")
        return df
    if 'sales' not in df.columns or 'id' not in df.columns:
        logger.error("'sales' or 'id' column missing for filtering by demand.")
        raise ValueError("'sales' or 'id' column missing for filtering by demand.")

    demand_counts_per_id = df[df['sales'] > 0].groupby('id')['sales'].count()
    valid_ids = demand_counts_per_id[demand_counts_per_id >= min_periods].index.tolist()
    original_item_count = df['id'].nunique()
    df_filtered = df[df['id'].isin(valid_ids)].copy()
    
    filtered_item_count = df_filtered['id'].nunique()
    logger.info(f"Filtered from {original_item_count} to {filtered_item_count} items.")
    if original_item_count > 0 and filtered_item_count == 0:
        logger.warning("No items remaining after filtering by demand criteria.")
    return df_filtered

def save_preprocessed_data(
    train_df: pd.DataFrame, test_df: pd.DataFrame, 
    output_dir_str: str, metadata: Dict[str, Any], 
    string_label_encoded_cols: List[str], # Pass the list of cols that were label encoded
    final_categorical_list_for_model: List[str], 
    calculated_feature_specs: Dict[str, Any]
):
    output_path = Path(output_dir_str)
    output_path.mkdir(parents=True, exist_ok=True)
    
    train_path = output_path / config.DATA_CONFIG["train_file_name"]
    test_path = output_path / config.DATA_CONFIG["test_file_name"]
    # This is the crucial file for LSTM training, ensuring it's derived from encoded data
    train_val_path = output_path / config.DATA_CONFIG["train_val_file_name"] 
    val_path = output_path / config.DATA_CONFIG["validation_file_name"] 

    logger.info(f"Saving train_df ({train_df.shape}) to {train_path}")
    train_df.to_csv(train_path, index=False)
    logger.info(f"Saving test_df ({test_df.shape}) to {test_path} (also as validation_file to {val_path})")
    test_df.to_csv(test_path, index=False)
    test_df.to_csv(val_path, index=False) 

    if not train_df.empty and not test_df.empty and \
       all(col in train_df.columns for col in ['id', 'date']) and \
       all(col in test_df.columns for col in ['id', 'date']):
        combined_train_val_df = pd.concat([train_df, test_df]).sort_values(['id', 'date']).reset_index(drop=True)
        
        logger.info(f"Sanity check BEFORE saving combined_train_val_df to {train_val_path}:")
        # Check the specific columns that were supposed to be label-encoded
        for col_to_check in string_label_encoded_cols: 
            if col_to_check in combined_train_val_df.columns:
                logger.info(f"  Col '{col_to_check}': dtype={combined_train_val_df[col_to_check].dtype}, "
                            f"sample unique values: {combined_train_val_df[col_to_check].unique()[:5]}")
                if not pd.api.types.is_integer_dtype(combined_train_val_df[col_to_check].dtype):
                    logger.error(f"    CRITICAL CHECK FAILED: Column '{col_to_check}' in combined_train_val_df is NOT INTEGER before saving to {train_val_path}!")
            else:
                logger.warning(f"  Col '{col_to_check}' (expected to be encoded) not in combined_train_val_df for sanity check.")
        
        logger.info(f"Saving COMBINED AND ENCODED train_val data ({combined_train_val_df.shape}) to {train_val_path}")
        combined_train_val_df.to_csv(train_val_path, index=False)

        # --- DISK READ-BACK VERIFICATION ---
        logger.info(f"VERIFYING FILE ON DISK: Reading back {train_val_path} immediately after save...")
        try:
            # Read back the critical columns as object to see their raw content, then try to convert
            dtype_to_read_as_object = {col: 'object' for col in string_label_encoded_cols}
            df_read_back = pd.read_csv(train_val_path, dtype=dtype_to_read_as_object)
            logger.info(f"  Successfully read back {train_val_path}. Shape: {df_read_back.shape}")
            all_verified_ok = True
            for col_to_check in string_label_encoded_cols:
                if col_to_check in df_read_back.columns:
                    disk_sample = df_read_back[col_to_check].dropna().unique()[:5]
                    logger.info(f"    VERIFY Col '{col_to_check}' from disk: dtype={df_read_back[col_to_check].dtype}, sample: {disk_sample}")
                    # Try to convert to numeric; if it fails on "SYNTH_XXX", the save was bad.
                    try:
                        pd.to_numeric(df_read_back[col_to_check], errors='raise')
                        logger.info(f"      SUCCESS: Column '{col_to_check}' from disk appears numeric-convertible.")
                    except ValueError:
                        logger.error(f"      DISK VERIFICATION FAILED FOR '{col_to_check}'. File on disk still contains non-numeric strings like '{disk_sample[0]}'.")
                        all_verified_ok = False
                else:
                    logger.warning(f"    VERIFY Col '{col_to_check}' not in df_read_back.")
            if not all_verified_ok:
                logger.critical("CRITICAL: Disk verification failed. The CSV file does not contain pure integer codes as expected.")
                # raise RuntimeError("CSV file verification failed after save. Encoded data not written correctly.")
        except Exception as e_readback:
            logger.error(f"  Failed to read back or verify {train_val_path}. Error: {e_readback}")
        # --- END OF DISK READ-BACK ---
    else:
        logger.error("Cannot create combined_train_val_df for LSTM. Train/test empty or missing 'id'/'date'.")
        raise ValueError("Failed to create combined train_val_df for LSTM training.")

    metadata_path = output_path / config.DATA_CONFIG["metadata_file_name"]
    with open(metadata_path, 'w') as f: json.dump(metadata, f, indent=2, default=str) 
    
    embedding_info = {
        'categorical_features': final_categorical_list_for_model,
        'feature_specs': calculated_feature_specs,
        'preprocessing_date': datetime.now().isoformat(),
        'data_type': 'synthetic_monte_carlo_simplified_features'
    }
    embedding_path = output_path / config.DATA_CONFIG["embedding_info_pickle_name"]
    with open(embedding_path, 'wb') as f: pickle.dump(embedding_info, f)
    
    logger.info(f"Saved all preprocessed data to {output_path}")
    logger.info(f"  Embedding Info -> Categoricals for LSTM: {final_categorical_list_for_model}")
    logger.info(f"  Embedding Info -> Feature specs for: {list(calculated_feature_specs.keys())}")

def main():
    args = parse_arguments()
    start_time = datetime.now()
    logger.info("=" * 80 + f"\nSTART: MONTE CARLO SYNTHETIC DATA PREPROCESSING (SIMPLIFIED)\nTime: {start_time.strftime('%Y-%m-%d %H:%M:%S')}\n" + "=" * 80)
    logger.info(f"Config: {config.EXPERIMENT_TAG}, Raw Data Dir: {args.data_dir}, Output Dir: {args.output_dir}")
    if args.limit_items: logger.info(f"Limiting processing to first {args.limit_items} items.")
    
    # Check if output files already exist
    output_path = Path(args.output_dir)
    train_val_path = output_path / config.DATA_CONFIG["train_val_file_name"]
    metadata_path = output_path / config.DATA_CONFIG["metadata_file_name"]
    
    if not args.force and train_val_path.exists() and metadata_path.exists():
        logger.info(f"Output files already exist at {output_path}. Use --force to reprocess.")
        logger.info("Preprocessing skipped.")
        return
    
    if args.force and train_val_path.exists():
        logger.info("--force flag used. Reprocessing existing data.")
    
    try:
        loaded_data = load_synthetic_m5_data(args.data_dir)
        if loaded_data is None or loaded_data[0] is None or loaded_data[1] is None:
            logger.critical("Failed to load initial sales or calendar data. Aborting."); return
        sales_df, calendar_df, prices_df = loaded_data 
        
        if args.limit_items and args.limit_items > 0:
            unique_ids_to_process = sales_df['id'].unique()[:args.limit_items]
            sales_df = sales_df[sales_df['id'].isin(unique_ids_to_process)].copy()
            logger.info(f"Limited to {len(unique_ids_to_process)} unique 'id's for processing.")
        
        df_long = create_synthetic_long_format(sales_df, calendar_df)
        if df_long is None or df_long.empty: logger.critical("Long format failed. Aborting."); return

        df_filtered = filter_items_by_demand(df_long, args.min_periods)
        if df_filtered.empty: logger.critical("No items after filtering. Aborting."); return

        df_with_features = apply_simplified_feature_engineering(df_filtered, calendar_df)
        if df_with_features.empty: logger.critical("No features generated. Aborting."); return
        
        df_encoded, category_encoders_dict = encode_categorical_features(df_with_features)
        string_label_encoded_cols = list(category_encoders_dict.keys()) 
        
        other_integer_categorical_cols = []
        if config.LSTM_CONFIG.get("use_calendar_features", True):
             potential_calendar_cats = ['weekday', 'month', 'quarter', 'year', 'week_of_year'] 
             for col in potential_calendar_cats:
                 if col in df_encoded.columns and pd.api.types.is_integer_dtype(df_encoded[col].dtype) and \
                    col not in string_label_encoded_cols: 
                     other_integer_categorical_cols.append(col)
        
        final_cat_list_for_model, feature_specs_for_embedding = get_categorical_feature_specs_for_embedding(
            df_encoded, string_label_encoded_cols, other_integer_categorical_cols
        )
        
        splits = create_train_test_splits(df_encoded, config.DATA_CONFIG.get("test_days", 28))
        if splits is None or splits[0].empty or splits[1].empty:
            logger.critical("Train or test DataFrame empty after split. Aborting."); return
        train_df, test_df = splits

        metadata = {
            "timestamp": datetime.now().isoformat(), "source": "synthetic_mc",
            "config_tag": config.EXPERIMENT_TAG, "input_data_dir": str(args.data_dir),
            "raw_item_count_from_input_sales_df": sales_df['id'].nunique(),
            "filtered_item_count_processed": df_filtered['id'].nunique(),
            "total_calendar_days_available": len(calendar_df),
            "days_in_train_set": train_df['date'].nunique() if not train_df.empty else 0, 
            "days_in_test_set": test_df['date'].nunique() if not test_df.empty else 0,
            "categorical_features_in_embedding_info": final_cat_list_for_model 
        }
        save_preprocessed_data(
            train_df, test_df, args.output_dir, metadata, 
            string_label_encoded_cols, # Pass the list of cols that were label encoded for verification
            final_cat_list_for_model, 
            feature_specs_for_embedding
        )
        
        end_time = datetime.now()
        logger.info("=" * 80 + f"\nPREPROCESSING COMPLETED SUCCESSFULLY\nTime: {end_time.strftime('%Y-%m-%d %H:%M:%S')}\nDuration: {end_time - start_time}\n" + "=" * 80)
        
    except FileNotFoundError as e: logger.error(f"Data file not found: {e}", exc_info=True)
    except ValueError as e: logger.error(f"ValueError in preprocessing: {e}", exc_info=True)
    except Exception as e: logger.error(f"Unexpected error: {e}", exc_info=True)
        
if __name__ == "__main__":
    main()