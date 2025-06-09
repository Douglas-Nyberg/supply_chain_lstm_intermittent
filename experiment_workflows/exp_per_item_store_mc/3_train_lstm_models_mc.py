#////////////////////////////////////////////////////////////////////////////////#
# File:         3_train_lstm_models_mc.py                                        #
# Author:       Douglas Nyberg                                                   #
# Email:        douglas1.nyberg@gmail.com                                        #
# Date:         2025-06-04 # Correct get_lstm_fold_data call                     #
#////////////////////////////////////////////////////////////////////////////////#
#!/usr/bin/env python3
"""
Trains one LSTM model per series (item-store 'id') for the synthetic MC data.
"""

import argparse
import json
import logging
import pickle
import sys
import time
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Tuple, Optional, Any
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent.parent
sys.path.append(str(PROJECT_ROOT))

from src.models.lstm import LSTMModel, QuantileLoss 
import config_mc_master as config
from src.utils import set_random_seed, sanitize_for_path
from src.cv_utils import IndexBasedWalkForwardValidator, get_lstm_fold_data, aggregate_cv_metrics
from src.feature_engineering import create_lstm_sequences_for_series

config.LOGS_DIR.mkdir(parents=True, exist_ok=True)
for handler in logging.root.handlers[:]: logging.root.removeHandler(handler)
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
    parser = argparse.ArgumentParser(description="Train Per-Series LSTM models for MC Synthetic Data")
    parser.add_argument("--input-dir", type=str, default=config.DATA_CONFIG["preprocessed_output_dir"], help="Dir of preprocessed data.")
    parser.add_argument("--parallel-jobs", type=int, default=config.WORKFLOW_CONFIG["parallel_jobs"], help="Num parallel jobs.")
    parser.add_argument("--limit-series", type=int, default=config.DATA_CONFIG["limit_items"], help="Limit num series for training.")
    parser.add_argument("--force", action="store_true", help="Force retraining.")
    parser.add_argument("--device", type=str, default=config.DEVICE_CONFIG["device_name_lstm"], choices=["cuda", "cpu"], help="Device.")
    return parser.parse_args()

def load_split_data_and_metadata(input_dir: Path) -> Tuple[Optional[pd.DataFrame], Optional[Dict], Optional[Dict]]:
    logger.info(f"Loading data and metadata from: {input_dir}")
    train_val_path = input_dir / config.DATA_CONFIG["train_val_file_name"] 
    meta_pickle_path = input_dir / config.DATA_CONFIG["embedding_info_pickle_name"]
    meta_json_path = input_dir / config.DATA_CONFIG["metadata_file_name"] 

    if not train_val_path.exists() or not meta_pickle_path.exists():
        missing = [p for p in [train_val_path, meta_pickle_path] if not p.exists()]
        logger.error(f"CRITICAL files not found: {missing}. Ensure 1_preprocess_exp_mc.py ran and "
                     f"'{config.DATA_CONFIG['train_val_file_name']}' points to the correct combined+encoded CSV.")
        return None, None, None

    try:
        with open(meta_pickle_path, 'rb') as f: embedding_data = pickle.load(f)
        logger.info(f"Successfully loaded {meta_pickle_path.name}.")
    except Exception as e:
        logger.error(f"Failed to load embedding_info pickle '{meta_pickle_path}': {e}", exc_info=True)
        return None, None, None

    categorical_cols_from_embedding = embedding_data.get('categorical_features', []) if isinstance(embedding_data, dict) else []
    if not isinstance(categorical_cols_from_embedding, list):
        try: categorical_cols_from_embedding = list(categorical_cols_from_embedding)
        except: categorical_cols_from_embedding = []
    logger.info(f"Categorical columns from embedding info (expected as int32 in CSV): {categorical_cols_from_embedding}")
    
    dtype_spec_for_read_csv = {col: 'int32' for col in categorical_cols_from_embedding if col}
    dtype_spec_for_read_csv['sales'] = 'float32' 

    try:
        logger.info(f"Reading {train_val_path} with explicit dtypes: {dtype_spec_for_read_csv}")
        train_val_df = pd.read_csv(train_val_path, parse_dates=['date'], dtype=dtype_spec_for_read_csv)
        logger.info(f"Loaded train_val data: {train_val_df.shape}")
        for col in categorical_cols_from_embedding:
            if col in train_val_df.columns:
                current_dtype = train_val_df[col].dtype
                logger.debug(f"  Column '{col}' after load with dtype spec: {current_dtype}")
                if not pd.api.types.is_integer_dtype(current_dtype) and not isinstance(current_dtype, pd.Int32Dtype): # check for nullable Int32 too
                    logger.error(f"CRITICAL: Column '{col}' is {current_dtype}, not int32/Int32, after loading {train_val_path}. "
                                 f"CSV file '{train_val_path}' likely contains non-integer strings. "
                                 f"Check 1_preprocess_exp_mc.py's output and disk verification logs.")
                    return None, None, None 
    except ValueError as ve: 
        logger.error(f"CRITICAL ERROR loading {train_val_path}. CSV does not contain integer-like values for specified categoricals. "
                     f"Error: {ve}. PLEASE VERIFY THE CSV CONTENT MANUALLY and 1_preprocess_exp_mc.py logs.", exc_info=True)
        return None, None, None
    except Exception as e:
        logger.error(f"General error loading {train_val_path}: {e}", exc_info=True)
        return None, None, None
    
    metadata_json = None 
    if meta_json_path.exists():
        try:
            with open(meta_json_path, 'r') as f: metadata_json = json.load(f)
            logger.info(f"Loaded {meta_json_path.name}.")
        except Exception as e: logger.warning(f"Could not load metadata_json: {e}")

    embedding_specs_for_model = {}
    if isinstance(embedding_data, dict) and 'feature_specs' in embedding_data:
        feature_specs_from_pickle = embedding_data.get('feature_specs', {})
        for feat_name in categorical_cols_from_embedding: 
            if feat_name in feature_specs_from_pickle:
                spec = feature_specs_from_pickle[feat_name]
                embedding_specs_for_model[feat_name] = (spec['vocab_size'], spec['embedding_dim'])
            else:
                logger.error(f"Missing embedding spec for categorical feature '{feat_name}'. Check preprocessing.")
                return None, None, None
        logger.info(f"Using embedding specs for: {list(embedding_specs_for_model.keys())}")
    else:
        logger.error("'feature_specs' not found in embedding_info pickle. Cannot configure LSTM."); return None, None, None

    return train_val_df, metadata_json, embedding_specs_for_model

class SeriesLSTMDataset(Dataset): # Same as your last provided version
    def __init__(self, X_cat_series: Dict[str, np.ndarray], X_num_series: Optional[np.ndarray], y_series: np.ndarray):
        self.X_cat_series_torch = {}
        for key, data in X_cat_series.items():
            if isinstance(data, pd.Series): data = data.to_numpy() 
            if not (isinstance(data, np.ndarray) and np.issubdtype(data.dtype, np.integer)):
                logger.error(f"Dataset Error: Cat feat '{key}' not int array. Dtype: {data.dtype if isinstance(data, np.ndarray) else type(data)}. Sample: {data[:3] if isinstance(data, np.ndarray) and data.size > 0 else 'N/A'}")
                try: data = np.array(data, dtype=np.int64)
                except ValueError as e_conv: raise TypeError(f"Cat feat '{key}' must be integer array for Dataset.") from e_conv
            self.X_cat_series_torch[key] = torch.tensor(data, dtype=torch.long)

        if X_num_series is None or X_num_series.size == 0:
            self.X_num_series_torch = torch.empty(len(y_series), 0, dtype=torch.float32)
        else:
            if isinstance(X_num_series, pd.DataFrame): X_num_series = X_num_series.to_numpy()
            if not (isinstance(X_num_series, np.ndarray) and np.issubdtype(X_num_series.dtype, np.floating)):
                logger.error(f"Dataset Error: Num feats not float array. Dtype: {X_num_series.dtype}")
                try: X_num_series = np.array(X_num_series, dtype=np.float32)
                except ValueError as e_conv_num: raise TypeError("Num feats must be float array.") from e_conv_num
            self.X_num_series_torch = torch.tensor(X_num_series, dtype=torch.float32)
        
        self.y_series_torch = torch.tensor(y_series, dtype=torch.float32)
        self.categorical_feature_keys = list(self.X_cat_series_torch.keys())

    def __len__(self) -> int: return len(self.y_series_torch)
    def __getitem__(self, index: int) -> Tuple[Dict[str, torch.Tensor], torch.Tensor, torch.Tensor]:
        cat_s = {k: self.X_cat_series_torch[k][index] for k in self.categorical_feature_keys}
        return cat_s, self.X_num_series_torch[index], self.y_series_torch[index]

def create_sequences_for_single_series( # Same as your last provided version
    series_df_chronological: pd.DataFrame, categorical_col_names: List[str],
    numerical_col_names: List[str], target_col_name: str,
    sequence_length: int, forecast_horizon: int
) -> Optional[Tuple[Dict[str, np.ndarray], np.ndarray, np.ndarray]]:
    for col in categorical_col_names: 
        if col in series_df_chronological.columns:
            is_int_dtype = pd.api.types.is_integer_dtype(series_df_chronological[col].dtype) or \
                           isinstance(series_df_chronological[col].dtype, (pd.Int32Dtype, pd.Int64Dtype))
            if not is_int_dtype:
                logger.error(f"Sequence Creation Error: Cat col '{col}' not integer ({series_df_chronological[col].dtype}).")
                return None
        else: 
            logger.error(f"Sequence Creation Error: Cat col '{col}' not found.")
            return None           
    return create_lstm_sequences_for_series(
        series_data=series_df_chronological, sequence_length=sequence_length,
        forecast_horizon=forecast_horizon, numerical_cols=numerical_col_names,
        categorical_cols=categorical_col_names, target_col=target_col_name
    )

def train_lstm_with_cv( # Corrected call to get_lstm_fold_data
    series_id: str, full_global_df: pd.DataFrame, 
    global_embedding_specs: Dict[str, Tuple[int, int]], numerical_col_names: List[str],
    categorical_col_names: List[str], lstm_hyperparams: Dict, device_name: str
) -> Tuple[Optional[LSTMModel], Optional[StandardScaler], float, Optional[int], Optional[Dict]]:
    validator = IndexBasedWalkForwardValidator(
        initial_train_size=config.CV_CONFIG["initial_train_size"],
        step_size=config.CV_CONFIG["step_size"], max_splits=config.CV_CONFIG["max_splits"],
        gap=config.CV_CONFIG.get("gap", 0)
    )
    # Extract series-specific data for length calculation
    series_df = full_global_df[full_global_df['id'] == series_id].sort_values('date').reset_index(drop=True)
    data_length = len(series_df)
    cv_splits = validator.get_split_indices(data_length)
    if not cv_splits:
        logger.warning(f"Series {series_id}: No CV splits (data len {data_length})."); return None, None, float('inf'), None, None
    
    fold_results_list, model_state_dicts = [], []
    best_val_loss, best_fold_idx = float('inf'), -1
    
    for idx, (train_idx_local, val_idx_local) in enumerate(cv_splits):
        logger.debug(f"  CV Fold {idx + 1}/{len(cv_splits)} for '{series_id}'")
        try:
            # Call get_lstm_fold_data with full global DataFrame so it can extract series by ID
            fold_data = get_lstm_fold_data(
                full_global_df,        # Full DataFrame containing all series
                series_id,             # The specific series ID to extract
                train_idx_local, 
                val_idx_local,
                lstm_hyperparams["sequence_length"],
                numerical_col_names,
                categorical_col_names,
                'sales'  # target_col
            )
            if not fold_data or fold_data['train_samples'] == 0 or fold_data['val_samples'] == 0:
                logger.warning(f"Fold {idx+1}: Insufficient sequences. Skipping."); continue
            
            train_ds = SeriesLSTMDataset(fold_data['train']['X_cat'], fold_data['train']['X_num'], fold_data['train']['y'])
            val_ds = SeriesLSTMDataset(fold_data['val']['X_cat'], fold_data['val']['X_num'], fold_data['val']['y'])
            num_num_feats = fold_data['train']['X_num'].shape[-1] if fold_data['train']['X_num'] is not None and fold_data['train']['X_num'].size > 0 else 0
            
            model = LSTMModel(
                numerical_input_dim=num_num_feats, embedding_specs=global_embedding_specs,
                hidden_dim=lstm_hyperparams["hidden_dim"], num_layers=lstm_hyperparams["num_layers"],
                output_dim=lstm_hyperparams["forecast_horizon"], dropout=lstm_hyperparams["dropout"],
                bidirectional=lstm_hyperparams["bidirectional"], use_attention=lstm_hyperparams["use_attention"],
                quantile_output=lstm_hyperparams["quantile_output"],
                quantile_levels=lstm_hyperparams["quantiles"] if lstm_hyperparams["quantile_output"] else None
            ).to(device_name)
            
            train_dl = DataLoader(train_ds, batch_size=lstm_hyperparams["batch_size"], shuffle=True, num_workers=0)
            val_dl = DataLoader(val_ds, batch_size=lstm_hyperparams["batch_size"], shuffle=False, num_workers=0)
            crit = QuantileLoss(lstm_hyperparams["quantiles"]) if lstm_hyperparams["quantile_output"] else nn.MSELoss()
            opt = optim.Adam(model.parameters(), lr=lstm_hyperparams["learning_rate"], weight_decay=lstm_hyperparams["weight_decay"])
            
            model.train()
            max_fold_epochs = min(config.CV_CONFIG.get("max_epochs_per_fold", 10), lstm_hyperparams["epochs"])
            for _ in range(max_fold_epochs):
                for bc, bn, by in train_dl:
                    bc, bn, by = {k:v.to(device_name) for k,v in bc.items()}, bn.to(device_name), by.to(device_name)
                    opt.zero_grad(); p = model(bc, bn); l = crit(p, by); l.backward(); opt.step()
            
            model.eval(); fold_val_loss = 0.0
            if len(val_dl) > 0:
                with torch.no_grad():
                    for bc, bn, by in val_dl:
                        bc, bn, by = {k:v.to(device_name) for k,v in bc.items()}, bn.to(device_name), by.to(device_name)
                        fold_val_loss += crit(model(bc, bn), by).item()
                avg_loss = fold_val_loss / len(val_dl)
            else: avg_loss = float('inf')
            
            fold_results_list.append({'val_loss': float(avg_loss), 'fold': idx})
            model_state_dicts.append(model.state_dict())
            if avg_loss < best_val_loss: best_val_loss, best_fold_idx = avg_loss, idx
        except Exception as e: logger.warning(f"Fold {idx+1} Error for series '{series_id}': {e}", exc_info=True); continue # Log series_id
            
    if not fold_results_list or best_fold_idx == -1:
        logger.warning(f"Series {series_id}: CV training yielded no valid results."); return None, None, float('inf'), None, None
    
    cv_stats = aggregate_cv_metrics(fold_results_list)
    final_model_num_features = len(numerical_col_names) 
    best_model = LSTMModel(
        numerical_input_dim=final_model_num_features, embedding_specs=global_embedding_specs,
        hidden_dim=lstm_hyperparams["hidden_dim"], num_layers=lstm_hyperparams["num_layers"],
        output_dim=lstm_hyperparams["forecast_horizon"], dropout=lstm_hyperparams["dropout"],
        bidirectional=lstm_hyperparams["bidirectional"], use_attention=lstm_hyperparams["use_attention"],
        quantile_output=lstm_hyperparams["quantile_output"],
        quantile_levels=lstm_hyperparams["quantiles"] if lstm_hyperparams["quantile_output"] else None
    ).to(device_name)
    best_model.load_state_dict(model_state_dicts[best_fold_idx])
    
    scaler = None
    if numerical_col_names and not series_df[numerical_col_names].empty:
        scaler = StandardScaler().fit(series_df[numerical_col_names])
    logger.debug(f"Series {series_id}: CV best fold {best_fold_idx + 1}, loss={best_val_loss:.4f}")
    return best_model, scaler, best_val_loss, best_fold_idx, cv_stats

def train_lstm_for_single_series( # Same as your last provided version, but relies on corrected train_lstm_with_cv call
    id_value: str, series_train_val_df: pd.DataFrame, full_global_df: pd.DataFrame,
    global_embedding_specs: Dict[str, Tuple[int, int]], numerical_col_names: List[str],
    categorical_col_names: List[str], lstm_hyperparams: Dict, base_model_output_dir: Path,
    device_name: str, force_retrain: bool
) -> Dict:
    from datetime import datetime 
    import torch 
    from sklearn.preprocessing import StandardScaler 
    import pickle 
    from pathlib import Path 
    set_random_seed(config.RANDOM_SEED) 

    sanitized_id = sanitize_for_path(id_value)
    model_subdir = base_model_output_dir / sanitized_id
    model_subdir.mkdir(parents=True, exist_ok=True)
    model_file, scaler_file, meta_file = model_subdir/"lstm_model.pt", model_subdir/"scaler.pkl", model_subdir/"training_meta.json"
    
    if model_file.exists() and (scaler_file.exists() or not numerical_col_names) and meta_file.exists() and not force_retrain:
        logger.debug(f"Model for '{id_value}' exists. Skipping.")
        try:
            with open(meta_file, 'r') as f: meta = json.load(f)
            meta["status"] = "skipped_exists"; return meta
        except: pass

    logger.debug(f"Training LSTM for '{id_value}' on '{device_name}'.")
    model_to_save, series_scaler, best_loss, epochs_run, cv_summary = None, None, float('inf'), 0, None

    try:
        series_data_sorted = series_train_val_df.sort_values('date').reset_index(drop=True)
        if config.CV_CONFIG.get("use_cv", False):
            logger.debug(f"Series '{id_value}': Using CV.")
            model_to_save, series_scaler, best_loss, _, cv_summary = train_lstm_with_cv(
                id_value, full_global_df,  # Pass full DataFrame for CV context
                global_embedding_specs, numerical_col_names, categorical_col_names,
                lstm_hyperparams, device_name
            )
            epochs_run = config.CV_CONFIG.get("max_epochs_per_fold", 10) 
            if model_to_save is None: raise ValueError("CV failed to produce a model.")
        else: 
            data_len = len(series_data_sorted)
            val_iters = lstm_hyperparams.get("val_iterations", config.LSTM_CONFIG.get("val_iterations",1))
            val_size_ideal = lstm_hyperparams["forecast_horizon"] * val_iters
            val_size_max_prop = data_len // 5 
            val_size = min(max(1, val_size_ideal), val_size_max_prop if val_size_max_prop > 0 else max(0, data_len -1))
            min_train_len_for_seq = lstm_hyperparams["sequence_length"] 
            if data_len - val_size < min_train_len_for_seq :
                 val_size = max(0, data_len - min_train_len_for_seq) 
                 logger.warning(f"Series '{id_value}': Adjusted val_size to {val_size} to ensure min_train_len {min_train_len_for_seq} from total {data_len}")
            if data_len - val_size < min_train_len_for_seq or (val_size == 0 and data_len < min_train_len_for_seq + lstm_hyperparams["forecast_horizon"]):
                 raise ValueError(f"Insufficient data for train/val (total: {data_len}, train_min_needed_for_seq: {min_train_len_for_seq}, val_size: {val_size}).")
            
            train_df, val_df = series_data_sorted.iloc[:-val_size], series_data_sorted.iloc[-val_size:]
            if numerical_col_names and not train_df[numerical_col_names].empty: # Check if num_cols actually exist in train_df
                series_scaler = StandardScaler()
                train_num_scaled = series_scaler.fit_transform(train_df[numerical_col_names])
                train_num_df = pd.DataFrame(train_num_scaled, columns=numerical_col_names, index=train_df.index)
                if not val_df.empty and not val_df[numerical_col_names].empty:
                    val_num_scaled = series_scaler.transform(val_df[numerical_col_names])
                    val_num_df = pd.DataFrame(val_num_scaled, columns=numerical_col_names, index=val_df.index)
                else: val_num_df = pd.DataFrame(columns=numerical_col_names, index=val_df.index) 
            else: 
                series_scaler = None 
                train_num_df, val_num_df = pd.DataFrame(index=train_df.index), pd.DataFrame(index=val_df.index)

            train_seq_df = pd.concat([train_df[categorical_col_names + ['sales', 'date']], train_num_df], axis=1)
            context_for_val_sequences = train_seq_df.tail(lstm_hyperparams["sequence_length"])
            val_prep_df = pd.concat([val_df[categorical_col_names + ['sales', 'date']], val_num_df], axis=1)
            data_for_val_seq = pd.concat([context_for_val_sequences, val_prep_df]).sort_values('date').reset_index(drop=True)

            train_seqs = create_sequences_for_single_series(train_seq_df, categorical_col_names, numerical_col_names, 'sales', lstm_hyperparams["sequence_length"], lstm_hyperparams["forecast_horizon"])
            val_seqs = create_sequences_for_single_series(data_for_val_seq, categorical_col_names, numerical_col_names, 'sales', lstm_hyperparams["sequence_length"], lstm_hyperparams["forecast_horizon"])

            if not train_seqs or not val_seqs or train_seqs[2].shape[0]==0 or val_seqs[2].shape[0]==0:
                raise ValueError("Not enough data for train/val sequences after split.")

            train_ds, val_ds = SeriesLSTMDataset(*train_seqs), SeriesLSTMDataset(*val_seqs)
            train_dl = DataLoader(train_ds, lstm_hyperparams["batch_size"], shuffle=True, num_workers=0)
            val_dl = DataLoader(val_ds, lstm_hyperparams["batch_size"], shuffle=False, num_workers=0) if len(val_ds) > 0 else None
            
            num_numerical_features = len(numerical_col_names) if numerical_col_names else 0
            model_to_save = LSTMModel(
                numerical_input_dim=num_numerical_features, embedding_specs=global_embedding_specs,
                hidden_dim=lstm_hyperparams["hidden_dim"], num_layers=lstm_hyperparams["num_layers"],
                output_dim=lstm_hyperparams["forecast_horizon"], dropout=lstm_hyperparams["dropout"],
                bidirectional=lstm_hyperparams["bidirectional"], use_attention=lstm_hyperparams["use_attention"],
                quantile_output=lstm_hyperparams["quantile_output"],
                quantile_levels=lstm_hyperparams["quantiles"] if lstm_hyperparams["quantile_output"] else None
            ).to(device_name)
            opt = optim.Adam(model_to_save.parameters(), lr=lstm_hyperparams["learning_rate"], weight_decay=lstm_hyperparams["weight_decay"])
            crit = QuantileLoss(lstm_hyperparams["quantiles"]) if lstm_hyperparams["quantile_output"] else nn.MSELoss()
            sched = optim.lr_scheduler.ReduceLROnPlateau(opt, 'min', patience=lstm_hyperparams["patience"]//2, factor=0.5, verbose=False)
            
            patience_count = 0
            for epoch in range(1, lstm_hyperparams["epochs"] + 1):
                model_to_save.train(); e_train_loss = 0.0
                for bc, bn, by in train_dl:
                    bc_d, bn_d, by_d = {k:v.to(device_name) for k,v in bc.items()}, bn.to(device_name), by.to(device_name)
                    opt.zero_grad(); p=model_to_save(bc_d,bn_d); l=crit(p,by_d); l.backward()
                    torch.nn.utils.clip_grad_norm_(model_to_save.parameters(), lstm_hyperparams["gradient_clip"]); opt.step()
                    e_train_loss += l.item()
                
                avg_epoch_val_loss = float('inf')
                if val_dl : 
                    model_to_save.eval(); e_val_loss = 0.0
                    with torch.no_grad():
                        for bc_v, bn_v, by_v in val_dl:
                            bc_v_d={k:v.to(device_name) for k,v in bc_v.items()}
                            bn_v_d,by_v_d=bn_v.to(device_name),by_v.to(device_name)
                            e_val_loss += crit(model_to_save(bc_v_d,bn_v_d),by_v_d).item()
                    avg_epoch_val_loss = e_val_loss / len(val_dl)
                    sched.step(avg_epoch_val_loss)
                else: logger.warning(f"'{id_value}', Ep {epoch}: No val data. Using train loss for logging."); avg_epoch_val_loss = e_train_loss / len(train_dl)
                
                epochs_run = epoch
                if avg_epoch_val_loss < best_loss:
                    best_loss, patience_count = avg_epoch_val_loss, 0; torch.save(model_to_save.state_dict(), model_file)
                    logger.debug(f"'{id_value}', Ep {epoch}: New best val_loss: {best_loss:.4f}. Saved.")
                else: patience_count += 1
                if patience_count >= lstm_hyperparams["patience"] and val_dl : 
                    logger.info(f"'{id_value}': Early stop Ep {epoch}. Best val_loss: {best_loss:.4f}"); break
        
        if model_to_save is None: raise ValueError("Model is None after training.")
        if not model_file.exists() or (config.CV_CONFIG.get("use_cv", False) and best_loss != float('inf')) : 
            torch.save(model_to_save.state_dict(), model_file)
            logger.info(f"Final model state for '{id_value}' saved to {model_file}")

        if series_scaler and numerical_col_names:
            with open(scaler_file, 'wb') as f: pickle.dump(series_scaler, f)
        
        final_meta = {"id_value": id_value, "status": "success", "epochs_trained": epochs_run, 
                      "best_val_loss": float(best_loss) if best_loss != float('inf') else None, 
                      "model_file": model_file.name, 
                      "scaler_file": scaler_file.name if series_scaler and numerical_col_names else None,
                      "model_output_subdir": str(model_subdir.relative_to(base_model_output_dir.parent)),
                      "numerical_features_scaled": numerical_col_names, "categorical_features_used": categorical_col_names,
                      "embedding_specs_used": {k: list(v) for k, v in global_embedding_specs.items()},
                      "training_timestamp": datetime.now().isoformat(), "cv_enabled": config.CV_CONFIG.get("use_cv", False),
                      "cv_results": cv_summary}
        with open(meta_file, 'w') as f: json.dump(final_meta, f, indent=4)
        logger.info(f"Series '{id_value}': Training successful. Val loss: {best_loss:.4f}, Epochs: {epochs_run}.")
        return final_meta
    except Exception as e:
        logger.error(f"Series '{id_value}': Training failed: {e}", exc_info=True)
        return {"id_value": id_value, "status": "error_exception", "error_message": str(e)}

def main():
    args = parse_arguments()
    set_random_seed(config.RANDOM_SEED) 
    
    # Force CPU usage if specified
    if args.device == "cpu":
        import os
        os.environ["CUDA_VISIBLE_DEVICES"] = ""
        torch.cuda.is_available = lambda: False
        logger.info("🔧 Forced CPU-only mode: CUDA disabled")
    
    start_time_main = time.time()
    logger.info("="*70 + f"\nSTARTING LSTM TRAINING (MC SYNTHETIC)\nInput: {args.input_dir}\n" + "="*70)

    input_path = Path(args.input_dir)
    lstm_out_dir = config.get_model_storage_dir("lstm") 
    lstm_out_dir.mkdir(parents=True, exist_ok=True)
    logger.info(f"LSTM models output: {lstm_out_dir}")

    load_res = load_split_data_and_metadata(input_path)
    if load_res is None or load_res[0] is None: logger.critical("Data loading failed. Exiting."); return
    df_global, _, global_embed_specs = load_res
    if df_global is None : logger.critical("DataFrame is None after loading. Exiting."); return 
    if not global_embed_specs: logger.critical("Embedding specs missing. Exiting."); return

    cat_feats = [c for c in global_embed_specs.keys() if c in df_global.columns and pd.api.types.is_integer_dtype(df_global[c].dtype)]
    logger.info(f"Using {len(cat_feats)} integer categorical features for model: {cat_feats}")
    if len(cat_feats) != len(global_embed_specs.keys()):
        missing_specs_cols = set(global_embed_specs.keys()) - set(cat_feats)
        logger.warning(f"Mismatch: {len(global_embed_specs.keys())} specs, but only {len(cat_feats)} are valid int columns. Missing/invalid: {list(missing_specs_cols)}")

    num_feats = [c for c in df_global.columns if c not in (['id','date','sales'] + cat_feats) and pd.api.types.is_numeric_dtype(df_global[c])]
    logger.info(f"Using {len(num_feats)} numerical features for model: {num_feats}")
    
    series_ids = df_global['id'].unique()
    if args.limit_series and args.limit_series > 0 : series_ids = series_ids[:args.limit_series]
    logger.info(f"Total unique series to process: {len(series_ids)}")

    tasks = []
    for sid_val in series_ids: 
        series_data = df_global[df_global['id'] == sid_val].copy()
        if series_data.empty: logger.warning(f"No data for series '{sid_val}'. Skipping."); continue
        # Pass both series-specific data and full global DataFrame for CV context
        tasks.append((sid_val, series_data, # series_train_val_df for this specific series
                      df_global,         # full_global_df containing all series for CV context
                      global_embed_specs, num_feats, cat_feats, 
                      config.LSTM_CONFIG, lstm_out_dir, args.device, args.force))

    results = {}; 
    eff_jobs = args.parallel_jobs
    if args.device=="cuda" and args.parallel_jobs > 1: logger.warning("CUDA with >1 job, forcing sequential."); eff_jobs=1

    if eff_jobs <= 1 or not tasks:
        logger.info(f"Running LSTM training sequentially for {len(tasks)} tasks.")
        for task_args in tqdm(tasks, desc="Training LSTMs (Sequential)"):
            r = train_lstm_for_single_series(*task_args)
            if r: results[r.get("id_value", task_args[0])] = r 
    elif tasks:
        logger.info(f"Running LSTM training in parallel ({eff_jobs} CPU workers).")
        cpu_tasks = [list(t) for t in tasks]
        for task_set in cpu_tasks: task_set[8] = "cpu" # device_name index
        cpu_tasks_tuples = [tuple(t) for t in cpu_tasks] 
        
        with ProcessPoolExecutor(max_workers=eff_jobs) as executor:
            future_to_id = {executor.submit(train_lstm_for_single_series, *task_args_tuple): task_args_tuple[0] 
                            for task_args_tuple in cpu_tasks_tuples}
            for future in tqdm(as_completed(future_to_id), total=len(future_to_id), desc="Training LSTMs (Parallel)"):
                task_id = future_to_id[future]
                try:
                    res = future.result()
                    if res: results[res.get("id_value", task_id)] = res
                except Exception as e: 
                    logger.error(f"Parallel task for series '{task_id}' failed: {e}", exc_info=True)
                    results[task_id]={"id_value":task_id, "status":"error_future_exception", "error_message":str(e)}
    
    total_t = time.time()-start_time_main; logger.info(f"Overall LSTM training script completed in {total_t:.2f}s.")
    summary_out_dir = config.RESULTS_BASE_DIR/"lstm_training_summary"; summary_out_dir.mkdir(parents=True,exist_ok=True)
    ts_str = datetime.now().strftime('%Y%m%d_%H%M%S')
    summary_fpath = summary_out_dir/f"lstm_mc_training_summary_{ts_str}.json"
    
    n_ok=sum(1 for r in results.values() if r.get("status")=="success")
    n_err=sum(1 for r in results.values() if r.get("status","").startswith("error"))
    n_skip=sum(1 for r in results.values() if r.get("status")=="skipped_exists")

    summary_data = {"ts":datetime.now().isoformat(), "type":"lstm_mc", "series_total":len(series_ids),
                    "limit":args.limit_series or "all", "force":args.force, "device":args.device, "jobs":args.parallel_jobs,
                    "eff_jobs":eff_jobs, "stats":{"ok":n_ok, "err":n_err, "skip":n_skip},
                    "results":{k:v for k,v in results.items()}, "runtime_s":total_t, "params":config.LSTM_CONFIG,
                    "embed_specs_used":{k:list(v) if isinstance(v, tuple) else v for k,v in global_embed_specs.items()}
                   }
    with open(summary_fpath,'w') as f: json.dump(summary_data,f,indent=4)
    logger.info(f"Summary: {summary_fpath}")
    logger.info("="*70 + f"\nLSTM TRAINING COMPLETE! OK: {n_ok}, Err: {n_err}, Skip: {n_skip}\n" + "="*70)

if __name__ == "__main__":
    main()