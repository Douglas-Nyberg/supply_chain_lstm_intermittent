# Monte Carlo Per-Item-Store Synthetic Data Forecasting Experiment

This directory contains a complete workflow for comparing LSTM and classical forecasting methods using per-item-store modeling on synthetic demand data generated using Monte Carlo simulation.

## Overview

The workflow generates synthetic intermittent demand data using truncated normal distributions and hyperbolic secant squared (sech²) easing functions, then runs the same forecasting pipeline used for M5 data but with simplified features.

## Key Features

- **Synthetic Data Generation**: Uses `scripts/gen_mc_data.py` based on Noah's toy data generation approach
- **M5 Compatibility**: Converts synthetic data to M5 CSV format for pipeline compatibility
- **Simplified Features**: Excludes M5-specific features (events, SNAP, holidays, complex pricing)
- **Complete Pipeline**: Preprocessing, training, prediction, evaluation, and visualization

## Quick Start

### 1. Test the Complete Workflow

```bash
# Run the test script to verify everything works
cd experiment_workflows/exp_per_item_store_mc
python test_mc_workflow.py
```

This will:
- Generate synthetic data (50 items, 3 years)
- Convert to M5 format
- Run preprocessing
- Verify data integrity at each step

### 2. Run the Full Experiment

```bash
# Run the complete forecasting experiment
python run_mc_experiment.py --models-to-run lstm croston tsb
```

## Files and Components

### Core Generation Scripts
- **`scripts/gen_mc_data.py`**: Generates synthetic demand data in HDF5 format using config_mc_master.py
- **`convert_hdf5_to_m5.py`**: Converts HDF5 to M5-compatible CSV format

### Configuration
- **`config_mc_master.py`**: Master configuration with profiles for different experiment scales
- **`config_exp_per_item_store_mc.py`**: Experiment configuration with simplified features
- Key settings:
  - `use_price_features = False`
  - `use_event_features = False` 
  - `use_snap_features = False`
  - `use_calendar_features = True` (basic date features only)

### Workflow Scripts
1. **`1_preprocess_exp_per_item_store_mc.py`**: Simplified preprocessing for synthetic data
2. **`2_train_classical_models_mc.py`**: Train classical models (ARIMA, Croston, TSB, etc.)
3. **`3_train_lstm_models_mc.py`**: Train LSTM models with simplified features
4. **`4_predict_models_mc.py`**: Generate predictions for all models
5. **`5_evaluate_predictions_mc.py`**: Calculate performance metrics
6. **`6_statistical_comparison_mc.py`**: Statistical significance testing
7. **`7_visualize_results_mc.py`**: Generate plots and visualizations

### Test and Utilities
- **`test_mc_workflow.py`**: Comprehensive test script
- **`diff.md`**: Detailed comparison with original toy data approach

## Data Generation Parameters

The synthetic data is configured in `config_mc_master.py` MC_DATA_CONFIG:

```
num_items = 50                    # Number of synthetic items
num_years = 3                     # Years of demand history
HEIGHT_RANGE = {"LOW": 1, "HIGH": 15}     # Demand spike amplitudes  
DELTA_TIME_RANGE = {"LOW": 7, "HIGH": 60} # Days between spikes
MAX_EASE_RANGE = 3                # Sech² easing spread (days)
```

## Generated Data Structure

### HDF5 Output (from scripts/gen_mc_data.py)
- `daily_demand_item_0`, `daily_demand_item_1`, ... (daily demand series)
- `per_item_serviceable_amounts`, `per_item_unserviceable_amounts` (initial stock)
- `rand_procurement_cost_per_item`, `rand_repair_cost_per_item` (costs)
- `demand_parameters_item_0`, ... (generation parameters)

### M5 CSV Output (from convert_hdf5_to_m5.py)
- **`calendar.csv`**: Basic date features (date, weekday, month, year)
- **`sales_train_validation.csv`**: Demand data in M5 item hierarchy format
- **`sell_prices.csv`**: Optional constant placeholder prices

## Customization

### Modify Generation Parameters
Edit `config_mc_master.py` MC_DATA_CONFIG to change:
- Dataset size (`num_items`, `num_years`)
- Demand characteristics (`HEIGHT_RANGE`, `DELTA_TIME_RANGE`)
- Intermittency patterns (`MAX_EASE_RANGE`)

### Enable/Disable Features
Edit `config_exp_per_item_store_mc.py` to change:
- Feature types used in modeling
- Model hyperparameters
- Output directories

### Model Selection
Choose which models to run:
```bash
# Classical models only
python run_mc_experiment.py --models-to-run croston tsb arima

# LSTM only  
python run_mc_experiment.py --models-to-run lstm

# All available models
python run_mc_experiment.py --models-to-run lstm croston tsb arima ets ses
```

## Expected Results

The synthetic data creates realistic intermittent demand patterns that allow comparison of:

1. **Classical Methods**: ARIMA, Croston, TSB, ETS, SES, Moving Average
2. **Deep Learning**: LSTM with attention and embeddings

Key differences from M5 results:
- **Simplified patterns**: No external events or complex seasonality
- **Pure demand dynamics**: Focus on core forecasting capabilities
- **Controlled environment**: Known data generation process for analysis

## Output Structure

Results are saved in:
```
results/exp_per_item_store_mc_synthetic_demand_comparison/
├── [run_timestamp]/
│   ├── lstm/evaluation/          # LSTM performance metrics
│   ├── croston/evaluation/       # Croston performance metrics  
│   ├── tsb/evaluation/           # TSB performance metrics
│   └── visualizations/           # Comparison plots
```

## Troubleshooting

### Common Issues

1. **Import errors**: Ensure you're in the correct conda environment
2. **Missing dependencies**: Check `requirements.txt` in project root
3. **Data generation fails**: Verify `config_mc_master.py` parameters are valid
4. **Preprocessing errors**: Check that synthetic M5 data was created correctly

### Debug Mode
Run any script with increased logging:
```bash
python script_name.py --log-level DEBUG
```

### Quick Verification
```bash
# Test just data generation
python scripts/gen_mc_data.py -p quick_test -o test_output.h5 -s 42

# Test just conversion
python convert_hdf5_to_m5.py test_output.h5 test_m5_data/

# Test just preprocessing  
python 1_preprocess_exp_per_item_store_mc.py --data-dir test_m5_data/ --limit-items 5
```

## Performance Notes

- **Generation**: ~1-2 minutes for 50 items × 3 years
- **Conversion**: ~30 seconds for typical dataset
- **Preprocessing**: ~1-2 minutes with feature engineering
- **Training**: Classical models ~5-10 minutes, LSTM ~10-20 minutes
- **Total workflow**: ~30-45 minutes for complete experiment

For faster testing, use `-p quick_test` profile or use `--limit-items` flag in preprocessing.