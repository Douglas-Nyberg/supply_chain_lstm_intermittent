# Supply Chain Demand Forecasting: LSTM vs Classical Methods

So this is my implementation comparing deep learning with traditional forecasting methods on retail demand data. Basically trying to figure out if fancy neural networks are actually better than the old school statistical methods for predicting demand, especially for intermittent demand patterns.

## What this is about

The main research question is: when do LSTMs actually beat classical methods for retail forecasting? Im comparing:
- **LSTM networks**: Deep learning approach with embeddings for categorical features and attention mechanism
- **Classical methods**: ARIMA, Croston's method, TSB, ETS, Theta, and others designed for intermittent demand

The project includes two main experiments:
1. **Real M5 data**: Using actual Walmart sales data from the M5 competition
2. **Monte Carlo simulation**: Synthetic intermittent demand patterns for controlled testing

Trying to see when its worth using the computationally expensive neural nets vs just sticking with simpler methods.

## Quick Start

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

### 2. Get the M5 data

You gotta download the data from Kaggle yourself (its too big for github):

1. Go to https://www.kaggle.com/competitions/m5-forecasting-accuracy
2. Make a Kaggle account and accept the competition rules
3. Download these files from the Data tab:
   - `calendar.csv`
   - `sales_train_validation.csv` 
   - `sales_train_evaluation.csv`
   - `sell_prices.csv`
   - `sample_submission.csv`
4. Put all the csv files in `data/raw/` folder

### 3. Create the feature-engineered data

Once you have the raw data, you need to run the feature engineering script:

```bash
# This creates smaller data slices with all the features computed
python scripts/1_create_feature_rich_splices.py
```

This will create preprocessed datasets in `data/m5_splices/` that the models can actually use.

### 4. Configure and run experiments

Each experiment has its own config file that you need to check/update first:

For the M5 real data experiment:
```bash
# First check/edit the config file:
# experiment_workflows/exp1_per_item_store_models/config_exp1.py
# Update DATA_CONFIG['splice_file'] to point to your data slice

# Then run from the project root directory:
cd experiment_workflows/exp1_per_item_store_models
python run_unified_experiment.py
```

For the Monte Carlo synthetic data experiment:
```bash
# First check/edit the config file:
# experiment_workflows/exp_per_item_store_mc/config_mc_master.py
# Update paths and parameters as needed

# Then run from the project root directory:
cd experiment_workflows/exp_per_item_store_mc
python run_mc_experiment.py
```




## Whats implemented so far

### Models that actually work:
- **LSTM**: Per-item models with embedding layers and attention mechanism. Uses learned output scaling instead of BatchNorm to keep predictions in original sales units
- **ARIMA**: Auto-regressive integrated moving average (the classic)
- **Croston & TSB**: Specifically designed for intermittent demand
- **ETS**: Exponential smoothing state space models
- **Theta**: Theta method for decomposition
- **Naive methods**: Simple baselines for comparison

### Features ive added:
- 28-day ahead forecasting (M5 competition standard)
- Bayesian optimization for hyperparameters (way more efficent than grid search)
- Proper time series cross-validation with expanding window (no data leakage)
- Full M5 evaluation metrics including WRMSSE
- Statistical significance testing with paired t-tests
- 55 engineered features including lags, rolling stats, calendar, price, and intermittency metrics
- Per-item modeling approach (each series gets its own model)

## Project Structure

```
src/                          # All the core code
├── models/                   # Model implementations
├── evaluators/               # Metrics (theres so many)
├── hpo/                      # Hyperparameter optimization 
├── training/                 # Training scripts
└── cross_validation/         # Time series CV

experiment_workflows/         # Main experiment pipelines
├── exp1_per_item_store_models/  # M5 real data experiment
└── exp_per_item_store_mc/       # Monte Carlo synthetic experiment

data/                         # Data goes here (mostly gitignored)
├── m5_splices/              # Preprocessed data slices
├── synthetic_monte_carlo/   # Synthetic data
└── raw/                     # Original M5 files

scripts/                      # Data prep utilities
```

## How to actually use this

### Running full experiments

Both experiment workflows follow the same numbered pipeline:
0. Hyperparameter optimization (bayesian optimization)
1. Data preprocessing 
2. Train classical models (arima, croston, tsb, etc)
3. Train LSTM models (warning: takes a while)
4. Generate predictions
5. Calculate metrics (wrmsse, mae, rmse, etc)
6. Statistical comparisons
7. Visualizations

You can run the whole pipeline with `run_unified_experiment.py` or `run_mc_experiment.py`, or run individual scripts.

### Just want to optimize hyperparams?

Each experiment has HPO scripts (the ones starting with 0):
```
# For M5 experiment:
python experiment_workflows/exp1_per_item_store_models/0_run_hpo_lstm_bo.py

# For Monte Carlo:
python experiment_workflows/exp_per_item_store_mc/0_hpo_lstm_mc.py
```

### Training individual models

If you just want to test one model without the full pipeline, check the individual training scripts in each workflow directory. Start with small data slices to make sure everything works!

## Configuration

Each experiment has its own config file:
- `experiment_workflows/exp1_per_item_store_models/config_exp1.py` - M5 experiment settings
- `experiment_workflows/exp_per_item_store_mc/config_mc_master.py` - Monte Carlo settings

Global settings in `src/config.py`:
- Forecast horizon (28 days for M5)
- Cross-validation parameters
- File paths and directories
- Feature engineering flags

## Current research questions

1. When do LSTMs actually beat classical methods? 
2. How does performance change with intermittent vs smooth demand?
3. Which features matter most for neural networks?
4. Is the extra computation worth the accuracy gains?
5. How do models perform on different product categories?

## Known issues 

- GPU memory problems with large datasets (working on better batching)
- Some ARIMA models fail on really sparse series
- Cross-validation is slow with lots of items
- Need to add more ensemble methods
- Quantile forecasts still being improved

## Technical stuff

Built with:
- PyTorch for neural networks
- statsforecast for classical methods  
- scikit-learn for ML utilities
- pandas/numpy for data manipulation
- optuna for bayesian optimization

Check `requirements.txt` for everything you need. If somethings not working its probably a dependency issue.

## Contact

Questions or problems:

Douglas Nyberg  
Physics Department  
Purdue University  
dnyberg@purdue.edu

