# Project Structure

Ok so this is basically how I organized the supply chain forecasting project after working on it for a while. Its gotten pretty big as I added more experiments but ill try to explain the main parts.

## Main Folders

```
supply_chain/
├── src/                       # all the main code goes here
│   ├── cross_validation/      # time series CV stuff
│   ├── evaluators/            # different metrics for evaluation
│   ├── hpo/                   # hyperparameter optimization code
│   ├── models/                # where the actual models live (lstm & classical)
│   └── training/              # training scripts
├── data/                      # data storage (gitignored mostly)
│   ├── m5_splices/            # smaller data slices for experiments
│   ├── synthetic_monte_carlo/ # monte carlo simulation data
│   ├── raw/                   # original m5 competition files
│   └── preprocessed/          # processed data from experiments
├── experiment_workflows/      # full experiment pipelines
│   ├── exp1_per_item_store_models/  # main m5 experiment comparing models
│   └── exp_per_item_store_mc/       # monte carlo synthetic data experiment
├── scripts/                   # data prep utilities
├── trained_models/            # saved model files (created during runs)
├── predictions/               # model predictions (created during runs)
├── results/                   # evaluation results (created during runs)
├── logs/                      # experiment logs (created during runs)
├── docs/                      # extra documentation
├── requirements.txt           # python packages needed
├── CLAUDE.md                  # detailed code style guidelines
├── CHANGES.md                 # log of all code changes
└── README.md                  # main readme
```

## The Important Parts

### `src/` - Where the Stuff Happens

This is where all the core implementation code lives:
- `config.py` - configuration settings and paths
- `data_loader.py` - functions to load the m5 data
- `feature_engineering.py` - creates all the features (this ones pretty big)
- `cv_utils.py` - cross validation helper functions
- `utils.py` - random utility functions i needed
- `sbc.py` - simulation based calibration (still working on this)

The subdirectories have:
- `models/` - the lstm implementation (without BatchNorm for scale consistency) and classical methods (arima, croston, etc)
- `evaluators/` - calculates mae, rmse, wrmsse and other metrics
- `hpo/` - bayesian optimization code for tuning hyperparameters
- `training/` - scripts to actually train the models
- `cross_validation/` - expanding window cv for time series

### `data/` - Data Files

Most of this is gitignored cause the files are huge. Main subfolders:
- `m5_splices/` - pre-computed feature sets from the m5 competition data (different sizes for testing)
- `synthetic_monte_carlo/` - synthetic intermittent demand data for controlled experiments
- `raw/` - original m5 files if you need them
- `preprocessed/` - output from preprocessing scripts

### `experiment_workflows/` - Running Experiments

There are two main experiments:

**Experiment 1: M5 Real Data (`exp1_per_item_store_models/`)**
- uses actual walmart sales data from m5 competition
- compares lstm vs classical methods on real retail data
- has numbered scripts (0-7) that run the full pipeline

**Experiment 2: Monte Carlo Synthetic (`exp_per_item_store_mc/`)**
- uses synthetic intermittent demand patterns
- tests methods on controlled data with known properties
- same numbered pipeline structure

Both experiments start with hyperparameter optimization (script 0) using bayesian optimization, then proceed through:
1. preprocess data
2. train classical models (arima, croston, tsb, etc)
3. train lstm models (with embeddings and attention)
4. generate predictions
5. evaluate everything (wrmsse, mae, rmse, etc)
6. statistical comparison (paired t-tests)
7. visualizations

### `scripts/` - Data Utilities

Helper scripts for data preparation:
- `1_create_feature_rich_splices.py` - creates the m5 data subsets with all 55 features
- `gen_mc_data.py` - generates synthetic monte carlo demand data

## How to Use

Usually the workflow goes:
1. prepare your data (or use existing splices)
2. maybe run hyperparameter optimization if you want better params
3. run the experiment workflow scripts
4. check results in the output folders

The output folders get created automatically when you run stuff.

## Important Technical Notes

- lstm architecture deliberately excludes batchnorm to ensure predictions stay in original sales units
- target variables (sales) are never scaled during training - the model learns output scaling instead
- uses per-item models (each time series gets its own model) not global models
- time series cross-validation prevents data leakage with expanding window approach
- exactly 55 engineered features including lags, rolling stats, calendar, price, and intermittency metrics
- categorical features use embeddings in the lstm (stores, items, months, etc)

## Random Notes

- some scripts take command line args, check the argparse stuff at the top
- the m5 data is pretty big so i made smaller slices for testing
- lstm training can take a while depending on your gpu (or cpu if no gpu)
- logs go to the logs/ folder with timestamps
- all outputs are organized by experiment name and timestamp

Hopefully this helps make sense of the project structure! Let me know if anything is confusing.