# Item-Level Unified Experiment Workflow

This directory contains the primary experimental pipeline for comparing LSTM and classical forecasting methods on M5 data.

## Experiment Design

Compares:
- **LSTM**: Individual neural networks trained per item with embedding layers
- **Classical**: ARIMA, Croston, and TSB implementations via statsforecast

All models generate 28-day multi-step forecasts following M5 evaluation protocol.

## Pipeline Components

1. `1_preprocess_exp1.py` - Data splitting and feature engineering
2. `2_train_classical_models.py` - Classical model training
3. `3_train_lstm_models.py` - Neural network training
4. `4_predict_models.py` - Forecast generation
5. `5_evaluate_predictions.py` - Metric calculation
6. `6_statistical_comparison.py` - Statistical analysis
7. `7_visualize_results.py` - Visualizations (placeholder)
8. `run_unified_experiment.py` - Main experiment runner

## Execution

### Full Comparison
```bash
python run_unified_experiment.py \
    --models-to-run lstm arima croston \
    --run-tag "exp1_full"
```

### Model Subsets
```bash
# Neural models only
python run_unified_experiment.py \
    --models-to-run lstm \
    --run-tag "lstm_only"

# Classical models only
python run_unified_experiment.py \
    --models-to-run arima croston tsb \
    --run-tag "classical_only"
```

### Configuration

Modify `config_exp1.py` for:
- Data file paths
- Model parameters
- Item limits for development
- Cross-validation configuration

### Resuming Experiments
```bash
# Skip completed stages
python run_unified_experiment.py \
    --models-to-run lstm \
    --run-tag "existing" \
    --skip-preprocessing \
    --skip-training
```

## Output Organization

```
results/exp1_per_item_lstm_vs_classical_[RUN_TAG]/
├── lstm/evaluation/           # Neural network metrics
├── arima/evaluation/          # ARIMA metrics
├── croston/evaluation/        # Croston metrics
└── statistical_comparison/    # Comparative analysis

predictions/exp1_per_item_lstm_vs_classical_[RUN_TAG]/
├── lstm/                      # Neural forecasts
├── arima/                     # ARIMA forecasts
└── croston/                   # Croston forecasts

trained_models/exp1_per_item_lstm_vs_classical/
├── classical/                 # Statistical models
└── deep_learning/             # Neural models
```

## Cross-Validation

Enable time series CV by setting `CV_CONFIG["use_cv"] = True` in configuration.

## Implementation Notes

- Start with reduced `limit_items` for development
- Per-item modeling approach (individual models per series)
- Point and quantile forecast support
- Statistical testing includes paired comparisons and effect sizes
- Results tagged with run identifier for tracking