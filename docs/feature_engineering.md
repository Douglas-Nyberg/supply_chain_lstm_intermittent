# Feature Engineering for Supply Chain Forecasting

## Overview

This document describes the feature engineering pipeline for the M5 forecasting project, covering feature creation, processing, and selection strategies for both neural and classical models.

## M5 Dataset Structure

The M5 competition dataset consists of:

- **30,490 items** across 10 stores in 3 states (CA, TX, WI)
- **1,913 days** of daily sales data (2011-01-29 to 2016-06-19)
- **3 product categories**: FOODS, HOBBIES, HOUSEHOLD
- **7 departments** per category
- Hierarchical structure: State → Store → Category → Department → Item

## Data Splicing Strategy

Due to computational constraints, we create strategic subsets ("splices") of the full dataset:

### Splice Creation Process

1. Load M5 raw data (sales, calendar, prices)
2. Apply filtering criteria (store, category, intermittency level)
3. Select items based on coefficient of variation to ensure diverse patterns
4. Transform from wide format (items × days) to long format
5. Engineer features and save as CSV

### Splice Categories

- **Development splices**: Small datasets (10-50 items) for rapid prototyping
- **Training splices**: Medium datasets (100-300 items) for model development
- **Publication splices**: Large datasets (500-5000 items) for final results

## Forecasting Configuration

All models generate 28-day ahead predictions following M5 competition standards:

1. **Multi-step output**: Direct 28-day forecasting (not recursive)
2. **Feature alignment**: Lagged features properly aligned to avoid leakage
3. **Missing value handling**: NaN values in price/lag features imputed with 0.0

## Feature Schema

Processed data contains 55 columns organized as follows:

### Identifiers (9 columns)
```
id          - Full item-store identifier
item_id     - Product code
dept_id     - Department identifier
cat_id      - Category (FOODS/HOBBIES/HOUSEHOLD)
store_id    - Store code (e.g., CA_1)
state_id    - State code
d           - Day index (d_1 to d_1913)
date        - Calendar date
sales       - Target variable
```

### Predictive Features (46 columns)

#### Calendar Features (14 columns)
Temporal patterns and seasonality:

```
weekday          - Day name
wday             - Numeric weekday (1-7)
month            - Month number
year             - Year
day_of_week      - Zero-indexed weekday
day_of_month     - Day of month
day_of_year      - Julian day
quarter          - Quarter (1-4)
week_of_year     - Week number
is_weekend       - Binary weekend indicator

# Cyclical encodings
day_of_week_sin  - Sine encoding of weekday
day_of_week_cos  - Cosine encoding of weekday
month_sin        - Sine encoding of month
month_cos        - Cosine encoding of month
```

#### Event Features (11 columns)
Holiday and special event indicators:

```
event_name_1     - Primary event name
event_type_1     - Primary event category
event_name_2     - Secondary event (if applicable)
event_type_2     - Secondary event category

# SNAP indicators
snap_CA          - California SNAP issuance
snap_TX          - Texas SNAP issuance
snap_WI          - Wisconsin SNAP issuance

# Binary encodings
has_event_name_1 - Primary event indicator
has_event_type_1 - Primary event type indicator
has_event_name_2 - Secondary event indicator
has_event_type_2 - Secondary event type indicator
```

#### Price Features (5 columns)
Pricing and promotion indicators:

```
sell_price           - Current selling price
price_change         - Period-over-period change
price_relative_to_avg - Ratio to historical average
price_7d_avg         - 7-day moving average
is_promotion         - Promotional indicator
```

Note: Price data availability varies by item and store.

#### Lag Features (3 columns)
Historical sales at key intervals:

```
lag_7    - Sales 7 days prior
lag_14   - Sales 14 days prior
lag_28   - Sales 28 days prior
```

#### Rolling Statistics (12 columns)
Moving window aggregations:

```
# 7-day window
rolling_7_mean    - 7-day average
rolling_7_std     - 7-day standard deviation
rolling_7_max     - 7-day maximum
rolling_7_min     - 7-day minimum

# 14-day window
rolling_14_mean, rolling_14_std, rolling_14_max, rolling_14_min

# 28-day window
rolling_28_mean, rolling_28_std, rolling_28_max, rolling_28_min
```

#### Additional Features (1 column)
```
wm_yr_wk  - Walmart fiscal calendar week
```

## Processing Pipeline

### Data Transformation

1. **Wide to Long Format**: Convert M5's wide format (items × days matrix) to long format (row per item-day)
2. **Feature Generation**: Sequential application of feature engineering functions
3. **Validation**: Ensure temporal consistency and no data leakage

### Feature Creation Order

1. Calendar features (from date)
2. Event features (from calendar file)
3. Price features (from sell_prices file)
4. Lag features (from historical sales)
5. Rolling statistics (from lag features)

### Categorical Feature Handling

#### Label Encoding
For classical models, categorical variables are label-encoded to integers. This approach has limitations as it imposes arbitrary ordering.

#### Embedding Strategy
For LSTM models, we use learned embeddings for categorical features:

```python
categorical_specs = {
    'weekday': {'vocab_size': 7, 'embedding_dim': 3},
    'event_name_1': {'vocab_size': 12, 'embedding_dim': 5},
    'event_type_1': {'vocab_size': 5, 'embedding_dim': 3},
    'month': {'vocab_size': 12, 'embedding_dim': 4},
    'quarter': {'vocab_size': 4, 'embedding_dim': 2}
}
```

Embedding dimensions are chosen based on cardinality and expected complexity of relationships.

The LSTM architecture integrates embeddings directly into the model, concatenating embedded categorical features with numerical features before the LSTM layers. This allows the model to learn appropriate representations for each categorical variable during training.

### Feature Selection

Features are selected based on:
1. Exclusion of identifier columns
2. Inclusion of all numeric features
3. Proper handling of categorical encodings
4. Validation of data types

## Feature Dimensionality

### Classical Models
- Total columns: 55
- Metadata (excluded): 9
- Features after encoding: ~50 numeric features

### LSTM with Embeddings
- Categorical features: 7 (embedded to 21 dimensions total)
- Numerical features: 39
- Total LSTM input: 60 dimensions

The embedding approach provides richer representations without imposing artificial ordering on categorical variables.

## Design Rationale

### Feature Selection Principles
- **Temporal coverage**: Features capture multiple time scales (daily, weekly, monthly patterns)
- **Domain relevance**: Each feature has established importance in retail forecasting literature
- **Computational efficiency**: Balance between feature richness and training time

### Implementation Considerations
- **Modularity**: Separate functions for each feature type
- **Reproducibility**: Deterministic feature generation
- **Scalability**: Efficient computation for large datasets

## Implementation Files

- `src/feature_engineering.py` - Core feature engineering functions
- `scripts/1_create_feature_rich_splices.py` - Splice generation script
- `src/hpo/lstm_fitness.py` - Feature processing for model training

## Extension Guidelines

If adding new features please:
1. Follow the existing modular pattern
2. Document feature rationale and computation
3. Validate against data leakage
4. Update feature count documentation
5. Test on small splices before full datasets