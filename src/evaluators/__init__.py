#////////////////////////////////////////////////////////////////////////////////#
# File:         __init__.py                                                      #
# Author:       Douglas Nyberg                                                   #
# Email:        douglas1.nyberg@gmail.com                                        #
# Date:         2025-05-02                                                       #
# Description:  Evaluators package initialization for model evaluation metrics. #
#////////////////////////////////////////////////////////////////////////////////#

# Point forecast metrics from accuracy.py
from .accuracy import (
    calculate_rmse,
    calculate_mae,
    calculate_mape,
    calculate_smape,
    calculate_mase,
    calculate_rmsse,
    calculate_wrmsse,
    calculate_wrmsse_vectorized,
    calculate_geometric_mean_absolute_error,
    calculate_mean_relative_absolute_error,
    calculate_median_absolute_error,
    calculate_median_absolute_scaled_error,
    calculate_asymmetric_error,
    calculate_linex_error,
    calculate_msse
)

# Probabilistic/Quantile metrics from uncertainty.py
from .uncertainty import (
    calculate_pinball_loss,
    calculate_pinball_loss_sktime,
    spl_scale,
    calculate_spl,
    calculate_wspl,
    calculate_wspl_vectorized,
    calculate_coverage,
    calculate_empirical_coverage_sktime,
    calculate_pi_width,
    calculate_combined_uncertainty_metric,
    prepare_m5_forecasts,
    calculate_crps,
    calculate_log_loss,
    calculate_constraint_violation,
    calculate_interval_score,
    calculate_au_calibration,
    calculate_squared_distr_loss,
    create_interval_format_dataframe,
    calculate_prediction_interval_width_sktime,
    calculate_interval_score_sktime,
    calculate_constraint_violation_sktime
)

# M5 competition specific metrics
from .m5_official import (
    M5OfficialEvaluator,
    evaluate_model_on_official_m5,
    compare_with_m5_baselines
)

# Model comparison workflows
from .model_comparison import (
    ModelComparisonEvaluator,
    compare_models_wrmsse
)

__all__ = [
    # Point forecast metrics
    'calculate_rmse',
    'calculate_mae',
    'calculate_mape',
    'calculate_smape',
    'calculate_mase',
    'calculate_rmsse',
    'calculate_wrmsse',
    'calculate_wrmsse_vectorized',
    'calculate_geometric_mean_absolute_error',
    'calculate_mean_relative_absolute_error',
    'calculate_median_absolute_error',
    'calculate_median_absolute_scaled_error',
    'calculate_asymmetric_error',
    'calculate_linex_error',
    'calculate_msse',
    # Probabilistic metrics
    'calculate_pinball_loss',
    'calculate_pinball_loss_sktime',
    'spl_scale',
    'calculate_spl',
    'calculate_wspl',
    'calculate_wspl_vectorized',
    'calculate_coverage',
    'calculate_empirical_coverage_sktime',
    'calculate_pi_width',
    'calculate_combined_uncertainty_metric',
    'prepare_m5_forecasts',
    'calculate_crps',
    'calculate_log_loss',
    'calculate_constraint_violation',
    'calculate_interval_score',
    'calculate_au_calibration',
    'calculate_squared_distr_loss',
    'create_interval_format_dataframe',
    'calculate_prediction_interval_width_sktime',
    'calculate_interval_score_sktime',
    'calculate_constraint_violation_sktime',
    # M5 specific
    'M5OfficialEvaluator',
    'evaluate_model_on_official_m5',
    'compare_with_m5_baselines',
    # Model comparison
    'ModelComparisonEvaluator',
    'compare_models_wrmsse'
]