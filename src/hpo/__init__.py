#////////////////////////////////////////////////////////////////////////////////#
# File:         __init__.py                                                      #
# Author:       Douglas Nyberg                                                   #
# Email:        douglas1.nyberg@gmail.com                                        #
# Date:         2025-05-21                                                       #
# Description:  HPO package initialization for hyperparameter optimization.     #
#////////////////////////////////////////////////////////////////////////////////#


"""
Initialization for hyperparameter optimization package.
"""

# Only import essential modules to avoid dependency issues
try:
    from .bayesian_optimizer import BayesianOptimizer
except ImportError as e:
    print(f"Warning: Could not import BayesianOptimizer: {e}")

try:
    from .search_spaces import (
        get_lstm_param_space,
        get_lstm_param_space_limited,
        get_lstm_param_space_focused
    )
except ImportError as e:
    print(f"Warning: Could not import search spaces: {e}")

# Optional imports for advanced functionality
try:
    from .lstm_fitness import LSTMFitnessEvaluator, create_lstm_fitness_function
except ImportError:
    pass

try:
    from .classical_fitness import ClassicalFitnessEvaluator
except ImportError:
    pass