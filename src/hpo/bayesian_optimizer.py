#////////////////////////////////////////////////////////////////////////////////#
# File:         bayesian_optimizer.py                                            #
# Author:       Douglas Nyberg                                                   #
# Email:        douglas1.nyberg@gmail.com                                        #
# Date:         2025-04-15                                                       #
# Description:  Bayesian optimization for hyperparameter tuning of LSTM models.  #
# Affiliation:  Physics Department, Purdue University                            #
#////////////////////////////////////////////////////////////////////////////////#
"""
bayesian optimization for hyperparameter tuning of lstm models.

implements gaussian process regression with expected improvement acquisition
function to efficiently search hyperparameter space.
"""
import numpy as np
import warnings
from scipy.stats import norm
import random
import time
import os
import json
from typing import Dict, List, Tuple, Callable, Any, Optional, Union
from pathlib import Path
import logging
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern, WhiteKernel, ConstantKernel
import pandas as pd

# custom json encoder to handle numpy types
class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        """override default json encoder to handle numpy types"""
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, np.bool_):
            return bool(obj)
        return super(NumpyEncoder, self).default(obj)

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class BayesianOptimizer:
    """bayesian optimization with gaussian processes"""
    def __init__(
        self,
        param_space: Dict[str, Any],
        fitness_function: Callable,
        n_iterations: int = 20,
        n_initial_points: int = 5,
        exploration_weight: float = 0.1,
        maximize: bool = False,
        output_dir: Optional[str] = None,
        random_seed: Optional[int] = None,
        early_stopping_rounds: Optional[int] = None,
        early_stopping_threshold: Optional[float] = None,
        verbose: bool = True,
    ):
        # init with param space and fitness func
        self.param_space = param_space
        self.fitness_function = fitness_function
        self.n_iterations = n_iterations
        self.n_initial_points = n_initial_points
        self.exploration_weight = exploration_weight
        self.maximize = maximize
        self.verbose = verbose
        self.early_stopping_rounds = early_stopping_rounds
        self.early_stopping_threshold = early_stopping_threshold

        if self.verbose: 
            logger.info(f"BayesianOptimizer initialized with: n_iterations={self.n_iterations}, n_initial_points={self.n_initial_points}")
        
        # Set random seed if provided
        if random_seed is not None:
            np.random.seed(random_seed)
            random.seed(random_seed)
        
        # Setup output directory
        if output_dir:
            self.output_dir = Path(output_dir)
            os.makedirs(self.output_dir, exist_ok=True)
        else:
            self.output_dir = None
            
        # Initialize parameters for the optimizer
        self.param_types = {}
        self.param_ranges = {}
        self.categorical_maps = {}
        
        # Parse parameter space
        self._parse_param_space()
        
        # Initialize Gaussian Process with Matern kernel (better for hyperparameter optimization)
        # nu=2.5 corresponds to functions that are twice differentiable
        kernel = ConstantKernel() * Matern(length_scale=1.0, nu=2.5) + WhiteKernel(noise_level=0.1)
        self.gp = GaussianProcessRegressor(
            kernel=kernel,
            alpha=1e-6,  # Small regularization to prevent numerical issues
            normalize_y=True,  # Normalize target values
            n_restarts_optimizer=5,  # Number of restarts for GP hyperparameter optimization
            random_state=random_seed
        )
        
        # Initialize arrays to store observations
        self.X_observed = []  # Normalized parameter vectors
        self.y_observed = []  # Fitness values
        self.param_dicts = []  # Original parameter dictionaries
        
        # Initialize tracking variables
        self.best_params = None
        self.best_fitness = float('-inf') if maximize else float('inf')
        self.best_iteration = 0
        self.history = []
        
    def _parse_param_space(self):
        """Parse the parameter space and determine types and ranges."""
        for param_name, space in self.param_space.items():
            if isinstance(space, tuple):
                # Numeric parameter
                if len(space) == 2:
                    # Continuous parameter (min, max)
                    self.param_types[param_name] = 'continuous'
                    self.param_ranges[param_name] = (space[0], space[1])
                elif len(space) == 3:
                    if space[2] == 'log':
                        # Log-scale parameter (min, max, 'log')
                        self.param_types[param_name] = 'log'
                        self.param_ranges[param_name] = (np.log(space[0]), np.log(space[1]))
                    else:
                        # Discrete parameter (min, max, step)
                        self.param_types[param_name] = 'discrete'
                        self.param_ranges[param_name] = (space[0], space[1])
                        # Store step size for later reconstruction
                        self.categorical_maps[param_name] = {'step': space[2]}
            elif isinstance(space, list):
                # Categorical parameter
                self.param_types[param_name] = 'categorical'
                self.param_ranges[param_name] = (0, len(space) - 1)
                # Map indices to categorical values
                self.categorical_maps[param_name] = {i: val for i, val in enumerate(space)}
                
        if self.verbose:
            logger.info(f"parsed parameter space with {len(self.param_types)} parameters")
            
    def _normalize_params(self, params: Dict[str, Any]) -> np.ndarray:
        """normalize params to [0, 1] range for gp"""
        normalized = []
        
        for param_name in sorted(self.param_types.keys()):
            value = params[param_name]
            param_type = self.param_types[param_name]
            param_range = self.param_ranges[param_name]
            
            if param_type == 'categorical':
                # find the index of the categorical value
                for idx, val in self.categorical_maps[param_name].items():
                    if val == value:
                        normalized_value = idx / (len(self.categorical_maps[param_name]) - 1)
                        break
            elif param_type == 'log':
                # normalize in log space
                log_value = np.log(value)
                log_min, log_max = param_range
                normalized_value = (log_value - log_min) / (log_max - log_min)
            else:
                # Normalize continuous/discrete parameter
                min_val, max_val = param_range
                normalized_value = (value - min_val) / (max_val - min_val)
                
            normalized.append(normalized_value)
            
        return np.array(normalized).reshape(1, -1)
    
    def _denormalize_params(self, normalized_vector: np.ndarray) -> Dict[str, Any]:
        """convert normalized param vector back to parameter dict"""
        params = {}
        
        for i, param_name in enumerate(sorted(self.param_types.keys())):
            normalized_value = normalized_vector[i]
            param_type = self.param_types[param_name]
            param_range = self.param_ranges[param_name]
            
            if param_type == 'categorical':
                # convert to index and lookup categorical value
                idx = int(round(normalized_value * (len(self.categorical_maps[param_name]) - 1)))
                params[param_name] = self.categorical_maps[param_name][idx]
            elif param_type == 'log':
                # denormalize in log space and convert back
                log_min, log_max = param_range
                log_value = normalized_value * (log_max - log_min) + log_min
                params[param_name] = np.exp(log_value)
            elif param_type == 'discrete':
                # denormalize and round to nearest step
                min_val, max_val = param_range
                step = self.categorical_maps[param_name]['step']
                value = normalized_value * (max_val - min_val) + min_val
                # round to nearest multiple of step
                value = round(value / step) * step
                params[param_name] = int(value) if isinstance(min_val, int) else value
            else:
                # Denormalize continuous parameter
                min_val, max_val = param_range
                value = normalized_value * (max_val - min_val) + min_val
                params[param_name] = value
                
        return params
    
    def _sample_random_point(self) -> Dict[str, Any]:
        """
        Sample a random point from the parameter space.
        
        Returns:
            Dictionary of random parameters
        """
        params = {}
        
        for param_name, param_type in self.param_types.items():
            param_range = self.param_ranges[param_name]
            
            if param_type == 'categorical':
                # Sample from categorical values
                idx = random.randint(0, len(self.categorical_maps[param_name]) - 1)
                params[param_name] = self.categorical_maps[param_name][idx]
            elif param_type == 'log':
                # Sample in log space and convert back
                log_min, log_max = param_range
                log_value = random.uniform(log_min, log_max)
                params[param_name] = np.exp(log_value)
            elif param_type == 'discrete':
                # Sample discrete parameter
                min_val, max_val = param_range
                step = self.categorical_maps[param_name]['step']
                steps = int((max_val - min_val) / step) + 1
                value = min_val + random.randint(0, steps - 1) * step
                params[param_name] = value
            else:
                # Sample continuous parameter
                min_val, max_val = param_range
                params[param_name] = random.uniform(min_val, max_val)
                
        return params
    
    def _expected_improvement(self, X: np.ndarray) -> np.ndarray:
        """calculate expected improvement"""
        # Get mean and std from GP
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            mu, sigma = self.gp.predict(X, return_std=True)
            
        # if sigma is zero, ei is zero (no improvement expected)
        sigma = np.maximum(sigma, 1e-9)
            
        # get current best
        if self.maximize:
            best_f = np.max(self.y_observed)
            improvement = mu - best_f - self.exploration_weight
            Z = improvement / sigma
        else:
            best_f = np.min(self.y_observed)
            improvement = best_f - mu - self.exploration_weight
            Z = improvement / sigma
            
        # calculate ei
        ei = improvement * norm.cdf(Z) + sigma * norm.pdf(Z)
        
        # set ei to 0 if improvement is negative
        ei[improvement < 0] = 0.0
        
        return ei
    
    def _find_next_point(self) -> Dict[str, Any]:
        """find next point using acquisition func"""
        # generate random candidate points
        n_candidates = 1000
        candidates = []
        
        for _ in range(n_candidates):
            normalized_candidate = np.random.rand(len(self.param_types))
            candidates.append(normalized_candidate)
            
        candidates = np.vstack(candidates)
        
        # Calculate acquisition function for each candidate
        ei_values = self._expected_improvement(candidates)
        
        # Find candidate with highest EI
        best_idx = np.argmax(ei_values)
        best_candidate = candidates[best_idx]
        
        # Convert to parameter dictionary
        next_point = self._denormalize_params(best_candidate)
        
        return next_point
    
    def optimize(self) -> Dict[str, Any]:
        """run bayesian optimization"""
        start_time = time.time()
        
        # Variables for early stopping
        no_improvement_count = 0
        
        # Initialize with random points
        if self.verbose:
            logger.info(f"Initializing with {self.n_initial_points} random points...")
            
        for i in range(self.n_initial_points):
            iteration_start_time = time.time()
            
            # Sample random point
            params = self._sample_random_point()
            
            # Evaluate fitness
            try:
                # Add trial ID to params for logging
                params_with_trial_id = params.copy()
                params_with_trial_id['_trial_id_'] = i + 1
                fitness = self.fitness_function(params_with_trial_id)
                if self.verbose:
                    logger.info(f"Evaluated initial point {i+1}/{self.n_initial_points}: fitness = {fitness}")
            except Exception as e:
                logger.error(f"Error evaluating initial point {i+1}: {str(e)}")
                # Assign worst possible fitness
                fitness = float('-inf') if self.maximize else float('inf')
                
            # Store observation
            self.X_observed.append(self._normalize_params(params)[0])
            self.y_observed.append(fitness)
            self.param_dicts.append(params)
            
            # Update best parameters
            if (self.maximize and fitness > self.best_fitness) or \
               (not self.maximize and fitness < self.best_fitness):
                improvement = abs(fitness - self.best_fitness)
                self.best_params = params.copy()
                self.best_fitness = fitness
                self.best_iteration = i
                
                if self.verbose:
                    logger.info(f"New best parameters found during initialization: {self.best_params}")
                    logger.info(f"Fitness: {self.best_fitness}")
                
                # Reset early stopping counter
                if self.early_stopping_rounds is not None:
                    if self.early_stopping_threshold is None or improvement > self.early_stopping_threshold:
                        no_improvement_count = 0
                        
            # Save iteration statistics
            iteration_stats = {
                'iteration': i + 1,
                'params': params,
                'fitness': fitness,
                'best_fitness': self.best_fitness,
                'is_best': fitness == self.best_fitness,
                'time': time.time() - iteration_start_time
            }
            self.history.append(iteration_stats)
            
            # Save iteration results if output directory is specified
            if self.output_dir:
                self._save_iteration_results(i, params, fitness)
                
        # Convert to numpy arrays for GP
        X = np.vstack(self.X_observed)
        y = np.array(self.y_observed)
        
        # Fit Gaussian Process to observations
        if not self.maximize:
            # Minimize: GP works with minimization, no need to negate
            self.gp.fit(X, y)
        else:
            # Maximize: negate fitness to make it a minimization problem
            self.gp.fit(X, -y)
            
        # Run the main optimization loop
        for i in range(self.n_initial_points, self.n_iterations):
            iteration_start_time = time.time()
            
            # Find next point to evaluate
            params = self._find_next_point()
            
            # Evaluate fitness
            try:
                # Add trial ID to params for logging (offset by initial points)
                params_with_trial_id = params.copy()
                params_with_trial_id['_trial_id_'] = self.n_initial_points + i + 1
                fitness = self.fitness_function(params_with_trial_id)
                if self.verbose:
                    logger.info(f"Evaluated point {i+1}/{self.n_iterations}: fitness = {fitness}")
            except Exception as e:
                logger.error(f"Error evaluating point {i+1}: {str(e)}")
                # Assign worst possible fitness
                fitness = float('-inf') if self.maximize else float('inf')
                
            # Store observation
            normalized_params = self._normalize_params(params)[0]
            self.X_observed.append(normalized_params)
            self.y_observed.append(fitness)
            self.param_dicts.append(params)
            
            # Update GP
            X = np.vstack(self.X_observed)
            y = np.array(self.y_observed)
            
            if not self.maximize:
                self.gp.fit(X, y)
            else:
                self.gp.fit(X, -y)
                
            # Update best parameters
            if (self.maximize and fitness > self.best_fitness) or \
               (not self.maximize and fitness < self.best_fitness):
                improvement = abs(fitness - self.best_fitness)
                self.best_params = params.copy()
                self.best_fitness = fitness
                self.best_iteration = i
                
                if self.verbose:
                    logger.info(f"New best parameters found in iteration {i + 1}: {self.best_params}")
                    logger.info(f"Fitness: {self.best_fitness}")
                
                # Reset early stopping counter
                if self.early_stopping_rounds is not None:
                    if self.early_stopping_threshold is None or improvement > self.early_stopping_threshold:
                        no_improvement_count = 0
            else:
                # Increment counter if no improvement
                if self.early_stopping_rounds is not None:
                    no_improvement_count += 1
                    
            # Save iteration statistics
            iteration_stats = {
                'iteration': i + 1,
                'params': params,
                'fitness': fitness,
                'best_fitness': self.best_fitness,
                'is_best': fitness == self.best_fitness,
                'time': time.time() - iteration_start_time
            }
            self.history.append(iteration_stats)
            
            # Save iteration results if output directory is specified
            if self.output_dir:
                self._save_iteration_results(i, params, fitness)
                
            if self.verbose:
                logger.info(f"Iteration {i + 1}/{self.n_iterations} completed")
                logger.info(f"Best fitness: {self.best_fitness}")
                logger.info(f"Time: {iteration_stats['time']:.2f} seconds")
                
            # Check for early stopping
            if self.early_stopping_rounds is not None and no_improvement_count >= self.early_stopping_rounds:
                if self.verbose:
                    logger.info(f"Early stopping after {i + 1} iterations without improvement")
                break
                
        # Optimization completed
        total_time = time.time() - start_time
        
        if self.verbose:
            logger.info(f"Optimization completed in {total_time:.2f} seconds")
            logger.info(f"Best parameters found in iteration {self.best_iteration + 1}: {self.best_params}")
            logger.info(f"Best fitness: {self.best_fitness}")
            
        # Save final results if output directory is specified
        if self.output_dir:
            self._save_final_results(total_time)
            
        return self.best_params, self.best_fitness, self.history
    
    def _save_iteration_results(self, iteration: int, params: Dict[str, Any], fitness: float) -> None:
        """Save the results of an iteration to a file."""
        # Create directory for this iteration
        iteration_dir = self.output_dir / f"iteration_{iteration + 1}"
        os.makedirs(iteration_dir, exist_ok=True)
        
        # Save iteration data
        iteration_data = {
            'iteration': iteration + 1,
            'params': params,
            'fitness': fitness,
            'best_fitness': self.best_fitness,
            'is_best': fitness == self.best_fitness,
            'time': self.history[-1]['time']
        }
            
        with open(iteration_dir / "iteration_data.json", "w") as f:
            json.dump(iteration_data, f, indent=2, cls=NumpyEncoder)
            
        # Save best parameters so far
        best_data = {
            'params': self.best_params,
            'fitness': self.best_fitness,
            'iteration': self.best_iteration + 1
        }
        
        with open(iteration_dir / "best_so_far.json", "w") as f:
            json.dump(best_data, f, indent=2, cls=NumpyEncoder)
    
    def _save_final_results(self, total_time: float) -> None:
        """Save the final results of the optimization."""
        final_results = {
            'best_params': self.best_params,
            'best_fitness': self.best_fitness,
            'best_iteration': self.best_iteration + 1,
            'total_iterations': len(self.history),
            'total_time': total_time,
            'parameters': {
                'n_iterations': self.n_iterations,
                'n_initial_points': self.n_initial_points,
                'exploration_weight': self.exploration_weight,
                'maximize': self.maximize,
                'early_stopping_rounds': self.early_stopping_rounds,
                'early_stopping_threshold': self.early_stopping_threshold
            }
        }
        
        with open(self.output_dir / "final_results.json", "w") as f:
            json.dump(final_results, f, indent=2, cls=NumpyEncoder)
            
        # Save optimization progress for visualization
        progress_data = []
        for item in self.history:
            progress_data.append({
                'iteration': item['iteration'],
                'best_fitness': item['best_fitness'],
                'fitness': item['fitness'],
                'time': item['time']
            })
            
        with open(self.output_dir / "optimization_progress.json", "w") as f:
            json.dump(progress_data, f, indent=2, cls=NumpyEncoder)
            
        # Save all evaluated points
        all_points = []
        for params, fitness in zip(self.param_dicts, self.y_observed):
            point_data = params.copy()
            point_data['fitness'] = fitness
            all_points.append(point_data)
            
        with open(self.output_dir / "all_points.json", "w") as f:
            json.dump(all_points, f, indent=2, cls=NumpyEncoder)
            
    def get_optimization_summary(self) -> Dict[str, Any]:
        """
        Get a summary of the optimization results.
        
        Args:
            None
            
        Returns:
            Dictionary containing optimization summary with best parameters, fitness history,
            convergence information, and runtime statistics
        """
        if not self.history:
            return {
                'status': 'Not yet run',
                'best_params': None,
                'best_fitness': None
            }
            
        # Create a DataFrame for analysis
        df = pd.DataFrame([{
            'iteration': item['iteration'],
            'fitness': item['fitness'],
            'best_fitness': item['best_fitness'],
            'time': item['time']
        } for item in self.history])
        
        return {
            'status': 'Completed',
            'best_params': self.best_params,
            'best_fitness': self.best_fitness,
            'best_iteration': self.best_iteration + 1,
            'total_iterations': len(self.history),
            'avg_time_per_iteration': df['time'].mean(),
            'total_time': df['time'].sum(),
            'optimization_progress': df.to_dict('records')
        }