#////////////////////////////////////////////////////////////////////////////////#
# File:         run_unified_experiment.py                                        #
# Author:       Douglas Nyberg                                                   #
# Email:        douglas1.nyberg@gmail.com                                        #
# Date:         2025-04-12                                                       #
#////////////////////////////////////////////////////////////////////////////////#
#!/usr/bin/env python3
"""
Master experiment runner for item-level forecasting.

Runs: preprocessing, training, prediction, evaluation, visualization.
Compares classical vs LSTM methods.
"""

# imports
import argparse
import subprocess
import sys
import logging
from pathlib import Path
from datetime import datetime
import json
import time
from typing import Dict, List, Tuple, Optional, Any

# Set up project path
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent.parent
sys.path.append(str(PROJECT_ROOT))

# Import project-specific modules
import config_exp1 as config # Unified experiment configuration

# Setup logging
config.LOGS_DIR.mkdir(parents=True, exist_ok=True)
# Clear existing handlers if any
for handler in logging.root.handlers[:]: logging.root.removeHandler(handler)
logging.basicConfig(
    level=getattr(logging, config.LOGGING_CONFIG["level"]),
    format=config.LOGGING_CONFIG["format"],
    handlers=[
        logging.FileHandler(str(config.LOGGING_CONFIG["log_file"]), mode='w'), # Overwrite log for new run
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

# Define helper function to run sub-scripts
def run_script_step(script_name: str, script_args: List[str] = None) -> bool:
    """Executes a given script step and logs its success or failure."""
    if script_args is None:
        script_args = []
    
    command = [sys.executable, str(SCRIPT_DIR / script_name)] + script_args
    logger.info(f"Executing: {' '.join(command)}")
    
    try:
        process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        stdout, stderr = process.communicate()

        if stdout: logger.info(f"Output from {script_name}:\n{stdout.strip()}")
        if stderr: logger.error(f"Errors from {script_name}:\n{stderr.strip()}")
            
        if process.returncode != 0:
            logger.error(f"Script {script_name} failed with return code {process.returncode}.")
            return False
        logger.info(f"Script {script_name} completed successfully.")
        return True
    except FileNotFoundError:
        logger.error(f"Script {script_name} not found. Ensure it's in the same directory: {SCRIPT_DIR}")
        return False
    except Exception as e:
        logger.error(f"An unexpected error occurred while running {script_name}: {e}")
        return False

# Define main experiment workflow
def main():
    """Main entry point to run the unified experiment."""
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description="Run Unified M5 Forecasting Experiment.")
    parser.add_argument(
        "--models-to-run", nargs='+',
        default=config.DEFAULT_CLASSICAL_MODELS_TO_RUN + config.DEFAULT_DEEP_LEARNING_MODELS_TO_RUN,
        choices=config.AVAILABLE_MODELS["classical"] + config.AVAILABLE_MODELS["deep_learning"],
        help="Specific models to process in this run."
    )
    parser.add_argument(
        "--run-tag", type=str, default=f"run_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
        help="A unique tag for this specific experiment run (used in output paths)."
    )
    parser.add_argument("--skip-preprocessing", action="store_true", help="Skip data preprocessing.")
    parser.add_argument("--skip-training", action="store_true", help="Skip model training.")
    parser.add_argument("--skip-prediction", action="store_true", help="Skip prediction generation.")
    parser.add_argument("--skip-evaluation", action="store_true", help="Skip evaluation.")
    parser.add_argument("--skip-visualization", action="store_true", help="Skip result visualization.")
    parser.add_argument("--skip-statistical-comparison", action="store_true", help="Skip statistical comparison between models.")
    parser.add_argument("--force-steps", action="store_true", help="Pass --force to substeps like preprocessing/training.")
    
    args = parser.parse_args()

    run_specific_experiment_name = f"{config.EXPERIMENT_TAG}_{args.run_tag}"
    logger.info("=" * 70)
    logger.info(f"STARTING UNIFIED EXPERIMENT RUN: {run_specific_experiment_name}")
    logger.info(f"Models selected: {args.models_to_run}")
    logger.info("=" * 70)

    overall_success = True
    step_times = {}

    # data preprocessing
    if not args.skip_preprocessing:
        logger.info("--- Stage 1: Data Preprocessing ---")
        start_time = time.time()
        preprocess_args = ["--output-dir", str(config.DATA_CONFIG["preprocessed_output_dir"])]
        if args.force_steps: preprocess_args.append("--force")
        if config.DATA_CONFIG["limit_items"] is not None: # Pass limit_items from config if set
             preprocess_args.extend(["--limit-items", str(config.DATA_CONFIG["limit_items"])])

        if not run_script_step("1_preprocess_exp1.py", preprocess_args):
            logger.error("Preprocessing failed. Halting experiment.")
            sys.exit(1) # Critical step QED exit if fails
        step_times["preprocessing"] = time.time() - start_time
    else:
        logger.info("--- Stage 1: Data Preprocessing SKIPPED ---")

    # train models
    if not args.skip_training:
        logger.info("--- Stage 2: Model Training ---")
        start_time = time.time()
        classical_to_train = [m for m in args.models_to_run if m in config.AVAILABLE_MODELS["classical"]]
        lstm_to_train = [m for m in args.models_to_run if m == "lstm"]

        if classical_to_train:
            logger.info(f"Training classical models: {classical_to_train}")
            train_classical_args = ["--models-to-train"] + classical_to_train
            if args.force_steps: train_classical_args.append("--force")
            if config.DATA_CONFIG["limit_items"] is not None:
                 train_classical_args.extend(["--limit-items", str(config.DATA_CONFIG["limit_items"])])
            if not run_script_step("2_train_classical_models.py", train_classical_args):
                overall_success = False # Log failure but continue if other models might train
        
        if lstm_to_train: 
            logger.info("Training LSTM (per-item) models...")
            train_lstm_args = []
            if args.force_steps: train_lstm_args.append("--force")
            if config.DATA_CONFIG["limit_items"] is not None:
                train_lstm_args.extend(["--limit-items", str(config.DATA_CONFIG["limit_items"])])
            # Device for LSTM can be passed, though parallel CPU is default for per-item
            # train_lstm_args.extend(["--device", config.DEVICE_CONFIG["device_name_lstm"]])


            if not run_script_step("3_train_lstm_models.py", train_lstm_args):
                overall_success = False
        step_times["training"] = time.time() - start_time
    else:
        logger.info("--- Stage 2: Model Training SKIPPED ---")

    # generate predictions
    if not args.skip_prediction:
        logger.info("--- Stage 3: Prediction Generation ---")
        start_time = time.time()
        for model_type in args.models_to_run:
            predict_args = [
                "--model-type", model_type,
                "--run-experiment-name", run_specific_experiment_name
            ]
            if config.DATA_CONFIG["limit_items"] is not None:
                 predict_args.extend(["--limit-items", str(config.DATA_CONFIG["limit_items"])])
            if model_type == "lstm":
                 predict_args.extend(["--device", config.DEVICE_CONFIG["device_name_lstm"]])

            if not run_script_step("4_predict_models.py", predict_args):
                overall_success = False
        step_times["prediction"] = time.time() - start_time
    else:
        logger.info("--- Stage 3: Prediction Generation SKIPPED ---")

    # evaluate predictions
    if not args.skip_evaluation:
        logger.info("--- Stage 4: Evaluation ---")
        start_time = time.time()
        # Evaluate all models together in one call
        predictions_base_dir = config.PREDICTIONS_BASE_DIR / run_specific_experiment_name
        data_dir = config.PREPROCESSED_DATA_OUTPUT_DIR
        results_dir = config.RESULTS_BASE_DIR / run_specific_experiment_name
        
        eval_args = [
            "--predictions-dir", str(predictions_base_dir),
            "--data-dir", str(data_dir),
            "--output-dir", str(results_dir),
            "--models-to-evaluate"
        ] + args.models_to_run + ["--generate-plots"]
        
        if not run_script_step("5_evaluate_predictions.py", eval_args):
            overall_success = False
        step_times["evaluation"] = time.time() - start_time
    else:
        logger.info("--- Stage 4: Evaluation SKIPPED ---")

    # generate visualizations
    if not args.skip_visualization:
        logger.info("--- Stage 5: Visualization ---")
        start_time = time.time()
        viz_args = [
            "--run-experiment-name", run_specific_experiment_name,
            "--models-to-visualize"
        ] + args.models_to_run
        if not run_script_step("7_visualize_results.py", viz_args):
            overall_success = False
        step_times["visualization"] = time.time() - start_time
    else:
        logger.info("--- Stage 5: Visualization SKIPPED ---")

    # statistical comparison
    if not args.skip_statistical_comparison and len(args.models_to_run) > 1:
        logger.info("--- Stage 6: Statistical Comparison ---")
        start_time = time.time()
        comparison_args = [
            "--results-dir", str(config.RESULTS_BASE_DIR / run_specific_experiment_name),
            "--models"
        ] + args.models_to_run
        if not run_script_step("6_statistical_comparison.py", comparison_args):
            overall_success = False
        step_times["statistical_comparison"] = time.time() - start_time
    else:
        if args.skip_statistical_comparison:
            logger.info("--- Stage 6: Statistical Comparison SKIPPED ---")
        elif len(args.models_to_run) <= 1:
            logger.info("--- Stage 6: Statistical Comparison SKIPPED (need at least 2 models) ---")

    # final summary
    logger.info("=" * 70)
    if overall_success:
        logger.info(f"EXPERIMENT RUN '{run_specific_experiment_name}' COMPLETED SUCCESSFULLY.")
    else:
        logger.warning(f"EXPERIMENT RUN '{run_specific_experiment_name}' COMPLETED WITH ONE OR MORE FAILURES.")
    logger.info("Timings per stage (seconds):")
    for stage, t in step_times.items():
        logger.info(f"  - {stage.capitalize()}: {t:.2f}s")
    logger.info("Please check individual logs and output directories for detailed results.")
    logger.info("=" * 70)

# execute main workflow
if __name__ == "__main__":
    main()