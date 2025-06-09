#////////////////////////////////////////////////////////////////////////////////#
# File:         run_multi_seed_experiment.py                                    #
# Author:       Douglas Nyberg                                                  #
# Email:        douglas1.nyberg@gmail.com                                       #
# Date:         2025-06-06                                                      #
# Description:  Multi-seed experiment runner for stability testing             #
# Affiliation:  Physics Department, Purdue University                          #
#////////////////////////////////////////////////////////////////////////////////#
#!/usr/bin/env python3
"""
Multi-Seed Monte Carlo Experiment Runner

Automatically runs the complete MC experiment workflow with multiple seeds
to test result stability. Generates data, trains models, and evaluates
performance across different random seeds.

Usage:
    python run_multi_seed_experiment.py --profile quick_test --multi-seed
    python run_multi_seed_experiment.py --profile production --seed 42
    python run_multi_seed_experiment.py --profile production --seeds 42,123,456
"""

import argparse
import subprocess
import sys
import logging
from pathlib import Path
from typing import List, Dict, Any
import json
from datetime import datetime

# Add project root to path
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent.parent
sys.path.append(str(PROJECT_ROOT))

import config_mc_master as config

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def parse_arguments() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Multi-seed Monte Carlo experiment runner",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    # Configuration profile
    parser.add_argument(
        "-p", "--profile", 
        default="quick_test",
        choices=["quick_test", "moderate", "production", "hpo_intensive"],
        help="Configuration profile to use"
    )
    
    # Seed options (mutually exclusive)
    seed_group = parser.add_mutually_exclusive_group()
    seed_group.add_argument(
        "--seed", 
        type=int,
        help="Single seed to use (overrides config)"
    )
    seed_group.add_argument(
        "--seeds",
        type=str,
        help="Comma-separated list of seeds (e.g., '42,123,456')"
    )
    seed_group.add_argument(
        "--multi-seed",
        action="store_true",
        help="Use multi-seed mode from config"
    )
    
    # Experiment control
    parser.add_argument(
        "--steps",
        default="1,2,3,4,5,6,7",
        help="Comma-separated list of workflow steps to run (1-7)"
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show what would be run without executing"
    )
    parser.add_argument(
        "--continue-on-error",
        action="store_true",
        help="Continue with next seed if one fails"
    )
    
    return parser.parse_args()

def get_seeds_to_run(args: argparse.Namespace, config: Dict[str, Any]) -> List[int]:
    """Determine which seeds to use based on arguments and config."""
    if args.seed is not None:
        return [args.seed]
    elif args.seeds is not None:
        return [int(s.strip()) for s in args.seeds.split(",")]
    elif args.multi_seed:
        return config_mc_master.get_experiment_seeds(config["seed"])
    else:
        # Default: single seed from config
        return [config["seed"]["default_seed"]]

def run_workflow_step(step: int, seed: int, profile: str, dry_run: bool = False) -> bool:
    """
    Run a single workflow step with the given seed.
    
    Returns True if successful, False otherwise.
    """
    step_scripts = {
        1: "1_preprocess_exp_mc.py",
        2: "2_train_classical_models_mc.py", 
        3: "3_train_lstm_models_mc.py",
        4: "4_predict_models_mc.py",
        5: "5_evaluate_predictions_mc.py",
        6: "6_statistical_comparison_mc.py",
        7: "7_visualize_results_mc.py"
    }
    
    if step not in step_scripts:
        logger.error(f"Invalid step number: {step}")
        return False
    
    script_name = step_scripts[step]
    script_path = SCRIPT_DIR / script_name
    
    if not script_path.exists():
        logger.error(f"Script not found: {script_path}")
        return False
    
    # Build command with seed parameter
    cmd = [sys.executable, str(script_path)]
    
    # Add seed parameter if script supports it
    if step in [1, 2, 3]:  # Steps that might need seed parameter
        cmd.extend(["--seed", str(seed)])
    
    logger.info(f"Step {step} (seed {seed}): {script_name}")
    
    if dry_run:
        logger.info(f"[DRY RUN] Would run: {' '.join(cmd)}")
        return True
    
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)
        logger.info(f"✓ Step {step} completed successfully (seed {seed})")
        return True
    except subprocess.CalledProcessError as e:
        logger.error(f"✗ Step {step} failed (seed {seed})")
        logger.error(f"Return code: {e.returncode}")
        logger.error(f"Stderr: {e.stderr}")
        return False

def generate_synthetic_data(seed: int, profile: str, dry_run: bool = False) -> bool:
    """Generate synthetic data for the given seed."""
    mc_data_dir = config.MC_DATA_DIR
    
    # Create seed-specific file paths
    seed_suffix = f"_seed{seed}"
    hdf5_path = mc_data_dir / f"synthetic_data{seed_suffix}.h5"
    m5_data_dir = mc_data_dir / f"m5_format{seed_suffix}"
    
    logger.info(f"Generating synthetic data for seed {seed}")
    
    # Step 0a: Generate HDF5 data
    gen_mc_data_script = PROJECT_ROOT / "scripts" / "gen_mc_data.py"
    gen_cmd = [
        sys.executable, str(gen_mc_data_script),
        "-p", profile,
        "-o", str(hdf5_path),
        "-s", str(seed)
    ]
    
    if dry_run:
        logger.info(f"[DRY RUN] Would run: {' '.join(gen_cmd)}")
    else:
        try:
            result = subprocess.run(gen_cmd, capture_output=True, text=True, check=True)
            logger.info(f"✓ Generated HDF5 data for seed {seed}")
        except subprocess.CalledProcessError as e:
            logger.error(f"✗ Failed to generate HDF5 data for seed {seed}")
            logger.error(f"Stderr: {e.stderr}")
            return False
    
    # Step 0b: Convert to M5 format
    convert_cmd = [
        sys.executable, "convert_hdf5_to_m5.py",
        str(hdf5_path),
        str(m5_data_dir)
    ]
    
    if dry_run:
        logger.info(f"[DRY RUN] Would run: {' '.join(convert_cmd)}")
    else:
        try:
            result = subprocess.run(convert_cmd, capture_output=True, text=True, check=True)
            logger.info(f"✓ Converted to M5 format for seed {seed}")
        except subprocess.CalledProcessError as e:
            logger.error(f"✗ Failed to convert to M5 format for seed {seed}")
            logger.error(f"Stderr: {e.stderr}")
            return False
    
    return True

def main():
    """Main execution function."""
    args = parse_arguments()
    
    # Use the simplified config
    seeds = get_seeds_to_run(args, config)
    steps = [int(s.strip()) for s in args.steps.split(",")]
    
    logger.info("=" * 80)
    logger.info("MULTI-SEED MONTE CARLO EXPERIMENT")
    logger.info("=" * 80)
    logger.info(f"Profile: {args.profile}")
    logger.info(f"Seeds: {seeds}")
    logger.info(f"Steps: {steps}")
    logger.info(f"Dry run: {args.dry_run}")
    logger.info("=" * 80)
    
    # Track results
    results = {}
    start_time = datetime.now()
    
    # Run experiments for each seed
    for seed_idx, seed in enumerate(seeds):
        logger.info(f"\n{'='*20} SEED {seed} ({seed_idx+1}/{len(seeds)}) {'='*20}")
        
        results[seed] = {"success": True, "failed_steps": []}
        
        # Generate synthetic data for this seed
        if not generate_synthetic_data(seed, args.profile, args.dry_run):
            results[seed]["success"] = False
            results[seed]["failed_steps"].append("data_generation")
            if not args.continue_on_error:
                logger.error(f"Data generation failed for seed {seed}, stopping")
                break
            else:
                logger.warning(f"Data generation failed for seed {seed}, continuing...")
                continue
        
        # Run workflow steps
        for step in steps:
            success = run_workflow_step(step, seed, args.profile, args.dry_run)
            if not success:
                results[seed]["success"] = False
                results[seed]["failed_steps"].append(step)
                if not args.continue_on_error:
                    logger.error(f"Step {step} failed for seed {seed}, stopping")
                    break
                else:
                    logger.warning(f"Step {step} failed for seed {seed}, continuing...")
        
        if results[seed]["success"]:
            logger.info(f"✓ All steps completed successfully for seed {seed}")
        else:
            logger.error(f"✗ Some steps failed for seed {seed}: {results[seed]['failed_steps']}")
    
    # Print summary
    end_time = datetime.now()
    duration = end_time - start_time
    
    logger.info("\n" + "=" * 80)
    logger.info("EXPERIMENT SUMMARY")
    logger.info("=" * 80)
    logger.info(f"Total duration: {duration}")
    logger.info(f"Seeds processed: {len(results)}")
    
    successful_seeds = [seed for seed, result in results.items() if result["success"]]
    failed_seeds = [seed for seed, result in results.items() if not result["success"]]
    
    logger.info(f"Successful: {len(successful_seeds)}/{len(results)}")
    if successful_seeds:
        logger.info(f"  Seeds: {successful_seeds}")
    
    if failed_seeds:
        logger.info(f"Failed: {len(failed_seeds)}/{len(results)}")
        logger.info(f"  Seeds: {failed_seeds}")
        for seed in failed_seeds:
            logger.info(f"    Seed {seed} failed steps: {results[seed]['failed_steps']}")
    
    # Save results summary
    if not args.dry_run:
        summary_file = config_mc_master.MC_DATA_DIR / f"experiment_summary_{args.profile}_{start_time.strftime('%Y%m%d_%H%M%S')}.json"
        summary_data = {
            "profile": args.profile,
            "seeds": seeds,
            "steps": steps,
            "start_time": start_time.isoformat(),
            "end_time": end_time.isoformat(),
            "duration_seconds": duration.total_seconds(),
            "results": results
        }
        
        with open(summary_file, 'w') as f:
            json.dump(summary_data, f, indent=2)
        logger.info(f"Results saved to: {summary_file}")
    
    logger.info("=" * 80)
    
    # Exit with error code if any seeds failed
    if failed_seeds:
        sys.exit(1)

if __name__ == "__main__":
    main()