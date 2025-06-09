#////////////////////////////////////////////////////////////////////////////////#
# File:         6_statistical_comparison.py                                      #
# Author:       Douglas Nyberg                                                   #
# Email:        douglas1.nyberg@gmail.com                                        #
# Date:         2025-05-12                                                       #
#////////////////////////////////////////////////////////////////////////////////#
#!/usr/bin/env python3
"""statistical comparison between models"""

# imports
import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any
import warnings

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from statsmodels.stats.multicomp import pairwise_tukeyhsd

# project path
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent.parent
sys.path.append(str(PROJECT_ROOT))

# project modules
import config_exp1 as config
from src.utils import set_random_seed
from src.evaluators.model_comparison import compare_models_wrmsse

# logging
config.LOGS_DIR.mkdir(parents=True, exist_ok=True)
logging.basicConfig(
    level=getattr(logging, config.LOGGING_CONFIG["level"]),
    format=config.LOGGING_CONFIG["format"],
    handlers=[
        logging.FileHandler(config.LOGS_DIR / "statistical_comparison.log", mode='a'),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

# cli args
def parse_arguments() -> argparse.Namespace:
    """parse args"""
    parser = argparse.ArgumentParser(
        description="Statistical comparison between LSTM and Classical models"
    )
    
    parser.add_argument(
        "--results-dir", type=str,
        default="results",
        help="Directory containing evaluation results for all models"
    )
    
    parser.add_argument(
        "--models", nargs="+", 
        default=["lstm", "arima", "croston", "tsb"],
        help="List of models to compare"
    )
    
    parser.add_argument(
        "--significance-level", type=float, default=0.05,
        help="Significance level for statistical tests (default: 0.05)"
    )
    
    parser.add_argument(
        "--output-dir", type=str,
        default=None,
        help="Directory to save comparison results (default: results_dir/statistical_comparison)"
    )
    
    return parser.parse_args()


# load results
def load_model_results(results_dir: Path, model_name: str) -> Dict[str, Any]:
    """
    Load evaluation results for a specific model.
    
    args:
        results_dir: Base results directory
        model_name: Name of the model (e.g., 'lstm', 'arima')
        
    returns:
        Dictionary containing evaluation metrics and per-series results
    """
    # Check if evaluation results exist in the expected location
    # First try the unified evaluation summary (all models together)
    unified_summary_file = results_dir / "evaluation_summary.json"
    model_specific_metrics_file = results_dir / f"{model_name}_comprehensive_metrics.csv"
    
    if unified_summary_file.exists() and model_specific_metrics_file.exists():
        # Load unified summary and extract this model's results
        with open(unified_summary_file, 'r') as f:
            unified_summary = json.load(f)
        
        # Check if this model's results are in the aggregated results
        if model_name in unified_summary.get("aggregated_results", {}):
            summary = {"aggregated_metrics": unified_summary["aggregated_results"][model_name]}
        else:
            # Create a basic summary structure
            summary = {"aggregated_metrics": {}}
        
        series_metrics_file = model_specific_metrics_file
    else:
        # Fallback to old structure (individual model directories)
        model_eval_dir = results_dir / config.EXPERIMENT_NAME / model_name / "evaluation"
        summary_file = model_eval_dir / f"{model_name}_evaluation_summary.json"
        
        if not summary_file.exists():
            raise FileNotFoundError(f"Evaluation summary not found: {summary_file}")
        
        with open(summary_file, 'r') as f:
            summary = json.load(f)
        
        series_metrics_file = model_eval_dir / f"{model_name}_series_level_point_metrics.csv"
    if series_metrics_file.exists():
        series_metrics = pd.read_csv(series_metrics_file, index_col=0)
    else:
        series_metrics = None
        logger.warning(f"Per-series metrics not found for {model_name}")
    
    return {
        'model_name': model_name,
        'summary': summary,
        'series_metrics': series_metrics
    }


# Statistical Tests
def perform_paired_tests(
    model1_metrics: pd.Series, 
    model2_metrics: pd.Series,
    model1_name: str,
    model2_name: str,
    significance_level: float = 0.05
) -> Dict[str, Any]:
    """
    Perform paired statistical tests between two models.
    
    Uses both parametric (paired t-test) and non-parametric (Wilcoxon signed-rank) tests.
    """
    # Ensure same series are compared
    common_series = model1_metrics.index.intersection(model2_metrics.index)
    if len(common_series) == 0:
        raise ValueError("No common series found between models")
    
    m1_values = model1_metrics.loc[common_series].values
    m2_values = model2_metrics.loc[common_series].values
    
    # Calculate differences
    differences = m1_values - m2_values
    
    # Paired t-test (assumes normal distribution of differences)
    t_stat, t_pvalue = stats.ttest_rel(m1_values, m2_values)
    
    # Wilcoxon signed-rank test (non-parametric alternative)
    if len(differences) > 20:  # Wilcoxon needs sufficient sample size
        w_stat, w_pvalue = stats.wilcoxon(differences)
    else:
        w_stat, w_pvalue = np.nan, np.nan
    
    # Effect size (Cohen's d for paired samples)
    cohens_d = np.mean(differences) / (np.std(differences, ddof=1) + 1e-10)
    
    # Confidence interval for mean difference
    mean_diff = np.mean(differences)
    std_diff = np.std(differences, ddof=1)
    n = len(differences)
    t_critical = stats.t.ppf(1 - significance_level/2, n-1)
    ci_lower = mean_diff - t_critical * std_diff / np.sqrt(n)
    ci_upper = mean_diff + t_critical * std_diff / np.sqrt(n)
    
    return {
        'model1': model1_name,
        'model2': model2_name,
        'n_series': len(common_series),
        'mean_model1': np.mean(m1_values),
        'mean_model2': np.mean(m2_values),
        'mean_difference': mean_diff,
        'std_difference': std_diff,
        'ci_lower': ci_lower,
        'ci_upper': ci_upper,
        't_statistic': t_stat,
        't_pvalue': t_pvalue,
        'wilcoxon_statistic': w_stat,
        'wilcoxon_pvalue': w_pvalue,
        'cohens_d': cohens_d,
        'significant_t': t_pvalue < significance_level,
        'significant_wilcoxon': w_pvalue < significance_level if not np.isnan(w_pvalue) else None
    }


# Create comparison tables
def create_comparison_summary(
    all_results: Dict[str, Dict],
    metric_name: str = 'rmsse'
) -> pd.DataFrame:
    """
    Create a summary comparison table for all models.
    """
    summary_data = []
    
    for model_name, results in all_results.items():
        if metric_name == 'rmsse':
            value = results['summary']['rmsse']
            summary_data.append({
                'Model': model_name.upper(),
                'RMSSE': value,
                'Rank': 0 
            })
        else:
            # Handle per-series metrics
            if results['series_metrics'] is not None and metric_name in results['series_metrics'].columns:
                values = results['series_metrics'][metric_name]
                summary_data.append({
                    'Model': model_name.upper(),
                    'Mean': values.mean(),
                    'Std': values.std(),
                    'Min': values.min(),
                    'Max': values.max(),
                    'Median': values.median()
                })
    
    df = pd.DataFrame(summary_data)
    
    # Add ranking for WRMSSE  
    if 'WRMSSE' in df.columns:
        df['Rank'] = df['WRMSSE'].rank().astype(int)
        df = df.sort_values('WRMSSE')
    
    return df


# Visualization functions
def plot_metric_distributions(
    all_results: Dict[str, Dict],
    metric_name: str,
    output_path: Path
):
    """
    Create box plots comparing metric distributions across models.
    """
    # Prepare data for plotting
    plot_data = []
    
    for model_name, results in all_results.items():
        if results['series_metrics'] is not None and metric_name in results['series_metrics'].columns:
            values = results['series_metrics'][metric_name].values
            for val in values:
                plot_data.append({
                    'Model': model_name.upper(),
                    metric_name.upper(): val
                })
    
    if not plot_data:
        logger.warning(f"No data available for metric {metric_name}")
        return
    
    df_plot = pd.DataFrame(plot_data)
    
    # Create plot
    plt.figure(figsize=(10, 6))
    sns.boxplot(x='Model', y=metric_name.upper(), data=df_plot)
    plt.title(f'{metric_name.upper()} Distribution by Model')
    plt.ylabel(metric_name.upper())
    plt.tight_layout()
    plt.savefig(output_path / f'{metric_name}_distribution.png', dpi=300)
    plt.close()


def create_pairwise_comparison_heatmap(
    all_results: Dict[str, Dict],
    metric_name: str,
    significance_level: float,
    output_path: Path
):
    """
    Create a heatmap showing pairwise statistical comparisons.
    """
    models = list(all_results.keys())
    n_models = len(models)
    
    # Initialize matrices for p-values and effect sizes
    pvalue_matrix = np.ones((n_models, n_models))
    effect_matrix = np.zeros((n_models, n_models))
    
    # Perform pairwise comparisons
    for i, model1 in enumerate(models):
        for j, model2 in enumerate(models):
            if i != j and all_results[model1]['series_metrics'] is not None and \
               all_results[model2]['series_metrics'] is not None:
                
                m1_metrics = all_results[model1]['series_metrics'][metric_name]
                m2_metrics = all_results[model2]['series_metrics'][metric_name]
                
                test_results = perform_paired_tests(
                    m1_metrics, m2_metrics, model1, model2, significance_level
                )
                
                pvalue_matrix[i, j] = test_results['t_pvalue']
                effect_matrix[i, j] = test_results['cohens_d']
    
    # Create heatmap
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # P-value heatmap
    sns.heatmap(
        pvalue_matrix, 
        annot=True, 
        fmt='.3f',
        xticklabels=[m.upper() for m in models],
        yticklabels=[m.upper() for m in models],
        cmap='RdYlGn_r',
        vmin=0, vmax=0.1,
        ax=ax1
    )
    ax1.set_title(f'Pairwise P-values ({metric_name.upper()})')
    
    # Effect size heatmap
    sns.heatmap(
        effect_matrix,
        annot=True,
        fmt='.2f',
        xticklabels=[m.upper() for m in models],
        yticklabels=[m.upper() for m in models],
        cmap='coolwarm',
        center=0,
        ax=ax2
    )
    ax2.set_title(f"Cohen's d Effect Size ({metric_name.upper()})")
    
    plt.tight_layout()
    plt.savefig(output_path / f'{metric_name}_pairwise_comparison.png', dpi=300)
    plt.close()


# Main execution
def main():
    """Main function to run statistical comparison."""
    args = parse_arguments()
    set_random_seed(config.RANDOM_SEED)
    
    results_dir = Path(args.results_dir)
    output_dir = Path(args.output_dir) if args.output_dir else results_dir / "statistical_comparison"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    logger.info("="*70)
    logger.info("STARTING STATISTICAL COMPARISON")
    logger.info("="*70)
    logger.info(f"Results directory: {results_dir}")
    logger.info(f"Models to compare: {args.models}")
    logger.info(f"Significance level: {args.significance_level}")
    logger.info(f"Output directory: {output_dir}")
    
    # Load all model results
    all_results = {}
    for model in args.models:
        try:
            all_results[model] = load_model_results(results_dir, model)
            logger.info(f"Loaded results for {model}")
        except Exception as e:
            logger.error(f"Failed to load results for {model}: {e}")
    
    if len(all_results) < 2:
        logger.error("Need at least 2 models for comparison")
        return
    
    # Create WRMSSE comparison using the dedicated evaluator
    logger.info("\nCalculating WRMSSE using M5 competition standard...")
    
    # Use correct directory paths for the experiment
    project_root = results_dir.parent.parent.parent
    data_dir = str(project_root / "data" / "preprocessed" / f"exp1_{config.EXPERIMENT_TAG}")
    pred_dir = str(project_root / "predictions" / f"exp1_{config.EXPERIMENT_TAG}" / results_dir.name)
    
    wrmsse_summary = None
    try:
        wrmsse_results = compare_models_wrmsse(
            data_dir=data_dir,
            predictions_dir=pred_dir,
            models=args.models
        )
        
        # Create WRMSSE comparison table from direct calculation
        wrmsse_data = []
        for model, wrmsse in wrmsse_results.items():
            wrmsse_data.append({
                'Model': model.upper(),
                'WRMSSE': wrmsse,
                'Rank': 0  # Will be filled below
            })
        
        wrmsse_summary = pd.DataFrame(wrmsse_data)
        if not wrmsse_summary.empty:
            wrmsse_summary['Rank'] = wrmsse_summary['WRMSSE'].rank().astype(int)
            wrmsse_summary = wrmsse_summary.sort_values('WRMSSE')
            logger.info("\nWRMSSE Summary (M5 Competition Standard):")
            logger.info(wrmsse_summary.to_string())
            wrmsse_summary.to_csv(output_dir / 'wrmsse_comparison.csv', index=False)
        else:
            logger.warning("No WRMSSE results calculated - creating empty summary")
            wrmsse_summary = pd.DataFrame()
            
    except Exception as e:
        logger.warning(f"WRMSSE calculation failed: {e}")
        logger.info("WRMSSE calculation not available for this dataset")
        # Create empty summary as fallback
        wrmsse_summary = pd.DataFrame()
    
    # Perform pairwise statistical tests
    logger.info("\nPerforming pairwise statistical tests...")
    pairwise_results = []
    
    for metric in ['mae', 'rmse', 'mase']:
        logger.info(f"\nAnalyzing {metric.upper()}...")
        
        # Create visualizations
        plot_metric_distributions(all_results, metric, output_dir)
        create_pairwise_comparison_heatmap(all_results, metric, args.significance_level, output_dir)
        
        # Perform all pairwise comparisons
        models = list(all_results.keys())
        for i, model1 in enumerate(models):
            for j, model2 in enumerate(models[i+1:], i+1):
                if all_results[model1]['series_metrics'] is not None and \
                   all_results[model2]['series_metrics'] is not None:
                    
                    m1_metrics = all_results[model1]['series_metrics'][metric]
                    m2_metrics = all_results[model2]['series_metrics'][metric]
                    
                    test_result = perform_paired_tests(
                        m1_metrics, m2_metrics, model1, model2, args.significance_level
                    )
                    test_result['metric'] = metric
                    pairwise_results.append(test_result)
    
    # Save pairwise results
    pairwise_df = pd.DataFrame(pairwise_results)
    pairwise_df.to_csv(output_dir / 'pairwise_statistical_tests.csv', index=False)
    
    # Create summary
    logger.info("\nCreating Summary...")
    
    pub_summary = []
    for model, results in all_results.items():
        if results['series_metrics'] is not None:
            # Use WRMSSE from the calculated results if available
            wrmsse_val = 'N/A'
            if wrmsse_summary is not None and not wrmsse_summary.empty:
                model_wrmsse = wrmsse_summary[wrmsse_summary['Model'] == model.upper()]
                if not model_wrmsse.empty:
                    wrmsse_val = model_wrmsse['WRMSSE'].iloc[0]
            
            # TODO: MASE is broken for LSTM (getting 1600+ values) so commenting it out for now
            # TODO: need to fix whatever series is causing LSTM to go nuts 
            # TODO: also need to make sure all models train on same series (499 vs 500 issue)
            row = {
                'Model': model.upper(),
                'WRMSSE': f"{wrmsse_val:.4f}" if wrmsse_val != 'N/A' else 'N/A',
                'MAE': f"{results['series_metrics']['mae'].mean():.2f} ± {results['series_metrics']['mae'].std():.2f}",
                'RMSE': f"{results['series_metrics']['rmse'].mean():.2f} ± {results['series_metrics']['rmse'].std():.2f}",
                # 'MASE': f"{results['series_metrics']['mase'].mean():.3f} ± {results['series_metrics']['mase'].std():.3f}"
                'MASE': 'DISABLED'  # Temporarily disabled - LSTM values are insane
            }
            pub_summary.append(row)
    
    pub_df = pd.DataFrame(pub_summary)
    pub_df.to_csv(output_dir / 'summary.csv', index=False)
    
    logger.info("\nSummary:")
    logger.info(pub_df.to_string())
    
    # save report
    report = {
        'comparison_date': pd.Timestamp.now().isoformat(),
        'models_compared': args.models,
        'significance_level': args.significance_level,
        'wrmsse_ranking': wrmsse_summary.to_dict('records') if wrmsse_summary is not None and not wrmsse_summary.empty else [],
        'statistical_tests': pairwise_df.to_dict('records'),
        'summary': pub_df.to_dict('records')
    }
    
    with open(output_dir / 'statistical_comparison_report.json', 'w') as f:
        json.dump(report, f, indent=2)
    
    logger.info("="*70)
    logger.info("STATISTICAL COMPARISON COMPLETED")
    logger.info(f"Results saved to: {output_dir}")
    logger.info("="*70)


if __name__ == "__main__":
    main()