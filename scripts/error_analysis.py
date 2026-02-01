#!/usr/bin/env python3
"""
Generate confusion matrices and detailed error analysis for menu extraction.
"""

import sys
import json
import argparse
from pathlib import Path
from collections import defaultdict
from typing import Optional

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.models.schema import TextElementType


def bootstrap_confidence_interval(
    scores: list[float],
    confidence: float = 0.95,
    n_bootstrap: int = 1000,
) -> tuple[float, float, float]:
    """
    Compute bootstrap confidence interval for a metric.
    
    Returns:
    --------
    Tuple of (mean, lower_bound, upper_bound)
    """
    if not scores:
        return 0.0, 0.0, 0.0
    
    scores = np.array(scores)
    mean = np.mean(scores)
    
    if len(scores) < 2:
        return mean, mean, mean
    
    # Bootstrap resampling
    bootstrapped = []
    for _ in range(n_bootstrap):
        sample = np.random.choice(scores, size=len(scores), replace=True)
        bootstrapped.append(np.mean(sample))
    
    alpha = 1 - confidence
    lower = np.percentile(bootstrapped, alpha / 2 * 100)
    upper = np.percentile(bootstrapped, (1 - alpha / 2) * 100)
    
    return mean, lower, upper


def plot_confusion_matrix(
    y_true: list[str],
    y_pred: list[str],
    labels: Optional[list[str]] = None,
    output_path: Optional[Path] = None,
    title: str = "Classification Confusion Matrix",
) -> np.ndarray:
    """
    Generate and plot confusion matrix.
    
    Parameters:
    -----------
    y_true : True labels
    y_pred : Predicted labels
    labels : Label names (optional)
    output_path : Path to save figure
    title : Plot title
    
    Returns:
    --------
    Confusion matrix as numpy array
    """
    from sklearn.metrics import confusion_matrix
    
    if labels is None:
        labels = sorted(set(y_true) | set(y_pred))
    
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    
    # Plot
    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(
        cm, annot=True, fmt='d', cmap='Blues',
        xticklabels=labels, yticklabels=labels,
        ax=ax
    )
    ax.set_xlabel('Predicted')
    ax.set_ylabel('True')
    ax.set_title(title)
    plt.tight_layout()
    
    if output_path:
        plt.savefig(output_path, dpi=150)
        print(f"Saved confusion matrix to {output_path}")
    
    plt.close()
    
    return cm


def compute_per_stage_errors(
    results: list[dict],
) -> dict:
    """
    Compute error decomposition by pipeline stage.
    
    Expected results format:
    {
        'ocr_detections': int,
        'ocr_correct': int,
        'classification_correct': int,
        'matching_correct': int,
        'total_gt_items': int,
    }
    """
    totals = defaultdict(int)
    
    for r in results:
        totals['ocr_detections'] += r.get('ocr_detections', 0)
        totals['ocr_correct'] += r.get('ocr_correct', 0)
        totals['classification_total'] += r.get('classification_total', 0)
        totals['classification_correct'] += r.get('classification_correct', 0)
        totals['matching_total'] += r.get('matching_total', 0)
        totals['matching_correct'] += r.get('matching_correct', 0)
        totals['total_gt_items'] += r.get('total_gt_items', 0)
    
    # Compute rates
    ocr_recall = totals['ocr_correct'] / max(totals['total_gt_items'], 1)
    classification_acc = totals['classification_correct'] / max(totals['classification_total'], 1)
    matching_acc = totals['matching_correct'] / max(totals['matching_total'], 1)
    
    # Cumulative error
    # Items lost at each stage
    ocr_loss = 1 - ocr_recall
    classification_loss = ocr_recall * (1 - classification_acc)
    matching_loss = ocr_recall * classification_acc * (1 - matching_acc)
    
    return {
        'ocr_recall': ocr_recall,
        'ocr_error_rate': 1 - ocr_recall,
        'classification_accuracy': classification_acc,
        'classification_error_rate': 1 - classification_acc,
        'matching_accuracy': matching_acc,
        'matching_error_rate': 1 - matching_acc,
        'cumulative_ocr_loss': ocr_loss,
        'cumulative_classification_loss': ocr_loss + classification_loss,
        'cumulative_total_loss': ocr_loss + classification_loss + matching_loss,
    }


def generate_error_analysis_report(
    evaluation_results: dict,
    output_dir: Path,
) -> dict:
    """
    Generate comprehensive error analysis report.
    
    Parameters:
    -----------
    evaluation_results : Results from evaluation script
    output_dir : Directory to save outputs
    
    Returns:
    --------
    Analysis summary dict
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    results = evaluation_results.get('results', [])
    if not results:
        print("No results to analyze")
        return {}
    
    # Extract metrics
    f1_scores = [r['f1_score'] for r in results]
    precision_scores = [r['precision'] for r in results]
    recall_scores = [r['recall'] for r in results]
    price_accuracies = [r['price_accuracy'] for r in results]
    
    # Compute confidence intervals
    analysis = {
        'f1': {
            'mean': np.mean(f1_scores),
            'ci_95': bootstrap_confidence_interval(f1_scores),
        },
        'precision': {
            'mean': np.mean(precision_scores),
            'ci_95': bootstrap_confidence_interval(precision_scores),
        },
        'recall': {
            'mean': np.mean(recall_scores),
            'ci_95': bootstrap_confidence_interval(recall_scores),
        },
        'price_accuracy': {
            'mean': np.mean(price_accuracies),
            'ci_95': bootstrap_confidence_interval(price_accuracies),
        },
    }
    
    # Generate report text
    report_lines = [
        "=" * 60,
        "ERROR ANALYSIS REPORT",
        "=" * 60,
        "",
        f"Number of samples: {len(results)}",
        "",
        "METRICS WITH 95% CONFIDENCE INTERVALS",
        "-" * 40,
    ]
    
    for metric_name, metric_data in analysis.items():
        mean, lower, upper = metric_data['ci_95']
        report_lines.append(
            f"{metric_name.replace('_', ' ').title():20s}: "
            f"{mean:.1%} [{lower:.1%}, {upper:.1%}]"
        )
    
    report_lines.extend([
        "",
        "PER-SAMPLE BREAKDOWN",
        "-" * 40,
    ])
    
    for r in results:
        report_lines.append(
            f"  {r['model']:20s}: F1={r['f1_score']:.1%}, "
            f"Price={r['price_accuracy']:.1%}"
        )
    
    # Save report
    report_path = output_dir / "error_analysis_report.txt"
    with open(report_path, 'w') as f:
        f.write('\n'.join(report_lines))
    print(f"Saved report to {report_path}")
    
    # Save analysis JSON
    analysis_path = output_dir / "error_analysis.json"
    
    # Convert tuples to lists for JSON serialization
    json_analysis = {}
    for k, v in analysis.items():
        json_analysis[k] = {
            'mean': v['mean'],
            'ci_lower': v['ci_95'][1],
            'ci_upper': v['ci_95'][2],
        }
    
    with open(analysis_path, 'w') as f:
        json.dump(json_analysis, f, indent=2)
    print(f"Saved analysis to {analysis_path}")
    
    return analysis


def main():
    parser = argparse.ArgumentParser(description="Generate error analysis")
    parser.add_argument(
        "--results", "-r",
        type=Path,
        default=Path("models/evaluation_results.json"),
        help="Path to evaluation results JSON"
    )
    parser.add_argument(
        "--output", "-o",
        type=Path,
        default=Path("output/analysis"),
        help="Output directory for analysis"
    )
    args = parser.parse_args()
    
    # Load results
    if not args.results.exists():
        print(f"Results file not found: {args.results}")
        return
    
    with open(args.results) as f:
        results = json.load(f)
    
    # Generate analysis
    analysis = generate_error_analysis_report(results, args.output)
    
    print("\nAnalysis complete!")


if __name__ == "__main__":
    main()
