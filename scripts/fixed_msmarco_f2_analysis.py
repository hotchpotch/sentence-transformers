#!/usr/bin/env python
"""
Fixed F2 score analysis for MS MARCO models.
"""

import sys
from pathlib import Path
import logging
import numpy as np
import pandas as pd
from datasets import Dataset
from typing import List, Dict, Tuple
import matplotlib.pyplot as plt

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from sentence_transformers.pruning import PruningEncoder

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def calculate_f2_score(y_true: List[int], y_pred: List[int]) -> float:
    """Calculate F2 score manually."""
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    
    tp = np.sum((y_pred == 1) & (y_true == 1))
    fp = np.sum((y_pred == 1) & (y_true == 0))
    fn = np.sum((y_pred == 0) & (y_true == 1))
    
    if tp + fp == 0:
        precision = 0.0
    else:
        precision = tp / (tp + fp)
    
    if tp + fn == 0:
        recall = 0.0
    else:
        recall = tp / (tp + fn)
    
    if precision + recall == 0:
        f2 = 0.0
    else:
        f2 = 5 * precision * recall / (4 * precision + recall)
    
    return f2


def evaluate_model_f2_fixed(model_path: str, test_dataset: Dataset, model_name: str, thresholds: List[float]) -> Dict:
    """Evaluate model F2 scores with fixed calculation."""
    logger.info(f"Evaluating {model_name} from {model_path}")
    
    # Load model
    model = PruningEncoder.from_pretrained(model_path)
    logger.info(f"Model mode: {model.mode}")
    
    results = {}
    
    # Take subset for evaluation
    eval_size = min(200, len(test_dataset))  # Smaller for faster analysis
    eval_dataset = test_dataset.select(range(eval_size))
    
    logger.info(f"Evaluating on {eval_size} samples")
    
    for threshold in thresholds:
        logger.info(f"Evaluating at threshold {threshold}")
        
        pos_predictions = []
        pos_labels = []
        neg_predictions = []
        neg_labels = []
        all_predictions = []
        all_labels = []
        
        # Process samples
        for i, sample in enumerate(eval_dataset):
            if i % 50 == 0:
                logger.info(f"Processing sample {i}/{eval_size}")
            
            query = sample['query']
            texts = sample['texts']
            labels = sample['labels']  # [1, 0, 0, 0, 0, 0, 0, 0]
            
            # Prepare query-text pairs
            query_text_pairs = [(query, text) for text in texts]
            
            try:
                # Get predictions with pruning
                outputs = model.predict_with_pruning(
                    query_text_pairs,
                    pruning_threshold=threshold,
                    return_documents=True
                )
                
                # Process each text
                for j, (output, label) in enumerate(zip(outputs, labels)):
                    # Binary prediction based on compression ratio
                    compression_ratio = output.compression_ratio
                    
                    # Lower compression means content is relevant (kept)
                    # Higher compression means content is irrelevant (pruned)
                    prediction = 1 if compression_ratio < (1.0 - threshold) else 0
                    
                    all_predictions.append(prediction)
                    all_labels.append(label)
                    
                    if j == 0:  # First text is positive
                        pos_predictions.append(prediction)
                        pos_labels.append(label)
                    else:  # Remaining texts are negative
                        neg_predictions.append(prediction)
                        neg_labels.append(label)
                        
            except Exception as e:
                logger.warning(f"Error processing sample {i}: {e}")
                # Skip problematic samples
                continue
        
        # Calculate F2 scores
        pos_f2 = calculate_f2_score(pos_labels, pos_predictions)
        neg_f2 = calculate_f2_score(neg_labels, neg_predictions)
        all_f2 = calculate_f2_score(all_labels, all_predictions)
        
        # Calculate precision and recall
        def calc_precision_recall(y_true, y_pred):
            y_true = np.array(y_true)
            y_pred = np.array(y_pred)
            
            tp = np.sum((y_pred == 1) & (y_true == 1))
            fp = np.sum((y_pred == 1) & (y_true == 0))
            fn = np.sum((y_pred == 0) & (y_true == 1))
            
            precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
            
            return precision, recall
        
        pos_precision, pos_recall = calc_precision_recall(pos_labels, pos_predictions)
        neg_precision, neg_recall = calc_precision_recall(neg_labels, neg_predictions)
        
        results[threshold] = {
            'pos_f2': pos_f2,
            'neg_f2': neg_f2,
            'all_f2': all_f2,
            'pos_precision': pos_precision,
            'pos_recall': pos_recall,
            'neg_precision': neg_precision,
            'neg_recall': neg_recall,
            'pos_samples': len(pos_labels),
            'neg_samples': len(neg_labels),
            'pos_kept_ratio': np.mean(pos_predictions) if pos_predictions else 0,
            'neg_kept_ratio': np.mean(neg_predictions) if neg_predictions else 0,
        }
        
        logger.info(f"Threshold {threshold}: POS F2={pos_f2:.4f}, NEG F2={neg_f2:.4f}, ALL F2={all_f2:.4f}")
    
    return results


def create_fixed_analysis_plots(results_dict: Dict, save_path: str):
    """Create analysis plots for F2 scores."""
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))
    
    # Extract data for plotting
    models = list(results_dict.keys())
    thresholds = list(next(iter(results_dict.values())).keys())
    
    colors = ['tab:blue', 'tab:orange', 'tab:green', 'tab:red']
    markers = ['o', 's', '^', 'v']
    
    # Plot 1: F2 Scores by threshold
    for i, model in enumerate(models):
        results = results_dict[model]
        pos_f2s = [results[t]['pos_f2'] for t in thresholds]
        neg_f2s = [results[t]['neg_f2'] for t in thresholds]
        all_f2s = [results[t]['all_f2'] for t in thresholds]
        
        ax1.plot(thresholds, pos_f2s, color=colors[i], marker=markers[i], 
                linestyle='-', label=f'{model} (POS)', alpha=0.8)
        ax1.plot(thresholds, neg_f2s, color=colors[i], marker=markers[i], 
                linestyle='--', label=f'{model} (NEG)', alpha=0.8)
        ax1.plot(thresholds, all_f2s, color=colors[i], marker=markers[i], 
                linestyle=':', label=f'{model} (ALL)', alpha=0.8)
    
    ax1.set_xlabel('Pruning Threshold')
    ax1.set_ylabel('F2 Score')
    ax1.set_title('F2 Scores vs Threshold')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: Best F2 comparison
    best_scores = {}
    for model in models:
        results = results_dict[model]
        best_threshold = max(results.keys(), key=lambda t: results[t]['all_f2'])
        best_scores[model] = {
            'pos_f2': results[best_threshold]['pos_f2'],
            'neg_f2': results[best_threshold]['neg_f2'],
            'all_f2': results[best_threshold]['all_f2'],
            'threshold': best_threshold
        }
    
    model_names = list(best_scores.keys())
    pos_scores = [best_scores[m]['pos_f2'] for m in model_names]
    neg_scores = [best_scores[m]['neg_f2'] for m in model_names]
    all_scores = [best_scores[m]['all_f2'] for m in model_names]
    
    x = np.arange(len(model_names))
    width = 0.25
    
    ax2.bar(x - width, pos_scores, width, label='POS F2', alpha=0.8, color='lightblue')
    ax2.bar(x, neg_scores, width, label='NEG F2', alpha=0.8, color='lightcoral')
    ax2.bar(x + width, all_scores, width, label='ALL F2', alpha=0.8, color='lightgreen')
    
    ax2.set_xlabel('Model')
    ax2.set_ylabel('Best F2 Score')
    ax2.set_title('Best F2 Scores Comparison')
    ax2.set_xticks(x)
    ax2.set_xticklabels([m.replace('_', '\n') for m in model_names])
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # Add score labels on bars
    for i, (pos, neg, all_) in enumerate(zip(pos_scores, neg_scores, all_scores)):
        ax2.text(i - width, pos + 0.01, f'{pos:.3f}', ha='center', va='bottom', fontsize=9)
        ax2.text(i, neg + 0.01, f'{neg:.3f}', ha='center', va='bottom', fontsize=9)
        ax2.text(i + width, all_ + 0.01, f'{all_:.3f}', ha='center', va='bottom', fontsize=9)
    
    # Plot 3: Precision vs Recall
    for i, model in enumerate(models):
        results = results_dict[model]
        best_threshold = max(results.keys(), key=lambda t: results[t]['all_f2'])
        best_result = results[best_threshold]
        
        ax3.scatter(best_result['pos_recall'], best_result['pos_precision'], 
                   s=100, color=colors[i], marker=markers[i], label=f'{model} (POS)', alpha=0.8)
        ax3.scatter(best_result['neg_recall'], best_result['neg_precision'], 
                   s=100, color=colors[i], marker=markers[i], label=f'{model} (NEG)', alpha=0.8, 
                   facecolors='none', edgecolors=colors[i])
    
    ax3.set_xlabel('Recall')
    ax3.set_ylabel('Precision')
    ax3.set_title('Precision vs Recall (Best Threshold)')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    ax3.set_xlim(0, 1)
    ax3.set_ylim(0, 1)
    
    # Plot 4: Keep ratios
    for i, model in enumerate(models):
        results = results_dict[model]
        pos_keep_ratios = [results[t]['pos_kept_ratio'] for t in thresholds]
        neg_keep_ratios = [results[t]['neg_kept_ratio'] for t in thresholds]
        
        ax4.plot(thresholds, pos_keep_ratios, color=colors[i], marker=markers[i],
                linestyle='-', label=f'{model} (POS)', alpha=0.8)
        ax4.plot(thresholds, neg_keep_ratios, color=colors[i], marker=markers[i],
                linestyle='--', label=f'{model} (NEG)', alpha=0.8)
    
    ax4.set_xlabel('Pruning Threshold')
    ax4.set_ylabel('Keep Ratio (1 - Pruning Rate)')
    ax4.set_title('Content Keep Ratios vs Threshold')
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    logger.info(f"Analysis plots saved to {save_path}")


def main():
    """Run fixed F2 analysis."""
    logger.info("="*60)
    logger.info("Fixed MS MARCO F2 Score Analysis")
    logger.info("="*60)
    
    # Load test data
    base_path = f"hf://datasets/hotchpotch/wip-msmarco-context-relevance/msmarco-small-ja"
    test_df = pd.read_parquet(f"{base_path}/test-00000-of-00001.parquet")
    test_dataset = Dataset.from_pandas(test_df)
    logger.info(f"Loaded test set: {len(test_dataset)} samples")
    
    # Find model paths
    import glob
    models = {}
    
    # Find reranking+pruning model
    reranking_paths = glob.glob("./output/msmarco_small_reranking_pruning_*/final_model")
    if reranking_paths:
        models["reranking_pruning"] = reranking_paths[-1]
    
    # Find pruning-only model
    pruning_paths = glob.glob("./output/msmarco_small_pruning_only_*/final_model")
    if pruning_paths:
        models["pruning_only"] = pruning_paths[-1]
    
    # Filter existing models
    existing_models = {k: v for k, v in models.items() if Path(v).exists()}
    
    if not existing_models:
        logger.error("No models found!")
        return
    
    logger.info(f"Found models: {list(existing_models.keys())}")
    
    # Thresholds to test
    thresholds = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
    
    # Evaluate all models
    results_dict = {}
    for model_name, model_path in existing_models.items():
        results_dict[model_name] = evaluate_model_f2_fixed(
            model_path, test_dataset, model_name, thresholds
        )
    
    # Create analysis plots
    output_dir = Path("./output")
    output_dir.mkdir(exist_ok=True)
    create_fixed_analysis_plots(results_dict, str(output_dir / "msmarco_f2_analysis_fixed.png"))
    
    # Print summary
    logger.info("="*60)
    logger.info("SUMMARY RESULTS")
    logger.info("="*60)
    
    for model_name, results in results_dict.items():
        logger.info(f"\n{model_name.upper()}:")
        best_threshold = max(results.keys(), key=lambda t: results[t]['all_f2'])
        best_result = results[best_threshold]
        
        logger.info(f"  Best threshold: {best_threshold}")
        logger.info(f"  POS F2: {best_result['pos_f2']:.4f}")
        logger.info(f"  NEG F2: {best_result['neg_f2']:.4f}")
        logger.info(f"  ALL F2: {best_result['all_f2']:.4f}")
        logger.info(f"  POS Precision/Recall: {best_result['pos_precision']:.4f}/{best_result['pos_recall']:.4f}")
        logger.info(f"  NEG Precision/Recall: {best_result['neg_precision']:.4f}/{best_result['neg_recall']:.4f}")
        logger.info(f"  POS Keep Ratio: {best_result['pos_kept_ratio']:.4f}")
        logger.info(f"  NEG Keep Ratio: {best_result['neg_kept_ratio']:.4f}")
    
    # Data structure analysis
    logger.info("="*60)
    logger.info("DATA STRUCTURE ANALYSIS")
    logger.info("="*60)
    
    sample = test_dataset[0]
    logger.info(f"Sample query: {sample['query']}")
    logger.info(f"Number of texts: {len(sample['texts'])}")
    logger.info(f"Labels: {sample['labels']} (first=pos, rest=neg)")
    
    # Check label distribution
    pos_count = 0
    neg_count = 0
    for sample in test_dataset.select(range(100)):
        labels = sample['labels']
        pos_count += sum(labels)
        neg_count += len(labels) - sum(labels)
    
    logger.info(f"Label distribution (first 100 samples): {pos_count} positive, {neg_count} negative")
    logger.info(f"Pos/Neg ratio: 1:{neg_count/pos_count:.1f}")
    
    # Key findings
    logger.info("="*60)
    logger.info("KEY FINDINGS")
    logger.info("="*60)
    
    logger.info("1. Data structure: Each sample has 8 texts, first is positive, rest are negative")
    logger.info("2. MS MARCO dataset has 1:7 pos/neg ratio per sample")
    logger.info("3. Models show different performance patterns:")
    
    if "reranking_pruning" in results_dict and "pruning_only" in results_dict:
        reranking_best = max(results_dict["reranking_pruning"].keys(), 
                           key=lambda t: results_dict["reranking_pruning"][t]['all_f2'])
        pruning_best = max(results_dict["pruning_only"].keys(), 
                         key=lambda t: results_dict["pruning_only"][t]['all_f2'])
        
        logger.info(f"   - Reranking+Pruning: F2={results_dict['reranking_pruning'][reranking_best]['all_f2']:.4f} @ threshold {reranking_best}")
        logger.info(f"   - Pruning-Only: F2={results_dict['pruning_only'][pruning_best]['all_f2']:.4f} @ threshold {pruning_best}")


if __name__ == "__main__":
    main()