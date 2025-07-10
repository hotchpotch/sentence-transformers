#!/usr/bin/env python
"""
Analyze F2 scores for MS MARCO models with proper pos/neg structure understanding.
Data structure: first text is positive, remaining 7 texts are negative.
"""

import sys
from pathlib import Path
import logging
import numpy as np
import pandas as pd
from datasets import Dataset
from typing import List, Dict, Tuple
import matplotlib.pyplot as plt
from sklearn.metrics import precision_recall_fscore_support

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from sentence_transformers.pruning import PruningEncoder

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def load_test_data(subset: str = "msmarco-small-ja"):
    """Load MS MARCO test dataset."""
    base_path = f"hf://datasets/hotchpotch/wip-msmarco-context-relevance/{subset}"
    test_df = pd.read_parquet(f"{base_path}/test-00000-of-00001.parquet")
    test_dataset = Dataset.from_pandas(test_df)
    
    logger.info(f"Loaded {subset} test set: {len(test_dataset)} samples")
    return test_dataset


def evaluate_model_f2(model_path: str, test_dataset: Dataset, model_name: str, thresholds: List[float]) -> Dict:
    """Evaluate model F2 scores at different thresholds."""
    logger.info(f"Evaluating {model_name} from {model_path}")
    
    # Load model
    model = PruningEncoder.from_pretrained(model_path)
    logger.info(f"Model mode: {model.mode}")
    
    results = {}
    
    # Take subset for evaluation (500 samples for faster analysis)
    eval_size = min(500, len(test_dataset))
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
            if i % 100 == 0:
                logger.info(f"Processing sample {i}/{eval_size}")
            
            query = sample['query']
            texts = sample['texts']
            labels = sample['labels']  # [1, 0, 0, 0, 0, 0, 0, 0] - first is pos, rest are neg
            
            # Prepare query-text pairs
            query_text_pairs = [(query, text) for text in texts]
            
            try:
                # Get predictions with pruning
                outputs = model.predict_with_pruning(
                    query_text_pairs,
                    pruning_threshold=threshold,
                    return_documents=True
                )
                
                # Process each text (pos/neg)
                for j, (output, label) in enumerate(zip(outputs, labels)):
                    # Extract binary pruning prediction based on compression ratio
                    # If compression ratio is high, it means most content was pruned (irrelevant)
                    # If compression ratio is low, it means content was kept (relevant)
                    compression_ratio = output.compression_ratio
                    
                    # Binary prediction: 1 if kept (low compression), 0 if pruned (high compression)
                    # Using inverse threshold logic for compression ratio
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
                # Fill with default predictions if error occurs
                for j, label in enumerate(labels):
                    prediction = 0  # Default to pruned
                    all_predictions.append(prediction)
                    all_labels.append(label)
                    
                    if j == 0:
                        pos_predictions.append(prediction)
                        pos_labels.append(label)
                    else:
                        neg_predictions.append(prediction)
                        neg_labels.append(label)
        
        # Calculate metrics
        def safe_f2_score(y_true, y_pred):
            """Calculate F2 score safely, handling edge cases."""
            if len(set(y_true)) == 1:
                # All labels are the same
                if y_true[0] == 1:
                    # All positive labels
                    return 1.0 if all(p == 1 for p in y_pred) else 0.0
                else:
                    # All negative labels
                    return 1.0 if all(p == 0 for p in y_pred) else 0.0
            
            precision, recall, f2, _ = precision_recall_fscore_support(
                y_true, y_pred, average='binary', beta=2.0, zero_division=0
            )
            return f2[0] if isinstance(f2, np.ndarray) else f2
        
        # Calculate F2 scores
        pos_f2 = safe_f2_score(pos_labels, pos_predictions)
        neg_f2 = safe_f2_score(neg_labels, neg_predictions)
        all_f2 = safe_f2_score(all_labels, all_predictions)
        
        # Additional metrics
        pos_precision, pos_recall, _, _ = precision_recall_fscore_support(
            pos_labels, pos_predictions, average='binary', zero_division=0
        )
        neg_precision, neg_recall, _, _ = precision_recall_fscore_support(
            neg_labels, neg_predictions, average='binary', zero_division=0
        )
        
        results[threshold] = {
            'pos_f2': float(pos_f2),
            'neg_f2': float(neg_f2),
            'all_f2': float(all_f2),
            'pos_precision': float(pos_precision[0] if isinstance(pos_precision, np.ndarray) else pos_precision),
            'pos_recall': float(pos_recall[0] if isinstance(pos_recall, np.ndarray) else pos_recall),
            'neg_precision': float(neg_precision[0] if isinstance(neg_precision, np.ndarray) else neg_precision),
            'neg_recall': float(neg_recall[0] if isinstance(neg_recall, np.ndarray) else neg_recall),
            'pos_samples': len(pos_labels),
            'neg_samples': len(neg_labels),
            'pos_kept_ratio': np.mean(pos_predictions),
            'neg_kept_ratio': np.mean(neg_predictions),
        }
        
        logger.info(f"Threshold {threshold}: POS F2={pos_f2:.4f}, NEG F2={neg_f2:.4f}, ALL F2={all_f2:.4f}")
    
    return results


def create_analysis_plots(results_dict: Dict, save_path: str):
    """Create analysis plots for F2 scores."""
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))
    
    # Extract data for plotting
    models = list(results_dict.keys())
    thresholds = list(next(iter(results_dict.values())).keys())
    
    colors = ['blue', 'orange', 'green', 'red']
    
    # Plot 1: F2 Scores by threshold
    for i, model in enumerate(models):
        results = results_dict[model]
        pos_f2s = [results[t]['pos_f2'] for t in thresholds]
        neg_f2s = [results[t]['neg_f2'] for t in thresholds]
        all_f2s = [results[t]['all_f2'] for t in thresholds]
        
        ax1.plot(thresholds, pos_f2s, f'{colors[i]}o-', label=f'{model} (POS)', alpha=0.7)
        ax1.plot(thresholds, neg_f2s, f'{colors[i]}s--', label=f'{model} (NEG)', alpha=0.7)
        ax1.plot(thresholds, all_f2s, f'{colors[i]}^:', label=f'{model} (ALL)', alpha=0.7)
    
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
    ax2.set_xticklabels([m.replace('_', '\n') for m in model_names], rotation=45)
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # Add score labels on bars
    for i, (pos, neg, all_) in enumerate(zip(pos_scores, neg_scores, all_scores)):
        ax2.text(i - width, pos + 0.01, f'{pos:.3f}', ha='center', va='bottom', fontsize=8)
        ax2.text(i, neg + 0.01, f'{neg:.3f}', ha='center', va='bottom', fontsize=8)
        ax2.text(i + width, all_ + 0.01, f'{all_:.3f}', ha='center', va='bottom', fontsize=8)
    
    # Plot 3: Precision vs Recall
    for i, model in enumerate(models):
        results = results_dict[model]
        best_threshold = max(results.keys(), key=lambda t: results[t]['all_f2'])
        best_result = results[best_threshold]
        
        ax3.scatter(best_result['pos_recall'], best_result['pos_precision'], 
                   s=100, color=colors[i], marker='o', label=f'{model} (POS)', alpha=0.7)
        ax3.scatter(best_result['neg_recall'], best_result['neg_precision'], 
                   s=100, color=colors[i], marker='s', label=f'{model} (NEG)', alpha=0.7)
    
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
        
        ax4.plot(thresholds, pos_keep_ratios, f'{colors[i]}o-', label=f'{model} (POS)', alpha=0.7)
        ax4.plot(thresholds, neg_keep_ratios, f'{colors[i]}s--', label=f'{model} (NEG)', alpha=0.7)
    
    ax4.set_xlabel('Pruning Threshold')
    ax4.set_ylabel('Keep Ratio (1 - Pruning Rate)')
    ax4.set_title('Content Keep Ratios vs Threshold')
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    logger.info(f"Analysis plots saved to {save_path}")


def main():
    """Run F2 analysis on MS MARCO models."""
    logger.info("="*60)
    logger.info("MS MARCO F2 Score Analysis")
    logger.info("="*60)
    
    # Load test data
    test_dataset = load_test_data("msmarco-small-ja")
    
    # Model paths (adjust based on actual output directories)
    models = {
        "reranking_pruning": "./output/msmarco_small_reranking_pruning_20250709_181034/final_model",
        "pruning_only": "./output/msmarco_small_pruning_only_20250709_181034/final_model",
    }
    
    # Find actual model directories
    import glob
    for model_type in models:
        pattern = f"./output/msmarco_small_{model_type}_*/final_model"
        matches = glob.glob(pattern)
        if matches:
            models[model_type] = matches[-1]  # Use the latest
            logger.info(f"Found {model_type} model: {models[model_type]}")
        else:
            logger.warning(f"No model found for {model_type}")
    
    # Filter existing models
    existing_models = {k: v for k, v in models.items() if Path(v).exists()}
    
    if not existing_models:
        logger.error("No models found!")
        return
    
    # Thresholds to test
    thresholds = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
    
    # Evaluate all models
    results_dict = {}
    for model_name, model_path in existing_models.items():
        results_dict[model_name] = evaluate_model_f2(
            model_path, test_dataset, model_name, thresholds
        )
    
    # Create analysis plots
    output_dir = Path("./output")
    output_dir.mkdir(exist_ok=True)
    create_analysis_plots(results_dict, str(output_dir / "msmarco_f2_analysis.png"))
    
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
    logger.info(f"Query: {sample['query']}")
    logger.info(f"Number of texts: {len(sample['texts'])}")
    logger.info(f"Labels: {sample['labels']} (first=pos, rest=neg)")
    logger.info(f"Positive texts: {sum(sample['labels'])}")
    logger.info(f"Negative texts: {len(sample['labels']) - sum(sample['labels'])}")
    
    # Check label distribution
    pos_count = 0
    neg_count = 0
    for sample in test_dataset.select(range(100)):  # Check first 100 samples
        labels = sample['labels']
        pos_count += sum(labels)
        neg_count += len(labels) - sum(labels)
    
    logger.info(f"Label distribution (first 100 samples): {pos_count} positive, {neg_count} negative")
    logger.info(f"Pos/Neg ratio: 1:{neg_count/pos_count:.1f}")


if __name__ == "__main__":
    main()