#!/usr/bin/env python
"""Compare old models vs fixed models performance."""

import os
import sys
import torch
import pandas as pd
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
from datasets import Dataset
from sklearn.metrics import f1_score, precision_recall_fscore_support
import logging

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from sentence_transformers.pruning import PruningEncoder
from huggingface_hub import hf_hub_download

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def load_test_dataset():
    """Load test dataset."""
    # Download test file
    test_path = hf_hub_download(
        repo_id="hotchpotch/wip-msmarco-context-relevance",
        filename="msmarco-small-ja/test-00000-of-00001.parquet",
        repo_type="dataset"
    )
    
    test_df = pd.read_parquet(test_path)
    test_dataset = Dataset.from_pandas(test_df)
    
    # Limit to 200 samples for faster evaluation
    test_dataset = test_dataset.select(range(min(200, len(test_dataset))))
    
    logger.info(f"Loaded test set: {len(test_dataset)} samples")
    return test_dataset

def evaluate_model_at_thresholds(model_path, test_dataset, thresholds, mode):
    """Evaluate model at different thresholds."""
    logger.info(f"Evaluating {mode} from {model_path}")
    
    # Load model
    model = PruningEncoder.from_pretrained(
        model_path,
        device="cuda" if torch.cuda.is_available() else "cpu"
    )
    
    results = {}
    
    for threshold in thresholds:
        logger.info(f"Evaluating at threshold {threshold}")
        
        all_predictions = []
        all_labels = []
        compression_ratios = []
        
        for idx, sample in enumerate(test_dataset):
            if idx % 50 == 0:
                logger.info(f"Processing sample {idx}/{len(test_dataset)}")
            
            query = sample['query']
            texts = sample['texts']
            labels = sample['labels']
            
            # Create query-document pairs
            pairs = [(query, text) for text in texts]
            
            try:
                # Get predictions with pruning
                outputs = model.predict_with_pruning(
                    pairs,
                    pruning_threshold=threshold,
                    return_documents=True
                )
                
                # Extract compression ratios and predictions
                for output, label in zip(outputs, labels):
                    # Binary prediction based on compression ratio
                    # If compression_ratio < (1 - threshold), predict 1 (keep), else 0 (prune)
                    prediction = 1 if output.compression_ratio < (1.0 - threshold) else 0
                    
                    all_predictions.append(prediction)
                    all_labels.append(label)
                    compression_ratios.append(output.compression_ratio)
                    
            except Exception as e:
                logger.warning(f"Error processing sample {idx}: {e}")
                continue
        
        if not all_predictions:
            logger.warning(f"No valid predictions for threshold {threshold}")
            continue
            
        # Calculate metrics
        all_predictions = np.array(all_predictions)
        all_labels = np.array(all_labels)
        compression_ratios = np.array(compression_ratios)
        
        # Separate positive and negative samples
        pos_mask = all_labels == 1
        neg_mask = all_labels == 0
        
        # Calculate F2 scores
        beta = 2
        pos_p, pos_r, pos_f1, _ = precision_recall_fscore_support(all_labels[pos_mask], all_predictions[pos_mask], average='binary', zero_division=0)
        neg_p, neg_r, neg_f1, _ = precision_recall_fscore_support(all_labels[neg_mask], all_predictions[neg_mask], average='binary', zero_division=0)
        all_p, all_r, all_f1, _ = precision_recall_fscore_support(all_labels, all_predictions, average='binary', zero_division=0)
        
        # Calculate F2 manually
        pos_f2 = (1 + beta**2) * (pos_p * pos_r) / ((beta**2 * pos_p) + pos_r) if (pos_p + pos_r) > 0 else 0
        neg_f2 = (1 + beta**2) * (neg_p * neg_r) / ((beta**2 * neg_p) + neg_r) if (neg_p + neg_r) > 0 else 0
        all_f2 = (1 + beta**2) * (all_p * all_r) / ((beta**2 * all_p) + all_r) if (all_p + all_r) > 0 else 0
        
        # Calculate average compression ratios
        avg_compression = compression_ratios.mean()
        avg_deletion = 1.0 - avg_compression
        
        # Calculate keep ratios by class
        pos_keep_ratio = compression_ratios[pos_mask].mean() if pos_mask.any() else 0
        neg_keep_ratio = compression_ratios[neg_mask].mean() if neg_mask.any() else 0
        
        results[threshold] = {
            'pos_f2': pos_f2,
            'neg_f2': neg_f2,
            'all_f2': all_f2,
            'precision': all_p,
            'recall': all_r,
            'avg_compression': avg_compression,
            'avg_deletion': avg_deletion,
            'pos_keep_ratio': pos_keep_ratio,
            'neg_keep_ratio': neg_keep_ratio,
            'num_samples': len(all_predictions)
        }
        
        logger.info(f"Threshold {threshold}: F2={all_f2:.4f}, Deletion={avg_deletion:.2%}, POS_keep={pos_keep_ratio:.2%}, NEG_keep={neg_keep_ratio:.2%}")
    
    return results

def main():
    logger.info("=" * 80)
    logger.info("COMPARISON: Old Models vs Fixed Models")
    logger.info("=" * 80)
    
    # Load test dataset
    test_dataset = load_test_dataset()
    
    # Find models
    model_paths = {
        'reranking_pruning_fixed': None,
        'pruning_only_fixed': None,
        'reranking_pruning_old': None,
        'pruning_only_old': None
    }
    
    # Find fixed models
    for path in Path('./output').glob('*_fixed_*/final_model'):
        if 'reranking_pruning' in str(path):
            model_paths['reranking_pruning_fixed'] = str(path)
        elif 'pruning_only' in str(path):
            model_paths['pruning_only_fixed'] = str(path)
    
    # Find old models (most recent small dataset models without fixed)
    old_candidates = []
    for path in Path('./output').glob('msmarco*small*_*/final_model'):
        if '_fixed_' not in str(path):
            old_candidates.append(path)
    
    # Sort by modification time to get most recent
    old_candidates.sort(key=lambda x: x.stat().st_mtime, reverse=True)
    
    for path in old_candidates:
        if 'reranking_pruning' in str(path) and not model_paths['reranking_pruning_old']:
            model_paths['reranking_pruning_old'] = str(path)
        elif 'pruning_only' in str(path) and not model_paths['pruning_only_old']:
            model_paths['pruning_only_old'] = str(path)
    
    logger.info("Found models:")
    for k, v in model_paths.items():
        if v:
            logger.info(f"  {k}: {Path(v).parent.name}")
    
    # Evaluate models
    thresholds = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
    all_results = {}
    
    for model_type, model_path in model_paths.items():
        if model_path and os.path.exists(model_path):
            mode = 'reranking_pruning' if 'reranking' in model_type else 'pruning_only'
            all_results[model_type] = evaluate_model_at_thresholds(
                model_path, test_dataset, thresholds, mode
            )
    
    # Create comparison plots
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    
    # Plot 1: F2 Scores vs Threshold
    ax = axes[0, 0]
    for model_type, results in all_results.items():
        if results:
            f2_scores = [results[t]['all_f2'] for t in thresholds if t in results]
            thresholds_available = [t for t in thresholds if t in results]
            style = '-' if 'fixed' in model_type else '--'
            color = 'blue' if 'reranking' in model_type else 'orange'
            label = model_type.replace('_', ' ').title()
            ax.plot(thresholds_available, f2_scores, marker='o', linestyle=style, color=color, label=label)
    ax.set_xlabel('Pruning Threshold')
    ax.set_ylabel('F2 Score')
    ax.set_title('F2 Scores Comparison: Old vs Fixed Models')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Plot 2: Deletion Rate vs F2 Score
    ax = axes[0, 1]
    for model_type, results in all_results.items():
        if results:
            f2_scores = [results[t]['all_f2'] for t in thresholds if t in results]
            deletion_rates = [results[t]['avg_deletion'] * 100 for t in thresholds if t in results]
            style = '-' if 'fixed' in model_type else '--'
            color = 'blue' if 'reranking' in model_type else 'orange'
            label = model_type.replace('_', ' ').title()
            ax.plot(deletion_rates, f2_scores, marker='o', linestyle=style, color=color, label=label)
    ax.set_xlabel('Deletion Rate (%)')
    ax.set_ylabel('F2 Score')
    ax.set_title('F2 vs Deletion Rate Trade-off')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Plot 3: POS vs NEG Keep Ratios
    ax = axes[1, 0]
    for model_type, results in all_results.items():
        if results:
            pos_keep = [results[t]['pos_keep_ratio'] for t in thresholds if t in results]
            neg_keep = [results[t]['neg_keep_ratio'] for t in thresholds if t in results]
            thresholds_available = [t for t in thresholds if t in results]
            style = '-' if 'fixed' in model_type else '--'
            color = 'blue' if 'reranking' in model_type else 'orange'
            label_base = model_type.replace('_', ' ').title()
            ax.plot(thresholds_available, pos_keep, marker='o', linestyle=style, color=color, label=f'{label_base} (POS)')
            ax.plot(thresholds_available, neg_keep, marker='s', linestyle=style, color=color, alpha=0.5, label=f'{label_base} (NEG)')
    ax.set_xlabel('Pruning Threshold')
    ax.set_ylabel('Keep Ratio')
    ax.set_title('POS vs NEG Content Keep Ratios')
    ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    ax.grid(True, alpha=0.3)
    
    # Plot 4: Best Performance Comparison
    ax = axes[1, 1]
    model_groups = {'Old Models': [], 'Fixed Models': []}
    best_f2s = {'Old Models': [], 'Fixed Models': []}
    best_deletions = {'Old Models': [], 'Fixed Models': []}
    best_thresholds = {'Old Models': [], 'Fixed Models': []}
    
    for model_type, results in all_results.items():
        if results:
            # Find best F2 score
            best_t = max(results.keys(), key=lambda t: results[t]['all_f2'])
            best_f2 = results[best_t]['all_f2']
            best_del = results[best_t]['avg_deletion'] * 100
            
            if 'fixed' in model_type:
                model_groups['Fixed Models'].append(model_type)
                best_f2s['Fixed Models'].append(best_f2)
                best_deletions['Fixed Models'].append(best_del)
                best_thresholds['Fixed Models'].append(best_t)
            else:
                model_groups['Old Models'].append(model_type)
                best_f2s['Old Models'].append(best_f2)
                best_deletions['Old Models'].append(best_del)
                best_thresholds['Old Models'].append(best_t)
    
    # Create grouped bar chart
    x = np.arange(2)  # Two groups
    width = 0.15
    
    colors = ['blue', 'orange']
    for i, (group, models) in enumerate(model_groups.items()):
        for j, model in enumerate(models):
            f2 = best_f2s[group][j]
            deletion = best_deletions[group][j]
            threshold = best_thresholds[group][j]
            
            offset = (j - 0.5) * width
            ax.bar(i + offset, f2, width, label=model.replace('_', ' '), color=colors[j], alpha=0.8)
            ax.text(i + offset, f2 + 0.01, f't={threshold:.1f}\n{deletion:.0f}%', ha='center', fontsize=8)
    
    ax.set_xlabel('Model Groups')
    ax.set_ylabel('Best F2 Score')
    ax.set_title('Best Performance Comparison')
    ax.set_xticks(x)
    ax.set_xticklabels(['Old Models', 'Fixed Models'])
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('output/old_vs_fixed_comparison.png', dpi=300, bbox_inches='tight')
    logger.info("Comparison plots saved to output/old_vs_fixed_comparison.png")
    
    # Print detailed comparison
    logger.info("=" * 80)
    logger.info("DETAILED COMPARISON RESULTS")
    logger.info("=" * 80)
    
    # Compare at specific thresholds
    comparison_thresholds = [0.1, 0.2, 0.3, 0.4, 0.5]
    
    logger.info("\nF2 Score Comparison at Different Thresholds:")
    logger.info("-" * 80)
    
    for threshold in comparison_thresholds:
        logger.info(f"\nThreshold {threshold}:")
        logger.info(f"{'Model':<40} {'F2 Score':<10} {'Deletion':<10} {'POS Keep':<10} {'NEG Keep':<10}")
        logger.info("-" * 80)
        
        for model_type in ['pruning_only_old', 'pruning_only_fixed', 'reranking_pruning_old', 'reranking_pruning_fixed']:
            if model_type in all_results and threshold in all_results[model_type]:
                res = all_results[model_type][threshold]
                logger.info(f"{model_type:<40} {res['all_f2']:<10.4f} {res['avg_deletion']*100:<10.1f}% {res['pos_keep_ratio']*100:<10.1f}% {res['neg_keep_ratio']*100:<10.1f}%")
    
    # Compare best performance
    logger.info("\n" + "=" * 80)
    logger.info("BEST PERFORMANCE COMPARISON:")
    logger.info("=" * 80)
    
    for model_type, results in all_results.items():
        if results:
            best_t = max(results.keys(), key=lambda t: results[t]['all_f2'])
            best_res = results[best_t]
            logger.info(f"\n{model_type.upper()}:")
            logger.info(f"  Best threshold: {best_t}")
            logger.info(f"  F2 Score: {best_res['all_f2']:.4f}")
            logger.info(f"  Deletion Rate: {best_res['avg_deletion']:.2%}")
            logger.info(f"  POS Keep Ratio: {best_res['pos_keep_ratio']:.2%}")
            logger.info(f"  NEG Keep Ratio: {best_res['neg_keep_ratio']:.2%}")

if __name__ == "__main__":
    main()