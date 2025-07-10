#!/usr/bin/env python
"""F2 score analysis with compression ratio evaluation."""

import os
import sys
import torch
import pandas as pd
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
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
    logger.info(f"Model mode: {mode}")
    
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
            
            # Get predictions with pruning
            outputs = model.predict_with_pruning(
                pairs,
                pruning_threshold=threshold,
                return_documents=True
            )
            
            # Extract compression ratios
            for output, label in zip(outputs, labels):
                # Binary prediction based on compression ratio
                # If compression_ratio < (1 - threshold), predict 1 (keep), else 0 (prune)
                prediction = 1 if output.compression_ratio < (1.0 - threshold) else 0
                
                all_predictions.append(prediction)
                all_labels.append(label)
                compression_ratios.append(output.compression_ratio)
        
        # Calculate metrics
        all_predictions = np.array(all_predictions)
        all_labels = np.array(all_labels)
        compression_ratios = np.array(compression_ratios)
        
        # Separate positive and negative samples
        pos_mask = all_labels == 1
        neg_mask = all_labels == 0
        
        # Calculate F2 scores
        pos_f2 = f1_score(all_labels[pos_mask], all_predictions[pos_mask], beta=2) if pos_mask.any() else 0
        neg_f2 = f1_score(all_labels[neg_mask], all_predictions[neg_mask], beta=2) if neg_mask.any() else 0
        all_f2 = f1_score(all_labels, all_predictions, beta=2)
        
        # Calculate precision and recall
        precision, recall, _, _ = precision_recall_fscore_support(all_labels, all_predictions, average='binary')
        
        # Calculate average compression ratios
        avg_compression = compression_ratios.mean()
        avg_deletion = 1.0 - avg_compression
        
        results[threshold] = {
            'pos_f2': pos_f2,
            'neg_f2': neg_f2,
            'all_f2': all_f2,
            'precision': precision,
            'recall': recall,
            'avg_compression': avg_compression,
            'avg_deletion': avg_deletion,
            'compression_ratios': compression_ratios
        }
        
        logger.info(f"Threshold {threshold}: F2={all_f2:.4f}, Avg deletion={avg_deletion:.2%}")
    
    return results

def main():
    logger.info("=" * 60)
    logger.info("F2 Score Analysis with Compression Evaluation")
    logger.info("=" * 60)
    
    # Load test dataset
    test_dataset = load_test_dataset()
    
    # Find models
    model_dirs = {
        'reranking_pruning_fixed': None,
        'pruning_only_fixed': None,
        'reranking_pruning_old': None,
        'pruning_only_old': None
    }
    
    # Look for fixed models
    for path in Path('./output').glob('*_fixed_*/final_model'):
        if 'reranking_pruning' in str(path):
            model_dirs['reranking_pruning_fixed'] = str(path)
        elif 'pruning_only' in str(path):
            model_dirs['pruning_only_fixed'] = str(path)
    
    # Look for old models (most recent non-fixed)
    for path in sorted(Path('./output').glob('msmarco-small-ja_*/final_model')):
        if '_fixed_' not in str(path):
            if 'reranking_pruning' in str(path):
                if model_dirs['reranking_pruning_old'] is None:
                    model_dirs['reranking_pruning_old'] = str(path)
            elif 'pruning_only' in str(path):
                if model_dirs['pruning_only_old'] is None:
                    model_dirs['pruning_only_old'] = str(path)
    
    logger.info(f"Found models: {model_dirs}")
    
    # Evaluate models
    thresholds = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
    all_results = {}
    
    for model_type, model_path in model_dirs.items():
        if model_path and os.path.exists(model_path):
            mode = 'reranking_pruning' if 'reranking' in model_type else 'pruning_only'
            all_results[model_type] = evaluate_model_at_thresholds(
                model_path, test_dataset, thresholds, mode
            )
    
    # Create comparison plots
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    
    # Plot 1: F2 Scores vs Threshold
    ax = axes[0, 0]
    for model_type, results in all_results.items():
        if results:
            f2_scores = [results[t]['all_f2'] for t in thresholds]
            style = '-' if 'fixed' in model_type else '--'
            label = model_type.replace('_', ' ')
            ax.plot(thresholds, f2_scores, marker='o', linestyle=style, label=label)
    ax.set_xlabel('Pruning Threshold')
    ax.set_ylabel('F2 Score')
    ax.set_title('F2 Scores vs Threshold')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Plot 2: Deletion Rate vs Threshold
    ax = axes[0, 1]
    for model_type, results in all_results.items():
        if results:
            deletion_rates = [results[t]['avg_deletion'] * 100 for t in thresholds]
            style = '-' if 'fixed' in model_type else '--'
            label = model_type.replace('_', ' ')
            ax.plot(thresholds, deletion_rates, marker='o', linestyle=style, label=label)
    ax.set_xlabel('Pruning Threshold')
    ax.set_ylabel('Deletion Rate (%)')
    ax.set_title('Average Deletion Rate vs Threshold')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Plot 3: F2 vs Deletion Rate
    ax = axes[1, 0]
    for model_type, results in all_results.items():
        if results:
            f2_scores = [results[t]['all_f2'] for t in thresholds]
            deletion_rates = [results[t]['avg_deletion'] * 100 for t in thresholds]
            style = '-' if 'fixed' in model_type else '--'
            label = model_type.replace('_', ' ')
            ax.plot(deletion_rates, f2_scores, marker='o', linestyle=style, label=label)
    ax.set_xlabel('Deletion Rate (%)')
    ax.set_ylabel('F2 Score')
    ax.set_title('F2 Score vs Deletion Rate Trade-off')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Plot 4: Best threshold comparison
    ax = axes[1, 1]
    best_results = {}
    for model_type, results in all_results.items():
        if results:
            # Find best F2 score
            best_t = max(results.keys(), key=lambda t: results[t]['all_f2'])
            best_results[model_type] = {
                'threshold': best_t,
                'f2': results[best_t]['all_f2'],
                'deletion': results[best_t]['avg_deletion'] * 100
            }
    
    if best_results:
        models = list(best_results.keys())
        x = np.arange(len(models))
        
        f2_scores = [best_results[m]['f2'] for m in models]
        deletion_rates = [best_results[m]['deletion'] for m in models]
        
        width = 0.35
        ax.bar(x - width/2, f2_scores, width, label='F2 Score', alpha=0.8)
        ax2 = ax.twinx()
        ax2.bar(x + width/2, deletion_rates, width, label='Deletion Rate (%)', alpha=0.8, color='orange')
        
        ax.set_xlabel('Model')
        ax.set_ylabel('F2 Score')
        ax2.set_ylabel('Deletion Rate (%)')
        ax.set_title('Best Performance Comparison')
        ax.set_xticks(x)
        ax.set_xticklabels([m.replace('_', ' ') for m in models], rotation=45, ha='right')
        
        # Add threshold annotations
        for i, m in enumerate(models):
            t = best_results[m]['threshold']
            ax.text(i, f2_scores[i] + 0.01, f't={t}', ha='center', va='bottom')
    
    plt.tight_layout()
    plt.savefig('output/f2_compression_analysis_fixed.png', dpi=300, bbox_inches='tight')
    logger.info("Analysis plots saved to output/f2_compression_analysis_fixed.png")
    
    # Print detailed results
    logger.info("=" * 60)
    logger.info("DETAILED RESULTS")
    logger.info("=" * 60)
    
    for model_type, results in all_results.items():
        if results:
            logger.info(f"\n{model_type.upper()}:")
            # Find best threshold
            best_t = max(results.keys(), key=lambda t: results[t]['all_f2'])
            best_res = results[best_t]
            
            logger.info(f"  Best threshold: {best_t}")
            logger.info(f"  F2 Score: {best_res['all_f2']:.4f}")
            logger.info(f"  Deletion Rate: {best_res['avg_deletion']:.2%}")
            logger.info(f"  Precision/Recall: {best_res['precision']:.4f}/{best_res['recall']:.4f}")
            
            # Show performance at different deletion rates
            logger.info(f"\n  Performance at different deletion rates:")
            for t in [0.2, 0.3, 0.4, 0.5]:
                if t in results:
                    logger.info(f"    Threshold {t}: F2={results[t]['all_f2']:.4f}, Deletion={results[t]['avg_deletion']:.2%}")

if __name__ == "__main__":
    main()
