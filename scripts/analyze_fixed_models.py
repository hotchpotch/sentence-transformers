#!/usr/bin/env python
"""Analyze fixed models with corrected loss weights."""

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

def f2_score(y_true, y_pred, beta=2):
    """Calculate F2 score manually."""
    # Get precision and recall
    precision, recall, _, _ = precision_recall_fscore_support(y_true, y_pred, average='binary', zero_division=0)
    
    # Calculate F2 score
    if precision + recall == 0:
        return 0.0
    
    f2 = (1 + beta**2) * (precision * recall) / ((beta**2 * precision) + recall)
    return f2

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
                    # Binary prediction: keep if compression ratio is high (little pruning)
                    prediction = 1 if output.compression_ratio > 0.5 else 0
                    
                    all_predictions.append(prediction)
                    all_labels.append(label)
                    compression_ratios.append(output.compression_ratio)
                    
            except Exception as e:
                logger.warning(f"Error processing sample {idx}: {e}")
                # Skip this sample
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
        pos_f2 = f2_score(all_labels[pos_mask], all_predictions[pos_mask]) if pos_mask.any() else 0
        neg_f2 = f2_score(all_labels[neg_mask], all_predictions[neg_mask]) if neg_mask.any() else 0
        all_f2 = f2_score(all_labels, all_predictions)
        
        # Calculate precision and recall
        precision, recall, _, _ = precision_recall_fscore_support(all_labels, all_predictions, average='binary', zero_division=0)
        
        # Calculate average compression ratios
        avg_compression = compression_ratios.mean()
        avg_deletion = 1.0 - avg_compression
        
        # Calculate pos/neg accuracy
        pos_accuracy = (all_predictions[pos_mask] == all_labels[pos_mask]).mean() if pos_mask.any() else 0
        neg_accuracy = (all_predictions[neg_mask] == all_labels[neg_mask]).mean() if neg_mask.any() else 0
        
        results[threshold] = {
            'pos_f2': pos_f2,
            'neg_f2': neg_f2,
            'all_f2': all_f2,
            'precision': precision,
            'recall': recall,
            'pos_accuracy': pos_accuracy,
            'neg_accuracy': neg_accuracy,
            'avg_compression': avg_compression,
            'avg_deletion': avg_deletion,
            'compression_ratios': compression_ratios,
            'num_samples': len(all_predictions),
            'num_pos': pos_mask.sum(),
            'num_neg': neg_mask.sum()
        }
        
        logger.info(f"Threshold {threshold}: F2={all_f2:.4f}, Deletion={avg_deletion:.2%}, Pos_Acc={pos_accuracy:.4f}, Neg_Acc={neg_accuracy:.4f}")
    
    return results

def main():
    logger.info("=" * 60)
    logger.info("Fixed Models F2 Score Analysis")
    logger.info("=" * 60)
    
    # Load test dataset
    test_dataset = load_test_dataset()
    
    # Find fixed models
    model_paths = {}
    for path in Path('./output').glob('*_fixed_*/final_model'):
        if 'reranking_pruning' in str(path):
            model_paths['reranking_pruning_fixed'] = str(path)
        elif 'pruning_only' in str(path):
            model_paths['pruning_only_fixed'] = str(path)
    
    logger.info(f"Found models: {model_paths}")
    
    # Evaluate models
    thresholds = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
    all_results = {}
    
    for model_type, model_path in model_paths.items():
        if model_path and os.path.exists(model_path):
            mode = 'reranking_pruning' if 'reranking' in model_type else 'pruning_only'
            all_results[model_type] = evaluate_model_at_thresholds(
                model_path, test_dataset, thresholds, mode
            )
    
    # Create plots
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    
    # Plot 1: F2 Scores vs Threshold
    ax = axes[0, 0]
    for model_type, results in all_results.items():
        if results:
            f2_scores = [results[t]['all_f2'] for t in thresholds if t in results]
            thresholds_available = [t for t in thresholds if t in results]
            label = model_type.replace('_', ' ').title()
            ax.plot(thresholds_available, f2_scores, marker='o', label=label)
    ax.set_xlabel('Pruning Threshold')
    ax.set_ylabel('F2 Score')
    ax.set_title('F2 Scores vs Threshold (Fixed Models)')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Plot 2: Deletion Rate vs Threshold
    ax = axes[0, 1]
    for model_type, results in all_results.items():
        if results:
            deletion_rates = [results[t]['avg_deletion'] * 100 for t in thresholds if t in results]
            thresholds_available = [t for t in thresholds if t in results]
            label = model_type.replace('_', ' ').title()
            ax.plot(thresholds_available, deletion_rates, marker='o', label=label)
    ax.set_xlabel('Pruning Threshold')
    ax.set_ylabel('Deletion Rate (%)')
    ax.set_title('Average Deletion Rate vs Threshold')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Plot 3: Positive vs Negative Accuracy
    ax = axes[1, 0]
    for model_type, results in all_results.items():
        if results:
            pos_accs = [results[t]['pos_accuracy'] for t in thresholds if t in results]
            neg_accs = [results[t]['neg_accuracy'] for t in thresholds if t in results]
            thresholds_available = [t for t in thresholds if t in results]
            label = model_type.replace('_', ' ').title()
            ax.plot(thresholds_available, pos_accs, marker='o', linestyle='-', label=f'{label} (Pos)')
            ax.plot(thresholds_available, neg_accs, marker='s', linestyle='--', label=f'{label} (Neg)')
    ax.set_xlabel('Pruning Threshold')
    ax.set_ylabel('Accuracy')
    ax.set_title('Positive vs Negative Accuracy')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Plot 4: F2 vs Deletion Rate Trade-off
    ax = axes[1, 1]
    for model_type, results in all_results.items():
        if results:
            f2_scores = [results[t]['all_f2'] for t in thresholds if t in results]
            deletion_rates = [results[t]['avg_deletion'] * 100 for t in thresholds if t in results]
            label = model_type.replace('_', ' ').title()
            ax.plot(deletion_rates, f2_scores, marker='o', label=label)
    ax.set_xlabel('Deletion Rate (%)')
    ax.set_ylabel('F2 Score')
    ax.set_title('F2 Score vs Deletion Rate Trade-off')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('output/fixed_models_analysis.png', dpi=300, bbox_inches='tight')
    logger.info("Analysis plots saved to output/fixed_models_analysis.png")
    
    # Print detailed results
    logger.info("=" * 60)
    logger.info("DETAILED RESULTS")
    logger.info("=" * 60)
    
    for model_type, results in all_results.items():
        if results:
            logger.info(f"\n{model_type.upper().replace('_', ' ')}:")
            
            # Find best threshold based on F2 score
            best_t = max(results.keys(), key=lambda t: results[t]['all_f2'])
            best_res = results[best_t]
            
            logger.info(f"  Best threshold: {best_t}")
            logger.info(f"  All F2 Score: {best_res['all_f2']:.4f}")
            logger.info(f"  Pos F2 Score: {best_res['pos_f2']:.4f}")
            logger.info(f"  Neg F2 Score: {best_res['neg_f2']:.4f}")
            logger.info(f"  Deletion Rate: {best_res['avg_deletion']:.2%}")
            logger.info(f"  Precision/Recall: {best_res['precision']:.4f}/{best_res['recall']:.4f}")
            logger.info(f"  Pos Accuracy: {best_res['pos_accuracy']:.4f}")
            logger.info(f"  Neg Accuracy: {best_res['neg_accuracy']:.4f}")
            logger.info(f"  Sample distribution: {best_res['num_pos']} pos, {best_res['num_neg']} neg")
            
            # Show performance at key thresholds
            logger.info(f"\n  Performance at key thresholds:")
            for t in [0.2, 0.3, 0.4, 0.5]:
                if t in results:
                    r = results[t]
                    logger.info(f"    Threshold {t}: F2={r['all_f2']:.4f}, Del={r['avg_deletion']:.2%}, Pos_Acc={r['pos_accuracy']:.4f}, Neg_Acc={r['neg_accuracy']:.4f}")

if __name__ == "__main__":
    main()