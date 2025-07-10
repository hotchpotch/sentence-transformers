#!/usr/bin/env python
"""Detailed metrics analysis including precision, recall, F1, F2, and accuracy."""

import os
import sys
import torch
import pandas as pd
import numpy as np
from pathlib import Path
from datasets import Dataset
from sklearn.metrics import (
    precision_recall_fscore_support, 
    accuracy_score, 
    confusion_matrix,
    classification_report
)
import logging
from tabulate import tabulate

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from sentence_transformers.pruning import PruningEncoder
from huggingface_hub import hf_hub_download

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def load_test_dataset():
    """Load test dataset."""
    test_path = hf_hub_download(
        repo_id="hotchpotch/wip-msmarco-context-relevance",
        filename="msmarco-small-ja/test-00000-of-00001.parquet",
        repo_type="dataset"
    )
    
    test_df = pd.read_parquet(test_path)
    test_dataset = Dataset.from_pandas(test_df)
    
    # Use full test set for comprehensive analysis
    logger.info(f"Loaded test set: {len(test_dataset)} samples (FULL TEST SET)")
    return test_dataset

def calculate_detailed_metrics(y_true, y_pred):
    """Calculate comprehensive metrics."""
    # Basic metrics
    precision, recall, f1, support = precision_recall_fscore_support(
        y_true, y_pred, average='binary', zero_division=0
    )
    
    # F2 score (recall weighted)
    beta = 2
    f2 = (1 + beta**2) * (precision * recall) / ((beta**2 * precision) + recall) if (precision + recall) > 0 else 0
    
    # Accuracy
    accuracy = accuracy_score(y_true, y_pred)
    
    # Confusion matrix
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    
    # Per-class metrics
    precision_0, recall_0, f1_0, _ = precision_recall_fscore_support(
        y_true, y_pred, labels=[0], average='binary', pos_label=0, zero_division=0
    )
    precision_1, recall_1, f1_1, _ = precision_recall_fscore_support(
        y_true, y_pred, labels=[1], average='binary', pos_label=1, zero_division=0
    )
    
    # Calculate F2 for each class
    f2_0 = (1 + beta**2) * (precision_0 * recall_0) / ((beta**2 * precision_0) + recall_0) if (precision_0 + recall_0) > 0 else 0
    f2_1 = (1 + beta**2) * (precision_1 * recall_1) / ((beta**2 * precision_1) + recall_1) if (precision_1 + recall_1) > 0 else 0
    
    return {
        # Overall metrics
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'f2': f2,
        
        # Confusion matrix
        'true_positive': tp,
        'true_negative': tn,
        'false_positive': fp,
        'false_negative': fn,
        
        # Per-class metrics
        'class_0': {
            'precision': precision_0,
            'recall': recall_0,
            'f1': f1_0,
            'f2': f2_0,
            'support': (y_true == 0).sum()
        },
        'class_1': {
            'precision': precision_1,
            'recall': recall_1,
            'f1': f1_1,
            'f2': f2_1,
            'support': (y_true == 1).sum()
        }
    }

def evaluate_model_comprehensive(model_path, test_dataset, thresholds, mode):
    """Comprehensive evaluation of model."""
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
            
        # Calculate comprehensive metrics
        all_predictions = np.array(all_predictions)
        all_labels = np.array(all_labels)
        compression_ratios = np.array(compression_ratios)
        
        metrics = calculate_detailed_metrics(all_labels, all_predictions)
        
        # Add compression statistics
        metrics['avg_compression'] = compression_ratios.mean()
        metrics['avg_deletion'] = 1.0 - metrics['avg_compression']
        metrics['total_samples'] = len(all_predictions)
        
        # Compression by class
        pos_mask = all_labels == 1
        neg_mask = all_labels == 0
        metrics['pos_compression'] = compression_ratios[pos_mask].mean() if pos_mask.any() else 0
        metrics['neg_compression'] = compression_ratios[neg_mask].mean() if neg_mask.any() else 0
        
        results[threshold] = metrics
    
    return results

def print_detailed_results(results, model_name):
    """Print results in detailed tabular format."""
    print("\n" + "="*100)
    print(f"DETAILED METRICS FOR: {model_name}")
    print("="*100)
    
    # Find best threshold based on F2
    best_threshold = max(results.keys(), key=lambda t: results[t]['f2'])
    
    # Print overview table
    overview_data = []
    for threshold in sorted(results.keys()):
        m = results[threshold]
        overview_data.append([
            f"{threshold:.1f}",
            f"{m['accuracy']:.4f}",
            f"{m['precision']:.4f}",
            f"{m['recall']:.4f}",
            f"{m['f1']:.4f}",
            f"{m['f2']:.4f}",
            f"{m['avg_deletion']*100:.1f}%",
            "★" if threshold == best_threshold else ""
        ])
    
    print("\nOVERALL PERFORMANCE BY THRESHOLD:")
    print(tabulate(overview_data, 
                   headers=["Threshold", "Accuracy", "Precision", "Recall", "F1", "F2", "Deletion%", "Best"],
                   tablefmt="grid"))
    
    # Detailed analysis for best threshold
    best = results[best_threshold]
    print(f"\nDETAILED ANALYSIS AT BEST THRESHOLD {best_threshold}:")
    print("-"*100)
    
    # Confusion matrix
    print("\nCONFUSION MATRIX:")
    cm_data = [
        ["Predicted Negative", best['true_negative'], best['false_positive'], best['true_negative'] + best['false_positive']],
        ["Predicted Positive", best['false_negative'], best['true_positive'], best['false_negative'] + best['true_positive']],
        ["Total", best['true_negative'] + best['false_negative'], best['false_positive'] + best['true_positive'], best['total_samples']]
    ]
    print(tabulate(cm_data, 
                   headers=["", "Actual Negative", "Actual Positive", "Total"],
                   tablefmt="grid"))
    
    # Per-class metrics
    print("\nPER-CLASS METRICS:")
    class_data = [
        ["Class 0 (Negative)", 
         f"{best['class_0']['support']}", 
         f"{best['class_0']['support']/best['total_samples']*100:.1f}%",
         f"{best['class_0']['precision']:.4f}",
         f"{best['class_0']['recall']:.4f}",
         f"{best['class_0']['f1']:.4f}",
         f"{best['class_0']['f2']:.4f}",
         f"{best['neg_compression']*100:.1f}%"],
        ["Class 1 (Positive)", 
         f"{best['class_1']['support']}", 
         f"{best['class_1']['support']/best['total_samples']*100:.1f}%",
         f"{best['class_1']['precision']:.4f}",
         f"{best['class_1']['recall']:.4f}",
         f"{best['class_1']['f1']:.4f}",
         f"{best['class_1']['f2']:.4f}",
         f"{best['pos_compression']*100:.1f}%"]
    ]
    print(tabulate(class_data,
                   headers=["Class", "Count", "Ratio", "Precision", "Recall", "F1", "F2", "Keep%"],
                   tablefmt="grid"))
    
    # Accuracy breakdown
    print("\nACCURACY BREAKDOWN:")
    pos_correct = best['true_positive']
    pos_total = best['class_1']['support']
    neg_correct = best['true_negative'] 
    neg_total = best['class_0']['support']
    
    acc_data = [
        ["Positive samples", f"{pos_correct}/{pos_total}", f"{pos_correct/pos_total*100:.1f}%"],
        ["Negative samples", f"{neg_correct}/{neg_total}", f"{neg_correct/neg_total*100:.1f}%"],
        ["Overall", f"{pos_correct+neg_correct}/{best['total_samples']}", f"{best['accuracy']*100:.1f}%"]
    ]
    print(tabulate(acc_data,
                   headers=["Category", "Correct/Total", "Accuracy"],
                   tablefmt="grid"))
    
    # Key performance indicators at different thresholds
    print("\nKEY THRESHOLDS COMPARISON:")
    key_thresholds = [0.3, 0.5, 0.7]
    key_data = []
    for t in key_thresholds:
        if t in results:
            m = results[t]
            key_data.append([
                f"{t:.1f}",
                f"{m['f2']:.4f}",
                f"{m['avg_deletion']*100:.1f}%",
                f"{m['class_1']['recall']:.4f}",
                f"{m['class_0']['recall']:.4f}",
                f"{m['accuracy']:.4f}"
            ])
    
    print(tabulate(key_data,
                   headers=["Threshold", "F2", "Deletion%", "Pos Recall", "Neg Recall", "Accuracy"],
                   tablefmt="grid"))

def main():
    logger.info("Starting comprehensive metrics analysis...")
    
    # Load test dataset
    test_dataset = load_test_dataset()
    
    # Find models
    model_paths = {}
    for path in Path('./output').glob('*_fixed_*/final_model'):
        if 'reranking_pruning' in str(path):
            model_paths['reranking_pruning_fixed'] = str(path)
        elif 'pruning_only' in str(path):
            model_paths['pruning_only_fixed'] = str(path)
    
    logger.info(f"Found models: {list(model_paths.keys())}")
    
    # Evaluate models - key thresholds only for full dataset
    thresholds = [0.1, 0.2, 0.3, 0.5, 0.7]  # Key thresholds for comprehensive analysis
    
    for model_type, model_path in model_paths.items():
        if model_path and os.path.exists(model_path):
            mode = 'reranking_pruning' if 'reranking' in model_type else 'pruning_only'
            results = evaluate_model_comprehensive(
                model_path, test_dataset, thresholds, mode
            )
            
            model_name = model_type.replace('_', ' ').upper()
            print_detailed_results(results, model_name)
    
    print("\n" + "="*100)
    print("ANALYSIS COMPLETED")
    print("="*100)

if __name__ == "__main__":
    main()