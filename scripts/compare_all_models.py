#!/usr/bin/env python
"""
Compare all models: pruning-only vs reranking+pruning
Evaluate F2 scores for POS/NEG samples separately.
"""

import os
import sys
from pathlib import Path
import logging
import torch
import numpy as np
from datasets import load_dataset
from datetime import datetime
from sklearn.metrics import f1_score, precision_score, recall_score, fbeta_score
from collections import defaultdict
from tqdm import tqdm
import json

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from sentence_transformers.pruning import PruningEncoder

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def evaluate_chunk_predictions(predictions, ground_truth, beta=2):
    """Evaluate chunk-level predictions using F-beta score."""
    # Flatten all predictions and ground truth
    all_preds = []
    all_truths = []
    
    for pred, truth in zip(predictions, ground_truth):
        all_preds.extend(pred)
        all_truths.extend(truth)
    
    # Calculate metrics
    f2 = fbeta_score(all_truths, all_preds, beta=2, average='binary')
    f1 = f1_score(all_truths, all_preds, average='binary')
    precision = precision_score(all_truths, all_preds, average='binary', zero_division=0)
    recall = recall_score(all_truths, all_preds, average='binary', zero_division=0)
    
    # Calculate TP, TN, FP, FN
    all_preds = np.array(all_preds)
    all_truths = np.array(all_truths)
    
    tp = np.sum((all_preds == 1) & (all_truths == 1))
    tn = np.sum((all_preds == 0) & (all_truths == 0))
    fp = np.sum((all_preds == 1) & (all_truths == 0))
    fn = np.sum((all_preds == 0) & (all_truths == 1))
    
    return {
        'f2': f2,
        'f1': f1,
        'precision': precision,
        'recall': recall,
        'tp': tp,
        'tn': tn,
        'fp': fp,
        'fn': fn,
        'total_positive': tp + fn,
        'total_negative': tn + fp,
        'accuracy': (tp + tn) / (tp + tn + fp + fn) if (tp + tn + fp + fn) > 0 else 0
    }


def map_tokens_to_chunks(output, chunk_positions, text, tokenizer):
    """Map token-level predictions to chunk-level predictions."""
    # Get token mask from output
    if hasattr(output, 'pruning_masks') and output.pruning_masks is not None:
        token_mask = output.pruning_masks[0]  # First element
    else:
        # No mask available, assume all kept
        return [1] * len(chunk_positions)
    
    # Initialize chunk predictions
    chunk_predictions = []
    
    # For each chunk, check if any tokens are kept
    for chunk_start, chunk_end in chunk_positions:
        # Simple heuristic: if any tokens in chunk range are kept, keep the chunk
        chunk_kept = 0
        
        # Check token mask
        if len(token_mask) > 0:
            # Estimate chunk position in token sequence
            chunk_ratio_start = chunk_start / len(text)
            chunk_ratio_end = chunk_end / len(text)
            
            token_start = int(chunk_ratio_start * len(token_mask))
            token_end = int(chunk_ratio_end * len(token_mask))
            
            # Check if any tokens in range are kept
            if token_end > token_start:
                chunk_tokens = token_mask[token_start:token_end]
                if np.any(chunk_tokens):
                    chunk_kept = 1
        
        chunk_predictions.append(chunk_kept)
    
    return chunk_predictions


def predict_and_evaluate(model, dataset, batch_size=32, threshold=0.5, model_name=""):
    """Run predictions and evaluate on dataset."""
    pos_predictions = []
    pos_ground_truth = []
    neg_predictions = []
    neg_ground_truth = []
    
    logger.info(f"Processing {len(dataset)} samples for {model_name}...")
    
    for idx in tqdm(range(0, len(dataset), batch_size), desc=f"Evaluating {model_name}"):
        batch_end = min(idx + batch_size, len(dataset))
        batch = dataset[idx:batch_end]
        
        # Process each sample in batch
        for sample_idx in range(len(batch['query'])):
            query = batch['query'][sample_idx]
            texts = batch['texts'][sample_idx]
            labels = batch['labels'][sample_idx]
            chunks_pos = batch['chunks_pos'][sample_idx]
            relevant_chunks = batch['relevant_chunks'][sample_idx]
            
            # Create query-text pairs
            pairs = [(query, text) for text in texts]
            
            # Run prediction
            outputs = model.predict_with_pruning(
                pairs,
                batch_size=len(pairs),
                pruning_threshold=threshold,
                return_documents=False
            )
            
            # Process each text
            for i, (output, label, chunk_pos, rel_chunks) in enumerate(
                zip(outputs, labels, chunks_pos, relevant_chunks)
            ):
                # Create ground truth labels for chunks
                num_chunks = len(chunk_pos)
                ground_truth_chunks = [1 if j in rel_chunks else 0 for j in range(num_chunks)]
                
                # Get predicted chunks from token-level predictions
                predicted_chunks = map_tokens_to_chunks(
                    output, chunk_pos, texts[i], model.tokenizer
                )
                
                # Ensure same length
                if len(predicted_chunks) != len(ground_truth_chunks):
                    min_len = min(len(predicted_chunks), len(ground_truth_chunks))
                    predicted_chunks = predicted_chunks[:min_len]
                    ground_truth_chunks = ground_truth_chunks[:min_len]
                
                # Separate POS/NEG samples
                if label == 1:  # Positive sample
                    pos_predictions.append(predicted_chunks)
                    pos_ground_truth.append(ground_truth_chunks)
                else:  # Negative sample
                    neg_predictions.append(predicted_chunks)
                    neg_ground_truth.append(ground_truth_chunks)
    
    # Calculate metrics
    logger.info("Calculating metrics...")
    
    pos_metrics = evaluate_chunk_predictions(pos_predictions, pos_ground_truth) if pos_predictions else None
    neg_metrics = evaluate_chunk_predictions(neg_predictions, neg_ground_truth) if neg_predictions else None
    
    # Combined metrics
    all_predictions = pos_predictions + neg_predictions
    all_ground_truth = pos_ground_truth + neg_ground_truth
    all_metrics = evaluate_chunk_predictions(all_predictions, all_ground_truth)
    
    return pos_metrics, neg_metrics, all_metrics


def print_metrics(metrics, label):
    """Print metrics in a formatted way."""
    logger.info(f"\n{label} Metrics:")
    logger.info(f"  F2 Score: {metrics['f2']:.4f}")
    logger.info(f"  F1 Score: {metrics['f1']:.4f}")
    logger.info(f"  Precision: {metrics['precision']:.4f}")
    logger.info(f"  Recall: {metrics['recall']:.4f}")
    logger.info(f"  Accuracy: {metrics['accuracy']:.4f}")
    logger.info(f"  TP: {metrics['tp']}, TN: {metrics['tn']}, FP: {metrics['fp']}, FN: {metrics['fn']}")
    logger.info(f"  Total Positive: {metrics['total_positive']}, Total Negative: {metrics['total_negative']}")


def main():
    # Model configurations
    models = {
        "pruning_only": {
            "minimal": "./output/pruning_only_minimal_20250709_081603/final_model",
            "small": "./output/pruning_only_small_20250709_084354/final_model",
            "full": "./output/pruning_only_full_20250709_091455/final_model"
        },
        "reranking_pruning": {
            "minimal": "./output/reranking_pruning_minimal_20250709_103823/final_model",
            "small": "./output/reranking_pruning_small_20250709_104353/final_model",
            "full": "./output/reranking_pruning_full_20250709_112127/final_model"
        }
    }
    
    # Load test dataset (use minimal for quick testing)
    logger.info("Loading test dataset...")
    dataset = load_dataset(
        'hotchpotch/wip-query-context-pruner-with-teacher-scores',
        'ja-minimal'
    )
    test_dataset = dataset['validation']
    
    logger.info(f"Test dataset size: {len(test_dataset)}")
    
    # Test thresholds
    thresholds = [0.3, 0.5, 0.7]
    
    # Store all results
    all_results = {}
    
    # Evaluate all models
    for mode_name, mode_models in models.items():
        logger.info(f"\n{'='*60}")
        logger.info(f"Evaluating {mode_name.replace('_', ' ').title()} Models")
        logger.info(f"{'='*60}")
        
        all_results[mode_name] = {}
        
        for dataset_size, model_path in mode_models.items():
            if not os.path.exists(model_path):
                logger.warning(f"Model not found: {model_path}")
                continue
            
            logger.info(f"\n--- {dataset_size.upper()} Model ---")
            
            # Load model
            model = PruningEncoder.from_pretrained(
                model_path,
                device="cuda" if torch.cuda.is_available() else "cpu"
            )
            
            all_results[mode_name][dataset_size] = {}
            
            # Test different thresholds
            for threshold in thresholds:
                logger.info(f"\nEvaluating with threshold={threshold}")
                
                pos_metrics, neg_metrics, all_metrics = predict_and_evaluate(
                    model, test_dataset, batch_size=16, threshold=threshold,
                    model_name=f"{mode_name}_{dataset_size}"
                )
                
                all_results[mode_name][dataset_size][threshold] = {
                    'pos': pos_metrics,
                    'neg': neg_metrics,
                    'all': all_metrics
                }
                
                if pos_metrics:
                    print_metrics(pos_metrics, f"POS (threshold={threshold})")
                if neg_metrics:
                    print_metrics(neg_metrics, f"NEG (threshold={threshold})")
                print_metrics(all_metrics, f"ALL (threshold={threshold})")
    
    # Summary comparison
    logger.info(f"\n{'='*60}")
    logger.info("SUMMARY: Best F2 Scores Comparison")
    logger.info(f"{'='*60}")
    
    # Find best scores for each configuration
    summary = {}
    for mode_name in all_results:
        summary[mode_name] = {}
        for dataset_size in all_results[mode_name]:
            best_pos_f2 = 0
            best_neg_f2 = 0
            best_all_f2 = 0
            best_pos_config = None
            best_neg_config = None
            best_all_config = None
            
            for threshold in all_results[mode_name][dataset_size]:
                metrics = all_results[mode_name][dataset_size][threshold]
                if metrics['pos'] and metrics['pos']['f2'] > best_pos_f2:
                    best_pos_f2 = metrics['pos']['f2']
                    best_pos_config = threshold
                if metrics['neg'] and metrics['neg']['f2'] > best_neg_f2:
                    best_neg_f2 = metrics['neg']['f2']
                    best_neg_config = threshold
                if metrics['all']['f2'] > best_all_f2:
                    best_all_f2 = metrics['all']['f2']
                    best_all_config = threshold
            
            summary[mode_name][dataset_size] = {
                'best_pos': (best_pos_f2, best_pos_config),
                'best_neg': (best_neg_f2, best_neg_config),
                'best_all': (best_all_f2, best_all_config)
            }
    
    # Print summary table
    logger.info("\nBest F2 Scores by Model Type and Dataset Size:")
    logger.info("\n| Mode | Dataset | POS F2 | NEG F2 | ALL F2 |")
    logger.info("|------|---------|--------|--------|--------|")
    
    for mode_name in ['pruning_only', 'reranking_pruning']:
        for dataset_size in ['minimal', 'small', 'full']:
            if dataset_size in summary.get(mode_name, {}):
                s = summary[mode_name][dataset_size]
                logger.info(f"| {mode_name.replace('_', ' ')} | {dataset_size} | "
                          f"{s['best_pos'][0]:.4f} (@{s['best_pos'][1]}) | "
                          f"{s['best_neg'][0]:.4f} (@{s['best_neg'][1]}) | "
                          f"{s['best_all'][0]:.4f} (@{s['best_all'][1]}) |")
    
    # Direct comparison at threshold=0.5
    logger.info(f"\n{'='*60}")
    logger.info("Direct Comparison at threshold=0.5")
    logger.info(f"{'='*60}")
    
    threshold = 0.5
    for dataset_size in ['minimal', 'small', 'full']:
        logger.info(f"\n--- {dataset_size.upper()} Dataset Models ---")
        
        if (dataset_size in all_results.get('pruning_only', {}) and 
            dataset_size in all_results.get('reranking_pruning', {})):
            
            po_metrics = all_results['pruning_only'][dataset_size][threshold]
            rp_metrics = all_results['reranking_pruning'][dataset_size][threshold]
            
            logger.info("\n| Metric | Pruning Only | Reranking+Pruning | Improvement |")
            logger.info("|--------|--------------|-------------------|-------------|")
            
            for metric_type in ['pos', 'neg', 'all']:
                if metric_type in po_metrics and metric_type in rp_metrics:
                    po_f2 = po_metrics[metric_type]['f2'] if po_metrics[metric_type] else 0
                    rp_f2 = rp_metrics[metric_type]['f2'] if rp_metrics[metric_type] else 0
                    improvement = rp_f2 - po_f2
                    improvement_pct = (improvement / po_f2 * 100) if po_f2 > 0 else 0
                    
                    logger.info(f"| {metric_type.upper()} F2 | {po_f2:.4f} | {rp_f2:.4f} | "
                              f"{improvement:+.4f} ({improvement_pct:+.1f}%) |")
    
    # Save detailed results (convert numpy types to native Python types)
    results_file = f"./log/model_comparison_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    
    # Convert numpy types to native Python types
    def convert_to_native(obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, dict):
            return {k: convert_to_native(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [convert_to_native(item) for item in obj]
        return obj
    
    with open(results_file, 'w') as f:
        json.dump(convert_to_native(all_results), f, indent=2)
    logger.info(f"\nDetailed results saved to: {results_file}")


if __name__ == "__main__":
    main()