#!/usr/bin/env python
"""
Evaluate pruning performance on full test set with F2 scores for POS/NEG samples.
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
    """
    Evaluate chunk-level predictions using F-beta score.
    
    Args:
        predictions: List of predicted chunk labels (0/1)
        ground_truth: List of ground truth chunk labels (0/1)
        beta: Beta value for F-beta score (default=2 for F2)
    
    Returns:
        Dictionary with evaluation metrics
    """
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


def predict_and_evaluate(model, dataset, batch_size=32, threshold=0.5):
    """
    Run predictions and evaluate on dataset.
    
    Returns:
        pos_metrics: Metrics for positive samples
        neg_metrics: Metrics for negative samples
        all_metrics: Metrics for all samples
    """
    pos_predictions = []
    pos_ground_truth = []
    neg_predictions = []
    neg_ground_truth = []
    
    logger.info(f"Processing {len(dataset)} samples...")
    
    for idx in tqdm(range(0, len(dataset), batch_size), desc="Evaluating"):
        batch_end = min(idx + batch_size, len(dataset))
        batch = dataset[idx:batch_end]
        
        # Process each sample in batch
        for sample_idx in range(len(batch)):
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
                # We need to map token predictions to chunk predictions
                predicted_chunks = map_tokens_to_chunks(
                    output, chunk_pos, texts[i], model.tokenizer
                )
                
                # Ensure same length
                if len(predicted_chunks) != len(ground_truth_chunks):
                    # Pad or truncate
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


def map_tokens_to_chunks(output, chunk_positions, text, tokenizer):
    """
    Map token-level predictions to chunk-level predictions.
    
    Returns:
        List of 0/1 predictions for each chunk
    """
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
        # This is approximate since we don't have exact token-to-character mapping
        chunk_kept = 0
        
        # Check token mask
        if len(token_mask) > 0:
            # Estimate chunk position in token sequence
            # This is a rough approximation
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
    # Load full test dataset
    logger.info("Loading full test dataset...")
    dataset = load_dataset(
        'hotchpotch/wip-query-context-pruner-with-teacher-scores',
        'ja-minimal'
    )
    test_dataset = dataset['validation']
    
    logger.info(f"Test dataset size: {len(test_dataset)}")
    
    # Model paths
    reranking_model_path = "outputs/pruning-ja-minimal/checkpoint-412-best"
    pruning_only_model_path = "./output/pruning_only_minimal_20250709_081603/checkpoint-1200-best"
    
    # Test different thresholds
    thresholds = [0.3, 0.5, 0.7]
    
    results = {}
    
    # Evaluate reranking_pruning model
    logger.info(f"\n{'='*60}")
    logger.info("Evaluating Reranking+Pruning Model")
    logger.info(f"{'='*60}\n")
    
    model1 = PruningEncoder.from_pretrained(
        reranking_model_path,
        device="cuda" if torch.cuda.is_available() else "cpu"
    )
    
    results['reranking_pruning'] = {}
    
    for threshold in thresholds:
        logger.info(f"\nEvaluating with threshold={threshold}")
        pos_metrics, neg_metrics, all_metrics = predict_and_evaluate(
            model1, test_dataset, batch_size=16, threshold=threshold
        )
        
        results['reranking_pruning'][threshold] = {
            'pos': pos_metrics,
            'neg': neg_metrics,
            'all': all_metrics
        }
        
        if pos_metrics:
            print_metrics(pos_metrics, f"POS (threshold={threshold})")
        if neg_metrics:
            print_metrics(neg_metrics, f"NEG (threshold={threshold})")
        print_metrics(all_metrics, f"ALL (threshold={threshold})")
    
    # Evaluate pruning_only model
    logger.info(f"\n{'='*60}")
    logger.info("Evaluating Pruning-Only Model")
    logger.info(f"{'='*60}\n")
    
    model2 = PruningEncoder.from_pretrained(
        pruning_only_model_path,
        device="cuda" if torch.cuda.is_available() else "cpu"
    )
    
    results['pruning_only'] = {}
    
    for threshold in thresholds:
        logger.info(f"\nEvaluating with threshold={threshold}")
        pos_metrics, neg_metrics, all_metrics = predict_and_evaluate(
            model2, test_dataset, batch_size=16, threshold=threshold
        )
        
        results['pruning_only'][threshold] = {
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
    logger.info("Summary: Best F2 Scores")
    logger.info(f"{'='*60}\n")
    
    for model_name in ['reranking_pruning', 'pruning_only']:
        logger.info(f"\n{model_name.replace('_', ' ').title()}:")
        
        best_pos_f2 = 0
        best_pos_threshold = 0
        best_neg_f2 = 0
        best_neg_threshold = 0
        best_all_f2 = 0
        best_all_threshold = 0
        
        for threshold, metrics in results[model_name].items():
            if metrics['pos'] and metrics['pos']['f2'] > best_pos_f2:
                best_pos_f2 = metrics['pos']['f2']
                best_pos_threshold = threshold
            
            if metrics['neg'] and metrics['neg']['f2'] > best_neg_f2:
                best_neg_f2 = metrics['neg']['f2']
                best_neg_threshold = threshold
            
            if metrics['all']['f2'] > best_all_f2:
                best_all_f2 = metrics['all']['f2']
                best_all_threshold = threshold
        
        logger.info(f"  Best POS F2: {best_pos_f2:.4f} (threshold={best_pos_threshold})")
        logger.info(f"  Best NEG F2: {best_neg_f2:.4f} (threshold={best_neg_threshold})")
        logger.info(f"  Best ALL F2: {best_all_f2:.4f} (threshold={best_all_threshold})")
    
    # Detailed comparison at threshold=0.5
    logger.info(f"\n{'='*60}")
    logger.info("Detailed Comparison at threshold=0.5")
    logger.info(f"{'='*60}\n")
    
    threshold = 0.5
    
    logger.info("Metric | Reranking+Pruning | Pruning-Only | Difference")
    logger.info("-------|-------------------|--------------|------------")
    
    for metric_type in ['pos', 'neg', 'all']:
        if metric_type == 'pos' and (not results['reranking_pruning'][threshold]['pos'] or 
                                      not results['pruning_only'][threshold]['pos']):
            continue
        if metric_type == 'neg' and (not results['reranking_pruning'][threshold]['neg'] or 
                                      not results['pruning_only'][threshold]['neg']):
            continue
            
        m1 = results['reranking_pruning'][threshold][metric_type]
        m2 = results['pruning_only'][threshold][metric_type]
        
        logger.info(f"{metric_type.upper()} F2  | {m1['f2']:.4f}            | {m2['f2']:.4f}       | {m1['f2'] - m2['f2']:+.4f}")
        logger.info(f"{metric_type.upper()} Rec | {m1['recall']:.4f}            | {m2['recall']:.4f}       | {m1['recall'] - m2['recall']:+.4f}")
        logger.info(f"{metric_type.upper()} Pre | {m1['precision']:.4f}            | {m2['precision']:.4f}       | {m1['precision'] - m2['precision']:+.4f}")
        logger.info("-------|-------------------|--------------|------------")


if __name__ == "__main__":
    main()