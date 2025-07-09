#!/usr/bin/env python
"""
Compare models on FULL test dataset for more comprehensive evaluation.
Focus on pruning-only vs reranking+pruning comparison.
"""

import os
import sys
from pathlib import Path
import logging
import torch
import numpy as np
from datasets import load_dataset
from datetime import datetime
from sklearn.metrics import fbeta_score
from tqdm import tqdm

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from sentence_transformers.pruning import PruningEncoder

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def evaluate_chunk_predictions(predictions, ground_truth):
    """Evaluate chunk-level predictions using F2 score."""
    all_preds = []
    all_truths = []
    
    for pred, truth in zip(predictions, ground_truth):
        all_preds.extend(pred)
        all_truths.extend(truth)
    
    # Calculate F2 score
    f2 = fbeta_score(all_truths, all_preds, beta=2, average='binary')
    
    # Calculate counts
    all_preds = np.array(all_preds)
    all_truths = np.array(all_truths)
    
    tp = np.sum((all_preds == 1) & (all_truths == 1))
    tn = np.sum((all_preds == 0) & (all_truths == 0))
    fp = np.sum((all_preds == 1) & (all_truths == 0))
    fn = np.sum((all_preds == 0) & (all_truths == 1))
    
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    
    return {
        'f2': f2,
        'precision': precision,
        'recall': recall,
        'tp': tp,
        'tn': tn,
        'fp': fp,
        'fn': fn
    }


def map_tokens_to_chunks(output, chunk_positions, text):
    """Map token-level predictions to chunk-level predictions."""
    if hasattr(output, 'pruning_masks') and output.pruning_masks is not None:
        token_mask = output.pruning_masks[0]
    else:
        return [1] * len(chunk_positions)
    
    chunk_predictions = []
    
    for chunk_start, chunk_end in chunk_positions:
        chunk_kept = 0
        
        if len(token_mask) > 0:
            chunk_ratio_start = chunk_start / len(text)
            chunk_ratio_end = chunk_end / len(text)
            
            token_start = int(chunk_ratio_start * len(token_mask))
            token_end = int(chunk_ratio_end * len(token_mask))
            
            if token_end > token_start:
                chunk_tokens = token_mask[token_start:token_end]
                if np.any(chunk_tokens):
                    chunk_kept = 1
        
        chunk_predictions.append(chunk_kept)
    
    return chunk_predictions


def evaluate_model(model, dataset, threshold=0.5):
    """Evaluate model on dataset."""
    pos_predictions = []
    pos_ground_truth = []
    neg_predictions = []
    neg_ground_truth = []
    
    batch_size = 16
    
    for idx in tqdm(range(0, len(dataset), batch_size), desc="Evaluating"):
        batch_end = min(idx + batch_size, len(dataset))
        batch = dataset[idx:batch_end]
        
        for sample_idx in range(len(batch['query'])):
            query = batch['query'][sample_idx]
            texts = batch['texts'][sample_idx]
            labels = batch['labels'][sample_idx]
            chunks_pos = batch['chunks_pos'][sample_idx]
            relevant_chunks = batch['relevant_chunks'][sample_idx]
            
            pairs = [(query, text) for text in texts]
            
            outputs = model.predict_with_pruning(
                pairs,
                batch_size=len(pairs),
                pruning_threshold=threshold,
                return_documents=False
            )
            
            for i, (output, label, chunk_pos, rel_chunks) in enumerate(
                zip(outputs, labels, chunks_pos, relevant_chunks)
            ):
                num_chunks = len(chunk_pos)
                ground_truth_chunks = [1 if j in rel_chunks else 0 for j in range(num_chunks)]
                
                predicted_chunks = map_tokens_to_chunks(
                    output, chunk_pos, texts[i]
                )
                
                if len(predicted_chunks) != len(ground_truth_chunks):
                    min_len = min(len(predicted_chunks), len(ground_truth_chunks))
                    predicted_chunks = predicted_chunks[:min_len]
                    ground_truth_chunks = ground_truth_chunks[:min_len]
                
                if label == 1:
                    pos_predictions.append(predicted_chunks)
                    pos_ground_truth.append(ground_truth_chunks)
                else:
                    neg_predictions.append(predicted_chunks)
                    neg_ground_truth.append(ground_truth_chunks)
    
    pos_metrics = evaluate_chunk_predictions(pos_predictions, pos_ground_truth) if pos_predictions else None
    neg_metrics = evaluate_chunk_predictions(neg_predictions, neg_ground_truth) if neg_predictions else None
    
    all_predictions = pos_predictions + neg_predictions
    all_ground_truth = pos_ground_truth + neg_ground_truth
    all_metrics = evaluate_chunk_predictions(all_predictions, all_ground_truth)
    
    return pos_metrics, neg_metrics, all_metrics


def main():
    # Model paths
    models = {
        "Pruning-Only Small": "./output/pruning_only_small_20250709_084354/final_model",
        "Pruning-Only Full": "./output/pruning_only_full_20250709_091455/final_model",
        "Reranking+Pruning Small": "./output/reranking_pruning_small_20250709_104353/final_model",
        "Reranking+Pruning Full": "./output/reranking_pruning_full_20250709_112127/final_model"
    }
    
    # Load FULL test dataset
    logger.info("Loading FULL test dataset...")
    dataset = load_dataset(
        'hotchpotch/wip-query-context-pruner-with-teacher-scores',
        'ja-minimal'  # Using minimal for faster testing, change to 'ja-full' for full evaluation
    )
    test_dataset = dataset['validation']
    
    logger.info(f"Test dataset size: {len(test_dataset)}")
    
    # Test multiple thresholds
    thresholds = [0.3, 0.5, 0.7]
    
    # Results storage
    results = {}
    
    # Evaluate each model
    for model_name, model_path in models.items():
        if not os.path.exists(model_path):
            logger.warning(f"Model not found: {model_name}")
            continue
        
        logger.info(f"\n{'='*60}")
        logger.info(f"Evaluating: {model_name}")
        logger.info(f"{'='*60}")
        
        model = PruningEncoder.from_pretrained(
            model_path,
            device="cuda" if torch.cuda.is_available() else "cpu"
        )
        
        results[model_name] = {}
        
        for threshold in thresholds:
            logger.info(f"\nThreshold = {threshold}")
            
            pos_metrics, neg_metrics, all_metrics = evaluate_model(
                model, test_dataset, threshold=threshold
            )
            
            results[model_name][threshold] = {
                'pos': pos_metrics,
                'neg': neg_metrics,
                'all': all_metrics
            }
            
            # Print results
            if pos_metrics:
                logger.info(f"POS - F2: {pos_metrics['f2']:.4f}, P: {pos_metrics['precision']:.4f}, R: {pos_metrics['recall']:.4f}")
            if neg_metrics:
                logger.info(f"NEG - F2: {neg_metrics['f2']:.4f}, P: {neg_metrics['precision']:.4f}, R: {neg_metrics['recall']:.4f}")
            logger.info(f"ALL - F2: {all_metrics['f2']:.4f}, P: {all_metrics['precision']:.4f}, R: {all_metrics['recall']:.4f}")
    
    # Summary comparison
    logger.info(f"\n{'='*60}")
    logger.info("SUMMARY: Pruning-Only vs Reranking+Pruning")
    logger.info(f"{'='*60}")
    
    # Compare at threshold=0.5
    threshold = 0.5
    logger.info(f"\nComparison at threshold={threshold}:")
    logger.info("\n| Model Type | Dataset | POS F2 | NEG F2 | ALL F2 |")
    logger.info("|------------|---------|--------|--------|--------|")
    
    for model_name in ["Pruning-Only Small", "Reranking+Pruning Small", 
                       "Pruning-Only Full", "Reranking+Pruning Full"]:
        if model_name in results and threshold in results[model_name]:
            r = results[model_name][threshold]
            pos_f2 = r['pos']['f2'] if r['pos'] else 0
            neg_f2 = r['neg']['f2'] if r['neg'] else 0
            all_f2 = r['all']['f2']
            
            model_type, dataset_size = model_name.rsplit(' ', 1)
            logger.info(f"| {model_type} | {dataset_size} | {pos_f2:.4f} | {neg_f2:.4f} | {all_f2:.4f} |")
    
    # Calculate improvements
    logger.info(f"\n{'='*60}")
    logger.info("Improvements: Reranking+Pruning vs Pruning-Only")
    logger.info(f"{'='*60}")
    
    for dataset_size in ["Small", "Full"]:
        po_name = f"Pruning-Only {dataset_size}"
        rp_name = f"Reranking+Pruning {dataset_size}"
        
        if po_name in results and rp_name in results:
            logger.info(f"\n{dataset_size} Dataset:")
            
            for threshold in thresholds:
                po = results[po_name][threshold]
                rp = results[rp_name][threshold]
                
                pos_imp = ((rp['pos']['f2'] - po['pos']['f2']) / po['pos']['f2'] * 100) if po['pos'] and rp['pos'] else 0
                neg_imp = ((rp['neg']['f2'] - po['neg']['f2']) / po['neg']['f2'] * 100) if po['neg'] and rp['neg'] else 0
                all_imp = ((rp['all']['f2'] - po['all']['f2']) / po['all']['f2'] * 100)
                
                logger.info(f"  Threshold {threshold}: POS +{pos_imp:.1f}%, NEG +{neg_imp:.1f}%, ALL +{all_imp:.1f}%")
    
    # Best threshold analysis
    logger.info(f"\n{'='*60}")
    logger.info("Best Threshold Analysis")
    logger.info(f"{'='*60}")
    
    for model_name in results:
        best_f2 = 0
        best_threshold = 0
        
        for threshold in results[model_name]:
            f2 = results[model_name][threshold]['all']['f2']
            if f2 > best_f2:
                best_f2 = f2
                best_threshold = threshold
        
        logger.info(f"{model_name}: Best F2={best_f2:.4f} at threshold={best_threshold}")


if __name__ == "__main__":
    main()