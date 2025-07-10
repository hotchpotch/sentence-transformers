#!/usr/bin/env python
"""
Debug MS MARCO predictions to understand why F2 scores are 0.
"""

import sys
from pathlib import Path
import logging
import numpy as np
import pandas as pd
from datasets import Dataset

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from sentence_transformers.pruning import PruningEncoder

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def debug_model_predictions():
    """Debug model predictions to understand the issue."""
    logger.info("="*60)
    logger.info("Debugging MS MARCO Model Predictions")
    logger.info("="*60)
    
    # Load test data
    base_path = f"hf://datasets/hotchpotch/wip-msmarco-context-relevance/msmarco-small-ja"
    test_df = pd.read_parquet(f"{base_path}/test-00000-of-00001.parquet")
    test_dataset = Dataset.from_pandas(test_df)
    
    # Load model
    import glob
    model_paths = glob.glob("./output/msmarco_small_reranking_pruning_*/final_model")
    if not model_paths:
        logger.error("No reranking+pruning model found!")
        return
    
    model_path = model_paths[-1]
    logger.info(f"Loading model from: {model_path}")
    
    model = PruningEncoder.from_pretrained(model_path)
    logger.info(f"Model mode: {model.mode}")
    
    # Analyze a few samples
    for i in range(5):
        sample = test_dataset[i]
        logger.info(f"\n--- Sample {i} ---")
        logger.info(f"Query: {sample['query']}")
        logger.info(f"Labels: {sample['labels']} (first=pos, rest=neg)")
        
        # Prepare query-text pairs
        query = sample['query']
        texts = sample['texts']
        labels = sample['labels']
        
        logger.info(f"Number of texts: {len(texts)}")
        
        # Test a few texts
        for j in range(min(3, len(texts))):
            text = texts[j]
            label = labels[j]
            
            logger.info(f"\nText {j} (label={label}):")
            logger.info(f"Text preview: {text[:100]}...")
            
            try:
                # Test with different thresholds
                for threshold in [0.3, 0.5, 0.7]:
                    outputs = model.predict_with_pruning(
                        [(query, text)],
                        pruning_threshold=threshold,
                        return_documents=True
                    )
                    
                    output = outputs[0]
                    
                    if hasattr(output, 'ranking_scores'):
                        ranking_score = output.ranking_scores
                    else:
                        ranking_score = "N/A"
                    
                    compression_ratio = output.compression_ratio
                    
                    # Binary prediction based on compression ratio
                    # High compression = content pruned = irrelevant (0)
                    # Low compression = content kept = relevant (1)
                    prediction = 1 if compression_ratio < (1.0 - threshold) else 0
                    
                    logger.info(f"  Threshold {threshold}: "
                              f"compression={compression_ratio:.3f}, "
                              f"ranking={ranking_score}, "
                              f"prediction={prediction}")
                    
            except Exception as e:
                logger.error(f"Error processing text {j}: {e}")
    
    # Check prediction distribution
    logger.info("\n" + "="*60)
    logger.info("Prediction Distribution Analysis")
    logger.info("="*60)
    
    threshold = 0.5
    predictions = []
    true_labels = []
    compression_ratios = []
    
    # Process subset
    for i in range(50):  # Check 50 samples
        sample = test_dataset[i]
        query = sample['query']
        texts = sample['texts']
        labels = sample['labels']
        
        for j, (text, label) in enumerate(zip(texts, labels)):
            try:
                outputs = model.predict_with_pruning(
                    [(query, text)],
                    pruning_threshold=threshold,
                    return_documents=True
                )
                
                compression_ratio = outputs[0].compression_ratio
                prediction = 1 if compression_ratio < (1.0 - threshold) else 0
                
                predictions.append(prediction)
                true_labels.append(label)
                compression_ratios.append(compression_ratio)
                
            except Exception as e:
                logger.warning(f"Error in sample {i}, text {j}: {e}")
    
    # Analyze results
    predictions = np.array(predictions)
    true_labels = np.array(true_labels)
    compression_ratios = np.array(compression_ratios)
    
    logger.info(f"Total samples analyzed: {len(predictions)}")
    logger.info(f"True positive labels: {np.sum(true_labels)}")
    logger.info(f"True negative labels: {len(true_labels) - np.sum(true_labels)}")
    logger.info(f"Predicted positive: {np.sum(predictions)}")
    logger.info(f"Predicted negative: {len(predictions) - np.sum(predictions)}")
    
    logger.info(f"Compression ratio stats:")
    logger.info(f"  Mean: {np.mean(compression_ratios):.3f}")
    logger.info(f"  Std: {np.std(compression_ratios):.3f}")
    logger.info(f"  Min: {np.min(compression_ratios):.3f}")
    logger.info(f"  Max: {np.max(compression_ratios):.3f}")
    
    # Check prediction logic
    logger.info(f"\nPrediction logic check (threshold={threshold}):")
    logger.info(f"Compression threshold: {1.0 - threshold}")
    logger.info(f"Compressions < threshold: {np.sum(compression_ratios < (1.0 - threshold))}")
    logger.info(f"Compressions >= threshold: {np.sum(compression_ratios >= (1.0 - threshold))}")
    
    # Correlation analysis
    pos_mask = true_labels == 1
    neg_mask = true_labels == 0
    
    if np.sum(pos_mask) > 0:
        pos_compressions = compression_ratios[pos_mask]
        logger.info(f"\nPositive samples compression: {np.mean(pos_compressions):.3f} ± {np.std(pos_compressions):.3f}")
    
    if np.sum(neg_mask) > 0:
        neg_compressions = compression_ratios[neg_mask]
        logger.info(f"Negative samples compression: {np.mean(neg_compressions):.3f} ± {np.std(neg_compressions):.3f}")
    
    # Confusion matrix
    logger.info(f"\nConfusion matrix:")
    tp = np.sum((predictions == 1) & (true_labels == 1))
    fp = np.sum((predictions == 1) & (true_labels == 0))
    tn = np.sum((predictions == 0) & (true_labels == 0))
    fn = np.sum((predictions == 0) & (true_labels == 1))
    
    logger.info(f"  TP: {tp}, FP: {fp}")
    logger.info(f"  FN: {fn}, TN: {tn}")
    
    if tp + fp > 0:
        precision = tp / (tp + fp)
    else:
        precision = 0
    
    if tp + fn > 0:
        recall = tp / (tp + fn)
    else:
        recall = 0
    
    if precision + recall > 0:
        f1 = 2 * precision * recall / (precision + recall)
        f2 = 5 * precision * recall / (4 * precision + recall)
    else:
        f1 = f2 = 0
    
    logger.info(f"Precision: {precision:.4f}")
    logger.info(f"Recall: {recall:.4f}")
    logger.info(f"F1: {f1:.4f}")
    logger.info(f"F2: {f2:.4f}")


if __name__ == "__main__":
    debug_model_predictions()