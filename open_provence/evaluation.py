#!/usr/bin/env python3
"""
Evaluation metrics for chunk-based Provence predictions
"""

from typing import List, Dict, Any, Tuple
import numpy as np
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, classification_report


def evaluate_chunk_predictions(
    true_chunks: List[List[int]], 
    pred_chunks: List[List[int]]
) -> Dict[str, float]:
    """
    Evaluate chunk-level predictions.
    
    Args:
        true_chunks: Ground truth chunk relevance [query][chunk_idx] -> 0/1
        pred_chunks: Predicted chunk relevance [query][chunk_idx] -> 0/1
        
    Returns:
        Dictionary with evaluation metrics
    """
    # Flatten all chunks for overall metrics
    all_true = []
    all_pred = []
    
    for true_query, pred_query in zip(true_chunks, pred_chunks):
        # Ensure same length (pad with 0s if needed)
        max_len = max(len(true_query), len(pred_query))
        true_padded = true_query + [0] * (max_len - len(true_query))
        pred_padded = pred_query + [0] * (max_len - len(pred_query))
        
        all_true.extend(true_padded)
        all_pred.extend(pred_padded)
    
    # Calculate metrics
    accuracy = accuracy_score(all_true, all_pred)
    precision, recall, f1, _ = precision_recall_fscore_support(
        all_true, all_pred, average='binary', zero_division=0
    )
    
    # Calculate per-class metrics
    precision_classes, recall_classes, f1_classes, support = precision_recall_fscore_support(
        all_true, all_pred, average=None, zero_division=0
    )
    
    results = {
        'chunk_accuracy': accuracy,
        'chunk_precision': precision,
        'chunk_recall': recall,
        'chunk_f1': f1,
        'irrelevant_precision': precision_classes[0] if len(precision_classes) > 0 else 0.0,
        'irrelevant_recall': recall_classes[0] if len(recall_classes) > 0 else 0.0,
        'irrelevant_f1': f1_classes[0] if len(f1_classes) > 0 else 0.0,
        'relevant_precision': precision_classes[1] if len(precision_classes) > 1 else 0.0,
        'relevant_recall': recall_classes[1] if len(recall_classes) > 1 else 0.0,
        'relevant_f1': f1_classes[1] if len(f1_classes) > 1 else 0.0,
        'total_chunks': len(all_true),
        'relevant_chunks': sum(all_true),
        'predicted_relevant': sum(all_pred),
    }
    
    return results


def evaluate_ranking_performance(
    true_labels: List[float],
    pred_scores: List[float],
    threshold: float = 0.5
) -> Dict[str, float]:
    """
    Evaluate ranking performance.
    
    Args:
        true_labels: Ground truth relevance scores/labels
        pred_scores: Predicted ranking scores
        threshold: Threshold for binary classification
        
    Returns:
        Dictionary with ranking evaluation metrics
    """
    # Convert to binary if needed
    if all(label in [0, 1] for label in true_labels):
        true_binary = true_labels
    else:
        true_binary = [1 if label > threshold else 0 for label in true_labels]
    
    pred_binary = [1 if score > threshold else 0 for score in pred_scores]
    
    # Calculate metrics
    accuracy = accuracy_score(true_binary, pred_binary)
    precision, recall, f1, _ = precision_recall_fscore_support(
        true_binary, pred_binary, average='binary', zero_division=0
    )
    
    # Calculate correlation if continuous scores
    correlation = np.corrcoef(true_labels, pred_scores)[0, 1] if len(set(true_labels)) > 2 else 0.0
    
    results = {
        'ranking_accuracy': accuracy,
        'ranking_precision': precision,
        'ranking_recall': recall,
        'ranking_f1': f1,
        'ranking_correlation': correlation,
        'avg_pred_score': np.mean(pred_scores),
        'avg_true_score': np.mean(true_labels),
    }
    
    return results


def comprehensive_evaluation(
    true_chunks: List[List[int]],
    pred_chunks: List[List[int]], 
    true_ranking: List[float],
    pred_ranking: List[float],
    ranking_threshold: float = 0.5
) -> Dict[str, Any]:
    """
    Comprehensive evaluation combining chunk and ranking metrics.
    
    Args:
        true_chunks: Ground truth chunk relevance
        pred_chunks: Predicted chunk relevance
        true_ranking: Ground truth ranking scores/labels
        pred_ranking: Predicted ranking scores
        ranking_threshold: Threshold for ranking binary classification
        
    Returns:
        Combined evaluation results
    """
    chunk_results = evaluate_chunk_predictions(true_chunks, pred_chunks)
    ranking_results = evaluate_ranking_performance(true_ranking, pred_ranking, ranking_threshold)
    
    # Calculate chunk-level compression statistics
    total_chunks = sum(len(chunks) for chunks in true_chunks)
    kept_chunks = sum(sum(chunks) for chunks in pred_chunks)
    compression_ratio = 1.0 - (kept_chunks / total_chunks) if total_chunks > 0 else 0.0
    
    # Calculate per-query statistics
    per_query_stats = []
    for i, (true_q, pred_q) in enumerate(zip(true_chunks, pred_chunks)):
        if len(true_q) > 0:
            query_accuracy = accuracy_score(true_q, pred_q[:len(true_q)])
            query_compression = 1.0 - (sum(pred_q[:len(true_q)]) / len(true_q))
            per_query_stats.append({
                'query_idx': i,
                'accuracy': query_accuracy,
                'compression': query_compression,
                'num_chunks': len(true_q),
                'true_relevant': sum(true_q),
                'pred_relevant': sum(pred_q[:len(true_q)])
            })
    
    results = {
        **chunk_results,
        **ranking_results,
        'compression_ratio': compression_ratio,
        'avg_query_accuracy': np.mean([stat['accuracy'] for stat in per_query_stats]) if per_query_stats else 0.0,
        'avg_query_compression': np.mean([stat['compression'] for stat in per_query_stats]) if per_query_stats else 0.0,
        'per_query_stats': per_query_stats
    }
    
    return results


def print_evaluation_report(results: Dict[str, Any]) -> None:
    """
    Print a formatted evaluation report.
    
    Args:
        results: Results from comprehensive_evaluation
    """
    print("=== Chunk-based Evaluation Report ===\n")
    
    print("📊 Chunk-level Performance:")
    print(f"  Accuracy: {results['chunk_accuracy']:.3f}")
    print(f"  Precision: {results['chunk_precision']:.3f}")
    print(f"  Recall: {results['chunk_recall']:.3f}")
    print(f"  F1: {results['chunk_f1']:.3f}")
    
    print(f"\n📈 Ranking Performance:")
    print(f"  Accuracy: {results['ranking_accuracy']:.3f}")
    print(f"  Precision: {results['ranking_precision']:.3f}")
    print(f"  Recall: {results['ranking_recall']:.3f}")
    print(f"  F1: {results['ranking_f1']:.3f}")
    print(f"  Correlation: {results['ranking_correlation']:.3f}")
    
    print(f"\n✂️  Compression Statistics:")
    print(f"  Overall compression: {results['compression_ratio']:.1%}")
    print(f"  Avg query compression: {results['avg_query_compression']:.1%}")
    print(f"  Total chunks: {results['total_chunks']}")
    print(f"  Relevant chunks: {results['relevant_chunks']}")
    print(f"  Predicted relevant: {results['predicted_relevant']}")
    
    print(f"\n🎯 Summary:")
    print(f"  Avg query accuracy: {results['avg_query_accuracy']:.3f}")
    print(f"  Chunk precision: {results['chunk_precision']:.3f}")
    print(f"  Chunk recall: {results['chunk_recall']:.3f}")
    print(f"  Ranking F1: {results['ranking_f1']:.3f}")


def evaluate_multiple_thresholds(
    true_chunks: List[List[int]],
    pred_chunks_by_threshold: Dict[str, List[List[int]]],
    true_ranking: List[float],
    pred_ranking: List[float]
) -> Dict[str, Dict[str, Any]]:
    """
    Evaluate performance across multiple threshold combinations.
    
    Args:
        true_chunks: Ground truth chunk relevance
        pred_chunks_by_threshold: Predicted chunks for each threshold combination
        true_ranking: Ground truth ranking scores
        pred_ranking: Predicted ranking scores
        
    Returns:
        Results for each threshold combination
    """
    results = {}
    
    for threshold_name, pred_chunks in pred_chunks_by_threshold.items():
        results[threshold_name] = comprehensive_evaluation(
            true_chunks, pred_chunks, true_ranking, pred_ranking
        )
    
    return results