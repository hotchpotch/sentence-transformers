#!/usr/bin/env python3
"""
Evaluate Naver Provence model on test sets.

This script evaluates the Provence reranker model's pruning performance.
It calculates precision, recall, exact match, F1, and F2 scores.

Usage:
    # Evaluate Provence model on test set
    python scripts/provence_eval_test_set.py \
        --subset msmarco-minimal-ja \
        --threshold 0.5
    
    # Evaluate with multiple thresholds (not supported by Provence)
    python scripts/provence_eval_test_set.py \
        --subset msmarco-small-ja \
        --max_samples 100
"""

import argparse
import json
import logging
from pathlib import Path
from typing import Dict, List, Tuple, Any
import numpy as np
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from datasets import load_dataset
import torch
from transformers import AutoModel
from tqdm import tqdm

logging.basicConfig(
    format='%(asctime)s - %(levelname)s - %(message)s',
    level=logging.INFO
)
logger = logging.getLogger(__name__)


def calculate_pruning_metrics(
    predicted_chunks: List[List[int]], 
    true_chunks: List[List[int]],
    context_spans: List[List[List[int]]]
) -> Dict[str, float]:
    """
    Calculate pruning metrics including precision, recall, F1, F2, and exact match.
    
    Args:
        predicted_chunks: Binary predictions for each chunk [batch_size, num_chunks]
        true_chunks: Binary labels for each chunk (indices format)
        context_spans: Token positions for each chunk
        
    Returns:
        Dictionary containing all metrics
    """
    all_predictions = []
    all_labels = []
    exact_matches = []
    
    for pred_chunks, relevant_indices, spans in zip(predicted_chunks, true_chunks, context_spans):
        # Convert relevant_indices to binary format
        num_chunks = len(pred_chunks)
        true_binary = [0] * num_chunks
        for idx in relevant_indices:
            if idx < num_chunks:
                true_binary[idx] = 1
        
        # Add to overall lists
        all_predictions.extend(pred_chunks)
        all_labels.extend(true_binary)
        
        # Check exact match (all chunks predicted correctly)
        exact_match = (np.array(pred_chunks) == np.array(true_binary)).all()
        exact_matches.append(exact_match)
    
    # Convert to numpy arrays
    all_predictions = np.array(all_predictions)
    all_labels = np.array(all_labels)
    
    # Calculate metrics
    precision, recall, f1, _ = precision_recall_fscore_support(
        all_labels, all_predictions, average='binary', zero_division=0
    )
    
    # Calculate F2 score (emphasizes recall more than precision)
    beta = 2.0
    f2 = (1 + beta**2) * precision * recall / (beta**2 * precision + recall) if (beta**2 * precision + recall) > 0 else 0.0
    
    # Calculate accuracy
    accuracy = accuracy_score(all_labels, all_predictions)
    
    # Calculate compression ratio (how many chunks were pruned)
    total_chunks = len(all_predictions)
    kept_chunks = all_predictions.sum()
    compression_ratio = 1.0 - (kept_chunks / total_chunks) if total_chunks > 0 else 0.0
    
    return {
        'precision': float(precision),
        'recall': float(recall),
        'f1': float(f1),
        'f2': float(f2),
        'accuracy': float(accuracy),
        'exact_match': float(np.mean(exact_matches)),
        'compression_ratio': float(compression_ratio),
        'total_chunks': int(total_chunks),
        'kept_chunks': int(kept_chunks),
        'keep_ratio': float(kept_chunks / total_chunks) if total_chunks > 0 else 0.0
    }


def process_sample_with_provence(
    model,
    query: str,
    texts: List[str],
    context_spans: List[List[Tuple[int, int]]],
    threshold: float = 0.5
) -> Tuple[List[List[int]], List[float]]:
    """
    Process a sample using Provence model.
    
    Args:
        model: Provence model
        query: Query text
        texts: List of context texts (usually just one)
        context_spans: List of span positions for each text
        threshold: Not used by Provence (kept for compatibility)
        
    Returns:
        Tuple of (chunk_predictions, ranking_scores)
    """
    all_chunk_predictions = []
    all_ranking_scores = []
    
    # Process each text (document)
    for text_idx, text in enumerate(texts):
        spans = context_spans[text_idx]
        
        # Extract spans from text
        context_chunks = []
        for start, end in spans:
            chunk = text[start:end]
            context_chunks.append(chunk)
        
        # Provence processes the entire text at once
        try:
            with torch.no_grad():
                output = model.process(query, text)
            
            # Get results
            reranking_score = output.get('reranking_score', 0.0)
            pruned_context = output.get('pruned_context', '')
            compression_rate = output.get('compression_rate', 0.0)
            
            all_ranking_scores.append(float(reranking_score))
            
            # Determine which chunks were kept
            # Provence returns pruned text, so we need to match chunks
            chunk_predictions = []
            
            for chunk in context_chunks:
                # Simple heuristic: if chunk text appears in pruned_context, it was kept
                # This is not perfect but works for most cases
                if chunk in pruned_context:
                    chunk_predictions.append(1)  # Keep
                else:
                    chunk_predictions.append(0)  # Prune
            
            # If pruned_context is empty, all chunks were pruned
            if len(pruned_context) == 0:
                chunk_predictions = [0] * len(context_chunks)
            # If pruned_context equals original text, all chunks were kept
            elif pruned_context == text:
                chunk_predictions = [1] * len(context_chunks)
            
            all_chunk_predictions.append(chunk_predictions)
            
        except Exception as e:
            logger.error(f"Error processing with Provence: {e}")
            # Default to keeping all chunks on error
            chunk_predictions = [1] * len(context_chunks)
            all_chunk_predictions.append(chunk_predictions)
            all_ranking_scores.append(0.0)
    
    return all_chunk_predictions, all_ranking_scores


def evaluate_provence_on_dataset(
    model,
    dataset_name: str,
    subset: str,
    threshold: float = 0.5,
    batch_size: int = 1,  # Provence doesn't support batching
    max_samples: int = None,
    teacher_model_name: str = "japanese-reranker-xsmall-v2"
) -> Dict[str, Any]:
    """
    Evaluate Provence model on a dataset.
    
    Args:
        model: Provence model
        dataset_name: Name of the dataset
        subset: Dataset subset to use
        threshold: Threshold (not used by Provence but kept for compatibility)
        batch_size: Batch size (Provence processes one at a time)
        max_samples: Maximum number of samples to evaluate
        teacher_model_name: Name of teacher model for score column
        
    Returns:
        Dictionary containing evaluation results
    """
    # Load dataset
    logger.info(f"Loading dataset: {dataset_name}:{subset}")
    dataset = load_dataset(dataset_name, subset)
    test_data = dataset.get('test', dataset.get('validation'))
    
    if test_data is None:
        raise ValueError(f"No test or validation split found in dataset")
    
    if max_samples:
        test_data = test_data.select(range(min(max_samples, len(test_data))))
    
    logger.info(f"Evaluating on {len(test_data)} samples")
    
    # Collect predictions and labels
    all_predicted_chunks = []
    all_true_chunks = []
    all_context_spans = []
    ranking_scores = []
    teacher_scores = []
    
    # Process samples one by one (Provence doesn't batch)
    for i in tqdm(range(len(test_data)), desc="Evaluating"):
        sample = test_data[i]
        
        # Extract data from sample
        query = sample['query']  # Changed from 'queries' to 'query'
        texts = sample['texts']
        context_spans = sample['context_spans']
        context_spans_relevance = sample['context_spans_relevance']
        
        # Process with Provence
        chunk_predictions_list, rank_scores = process_sample_with_provence(
            model, query, texts, context_spans, threshold
        )
        
        # Collect results
        for text_idx in range(len(texts)):
            if text_idx < len(chunk_predictions_list):
                all_predicted_chunks.append(chunk_predictions_list[text_idx])
                all_true_chunks.append(context_spans_relevance[text_idx])
                all_context_spans.append(context_spans[text_idx])
        
        # Collect ranking scores
        ranking_scores.extend(rank_scores)
        
        # Collect teacher scores if available
        teacher_col = f'teacher_scores.{teacher_model_name}'
        if teacher_col in sample:
            teacher_scores.extend(sample[teacher_col])
    
    # Calculate metrics
    pruning_metrics = calculate_pruning_metrics(
        all_predicted_chunks, 
        all_true_chunks,
        all_context_spans
    )
    
    results = {
        'model': 'naver/provence-reranker-debertav3-v1',
        'dataset': f"{dataset_name}:{subset}",
        'num_samples': len(test_data),
        'threshold': f"Provence (auto)",
        'pruning_metrics': pruning_metrics
    }
    
    # Add ranking metrics if available
    if ranking_scores and teacher_scores:
        # Ensure same length
        min_len = min(len(ranking_scores), len(teacher_scores))
        ranking_scores = ranking_scores[:min_len]
        teacher_scores = teacher_scores[:min_len]
        
        if len(ranking_scores) == len(teacher_scores):
            ranking_scores_arr = np.array(ranking_scores)
            teacher_scores_arr = np.array(teacher_scores)
            
            # Calculate MSE
            mse = np.mean((ranking_scores_arr - teacher_scores_arr) ** 2)
            
            # Calculate correlation
            from scipy.stats import pearsonr, spearmanr
            try:
                pearson_corr, _ = pearsonr(ranking_scores_arr, teacher_scores_arr)
                spearman_corr, _ = spearmanr(ranking_scores_arr, teacher_scores_arr)
            except:
                pearson_corr = 0.0
                spearman_corr = 0.0
            
            results['ranking_metrics'] = {
                'mse': float(mse),
                'rmse': float(np.sqrt(mse)),
                'pearson_correlation': float(pearson_corr),
                'spearman_correlation': float(spearman_corr),
                'num_scores': len(ranking_scores)
            }
    
    return results


def main():
    parser = argparse.ArgumentParser(description="Evaluate Provence model on test sets")
    parser.add_argument(
        "--model_name",
        type=str,
        default="naver/provence-reranker-debertav3-v1",
        help="Provence model name (default: naver/provence-reranker-debertav3-v1)"
    )
    parser.add_argument(
        "--dataset_name",
        type=str,
        default="hotchpotch/wip-msmarco-context-relevance",
        help="Dataset name"
    )
    parser.add_argument(
        "--subset",
        type=str,
        default="msmarco-minimal-ja",
        help="Dataset subset (e.g., msmarco-minimal-ja, msmarco-small-ja, msmarco-full-ja)"
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.5,
        help="Threshold (not used by Provence but kept for compatibility)"
    )
    parser.add_argument(
        "--max_samples",
        type=int,
        default=None,
        help="Maximum number of samples to evaluate"
    )
    parser.add_argument(
        "--teacher_model_name",
        type=str,
        default="japanese-reranker-xsmall-v2",
        help="Teacher model name for score column"
    )
    parser.add_argument(
        "--output_file",
        type=str,
        default=None,
        help="Path to save results JSON"
    )
    
    args = parser.parse_args()
    
    # Load Provence model
    logger.info(f"Loading Provence model: {args.model_name}")
    model = AutoModel.from_pretrained(args.model_name, trust_remote_code=True)
    if hasattr(model, 'eval'):
        model.eval()
    
    # Evaluate
    results = evaluate_provence_on_dataset(
        model,
        args.dataset_name,
        args.subset,
        args.threshold,
        batch_size=1,  # Provence processes one at a time
        max_samples=args.max_samples,
        teacher_model_name=args.teacher_model_name
    )
    
    # Print results
    print(f"\n{'='*60}")
    print(f"Model: {results['model']}")
    print(f"Dataset: {results['dataset']}")
    print(f"Threshold: {results['threshold']}")
    print(f"Samples: {results['num_samples']}")
    print(f"\nPruning Metrics:")
    for metric, value in results['pruning_metrics'].items():
        if isinstance(value, float):
            print(f"  {metric:20s}: {value:.4f}")
        else:
            print(f"  {metric:20s}: {value}")
    
    if 'ranking_metrics' in results:
        print(f"\nRanking Metrics:")
        rm = results['ranking_metrics']
        print(f"  MSE with teacher:     {rm['mse']:.6f}")
        print(f"  RMSE with teacher:    {rm['rmse']:.4f}")
        print(f"  Pearson correlation:  {rm['pearson_correlation']:.4f}")
        print(f"  Spearman correlation: {rm['spearman_correlation']:.4f}")
        print(f"  Number of scores:     {rm['num_scores']}")
    
    # Save results if requested
    if args.output_file:
        output_path = Path(args.output_file)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        output_data = {
            'model_name': args.model_name,
            'dataset': f"{args.dataset_name}:{args.subset}",
            'results': results
        }
        
        with open(output_path, 'w') as f:
            json.dump(output_data, f, indent=2)
        logger.info(f"Results saved to: {output_path}")


if __name__ == "__main__":
    main()