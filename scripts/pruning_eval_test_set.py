#!/usr/bin/env python3
"""
Evaluate trained PruningEncoder models on test sets.

This script evaluates both reranking and pruning performance, with special focus on pruning metrics.
It calculates precision, recall, exact match, F1, and F2 scores at different thresholds.

Usage:
    # Evaluate a specific model on test set
    python scripts/pruning_eval_test_set.py \
        --model_path ./output/pruning-models/japanese-reranker-xsmall-v2-reranking_pruning-msmarco-minimal-ja-250710135711/final_model \
        --subset msmarco-minimal-ja \
        --threshold 0.5
    
    # Evaluate with multiple thresholds
    python scripts/pruning_eval_test_set.py \
        --model_path ./output/pruning-models/model/final_model \
        --subset msmarco-small-ja \
        --thresholds 0.3 0.4 0.5 0.6 0.7
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
from tqdm import tqdm

from sentence_transformers.pruning import PruningEncoder, PruningDataCollator

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
        true_chunks: True relevance labels for each chunk (indices of relevant chunks)
        context_spans: Token positions for each chunk
        
    Returns:
        Dictionary containing all metrics
    """
    all_predictions = []
    all_labels = []
    exact_matches = []
    
    # For debugging
    total_texts_processed = 0
    
    for pred_chunks, true_chunk_indices, spans in zip(predicted_chunks, true_chunks, context_spans):
        # pred_chunks: [0, 0, 1, 0, 1, 1] (binary predictions for each chunk)
        # true_chunk_indices: [2, 4, 5] (indices of relevant chunks)
        
        num_chunks = len(pred_chunks)
        
        # Convert true chunk indices to binary array
        true_binary = [0] * num_chunks
        for idx in true_chunk_indices:
            if idx < num_chunks:
                true_binary[idx] = 1
        
        # Example: if true_chunk_indices = [2, 4, 5] and num_chunks = 6
        # true_binary = [0, 0, 1, 0, 1, 1]
        
        # Add to overall lists
        all_predictions.extend(pred_chunks)
        all_labels.extend(true_binary)
        
        # Check exact match (all chunks predicted correctly)
        exact_match = (np.array(pred_chunks) == np.array(true_binary)).all()
        exact_matches.append(exact_match)
        
        total_texts_processed += 1
    
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
        'kept_chunks': int(kept_chunks)
    }


def evaluate_model_on_dataset(
    model: PruningEncoder,
    dataset_name: str,
    subset: str,
    threshold: float = 0.5,
    batch_size: int = 64,
    max_samples: int = None,
    teacher_model_name: str = "japanese-reranker-xsmall-v2"
) -> Dict[str, Any]:
    """
    Evaluate a PruningEncoder model on a dataset.
    
    Args:
        model: Trained PruningEncoder model
        dataset_name: Name of the dataset
        subset: Dataset subset to use
        threshold: Threshold for pruning decisions
        batch_size: Batch size for evaluation
        max_samples: Maximum number of samples to evaluate (None for all)
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
    
    # Create data collator
    data_collator = PruningDataCollator(
        tokenizer=model.tokenizer,
        max_length=model.max_length,
        mode=model.mode,
        scores_column=f'teacher_scores.{teacher_model_name}',
        chunks_pos_column='context_spans',
        relevant_chunks_column='context_relevance'
    )
    
    # Collect predictions and labels
    all_predicted_chunks = []
    all_true_chunks = []
    all_context_spans = []
    ranking_scores = []
    teacher_scores = []
    
    # Process in batches
    model.eval()
    with torch.no_grad():
        for i in tqdm(range(0, len(test_data), batch_size), desc="Evaluating"):
            batch_indices = range(i, min(i + batch_size, len(test_data)))
            batch_data = [test_data[idx] for idx in batch_indices]
            
            # Prepare batch
            batch = data_collator(batch_data)
            
            # Move to device
            sentence_features = batch['sentence_features']
            for key in sentence_features[0]:
                if isinstance(sentence_features[0][key], torch.Tensor):
                    sentence_features[0][key] = sentence_features[0][key].to(model.device)
            
            # Get model predictions
            # Extract input_ids and attention_mask from sentence_features
            input_ids = sentence_features[0]['input_ids']
            attention_mask = sentence_features[0]['attention_mask']
            
            outputs = model.forward(
                input_ids=input_ids,
                attention_mask=attention_mask
            )
            
            # Process outputs based on mode
            if model.mode == "reranking_pruning":
                # Handle dict outputs
                if isinstance(outputs, dict):
                    # Get ranking scores
                    if 'ranking_logits' in outputs:
                        ranking_scores.extend(outputs['ranking_logits'].cpu().numpy().flatten().tolist())
                        
                        # Collect teacher scores for comparison
                        for sample in batch_data:
                            teacher_col = f'teacher_scores.{teacher_model_name}'
                            if teacher_col in sample:
                                teacher_scores.extend(sample[teacher_col])
                    
                    # Get pruning predictions
                    if 'pruning_logits' in outputs:
                        # Apply softmax to get probabilities
                        pruning_probs = torch.softmax(outputs['pruning_logits'], dim=-1)
                        keep_probs = pruning_probs[:, :, 1]  # Probability of keeping token
                    
                        # Process each sample in batch
                        for j, sample in enumerate(batch_data):
                            spans_per_text = sample['context_spans']  # List of spans for each text
                            true_chunks_per_text = sample['context_relevance']  # List of relevant chunk indices for each text
                            sample_keep_probs = keep_probs[j].cpu().numpy()
                            
                            # Process each text in the sample
                            for text_idx in range(len(spans_per_text)):
                                text_spans = spans_per_text[text_idx]
                                true_relevant_indices = true_chunks_per_text[text_idx]
                                
                                # Calculate average probability for each chunk in this text
                                chunk_predictions = []
                                for chunk_idx, (span_start, span_end) in enumerate(text_spans):
                                    # Get average probability for this chunk
                                    chunk_prob = sample_keep_probs[span_start:span_end].mean()
                                    # Apply threshold
                                    chunk_pred = 1 if chunk_prob >= threshold else 0
                                    chunk_predictions.append(chunk_pred)
                                
                                all_predicted_chunks.append(chunk_predictions)
                                all_true_chunks.append(true_relevant_indices)
                                all_context_spans.append(text_spans)
            
            elif model.mode == "pruning_only":
                # Handle pruning-only mode
                if isinstance(outputs, dict) and 'pruning_logits' in outputs:
                    pruning_probs = torch.softmax(outputs['pruning_logits'], dim=-1)
                    keep_probs = pruning_probs[:, :, 1]
                    
                    for j, sample in enumerate(batch_data):
                        spans_per_text = sample['context_spans']
                        true_chunks_per_text = sample['context_relevance']
                        sample_keep_probs = keep_probs[j].cpu().numpy()
                        
                        # Process each text in the sample
                        for text_idx in range(len(spans_per_text)):
                            text_spans = spans_per_text[text_idx]
                            true_relevant_indices = true_chunks_per_text[text_idx]
                            
                            chunk_predictions = []
                            for chunk_idx, (span_start, span_end) in enumerate(text_spans):
                                chunk_prob = sample_keep_probs[span_start:span_end].mean()
                                chunk_pred = 1 if chunk_prob >= threshold else 0
                                chunk_predictions.append(chunk_pred)
                            
                            all_predicted_chunks.append(chunk_predictions)
                            all_true_chunks.append(true_relevant_indices)
                            all_context_spans.append(text_spans)
    
    # Calculate metrics
    pruning_metrics = calculate_pruning_metrics(
        all_predicted_chunks, 
        all_true_chunks,
        all_context_spans
    )
    
    results = {
        'dataset': f"{dataset_name}:{subset}",
        'num_samples': len(test_data),
        'threshold': threshold,
        'pruning_metrics': pruning_metrics
    }
    
    # Add ranking metrics if available
    if ranking_scores and teacher_scores and model.mode == "reranking_pruning":
        # Ensure same length
        if len(ranking_scores) == len(teacher_scores):
            ranking_scores_arr = np.array(ranking_scores)
            teacher_scores_arr = np.array(teacher_scores)
            
            # Calculate MSE (Mean Squared Error)
            mse = np.mean((ranking_scores_arr - teacher_scores_arr) ** 2)
            
            # Calculate correlation
            from scipy.stats import pearsonr, spearmanr
            pearson_corr, _ = pearsonr(ranking_scores_arr, teacher_scores_arr)
            spearman_corr, _ = spearmanr(ranking_scores_arr, teacher_scores_arr)
            
            results['ranking_metrics'] = {
                'mse': float(mse),
                'rmse': float(np.sqrt(mse)),
                'pearson_correlation': float(pearson_corr),
                'spearman_correlation': float(spearman_corr),
                'num_scores': len(ranking_scores)
            }
        else:
            logger.warning(f"Ranking scores ({len(ranking_scores)}) and teacher scores ({len(teacher_scores)}) have different lengths")
    
    return results


def evaluate_multiple_thresholds(
    model: PruningEncoder,
    dataset_name: str,
    subset: str,
    thresholds: List[float],
    batch_size: int = 64,
    max_samples: int = None,
    teacher_model_name: str = "japanese-reranker-xsmall-v2"
) -> List[Dict[str, Any]]:
    """Evaluate model at multiple thresholds."""
    results = []
    for threshold in thresholds:
        logger.info(f"Evaluating at threshold: {threshold}")
        result = evaluate_model_on_dataset(
            model, dataset_name, subset, threshold, 
            batch_size, max_samples, teacher_model_name
        )
        results.append(result)
    return results


def main():
    parser = argparse.ArgumentParser(description="Evaluate PruningEncoder on test sets")
    parser.add_argument(
        "--model_path", 
        type=str, 
        required=True,
        help="Path to trained model"
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
        help="Single threshold for pruning decisions"
    )
    parser.add_argument(
        "--thresholds",
        type=float,
        nargs="+",
        help="Multiple thresholds to evaluate"
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=64,
        help="Batch size for evaluation (default: 64, larger values may cause OOM)"
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
    
    # Load model
    logger.info(f"Loading model from: {args.model_path}")
    model = PruningEncoder.from_pretrained(args.model_path)
    model.eval()
    
    # Determine thresholds to evaluate
    if args.thresholds:
        thresholds = args.thresholds
    else:
        thresholds = [args.threshold]
    
    # Evaluate
    if len(thresholds) == 1:
        results = evaluate_model_on_dataset(
            model, args.dataset_name, args.subset, thresholds[0],
            args.batch_size, args.max_samples, args.teacher_model_name
        )
        results_list = [results]
    else:
        results_list = evaluate_multiple_thresholds(
            model, args.dataset_name, args.subset, thresholds,
            args.batch_size, args.max_samples, args.teacher_model_name
        )
    
    # Print results
    for results in results_list:
        print(f"\n{'='*60}")
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
    
    # Find best threshold based on F2 score
    if len(results_list) > 1:
        best_idx = max(range(len(results_list)), 
                      key=lambda i: results_list[i]['pruning_metrics']['f2'])
        best_result = results_list[best_idx]
        print(f"\n{'='*60}")
        print(f"Best threshold based on F2 score: {best_result['threshold']}")
        print(f"F2 score: {best_result['pruning_metrics']['f2']:.4f}")
    
    # Save results if requested
    if args.output_file:
        output_path = Path(args.output_file)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        output_data = {
            'model_path': args.model_path,
            'dataset': f"{args.dataset_name}:{args.subset}",
            'results': results_list
        }
        
        with open(output_path, 'w') as f:
            json.dump(output_data, f, indent=2)
        logger.info(f"Results saved to: {output_path}")


if __name__ == "__main__":
    main()