#!/usr/bin/env python3
"""
Evaluate chunk-based performance for Provence models
"""

import argparse
import json
from pathlib import Path
from typing import Dict, List, Any

from datasets import load_dataset
import numpy as np
from tqdm import tqdm

from sentence_transformers.provence import ProvenceEncoder
from sentence_transformers.provence.evaluation import (
    comprehensive_evaluation, 
    print_evaluation_report,
    evaluate_multiple_thresholds
)


def load_model(model_path: str) -> ProvenceEncoder:
    """Load a trained Provence model"""
    print(f"Loading model from: {model_path}")
    model = ProvenceEncoder.from_pretrained(model_path)
    model.eval()
    return model


def load_evaluation_data(dataset_name: str, split: str = "train", max_samples: int = None) -> List[Dict[str, Any]]:
    """Load evaluation data with chunk positions and relevance labels"""
    dataset = load_dataset('hotchpotch/wip-query-context-pruner-with-teacher-scores', dataset_name)
    data = dataset[split]
    
    if max_samples:
        data = data.select(range(min(max_samples, len(data))))
    
    # Convert to list of examples
    examples = []
    for example in data:
        # Process each query-text pair
        for i, text in enumerate(example['texts']):
            if i < len(example['chunks_pos']) and i < len(example['relevant_chunks']):
                chunks_pos = example['chunks_pos'][i]
                relevant_chunks = example['relevant_chunks'][i]
                
                # Only include examples with chunks
                if chunks_pos and relevant_chunks:
                    examples.append({
                        'query': example['query'],
                        'text': text,
                        'chunks_pos': chunks_pos,
                        'relevant_chunks': relevant_chunks,
                        'score': example['teacher_scores_japanese-reranker-xsmall-v2'][i] if i < len(example['teacher_scores_japanese-reranker-xsmall-v2']) else 0.0
                    })
    
    return examples


def evaluate_chunk_performance(
    model: ProvenceEncoder,
    examples: List[Dict[str, Any]],
    token_threshold: float = 0.5,
    chunk_threshold: float = 0.5,
    batch_size: int = 32
) -> Dict[str, Any]:
    """Evaluate chunk-based performance"""
    
    print(f"Evaluating {len(examples)} examples with token_threshold={token_threshold}, chunk_threshold={chunk_threshold}")
    
    # Prepare data
    sentences = [(ex['query'], ex['text']) for ex in examples]
    chunk_positions = [ex['chunks_pos'] for ex in examples]
    true_ranking = [ex['score'] for ex in examples]
    true_chunks = [ex['relevant_chunks'] for ex in examples]
    
    # Get predictions
    print("Running inference...")
    outputs = model.predict_context(
        sentences,
        chunk_positions,
        batch_size=batch_size,
        token_threshold=token_threshold,
        chunk_threshold=chunk_threshold,
        show_progress_bar=True
    )
    
    # Extract predictions
    pred_ranking = [output.ranking_scores for output in outputs]
    pred_chunks = [output.chunk_predictions.tolist() for output in outputs]
    
    # Convert relevant_chunks indices to binary labels
    aligned_true_chunks = []
    aligned_pred_chunks = []
    
    for true_chunk_indices, pred_chunk_list, chunk_pos in zip(true_chunks, pred_chunks, chunk_positions):
        # Convert indices to binary labels
        num_chunks = len(chunk_pos)
        true_binary = [0] * num_chunks
        for idx in true_chunk_indices:
            if idx < num_chunks:
                true_binary[idx] = 1
        
        # Ensure pred_chunks has the same length
        pred_binary = pred_chunk_list[:num_chunks] + [0] * max(0, num_chunks - len(pred_chunk_list))
        
        aligned_true_chunks.append(true_binary)
        aligned_pred_chunks.append(pred_binary)
    
    # Evaluate
    results = comprehensive_evaluation(
        aligned_true_chunks, 
        aligned_pred_chunks,
        true_ranking,
        pred_ranking
    )
    
    return results


def main():
    parser = argparse.ArgumentParser(description="Evaluate chunk-based performance")
    parser.add_argument("--model_path", type=str, required=True, help="Path to trained model")
    parser.add_argument("--dataset", type=str, default="ja-minimal", 
                       choices=["ja-minimal", "ja-small", "ja-full"],
                       help="Dataset to evaluate on")
    parser.add_argument("--max_samples", type=int, default=None, help="Maximum samples to evaluate")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size for inference")
    parser.add_argument("--output_file", type=str, default=None, help="Output results to JSON file")
    
    # Threshold options
    parser.add_argument("--token_threshold", type=float, default=0.5, 
                       help="Threshold for token-level classification")
    parser.add_argument("--chunk_threshold", type=float, default=0.5,
                       help="Minimum ratio of tokens for chunk classification")
    parser.add_argument("--multiple_thresholds", action="store_true",
                       help="Evaluate multiple threshold combinations")
    
    args = parser.parse_args()
    
    # Load model
    model = load_model(args.model_path)
    
    # Load data
    print(f"Loading {args.dataset} dataset...")
    examples = load_evaluation_data(args.dataset, max_samples=args.max_samples)
    print(f"Loaded {len(examples)} examples")
    
    if args.multiple_thresholds:
        # Evaluate multiple threshold combinations
        threshold_combinations = [
            ("0.1_0.1", 0.1, 0.1),  # Lenient
            ("0.3_0.3", 0.3, 0.3),  # Moderate-low
            ("0.5_0.5", 0.5, 0.5),  # Moderate
            ("0.7_0.7", 0.7, 0.7),  # Moderate-high
            ("0.9_0.9", 0.9, 0.9),  # Strict
            ("0.5_0.3", 0.5, 0.3),  # Mixed: moderate token, lenient chunk
            ("0.3_0.5", 0.3, 0.5),  # Mixed: lenient token, moderate chunk
        ]
        
        all_results = {}
        for name, token_th, chunk_th in threshold_combinations:
            print(f"\n=== Evaluating {name} (token={token_th}, chunk={chunk_th}) ===")
            results = evaluate_chunk_performance(
                model, examples, token_th, chunk_th, args.batch_size
            )
            all_results[name] = results
            print_evaluation_report(results)
        
        # Save all results
        if args.output_file:
            output_path = Path(args.output_file)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            with open(output_path, 'w') as f:
                json.dump(all_results, f, indent=2, default=str)
            print(f"\nResults saved to: {output_path}")
    else:
        # Single evaluation
        results = evaluate_chunk_performance(
            model, examples, args.token_threshold, args.chunk_threshold, args.batch_size
        )
        
        print_evaluation_report(results)
        
        # Save results
        if args.output_file:
            output_path = Path(args.output_file)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            with open(output_path, 'w') as f:
                json.dump(results, f, indent=2, default=str)
            print(f"\nResults saved to: {output_path}")


if __name__ == "__main__":
    main()