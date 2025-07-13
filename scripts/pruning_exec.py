#!/usr/bin/env python3
"""
CLI tool for text pruning using PruningEncoder models.

Usage:
    # Evaluate with default test data
    python scripts/pruning_exec.py -m model_path
    
    # Evaluate with custom JSON file
    python scripts/pruning_exec.py -m model_path -j custom_test.json
    
    # Single query evaluation
    python scripts/pruning_exec.py -m model_path -q "query" -c "context1" "context2" ...
    
Example:
    # Use default evaluation data (pruning-config/pruning_data_ja.json)
    python scripts/pruning_exec.py \
        -m ./output/pruning-models/model/final_model
        
    # Single query example
    python scripts/pruning_exec.py \
        -m ./output/pruning-models/model/final_model \
        -q "What is machine learning?" \
        -c "Machine learning is a branch of AI." "It uses algorithms to learn from data."
"""

import argparse
import sys
import json
from pathlib import Path
from typing import List, Dict, Tuple, Optional
import numpy as np
import torch
from transformers import AutoTokenizer

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from sentence_transformers.pruning import PruningEncoder


def format_context_with_del(context: str, should_delete: bool) -> str:
    """Format context with <del> tags if it should be deleted."""
    if should_delete:
        return f"<del>{context}</del>"
    else:
        return context


def process_json_input(json_file: str, model_path: str, thresholds: List[float], use_majority: bool, batch_size: int):
    """Process multiple query-context pairs from JSON file."""
    with open(json_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    # Load model once
    model = PruningEncoder.from_pretrained(model_path)
    model.eval()
    
    results = []
    
    for item in data:
        query = item['query']
        contexts = item['contexts']
        labels = item.get('labels', None)
        
        # Process this query-context pair
        result = evaluate_single_example(model, query, contexts, thresholds, use_majority)
        result['labels'] = labels
        result['reasoning'] = item.get('reasoning', '')
        results.append(result)
    
    return results


def evaluate_single_example(model, query: str, contexts: List[str], thresholds: List[float], use_majority: bool) -> Dict:
    """Evaluate a single query-context example."""
    # Ensure contexts end with period
    processed_contexts = []
    for ctx in contexts:
        if not ctx.endswith('。') and not ctx.endswith('.'):
            ctx = ctx + '。'
        processed_contexts.append(ctx)
    
    # Create single input: query [SEP] context1 context2 context3...
    combined_text = query + model.tokenizer.sep_token + ''.join(processed_contexts)
    
    # Tokenize
    encoding = model.tokenizer(
        combined_text,
        padding=False,
        truncation=True,
        max_length=model.max_length,
        return_tensors='pt'
    )
    
    # Move to device
    encoding = {k: v.to(model.device) for k, v in encoding.items()}
    
    # Get predictions
    with torch.no_grad():
        outputs = model(
            input_ids=encoding['input_ids'],
            attention_mask=encoding['attention_mask'],
            return_dict=True
        )
        
        # Get scores based on mode
        if model.mode == "reranking_pruning":
            ranking_logits = outputs["ranking_logits"]
            ranking_score = torch.sigmoid(ranking_logits).cpu().item()
            pruning_logits = outputs["pruning_logits"]
        else:  # pruning_only
            ranking_score = None
            pruning_logits = outputs["pruning_logits"]
        
        # Get pruning probabilities
        pruning_probs = torch.sigmoid(pruning_logits).cpu().numpy()
        
        # If binary classification, take the "keep" probability (index 1)
        if pruning_probs.ndim == 3 and pruning_probs.shape[-1] == 2:
            pruning_probs = pruning_probs[0, :, 1]  # Shape: (seq_len,)
        else:
            pruning_probs = pruning_probs[0]  # Already (seq_len,)
    
    # Convert tokens to text to find context boundaries
    tokens = model.tokenizer.convert_ids_to_tokens(encoding['input_ids'][0])
    
    # Find SEP token position
    sep_positions = []
    for i, token in enumerate(tokens):
        if token == model.tokenizer.sep_token:
            sep_positions.append(i)
    
    if not sep_positions:
        return {
            'query': query,
            'contexts': contexts,
            'error': 'Could not find SEP token',
            'predictions': {}
        }
    
    # Context starts after the first SEP token
    context_start = sep_positions[0] + 1
    
    # Find context boundaries by looking for sentence endings
    context_boundaries = [context_start]
    current_text = ""
    
    for i in range(context_start, len(tokens)):
        if tokens[i] in ['<s>', '</s>', '<pad>', model.tokenizer.pad_token]:
            continue
            
        token_text = model.tokenizer.convert_tokens_to_string([tokens[i]])
        current_text += token_text
        
        # Check if we've reached the end of a context
        if current_text.endswith('。') or current_text.endswith('.'):
            context_boundaries.append(i + 1)
            current_text = ""
    
    # Ensure we have the right number of boundaries
    if len(context_boundaries) > len(processed_contexts) + 1:
        context_boundaries = context_boundaries[:len(processed_contexts) + 1]
    
    # Process each threshold
    predictions_by_threshold = {}
    
    for threshold in thresholds:
        predictions = []
        
        # Calculate keep/delete for each context
        for i, context in enumerate(processed_contexts):
            if i < len(context_boundaries) - 1:
                start_pos = context_boundaries[i]
                end_pos = context_boundaries[i + 1]
                
                if use_majority:
                    # Original majority voting approach
                    context_probs = pruning_probs[start_pos:end_pos]
                    kept_tokens = np.sum(context_probs >= threshold)
                    total_tokens = len(context_probs)
                    is_kept = kept_tokens >= (total_tokens / 2)
                else:
                    # Provence-style averaging approach
                    is_kept = sentence_rounding_provence(
                        pruning_probs, start_pos, end_pos, threshold
                    )
                
                predictions.append(1 if is_kept else 0)
            else:
                # If we couldn't determine boundaries, keep the context
                predictions.append(1)
        
        predictions_by_threshold[threshold] = predictions
    
    return {
        'query': query,
        'contexts': contexts,
        'ranking_score': ranking_score,
        'predictions': predictions_by_threshold
    }


def sentence_rounding_provence(probabilities: np.ndarray, start_pos: int, end_pos: int, threshold: float) -> bool:
    """
    Provence-style sentence rounding: use average score for the entire sentence.
    
    Args:
        probabilities: Token-level keep probabilities
        start_pos: Start position in the probability array
        end_pos: End position in the probability array
        threshold: Threshold for keeping the sentence
        
    Returns:
        True if sentence should be kept, False if it should be deleted
    """
    # Get probabilities for this sentence
    sentence_probs = probabilities[start_pos:end_pos]
    
    # Calculate average probability (Provence approach)
    if len(sentence_probs) > 0:
        avg_prob = np.mean(sentence_probs)
        return avg_prob >= threshold
    else:
        # If no tokens, keep the sentence
        return True


def main():
    parser = argparse.ArgumentParser(description='Execute pruning on query-context pairs')
    parser.add_argument('-m', '--model', required=True, help='Path to the pruning model')
    
    # Input options
    input_group = parser.add_mutually_exclusive_group()
    input_group.add_argument('-q', '--query', help='Query text')
    input_group.add_argument('-j', '--json', help='Path to JSON file with query-context pairs')
    
    parser.add_argument('-c', '--contexts', nargs='+', help='List of context texts (required with -q)')
    parser.add_argument('--thresholds', nargs='+', type=float, default=[0.3],
                      help='Pruning thresholds (default: 0.3)')
    parser.add_argument('--batch-size', type=int, default=32, help='Batch size for inference')
    parser.add_argument('--use-majority', action='store_true', 
                      help='Use majority voting instead of Provence-style averaging')
    parser.add_argument('--output', help='Output JSON file for results (only with -j)')
    
    args = parser.parse_args()
    
    # Set default JSON file if no input specified
    if not args.query and not args.json:
        default_json = Path(__file__).parent.parent / 'pruning-config' / 'pruning_data_ja.json'
        if default_json.exists():
            args.json = str(default_json)
            print(f"Using default evaluation data: {args.json}")
        else:
            parser.error("No input specified. Use -q for single query or -j for JSON file.")
    
    # Validate arguments
    if args.query and not args.contexts:
        parser.error("-q/--query requires -c/--contexts")
    if args.output and not args.json:
        parser.error("--output can only be used with -j/--json")
    
    if args.json:
        # Process JSON input
        results = process_json_input(args.json, args.model, args.thresholds, args.use_majority, args.batch_size)
        
        # Calculate statistics
        for threshold in args.thresholds:
            total = 0
            correct = 0
            tp = 0  # True Positives (correctly kept)
            fp = 0  # False Positives (incorrectly kept)
            tn = 0  # True Negatives (correctly deleted)
            fn = 0  # False Negatives (incorrectly deleted)
            
            for result in results:
                if result.get('labels') is not None and threshold in result['predictions']:
                    predictions = result['predictions'][threshold]
                    labels = result['labels']
                    
                    for pred, label in zip(predictions, labels):
                        total += 1
                        if pred == label:
                            correct += 1
                            if pred == 1:
                                tp += 1
                            else:
                                tn += 1
                        else:
                            if pred == 1:
                                fp += 1
                            else:
                                fn += 1
            
            # Calculate metrics
            accuracy = correct / total if total > 0 else 0
            precision = tp / (tp + fp) if (tp + fp) > 0 else 0
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0
            f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
            f2 = 5 * precision * recall / (4 * precision + recall) if (4 * precision + recall) > 0 else 0
            
            print(f"\n=== Results for threshold {threshold} ===")
            print(f"Total examples: {len(results)}")
            print(f"Total contexts: {total}")
            print(f"Accuracy: {accuracy:.4f} ({correct}/{total})")
            print(f"Precision: {precision:.4f}")
            print(f"Recall: {recall:.4f}")
            print(f"F1 Score: {f1:.4f}")
            print(f"F2 Score: {f2:.4f}")
            print(f"Confusion Matrix:")
            print(f"  TP (correctly kept): {tp}")
            print(f"  FP (incorrectly kept): {fp}")
            print(f"  TN (correctly deleted): {tn}")
            print(f"  FN (incorrectly deleted): {fn}")
        
        # Save detailed results if output file specified
        if args.output:
            output_data = {
                'model': args.model,
                'thresholds': args.thresholds,
                'use_majority': args.use_majority,
                'results': results
            }
            with open(args.output, 'w', encoding='utf-8') as f:
                json.dump(output_data, f, ensure_ascii=False, indent=2)
            print(f"\nDetailed results saved to: {args.output}")
    
    else:
        # Single query-context processing (original behavior)
        model = PruningEncoder.from_pretrained(args.model)
        model.eval()
        
        result = evaluate_single_example(model, args.query, args.contexts, args.thresholds, args.use_majority)
        
        # Print results for each threshold
        for threshold in args.thresholds:
            formatted_contexts = []
            predictions = result['predictions'].get(threshold, [])
            
            for i, (context, pred) in enumerate(zip(args.contexts, predictions)):
                if pred == 0:  # Deleted
                    formatted_contexts.append(f"<del>{context}</del>")
                else:
                    formatted_contexts.append(context)
            
            if result['ranking_score'] is not None:
                print(f"th: {threshold}, q: {args.query}, score: {result['ranking_score']:.2f}, contexts: {' '.join(formatted_contexts)}")
            else:
                print(f"th: {threshold}, q: {args.query}, contexts: {' '.join(formatted_contexts)}")


if __name__ == "__main__":
    main()