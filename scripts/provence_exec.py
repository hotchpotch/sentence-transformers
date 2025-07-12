#!/usr/bin/env python3
"""
Test Naver Provence reranker model with query-context pairs.

Usage:
    python scripts/provence_exec.py -q "query" -c "context1" "context2" ...
"""

import argparse
import json
from transformers import AutoModel
import torch
import numpy as np


def process_json_file(json_file, model, threshold=0.5):
    """Process JSON file with query-context pairs and evaluate with Provence model."""
    with open(json_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    results = []
    
    for item in data:
        query = item['query']
        contexts = item['contexts'] 
        labels = item.get('labels', None)
        
        # Process each context individually with Provence
        predictions = []
        scores = []
        pruned_contexts = []
        
        for context in contexts:
            try:
                with torch.no_grad():
                    output = model.process(query, context)
                
                # Extract results
                reranking_score = output.get('reranking_score', 0.0)
                pruned_context = output.get('pruned_context', context)
                
                # Determine if context should be kept based on whether it was pruned
                # If pruned_context is significantly shorter, it was pruned (deleted)
                is_pruned = len(pruned_context) < len(context) * 0.5
                prediction = 0 if is_pruned else 1
                
                predictions.append(prediction)
                scores.append(reranking_score)
                pruned_contexts.append(pruned_context)
                
            except Exception as e:
                print(f"Error processing context: {e}")
                predictions.append(1)  # Default to keep on error
                scores.append(0.0)
                pruned_contexts.append(context)
        
        results.append({
            'query': query,
            'contexts': contexts,
            'predictions': predictions,
            'scores': scores,
            'pruned_contexts': pruned_contexts,
            'labels': labels
        })
    
    return results


def calculate_metrics(results):
    """Calculate evaluation metrics from results."""
    all_predictions = []
    all_labels = []
    
    for result in results:
        if result['labels'] is not None:
            all_predictions.extend(result['predictions'])
            all_labels.extend(result['labels'])
    
    if not all_predictions:
        return None
    
    # Convert to numpy arrays
    predictions = np.array(all_predictions)
    labels = np.array(all_labels)
    
    # Calculate metrics
    tp = np.sum((predictions == 1) & (labels == 1))
    fp = np.sum((predictions == 1) & (labels == 0))
    tn = np.sum((predictions == 0) & (labels == 0))
    fn = np.sum((predictions == 0) & (labels == 1))
    
    accuracy = (tp + tn) / len(labels) if len(labels) > 0 else 0
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
    f2 = 5 * precision * recall / (4 * precision + recall) if (4 * precision + recall) > 0 else 0
    
    return {
        'accuracy': float(accuracy),
        'precision': float(precision),
        'recall': float(recall),
        'f1': float(f1),
        'f2': float(f2),
        'tp': int(tp),
        'fp': int(fp),
        'tn': int(tn),
        'fn': int(fn),
        'total': int(len(labels))
    }


def main():
    parser = argparse.ArgumentParser(description='Test Provence reranker model')
    
    # Input options
    input_group = parser.add_mutually_exclusive_group(required=True)
    input_group.add_argument('-q', '--query', help='Query text')
    input_group.add_argument('-j', '--json', help='Path to JSON file with query-context pairs')
    
    parser.add_argument('-c', '--contexts', nargs='+', help='List of context texts (required with -q)')
    parser.add_argument('--model', default='naver/provence-reranker-debertav3-v1', 
                        help='Model name (default: naver/provence-reranker-debertav3-v1)')
    parser.add_argument('--threshold', type=float, default=0.5,
                        help='Threshold for pruning decision (default: 0.5)')
    parser.add_argument('--output', help='Output JSON file for results (only with -j)')
    
    args = parser.parse_args()
    
    # Validate arguments
    if args.query and not args.contexts:
        parser.error("-q/--query requires -c/--contexts")
    
    # Load model
    print(f"Loading model: {args.model}")
    model = AutoModel.from_pretrained(args.model, trust_remote_code=True)
    
    # Set to eval mode
    if hasattr(model, 'eval'):
        model.eval()
    
    if args.json:
        # Process JSON file
        print(f"Processing JSON file: {args.json}")
        results = process_json_file(args.json, model, args.threshold)
        
        # Calculate metrics
        metrics = calculate_metrics(results)
        
        if metrics:
            print("\n" + "=" * 80)
            print("Evaluation Results")
            print("=" * 80)
            print(f"Total examples: {len(results)}")
            print(f"Total contexts: {metrics['total']}")
            print(f"Accuracy: {metrics['accuracy']:.4f}")
            print(f"Precision: {metrics['precision']:.4f}")
            print(f"Recall: {metrics['recall']:.4f}")
            print(f"F1 Score: {metrics['f1']:.4f}")
            print(f"F2 Score: {metrics['f2']:.4f}")
            print(f"\nConfusion Matrix:")
            print(f"  TP (correctly kept): {metrics['tp']}")
            print(f"  FP (incorrectly kept): {metrics['fp']}")
            print(f"  TN (correctly deleted): {metrics['tn']}")
            print(f"  FN (incorrectly deleted): {metrics['fn']}")
            
            # Show some examples
            print("\n" + "=" * 80)
            print("Example Results (first 3)")
            print("=" * 80)
            for i, result in enumerate(results[:3]):
                print(f"\nExample {i+1}:")
                print(f"Query: {result['query']}")
                print(f"Predictions: {result['predictions']}")
                print(f"Labels: {result['labels']}")
                print(f"Scores: [" + ", ".join(f"{s:.3f}" for s in result['scores']) + "]")
        
        # Save results if requested
        if args.output:
            output_data = {
                'model': args.model,
                'threshold': args.threshold,
                'metrics': metrics,
                'results': results
            }
            with open(args.output, 'w', encoding='utf-8') as f:
                json.dump(output_data, f, ensure_ascii=False, indent=2)
            print(f"\nDetailed results saved to: {args.output}")
    
    else:
        # Original single query processing
        print(f"\nQuery: {args.query}")
        print("=" * 80)
        
        results = []
        for i, context in enumerate(args.contexts):
            print(f"\nContext {i+1}: {context[:100]}..." if len(context) > 100 else f"\nContext {i+1}: {context}")
            
            try:
                # Process with Provence model
                with torch.no_grad():
                    output = model.process(args.query, context)
                
                # Extract results
                reranking_score = output.get('reranking_score', 'N/A')
                pruned_context = output.get('pruned_context', 'N/A')
                
                print(f"  Reranking score: {reranking_score}")
                print(f"  Pruned context: {pruned_context}")
                
                results.append({
                    'context': context,
                    'score': reranking_score,
                    'pruned': pruned_context
                })
                
            except Exception as e:
                print(f"  Error processing context: {e}")
                results.append({
                    'context': context,
                    'score': None,
                    'pruned': None
                })
        
        # Sort by score and display summary
        print("\n" + "=" * 80)
        print("Summary (sorted by relevance score):")
        print("=" * 80)
        
        # Filter out errors and sort
        valid_results = [r for r in results if r['score'] is not None]
        valid_results.sort(key=lambda x: x['score'], reverse=True)
        
        for i, result in enumerate(valid_results):
            context_preview = result['context'][:60] + "..." if len(result['context']) > 60 else result['context']
            print(f"{i+1}. Score: {result['score']:.4f} - {context_preview}")
            if result['pruned'] != result['context']:
                pruned_preview = result['pruned'][:60] + "..." if len(result['pruned']) > 60 else result['pruned']
                print(f"   Pruned: {pruned_preview}")


if __name__ == "__main__":
    main()