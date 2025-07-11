#!/usr/bin/env python3
"""
Test Naver Provence reranker model with query-context pairs.

Usage:
    python scripts/provence_exec.py -q "query" -c "context1" "context2" ...
"""

import argparse
from transformers import AutoModel
import torch


def main():
    parser = argparse.ArgumentParser(description='Test Provence reranker model')
    parser.add_argument('-q', '--query', required=True, help='Query text')
    parser.add_argument('-c', '--contexts', nargs='+', required=True, help='List of context texts')
    parser.add_argument('--model', default='naver/provence-reranker-debertav3-v1', 
                        help='Model name (default: naver/provence-reranker-debertav3-v1)')
    
    args = parser.parse_args()
    
    # Load model
    print(f"Loading model: {args.model}")
    model = AutoModel.from_pretrained(args.model, trust_remote_code=True)
    
    # Set to eval mode
    if hasattr(model, 'eval'):
        model.eval()
    
    # Process each context
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