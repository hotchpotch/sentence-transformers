#!/usr/bin/env python3
"""
Simple pruning effect visualization using PruningEncoder models.
Shows results in format: query: {query} contexts: <del correct/incorrect>{deleted}</del> {kept}
"""

import json
import argparse
import sys
from pathlib import Path
import numpy as np
import torch

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from sentence_transformers.pruning import PruningEncoder


def analyze_pruning_simple(json_file, model_path, threshold=0.5, num_samples=10):
    """Show pruning results in simple format with correctness."""
    
    # Load data
    with open(json_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    # Load model
    print(f"Loading model: {model_path}")
    try:
        model = PruningEncoder.from_pretrained(model_path)
        model.eval()
        print(f"✓ Model loaded (mode: {model.mode})")
    except Exception as e:
        print(f"✗ Failed to load model: {e}")
        return
    
    print(f"\nThreshold: {threshold}")
    print("=" * 80)
    
    total_contexts = 0
    total_deleted = 0
    correct_deletions = 0
    incorrect_deletions = 0
    correct_keeps = 0
    incorrect_keeps = 0
    
    for i, item in enumerate(data[:num_samples]):
        if i >= num_samples:
            break
            
        query = item['query']
        contexts = item['contexts']
        labels = item.get('labels', [1] * len(contexts))
        
        # Process contexts with sentence ending
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
            
            # Get pruning probabilities
            pruning_logits = outputs["pruning_logits"]
            pruning_probs = torch.sigmoid(pruning_logits).cpu().numpy()
            
            # If binary classification, take the "keep" probability (index 1)
            if pruning_probs.ndim == 3 and pruning_probs.shape[-1] == 2:
                pruning_probs = pruning_probs[0, :, 1]  # Shape: (seq_len,)
            else:
                pruning_probs = pruning_probs[0]  # Already (seq_len,)
        
        # Convert tokens to find context boundaries
        tokens = model.tokenizer.convert_ids_to_tokens(encoding['input_ids'][0])
        
        # Find SEP token position
        sep_positions = []
        for idx, token in enumerate(tokens):
            if token == model.tokenizer.sep_token:
                sep_positions.append(idx)
        
        if not sep_positions:
            continue
        
        # Context starts after the first SEP token
        context_start = sep_positions[0] + 1
        
        # Find context boundaries by looking for sentence endings
        context_boundaries = [context_start]
        current_text = ""
        
        for idx in range(context_start, len(tokens)):
            if tokens[idx] in ['<s>', '</s>', '<pad>', model.tokenizer.pad_token]:
                continue
                
            token_text = model.tokenizer.convert_tokens_to_string([tokens[idx]])
            current_text += token_text
            
            # Check if we've reached the end of a context
            if current_text.endswith('。') or current_text.endswith('.'):
                context_boundaries.append(idx + 1)
                current_text = ""
        
        # Ensure we have the right number of boundaries
        if len(context_boundaries) > len(processed_contexts) + 1:
            context_boundaries = context_boundaries[:len(processed_contexts) + 1]
        
        # Format output
        formatted_contexts = []
        deleted_count = 0
        
        for j, (context, label) in enumerate(zip(contexts, labels)):
            total_contexts += 1
            
            if j < len(context_boundaries) - 1:
                start_pos = context_boundaries[j]
                end_pos = context_boundaries[j + 1]
                
                # Calculate average probability for this context
                context_probs = pruning_probs[start_pos:end_pos]
                avg_prob = np.mean(context_probs) if len(context_probs) > 0 else 1.0
                is_kept = avg_prob >= threshold
                
                if is_kept:
                    # Kept (prediction = 1)
                    if label == 1:
                        # Correctly kept
                        formatted_contexts.append(context)
                        correct_keeps += 1
                    else:
                        # Incorrectly kept (should have been deleted)
                        formatted_contexts.append(context)
                        incorrect_keeps += 1
                else:
                    # Deleted (prediction = 0)
                    if label == 0:
                        # Correctly deleted
                        formatted_contexts.append(f"<del correct>{context}</del>")
                        correct_deletions += 1
                    else:
                        # Incorrectly deleted (should have been kept)
                        formatted_contexts.append(f"<del incorrect>{context}</del>")
                        incorrect_deletions += 1
                    deleted_count += 1
                    total_deleted += 1
            else:
                # If we couldn't determine boundaries, keep by default
                if label == 1:
                    formatted_contexts.append(context)
                    correct_keeps += 1
                else:
                    formatted_contexts.append(context)
                    incorrect_keeps += 1
        
        # Print result
        print(f"query: {query}")
        print(f"contexts: {' '.join(formatted_contexts)}")
        if deleted_count > 0 or incorrect_keeps > 0:
            print(f"(deleted: {deleted_count}/{len(contexts)}, errors: {sum(1 for c in formatted_contexts if 'incorrect' in c)})")
        print()
    
    # Summary
    print("=" * 80)
    deletion_rate = (total_deleted / total_contexts * 100) if total_contexts > 0 else 0
    accuracy = ((correct_deletions + correct_keeps) / total_contexts * 100) if total_contexts > 0 else 0
    
    # Calculate confusion matrix values
    tp = correct_keeps  # True Positives: correctly kept (label=1, pred=1)
    fp = incorrect_keeps  # False Positives: incorrectly kept (label=0, pred=1)
    tn = correct_deletions  # True Negatives: correctly deleted (label=0, pred=0)
    fn = incorrect_deletions  # False Negatives: incorrectly deleted (label=1, pred=0)
    
    # Calculate metrics
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
    f2 = 5 * precision * recall / (4 * precision + recall) if (4 * precision + recall) > 0 else 0
    
    print(f"Summary:")
    print(f"  Total contexts: {total_contexts}")
    print(f"  Deleted: {total_deleted} ({deletion_rate:.1f}%)")
    print(f"  Overall accuracy: {accuracy:.1f}%")
    
    print(f"\nConfusion Matrix:")
    print(f"  TP (correctly kept): {tp}")
    print(f"  FP (incorrectly kept): {fp}")
    print(f"  TN (correctly deleted): {tn}")
    print(f"  FN (incorrectly deleted): {fn}")
    
    print(f"\nMetrics:")
    print(f"  Precision: {precision:.4f}")
    print(f"  Recall: {recall:.4f}")
    print(f"  F1 Score: {f1:.4f}")
    print(f"  F2 Score: {f2:.4f}")
    
    if fn > 0:
        print(f"\n⚠️  WARNING: {fn} important contexts were incorrectly deleted!")
    else:
        print(f"\n✅ Good: No important contexts were incorrectly deleted.")


def main():
    parser = argparse.ArgumentParser(description='Simple pruning visualization with correctness')
    parser.add_argument('--json', default='pruning-config/pruning_data_ja.json',
                        help='Path to JSON data file')
    parser.add_argument('--model', required=True,
                        help='Path to trained PruningEncoder model')
    parser.add_argument('--threshold', type=float, default=0.5,
                        help='Pruning threshold (default: 0.5)')
    parser.add_argument('--samples', type=int, default=10,
                        help='Number of samples to show (default: 10)')
    
    args = parser.parse_args()
    
    analyze_pruning_simple(args.json, args.model, args.threshold, args.samples)


if __name__ == "__main__":
    main()