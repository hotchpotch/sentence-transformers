#!/usr/bin/env python3
"""
CLI tool for text pruning using PruningEncoder models.

Usage:
    python scripts/pruning_exec.py -m model_path -q "query" -c "context1" "context2" ...
    
Example:
    python scripts/pruning_exec.py \
        -m ./output/pruning-models/model/final_model \
        -q "What is machine learning?" \
        -c "Machine learning is a branch of AI." "It uses algorithms to learn from data."
"""

import argparse
import sys
from pathlib import Path
from typing import List, Dict, Tuple
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


def main():
    parser = argparse.ArgumentParser(description='Execute pruning on query-context pairs')
    parser.add_argument('-m', '--model', required=True, help='Path to the pruning model')
    parser.add_argument('-q', '--query', required=True, help='Query text')
    parser.add_argument('-c', '--contexts', nargs='+', required=True, help='List of context texts')
    parser.add_argument('--thresholds', nargs='+', type=float, default=[0.3],
                      help='Pruning thresholds (default: 0.3)')
    parser.add_argument('--batch-size', type=int, default=32, help='Batch size for inference')
    
    args = parser.parse_args()
    
    # Load model
    model = PruningEncoder.from_pretrained(args.model)
    model.eval()
    
    # Ensure contexts end with period
    contexts = []
    for ctx in args.contexts:
        if not ctx.endswith('。') and not ctx.endswith('.'):
            ctx = ctx + '。'
        contexts.append(ctx)
    
    # Create single input: query [SEP] context1 context2 context3...
    combined_text = args.query + model.tokenizer.sep_token + ''.join(contexts)
    
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
        print("Error: Could not find SEP token")
        return
    
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
    if len(context_boundaries) > len(contexts) + 1:
        context_boundaries = context_boundaries[:len(contexts) + 1]
    
    # Process each threshold
    for threshold in args.thresholds:
        formatted_contexts = []
        
        # Calculate keep/delete for each context
        for i, context in enumerate(contexts):
            if i < len(context_boundaries) - 1:
                start_pos = context_boundaries[i]
                end_pos = context_boundaries[i + 1]
                
                # Get probabilities for this context
                context_probs = pruning_probs[start_pos:end_pos]
                
                # Context is kept if majority of tokens are above threshold
                kept_tokens = np.sum(context_probs >= threshold)
                total_tokens = len(context_probs)
                is_deleted = kept_tokens < (total_tokens / 2)
                
                formatted_context = format_context_with_del(context, is_deleted)
                formatted_contexts.append(formatted_context)
            else:
                # If we couldn't determine boundaries, keep the context
                formatted_contexts.append(context)
        
        # Print simple format
        if ranking_score is not None:
            print(f"th: {threshold}, q: {args.query}, score: {ranking_score:.2f}, contexts: {' '.join(formatted_contexts)}")
        else:
            print(f"th: {threshold}, q: {args.query}, contexts: {' '.join(formatted_contexts)}")


if __name__ == "__main__":
    main()