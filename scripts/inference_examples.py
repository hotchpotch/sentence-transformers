#!/usr/bin/env python3
"""
Show inference examples with pruned texts at threshold 0.5.
"""

import logging
from pathlib import Path
import torch
from sentence_transformers.provence.encoder import ProvenceEncoder
from sentence_transformers.provence.data_collator_chunk_based import ProvenceChunkBasedDataCollator
from transformers import AutoTokenizer
from datasets import load_from_disk
import numpy as np

# Setup logging
logging.basicConfig(
    format='%(asctime)s - %(levelname)s - %(name)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S',
    level=logging.INFO
)
logger = logging.getLogger(__name__)


def get_pruned_text(model, tokenizer, query, text, threshold=0.5):
    """Get pruned text based on model predictions."""
    model.eval()
    
    # Tokenize
    inputs = tokenizer(
        [[query, text]],
        padding=True,
        truncation=True,
        max_length=512,
        return_tensors="pt"
    ).to(model.device)
    
    with torch.no_grad():
        outputs = model(
            input_ids=inputs['input_ids'],
            attention_mask=inputs['attention_mask']
        )
        
        # Get pruning predictions
        pruning_logits = outputs['pruning_logits']
        pruning_probs = torch.softmax(pruning_logits, dim=-1)
        keep_probs = pruning_probs[0, :, 1]  # Probability of keeping
        
        # Get token predictions
        mask = inputs['attention_mask'][0] == 1
        predictions = (keep_probs > threshold).float()
        
        # Find document boundaries (after second </s>)
        sep_token_id = tokenizer.sep_token_id
        sep_positions = (inputs['input_ids'][0] == sep_token_id).nonzero(as_tuple=True)[0]
        
        if len(sep_positions) >= 3:
            doc_start = sep_positions[1].item() + 1
            doc_end = sep_positions[2].item()
            
            # Get document tokens and predictions
            doc_token_ids = inputs['input_ids'][0, doc_start:doc_end]
            doc_predictions = predictions[doc_start:doc_end]
            
            # Decode only kept tokens
            kept_token_ids = []
            for token_id, keep in zip(doc_token_ids, doc_predictions):
                if keep:
                    kept_token_ids.append(token_id.item())
            
            # Decode
            original_text = tokenizer.decode(doc_token_ids, skip_special_tokens=True)
            pruned_text = tokenizer.decode(kept_token_ids, skip_special_tokens=True) if kept_token_ids else ""
            
            # Calculate stats
            num_tokens = len(doc_token_ids)
            num_kept = len(kept_token_ids)
            compression = 1.0 - (num_kept / num_tokens) if num_tokens > 0 else 0.0
            
            return {
                'original': original_text,
                'pruned': pruned_text,
                'num_tokens': num_tokens,
                'num_kept': num_kept,
                'compression_ratio': compression
            }
    
    return None


def main():
    logger.info("Running inference examples...")
    
    # Paths
    model_path = "outputs/provence-minimal-dynamic/checkpoint-750-best"
    dataset_path = "tmp/datasets/dev-dataset/minimal-5k-simple"
    
    # Load dataset
    logger.info(f"Loading dataset from {dataset_path}")
    dataset = load_from_disk(dataset_path)
    test_dataset = dataset['test']
    
    # Load model
    logger.info(f"Loading model from {model_path}")
    model = ProvenceEncoder.from_pretrained(model_path)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    logger.info(f"Model loaded on {device}")
    
    # Initialize tokenizer
    tokenizer = model.tokenizer
    
    # Process first 5 examples from test set
    threshold = 0.5
    logger.info(f"\nShowing pruned texts at threshold {threshold}:\n")
    
    for idx in range(min(5, len(test_dataset))):
        example = test_dataset[idx]
        query = example['query']
        
        print(f"\n{'='*80}")
        print(f"Example {idx + 1}")
        print(f"{'='*80}")
        print(f"\nQuery: {query}")
        
        # Process each text
        for text_idx, text in enumerate(example['texts']):
            label = example['ranking_labels'][text_idx]
            teacher_score = example['teacher_scores'][text_idx]
            relevant_chunks = example['relevant_chunks'][text_idx]
            
            print(f"\n--- Text {text_idx + 1} (Label: {label}, Teacher Score: {teacher_score:.3f}) ---")
            print(f"Relevant chunks: {relevant_chunks}")
            
            # Get pruned version
            result = get_pruned_text(model, tokenizer, query, text, threshold)
            
            if result:
                print(f"\nOriginal ({result['num_tokens']} tokens):")
                print(text[:500] + "..." if len(text) > 500 else text)
                
                print(f"\nPruned ({result['num_kept']} tokens, {result['compression_ratio']:.1%} compression):")
                print(result['pruned'][:500] + "..." if len(result['pruned']) > 500 else result['pruned'])
            else:
                print("Error processing text")
            
            # Only show first 3 texts per query for brevity
            if text_idx >= 2:
                print(f"\n... and {len(example['texts']) - 3} more texts")
                break


if __name__ == "__main__":
    main()