#!/usr/bin/env python3
"""
Test script for chunk-based collator with visual output.
"""

import torch
from transformers import AutoTokenizer
from sentence_transformers.provence.data_collator_chunk_based import ProvenceChunkBasedDataCollator
from datasets import load_from_disk
import logging

logging.basicConfig(level=logging.INFO)


def visualize_chunk_labels(tokenizer, encoded_inputs, pruning_labels, chunks_pos, relevant_chunks, offset_mapping, pair_idx=0):
    """Visualize pruning labels for a specific pair with chunk information."""
    token_ids = encoded_inputs['input_ids'][pair_idx]
    tokens = tokenizer.convert_ids_to_tokens(token_ids)
    labels = pruning_labels[pair_idx]
    offsets = offset_mapping[pair_idx] if offset_mapping is not None else None
    
    print(f"\n--- Pair {pair_idx} Visualization ---")
    print(f"Chunks: {len(chunks_pos)} total")
    print(f"Relevant chunks: {relevant_chunks}")
    print("\nToken | Label | Text | Offset")
    print("-" * 60)
    
    for i, (token, label) in enumerate(zip(tokens, labels)):
        if token == '<pad>':
            break
        label_str = "KEEP" if label == 1 else "PRUNE"
        offset_str = f"{offsets[i].tolist()}" if offsets is not None else "N/A"
        print(f"{i:5d} | {label_str:5s} | {token:15s} | {offset_str}")


def main():
    # Initialize tokenizer and collator
    tokenizer = AutoTokenizer.from_pretrained('intfloat/multilingual-e5-small')
    collator = ProvenceChunkBasedDataCollator(
        tokenizer=tokenizer,
        max_length=256,  # Longer to see more chunks
        padding=True,
        truncation=True
    )
    
    # Load processed minimal dataset
    dataset_path = "tmp/datasets/dev-dataset/minimal-5k-simple"
    dataset = load_from_disk(dataset_path)
    
    print(f"Loaded dataset from {dataset_path}")
    print(f"Train examples: {len(dataset['train'])}")
    
    # Get a few examples from the dataset
    examples = []
    for i in range(3):  # Take 3 examples
        example = dataset['train'][i]
        # Take first 2 texts only for visualization
        examples.append({
            'query': example['query'],
            'texts': example['texts'][:2],
            'ranking_labels': example['ranking_labels'][:2],
            'teacher_scores': example['teacher_scores'][:2],
            'chunks_pos': example['chunks_pos'][:2],
            'relevant_chunks': example['relevant_chunks'][:2]
        })
    
    # Process batch
    batch = collator(examples)
    
    # Extract results
    encoded_inputs = batch['sentence_features'][0]
    labels = batch['labels']
    pruning_labels = labels['pruning_labels']
    
    # Get offset mapping if available (for debugging)
    # Note: offset_mapping is removed during collation, so we need to regenerate it for visualization
    debug_pairs = []
    for example in examples:
        for text in example['texts']:
            debug_pairs.append([example['query'], text])
    
    debug_encoding = tokenizer(
        debug_pairs,
        padding='max_length',
        truncation=True,
        max_length=256,
        return_tensors='pt',
        return_offsets_mapping=True
    )
    offset_mappings = debug_encoding['offset_mapping']
    
    print("\nBatch processing results:")
    print(f"Number of pairs: {encoded_inputs['input_ids'].shape[0]}")
    print(f"Sequence length: {encoded_inputs['input_ids'].shape[1]}")
    print(f"Pruning labels shape: {pruning_labels.shape}")
    
    # Show details for each example
    pair_idx = 0
    for ex_idx, example in enumerate(examples):
        for text_idx in range(len(example['texts'])):
            print(f"\n{'='*80}")
            print(f"Example {ex_idx}, Text {text_idx}")
            print(f"Query: {example['query'][:50]}...")
            print(f"Text: {example['texts'][text_idx][:100]}...")
            print(f"Ranking label: {example['ranking_labels'][text_idx]}")
            print(f"Teacher score: {example['teacher_scores'][text_idx]:.3f}")
            print(f"Number of chunks: {len(example['chunks_pos'][text_idx])}")
            print(f"Relevant chunks: {example['relevant_chunks'][text_idx]}")
            
            # Show chunk details
            print("\nChunk details:")
            for chunk_idx, (start, end) in enumerate(example['chunks_pos'][text_idx][:5]):  # Show first 5 chunks
                text_snippet = example['texts'][text_idx][start:end]
                is_relevant = chunk_idx in example['relevant_chunks'][text_idx]
                print(f"  Chunk {chunk_idx} [{start}:{end}] {'[RELEVANT]' if is_relevant else ''}: {text_snippet[:50]}...")
            
            # Visualize pruning labels
            visualize_chunk_labels(
                tokenizer, 
                encoded_inputs, 
                pruning_labels,
                example['chunks_pos'][text_idx],
                example['relevant_chunks'][text_idx],
                offset_mappings,
                pair_idx
            )
            
            # Summary statistics
            total_tokens = (encoded_inputs['attention_mask'][pair_idx] == 1).sum().item()
            kept_tokens = pruning_labels[pair_idx].sum().item()
            keep_ratio = kept_tokens / total_tokens if total_tokens > 0 else 0
            print(f"\nSummary: {kept_tokens}/{total_tokens} tokens kept ({keep_ratio:.1%})")
            
            pair_idx += 1


if __name__ == "__main__":
    main()