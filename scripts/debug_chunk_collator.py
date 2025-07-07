#!/usr/bin/env python3
"""
Debug script for chunk-based collator to understand offset mapping.
"""

import torch
from transformers import AutoTokenizer
from sentence_transformers.provence.data_collator_chunk_based import ProvenceChunkBasedDataCollator
from datasets import load_from_disk
import logging

logging.basicConfig(level=logging.INFO)


def main():
    # Initialize tokenizer and collator
    tokenizer = AutoTokenizer.from_pretrained('intfloat/multilingual-e5-small')
    collator = ProvenceChunkBasedDataCollator(
        tokenizer=tokenizer,
        max_length=256,
        padding=True,
        truncation=True
    )
    
    # Load one example
    dataset_path = "tmp/datasets/dev-dataset/minimal-5k-simple"
    dataset = load_from_disk(dataset_path)
    
    # Get first example with relevant document
    example = dataset['train'][0]
    
    # Take only the first text for detailed analysis
    test_features = [{
        'query': example['query'],
        'texts': [example['texts'][0]],  # First text only
        'ranking_labels': [example['ranking_labels'][0]],
        'teacher_scores': [example['teacher_scores'][0]],
        'chunks_pos': [example['chunks_pos'][0]],
        'relevant_chunks': [example['relevant_chunks'][0]]
    }]
    
    print("Example details:")
    print(f"Query: {example['query']}")
    print(f"Text: {example['texts'][0][:100]}...")
    print(f"Ranking label: {example['ranking_labels'][0]}")
    print(f"Relevant chunks: {example['relevant_chunks'][0]}")
    print()
    
    # Show chunk details
    print("Chunk details:")
    text = example['texts'][0]
    for chunk_idx, (start, end) in enumerate(example['chunks_pos'][0]):
        is_relevant = chunk_idx in example['relevant_chunks'][0]
        chunk_text = text[start:end]
        print(f"Chunk {chunk_idx} [{start:3d}:{end:3d}] {'[REL]' if is_relevant else '     '}: {chunk_text[:60]}...")
    
    # Process with collator
    batch = collator(test_features)
    
    # Get the encoded output
    encoded = batch['sentence_features'][0]
    pruning_labels = batch['labels']['pruning_labels'][0]
    
    # Tokenize separately to get offset mapping
    pair = [[example['query'], example['texts'][0]]]
    debug_encoded = tokenizer(
        pair,
        padding=False,
        truncation=True,
        max_length=256,
        return_tensors='pt',
        return_offsets_mapping=True
    )
    
    tokens = tokenizer.convert_ids_to_tokens(debug_encoded['input_ids'][0])
    offsets = debug_encoded['offset_mapping'][0]
    
    # Find separator positions
    sep_token_id = tokenizer.sep_token_id
    sep_positions = (debug_encoded['input_ids'][0] == sep_token_id).nonzero(as_tuple=True)[0]
    
    print(f"\nTokenization info:")
    print(f"Total tokens: {len(tokens)}")
    print(f"SEP positions: {sep_positions.tolist()}")
    if len(sep_positions) >= 3:
        doc_start = sep_positions[1].item() + 1
        doc_end = sep_positions[2].item()
        print(f"Document token range: {doc_start} to {doc_end}")
    
    # Show token details
    print("\nToken details (showing relevant chunk region):")
    print("Idx  | Token           | Offset      | Label | In Chunk")
    print("-" * 65)
    
    # Focus on tokens around relevant chunks
    relevant_chunks = example['relevant_chunks'][0]
    if relevant_chunks:
        # Find tokens that should be marked
        first_rel_chunk = example['chunks_pos'][0][relevant_chunks[0]]
        start_char = first_rel_chunk[0]
        
        # Show tokens from a bit before to a bit after the first relevant chunk
        show_start = max(0, 100)  # Start showing from token 100
        show_end = min(len(tokens), 150)  # Show up to token 150
        
        for i in range(show_start, show_end):
            token = tokens[i]
            offset = offsets[i].tolist()
            label = pruning_labels[i].item() if i < len(pruning_labels) else -1
            
            # Check which chunk this token belongs to
            in_chunk = None
            for chunk_idx, (c_start, c_end) in enumerate(example['chunks_pos'][0]):
                if offset[0] >= c_start and offset[1] <= c_end:
                    in_chunk = chunk_idx
                    break
            
            chunk_str = f"Chunk {in_chunk}" if in_chunk is not None else ""
            if in_chunk in relevant_chunks:
                chunk_str += " [REL]"
            
            label_str = "KEEP" if label == 1 else "PRUNE" if label == 0 else "N/A"
            print(f"{i:4d} | {token:15s} | {str(offset):11s} | {label_str:5s} | {chunk_str}")
    
    # Summary
    total_tokens = len(pruning_labels)
    kept_tokens = pruning_labels.sum().item()
    print(f"\nSummary: {kept_tokens}/{total_tokens} tokens marked as KEEP ({kept_tokens/total_tokens*100:.1f}%)")


if __name__ == "__main__":
    main()