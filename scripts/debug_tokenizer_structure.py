#!/usr/bin/env python3
"""
Debug script to understand tokenizer structure.
"""

from transformers import AutoTokenizer
import torch


def main():
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained('intfloat/multilingual-e5-small')
    
    print(f"Tokenizer type: {type(tokenizer)}")
    print(f"Special tokens: {tokenizer.special_tokens_map}")
    print(f"CLS token: {tokenizer.cls_token} (ID: {tokenizer.cls_token_id})")
    print(f"SEP token: {tokenizer.sep_token} (ID: {tokenizer.sep_token_id})")
    print(f"PAD token: {tokenizer.pad_token} (ID: {tokenizer.pad_token_id})")
    
    # Test encoding
    query = "What is AI?"
    document = "Artificial Intelligence is a field of computer science."
    
    # Encode as pair
    encoded = tokenizer(
        [[query, document]],
        padding=True,
        truncation=True,
        max_length=128,
        return_tensors='pt',
        return_offsets_mapping=True
    )
    
    print("\n--- Encoded output ---")
    print(f"Input IDs shape: {encoded['input_ids'].shape}")
    print(f"Input IDs: {encoded['input_ids'][0][:20]}...")  # First 20 tokens
    
    # Decode tokens to see structure
    tokens = tokenizer.convert_ids_to_tokens(encoded['input_ids'][0])
    print(f"\nFirst 20 tokens: {tokens[:20]}")
    
    # Find special token positions
    input_ids = encoded['input_ids'][0]
    cls_positions = (input_ids == tokenizer.cls_token_id).nonzero(as_tuple=True)[0]
    sep_positions = (input_ids == tokenizer.sep_token_id).nonzero(as_tuple=True)[0]
    
    print(f"\nCLS positions: {cls_positions}")
    print(f"SEP positions: {sep_positions}")
    
    # Check offset mapping
    offset_mapping = encoded['offset_mapping'][0]
    print(f"\nOffset mapping shape: {offset_mapping.shape}")
    print("First 10 offsets:")
    for i in range(min(10, len(offset_mapping))):
        print(f"  Token {i}: offset {offset_mapping[i]}, token: '{tokens[i]}'")
    
    # Find where document starts
    print("\n--- Finding document start ---")
    if len(sep_positions) >= 1:
        first_sep = sep_positions[0].item()
        print(f"First SEP at position: {first_sep}")
        print(f"Tokens around first SEP:")
        for i in range(max(0, first_sep-2), min(len(tokens), first_sep+3)):
            print(f"  {i}: '{tokens[i]}'")
    
    # Alternative: single text encoding
    print("\n--- Single text encoding ---")
    single_encoded = tokenizer(
        query,
        padding=True,
        truncation=True,
        max_length=128,
        return_tensors='pt'
    )
    query_tokens = tokenizer.convert_ids_to_tokens(single_encoded['input_ids'][0])
    query_length = len([t for t in query_tokens if t not in ['<s>', '</s>', '<pad>']])
    print(f"Query tokens (excluding special): {query_length}")
    print(f"Query tokens: {query_tokens}")


if __name__ == "__main__":
    main()