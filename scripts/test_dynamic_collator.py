#!/usr/bin/env python3
"""
Test script for dynamic collator with visual output.
"""

import torch
from transformers import AutoTokenizer
from sentence_transformers.provence.data_collator_dynamic import ProvenceDynamicDataCollator
import logging

logging.basicConfig(level=logging.INFO)


def visualize_pruning_labels(tokenizer, encoded_inputs, pruning_labels, pair_idx=0):
    """Visualize pruning labels for a specific pair."""
    token_ids = encoded_inputs['input_ids'][pair_idx]
    tokens = tokenizer.convert_ids_to_tokens(token_ids)
    labels = pruning_labels[pair_idx]
    
    print(f"\n--- Pair {pair_idx} Visualization ---")
    print("Token | Label | Text")
    print("-" * 40)
    
    for i, (token, label) in enumerate(zip(tokens, labels)):
        if token == '<pad>':
            break
        label_str = "KEEP" if label == 1 else "PRUNE"
        print(f"{i:5d} | {label_str:5s} | {token}")


def main():
    # Initialize tokenizer and collator
    tokenizer = AutoTokenizer.from_pretrained('intfloat/multilingual-e5-small')
    collator = ProvenceDynamicDataCollator(
        tokenizer=tokenizer,
        max_length=128,
        padding=True,
        truncation=True
    )
    
    # Test data with Japanese text
    features = [
        {
            'query': '機械学習とは何ですか？',
            'texts': [
                '機械学習は人工知能の一分野で、データから学習するアルゴリズムを研究する分野です。',  # Relevant
                'Pythonは汎用的なプログラミング言語です。',  # Not relevant
                '深層学習はニューラルネットワークを使用した機械学習の手法です。',  # Relevant
            ],
            'ranking_labels': [1, 0, 1],
            'teacher_scores': [0.9, 0.1, 0.8]
        },
        {
            'query': 'パスタの作り方は？',
            'texts': [
                '水を沸騰させてパスタを茹でます。',  # Relevant
                '機械学習は面白い分野です。',  # Not relevant
            ],
            'ranking_labels': [1, 0],
            'teacher_scores': [0.95, 0.05]
        }
    ]
    
    # Process batch
    batch = collator(features)
    
    # Extract results
    encoded_inputs = batch['sentence_features'][0]
    labels = batch['labels']
    pruning_labels = labels['pruning_labels']
    
    print("Batch processing results:")
    print(f"Number of pairs: {encoded_inputs['input_ids'].shape[0]}")
    print(f"Sequence length: {encoded_inputs['input_ids'].shape[1]}")
    print(f"Pruning labels shape: {pruning_labels.shape}")
    
    # Visualize each pair
    pair_info = [
        ("Query 1 - Relevant doc about ML", 0),
        ("Query 1 - Irrelevant doc about Python", 1),
        ("Query 1 - Relevant doc about DL", 2),
        ("Query 2 - Relevant doc about pasta", 3),
        ("Query 2 - Irrelevant doc about ML", 4),
    ]
    
    for info, idx in pair_info:
        print(f"\n{'='*60}")
        print(f"{info}")
        visualize_pruning_labels(tokenizer, encoded_inputs, pruning_labels, idx)
    
    # Summary statistics
    print(f"\n{'='*60}")
    print("Summary Statistics:")
    for i, (info, idx) in enumerate(pair_info):
        total_tokens = (encoded_inputs['attention_mask'][idx] == 1).sum().item()
        kept_tokens = pruning_labels[idx].sum().item()
        keep_ratio = kept_tokens / total_tokens if total_tokens > 0 else 0
        print(f"{info}: {kept_tokens}/{total_tokens} tokens kept ({keep_ratio:.1%})")


if __name__ == "__main__":
    main()