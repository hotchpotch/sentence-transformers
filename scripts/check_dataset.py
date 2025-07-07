#!/usr/bin/env python3
"""Quick check of the dataset structure."""

from datasets import load_dataset
import json

print("Loading dataset info...")
try:
    # Try loading just a small portion first
    dataset = load_dataset("hotchpotch/wip-query-context-pruner", split="train", streaming=True)
    
    # Get first few samples
    samples = []
    for i, item in enumerate(dataset):
        samples.append(item)
        if i >= 4:  # Just get 5 samples
            break
    
    print(f"\nFirst sample structure:")
    print(json.dumps(samples[0], indent=2, ensure_ascii=False))
    
    print(f"\nColumn names: {list(samples[0].keys())}")
    
except Exception as e:
    print(f"Error loading dataset: {e}")
    print("\nTrying alternative approach...")
    
    # Check if it's a private dataset or needs authentication
    import huggingface_hub
    print(f"HF Hub version: {huggingface_hub.__version__}")