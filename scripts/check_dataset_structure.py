#!/usr/bin/env python3
"""Check dataset structure."""

from datasets import load_dataset

# Load dataset
dataset = load_dataset("hotchpotch/wip-query-context-pruner")

print("Dataset splits:", list(dataset.keys()))
print("\nTrain dataset info:")
print(f"Number of examples: {len(dataset['train'])}")
print(f"Features: {dataset['train'].features}")

# Check first example
print("\nFirst example keys:")
example = dataset['train'][0]
for key, value in example.items():
    if isinstance(value, list):
        print(f"  {key}: list of length {len(value)}")
        if value and isinstance(value[0], list):
            print(f"    First item: list of length {len(value[0])}")
    else:
        print(f"  {key}: {type(value).__name__}")

# Check actual data
print("\nFirst example data:")
print(f"  query: {example['query'][:50]}...")
print(f"  texts[0]: {example['texts'][0][:50]}...")
print(f"  labels: {example['labels']}")
print(f"  chunks[0]: {example['chunks'][0][:3]}...")  # First 3 chunks