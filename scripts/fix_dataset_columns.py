#!/usr/bin/env python3
"""
Fix dataset by removing reserved columns.
"""

from datasets import load_from_disk, DatasetDict

# Load dataset
dataset = load_from_disk("tmp/datasets/dev-dataset/minimal")

# Remove reserved columns
columns_to_remove = ['dataset_name', 'example_id']
for split in dataset:
    # Remove the reserved columns
    dataset[split] = dataset[split].remove_columns([col for col in columns_to_remove if col in dataset[split].column_names])

# Save the fixed dataset
dataset.save_to_disk("tmp/datasets/dev-dataset/minimal-fixed")
print("Dataset fixed and saved to tmp/datasets/dev-dataset/minimal-fixed")