#!/usr/bin/env python3
"""
Create training datasets for Provence model from hotchpotch/wip-query-context-pruner.
"""

import os
import json
from pathlib import Path
from typing import Dict, List
from datasets import load_dataset, Dataset, DatasetDict
import pandas as pd
from tqdm import tqdm


def analyze_dataset_distribution(dataset):
    """Analyze the distribution of dataset_name in the dataset."""
    dataset_names = [item['dataset_name'] for item in dataset]
    distribution = pd.Series(dataset_names).value_counts()
    print("\nDataset name distribution:")
    print(distribution)
    
    # Calculate proportions
    proportions = distribution / len(dataset_names)
    print("\nProportions:")
    for name, prop in proportions.items():
        print(f"{name}: {prop:.2%}")
    
    return proportions


def create_stratified_subset(dataset, target_size, proportions, seed=42):
    """Create a stratified subset maintaining dataset_name proportions."""
    # Group by dataset_name
    grouped = {}
    for idx, item in enumerate(dataset):
        dataset_name = item['dataset_name']
        if dataset_name not in grouped:
            grouped[dataset_name] = []
        grouped[dataset_name].append(idx)
    
    # Calculate samples per dataset_name
    samples_per_dataset = {}
    for name, prop in proportions.items():
        if name in grouped:
            samples_per_dataset[name] = max(1, int(target_size * prop))
    
    # Adjust to exactly match target_size
    total_samples = sum(samples_per_dataset.values())
    if total_samples != target_size:
        # Add or remove from the largest group
        largest_group = max(samples_per_dataset.items(), key=lambda x: x[1])[0]
        samples_per_dataset[largest_group] += (target_size - total_samples)
    
    # Sample indices
    import random
    random.seed(seed)
    
    selected_indices = []
    for name, count in samples_per_dataset.items():
        if name in grouped:
            indices = grouped[name]
            if len(indices) >= count:
                selected = random.sample(indices, count)
            else:
                # If not enough samples, take all and repeat some
                selected = indices * (count // len(indices) + 1)
                selected = selected[:count]
            selected_indices.extend(selected)
    
    # Create subset
    return dataset.select(selected_indices)


def split_dataset(dataset, valid_size, test_size):
    """Split dataset into train, validation, and test sets."""
    total_size = len(dataset)
    train_size = total_size - valid_size - test_size
    
    # Shuffle and split
    dataset = dataset.shuffle(seed=42)
    
    train_dataset = dataset.select(range(train_size))
    valid_dataset = dataset.select(range(train_size, train_size + valid_size))
    test_dataset = dataset.select(range(train_size + valid_size, total_size))
    
    return DatasetDict({
        'train': train_dataset,
        'validation': valid_dataset,
        'test': test_dataset
    })


def main():
    print("Loading dataset from hotchpotch/wip-query-context-pruner...")
    
    # Load the dataset
    dataset = load_dataset("hotchpotch/wip-query-context-pruner", split="train")
    print(f"Total dataset size: {len(dataset)}")
    
    # Print sample to understand structure
    print("\nFirst sample:")
    print(json.dumps(dataset[0], indent=2, ensure_ascii=False))
    
    # Analyze distribution
    proportions = analyze_dataset_distribution(dataset)
    
    # Define dataset configurations
    configs = [
        {
            'name': 'minimal',
            'total_size': 5000,
            'valid_size': 100,
            'test_size': 1000
        },
        {
            'name': 'small',
            'total_size': 50000,
            'valid_size': 2000,
            'test_size': 2000
        },
        {
            'name': 'full',
            'total_size': len(dataset),  # Use all available data
            'valid_size': 5000,
            'test_size': 5000
        }
    ]
    
    # Create datasets
    for config in configs:
        print(f"\n{'='*50}")
        print(f"Creating {config['name']} dataset...")
        
        if config['name'] == 'full':
            # Use the entire dataset
            subset = dataset
        else:
            # Create stratified subset
            subset = create_stratified_subset(
                dataset, 
                config['total_size'], 
                proportions
            )
        
        # Split into train/valid/test
        dataset_dict = split_dataset(
            subset,
            config['valid_size'],
            config['test_size']
        )
        
        # Print sizes
        print(f"Train size: {len(dataset_dict['train'])}")
        print(f"Validation size: {len(dataset_dict['validation'])}")
        print(f"Test size: {len(dataset_dict['test'])}")
        
        # Save dataset
        output_dir = f"tmp/datasets/dev-dataset/{config['name']}"
        print(f"Saving to {output_dir}...")
        dataset_dict.save_to_disk(output_dir)
        
        # Also save as JSON for inspection
        for split_name, split_dataset in dataset_dict.items():
            json_path = Path(output_dir) / f"{split_name}.json"
            # Save first 10 samples as JSON for inspection
            samples = [split_dataset[i] for i in range(min(10, len(split_dataset)))]
            with open(json_path, 'w', encoding='utf-8') as f:
                json.dump(samples, f, ensure_ascii=False, indent=2)
        
        print(f"✓ {config['name']} dataset created successfully!")


if __name__ == "__main__":
    main()