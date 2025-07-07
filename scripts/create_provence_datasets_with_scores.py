#!/usr/bin/env python3
"""
Create training datasets for Provence model with teacher scores.
"""

import os
import json
import random
from pathlib import Path
from typing import Dict, List, Tuple
from datasets import load_dataset, Dataset, DatasetDict
import pandas as pd
from tqdm import tqdm
import torch
from sentence_transformers import CrossEncoder
import numpy as np


def load_teacher_model():
    """Load the teacher reranker model."""
    print("Loading teacher model: hotchpotch/japanese-reranker-xsmall-v2...")
    model = CrossEncoder("hotchpotch/japanese-reranker-xsmall-v2")
    model.eval()
    return model


def generate_teacher_scores(model, query, texts, batch_size=16):
    """Generate teacher scores for query-text pairs."""
    pairs = [[query, text] for text in texts]
    
    # Compute scores in batches
    all_scores = []
    for i in range(0, len(pairs), batch_size):
        batch = pairs[i:i+batch_size]
        with torch.no_grad():
            scores = model.predict(batch)
        all_scores.extend(scores)
    
    return all_scores


def process_example(example, teacher_model=None):
    """Process a single example to add teacher scores and format for training."""
    query = example['query']
    texts = example['texts']
    labels = example['labels']
    chunks_pos = example['chunks_pos']
    relevant_chunks = example['relevant_chunks']
    
    # Generate teacher scores if model is provided
    if teacher_model is not None:
        teacher_scores = generate_teacher_scores(teacher_model, query, texts)
    else:
        teacher_scores = [0.0] * len(texts)
    
    # Create training examples for each query-text pair
    processed_examples = []
    for idx, (text, label, chunks, rel_chunks, teacher_score) in enumerate(
        zip(texts, labels, chunks_pos, relevant_chunks, teacher_scores)
    ):
        # Create pruning labels for each chunk (1 = keep, 0 = prune)
        pruning_labels = []
        if len(rel_chunks) > 0:
            # If there are relevant chunks marked
            for chunk_idx in range(len(chunks)):
                pruning_labels.append(1 if chunk_idx in rel_chunks else 0)
        else:
            # If no relevant chunks, use the document-level label
            pruning_labels = [label] * len(chunks)
        
        processed_example = {
            'query': query,
            'text': text,
            'ranking_label': label,
            'teacher_score': teacher_score,
            'pruning_labels': pruning_labels,
            'sentence_boundaries': chunks,
            'dataset_name': example['dataset_name'],
            'example_id': f"{example['id']}_{idx}"
        }
        processed_examples.append(processed_example)
    
    return processed_examples


def analyze_dataset_distribution(dataset):
    """Analyze the distribution of dataset_name in the dataset."""
    dataset_names = []
    for item in dataset:
        dataset_names.append(item['dataset_name'])
    
    distribution = pd.Series(dataset_names).value_counts()
    print("\nDataset name distribution:")
    print(distribution)
    
    # Calculate proportions
    proportions = distribution / len(dataset_names)
    print("\nProportions:")
    for name, prop in proportions.items():
        print(f"{name}: {prop:.2%}")
    
    return proportions.to_dict()


def create_stratified_subset(dataset, target_size, proportions, seed=42):
    """Create a stratified subset maintaining dataset_name proportions."""
    random.seed(seed)
    
    # Group indices by dataset_name
    grouped = {}
    for idx in range(len(dataset)):
        dataset_name = dataset[idx]['dataset_name']
        if dataset_name not in grouped:
            grouped[dataset_name] = []
        grouped[dataset_name].append(idx)
    
    # Calculate samples per dataset_name
    samples_per_dataset = {}
    remaining = target_size
    
    # Sort by proportion to handle rounding
    sorted_names = sorted(proportions.items(), key=lambda x: x[1], reverse=True)
    
    for name, prop in sorted_names:
        if name in grouped:
            ideal_count = int(target_size * prop)
            actual_count = min(ideal_count, len(grouped[name]), remaining)
            samples_per_dataset[name] = actual_count
            remaining -= actual_count
    
    # Distribute remaining samples
    while remaining > 0:
        for name in sorted_names:
            name = name[0]
            if name in grouped and len(grouped[name]) > samples_per_dataset.get(name, 0):
                samples_per_dataset[name] = samples_per_dataset.get(name, 0) + 1
                remaining -= 1
                if remaining == 0:
                    break
    
    # Sample indices
    selected_indices = []
    for name, count in samples_per_dataset.items():
        indices = grouped[name]
        if len(indices) >= count:
            selected = random.sample(indices, count)
        else:
            selected = indices
        selected_indices.extend(selected)
    
    # Shuffle the selected indices
    random.shuffle(selected_indices)
    
    return dataset.select(selected_indices)


def split_dataset(dataset, valid_size, test_size, seed=42):
    """Split dataset into train, validation, and test sets."""
    total_size = len(dataset)
    train_size = total_size - valid_size - test_size
    
    # Shuffle and split
    dataset = dataset.shuffle(seed=seed)
    
    train_dataset = dataset.select(range(train_size))
    valid_dataset = dataset.select(range(train_size, train_size + valid_size))
    test_dataset = dataset.select(range(train_size + valid_size, total_size))
    
    return DatasetDict({
        'train': train_dataset,
        'validation': valid_dataset,
        'test': test_dataset
    })


def process_and_save_dataset(dataset_dict, output_dir, teacher_model=None):
    """Process dataset and save with teacher scores."""
    processed_dict = {}
    
    for split_name, split_dataset in dataset_dict.items():
        print(f"Processing {split_name} split...")
        
        # Process all examples
        all_processed = []
        for example in tqdm(split_dataset, desc=f"Processing {split_name}"):
            processed = process_example(example, teacher_model)
            all_processed.extend(processed)
        
        # Create new dataset
        processed_dict[split_name] = Dataset.from_list(all_processed)
        print(f"{split_name}: {len(all_processed)} training examples created")
    
    # Save processed dataset
    processed_dataset = DatasetDict(processed_dict)
    processed_dataset.save_to_disk(output_dir)
    
    # Save sample as JSON for inspection
    for split_name, split_dataset in processed_dataset.items():
        json_path = Path(output_dir) / f"{split_name}_samples.json"
        samples = []
        for i in range(min(5, len(split_dataset))):
            sample = dict(split_dataset[i])
            samples.append(sample)
        
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(samples, f, ensure_ascii=False, indent=2)
    
    return processed_dataset


def main():
    print("=" * 70)
    print("Creating Provence training datasets with teacher scores")
    print("=" * 70)
    
    # Load teacher model
    teacher_model = load_teacher_model()
    
    # Load dataset
    print("\nLoading dataset from hotchpotch/wip-query-context-pruner...")
    dataset = load_dataset("hotchpotch/wip-query-context-pruner", split="train", streaming=False)
    print(f"Total dataset size: {len(dataset)} examples")
    
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
        # Skip full dataset for now due to size
    ]
    
    # Create datasets
    for config in configs:
        print(f"\n{'='*50}")
        print(f"Creating {config['name']} dataset...")
        
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
        print(f"\nOriginal splits:")
        print(f"  Train: {len(dataset_dict['train'])} queries")
        print(f"  Valid: {len(dataset_dict['validation'])} queries")
        print(f"  Test: {len(dataset_dict['test'])} queries")
        
        # Process and save with teacher scores
        output_dir = f"tmp/datasets/dev-dataset/{config['name']}"
        processed_dataset = process_and_save_dataset(
            dataset_dict, 
            output_dir, 
            teacher_model
        )
        
        print(f"\nProcessed splits (query-document pairs):")
        for split_name, split_data in processed_dataset.items():
            print(f"  {split_name}: {len(split_data)} examples")
        
        print(f"\n✓ {config['name']} dataset created successfully at {output_dir}")
    
    print("\n" + "="*70)
    print("Dataset creation completed!")
    print("="*70)


if __name__ == "__main__":
    main()