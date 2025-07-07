#!/usr/bin/env python3
"""
Create 100k samples small dataset from the original dataset.
"""

import logging
import random
from pathlib import Path
from datasets import load_dataset, Dataset, DatasetDict
from tqdm import tqdm

# Setup logging
logging.basicConfig(
    format='%(asctime)s - %(levelname)s - %(name)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S',
    level=logging.INFO
)
logger = logging.getLogger(__name__)

def main():
    logger.info("Creating 100k samples dataset...")
    
    # Configuration
    dataset_name = "hotchpotch/wip-query-context-pruner"
    output_path = "tmp/datasets/dev-dataset/small-100k"
    target_size = 100000
    
    logger.info(f"Loading dataset: {dataset_name}")
    
    # Load original dataset
    try:
        dataset = load_dataset(dataset_name)
        logger.info(f"Dataset loaded successfully")
        logger.info(f"Available splits: {list(dataset.keys())}")
        
        for split in dataset.keys():
            logger.info(f"  {split}: {len(dataset[split])} examples")
            
    except Exception as e:
        logger.error(f"Failed to load dataset: {e}")
        return
    
    # Prepare 100k dataset
    new_splits = {}
    
    if 'train' in dataset:
        train_data = dataset['train']
        logger.info(f"Original train size: {len(train_data)}")
        
        # Sample target_size examples
        if len(train_data) > target_size:
            # Shuffle and select
            indices = list(range(len(train_data)))
            random.seed(42)  # For reproducibility
            random.shuffle(indices)
            selected_indices = indices[:target_size]
            
            train_100k = train_data.select(selected_indices)
            logger.info(f"100k train size: {len(train_100k)}")
        else:
            train_100k = train_data
            logger.info(f"Using full train dataset: {len(train_100k)}")
        
        new_splits['train'] = train_100k
    
    # Create validation and test splits from train data
    if 'train' in new_splits:
        full_train = new_splits['train']
        train_size = len(full_train)
        
        # Use 80/10/10 split
        train_end = int(0.8 * train_size)
        val_end = int(0.9 * train_size)
        
        train_indices = list(range(train_end))
        val_indices = list(range(train_end, val_end))
        test_indices = list(range(val_end, train_size))
        
        new_splits['train'] = full_train.select(train_indices)
        new_splits['validation'] = full_train.select(val_indices)
        new_splits['test'] = full_train.select(test_indices)
        
        logger.info(f"Split sizes:")
        logger.info(f"  Train: {len(new_splits['train'])}")
        logger.info(f"  Validation: {len(new_splits['validation'])}")
        logger.info(f"  Test: {len(new_splits['test'])}")
    
    # Create DatasetDict
    small_dataset = DatasetDict(new_splits)
    
    # Save dataset
    output_path = Path(output_path)
    output_path.mkdir(parents=True, exist_ok=True)
    
    logger.info(f"Saving 100k dataset to {output_path}")
    small_dataset.save_to_disk(str(output_path))
    
    logger.info("100k dataset created successfully!")
    
    # Verify saved dataset
    logger.info("Verifying saved dataset...")
    try:
        loaded_dataset = DatasetDict.load_from_disk(str(output_path))
        for split, data in loaded_dataset.items():
            logger.info(f"  {split}: {len(data)} examples")
            
            # Show sample
            if len(data) > 0:
                sample = data[0]
                logger.info(f"  Sample keys: {list(sample.keys())}")
                
    except Exception as e:
        logger.error(f"Failed to verify dataset: {e}")

if __name__ == "__main__":
    main()