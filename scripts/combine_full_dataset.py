#!/usr/bin/env python3
"""
Combine all processed chunks into a final full dataset.
"""

import logging
from pathlib import Path
from datasets import load_from_disk, Dataset, DatasetDict, concatenate_datasets
import gc

# Setup logging
logging.basicConfig(
    format='%(asctime)s - %(levelname)s - %(name)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S',
    level=logging.INFO
)
logger = logging.getLogger(__name__)

def main():
    logger.info("Combining all processed chunks into final dataset...")
    
    # Paths
    first_67_path = "tmp/datasets/dev-dataset/full-batched-checkpoint/train_checkpoint.pkl"
    remaining_path = "tmp/datasets/dev-dataset/full-remaining"
    final_chunks_path = "tmp/datasets/dev-dataset/full-final" 
    output_path = "tmp/datasets/dev-dataset/full-batched"
    
    # Load original small datasets for validation and test splits
    small_dataset_path = "tmp/datasets/dev-dataset/small-100k-batched"
    small_dataset = load_from_disk(small_dataset_path)
    
    logger.info("Loading processed chunks...")
    
    # First, we need to reconstruct the first 67 chunks
    # Since the checkpoint file is corrupted, we'll use the 100k dataset as a starting point
    # and process the remaining chunks
    
    # Load all processed chunks
    all_train_examples = []
    
    # Check what we have in each directory
    logger.info("Checking available processed data...")
    
    # From full-remaining directory (chunks 68-97)
    remaining_dir = Path(remaining_path)
    if remaining_dir.exists():
        for chunk_file in sorted(remaining_dir.glob("train_chunks_*.arrow")):
            logger.info(f"Loading {chunk_file}")
            chunk_data = load_from_disk(str(chunk_file))
            all_train_examples.extend(chunk_data)
            logger.info(f"  Loaded {len(chunk_data)} examples")
    
    # From full-final directory (chunks 100-130)
    final_dir = Path(final_chunks_path)
    if final_dir.exists():
        for chunk_file in sorted(final_dir.glob("train_chunks_*.arrow")):
            logger.info(f"Loading {chunk_file}")
            chunk_data = load_from_disk(str(chunk_file))
            all_train_examples.extend(chunk_data)
            logger.info(f"  Loaded {len(chunk_data)} examples")
    
    logger.info(f"Total train examples loaded: {len(all_train_examples)}")
    
    # Since we're missing chunks 0-67, we'll create a partial dataset
    # For now, let's save what we have
    
    if len(all_train_examples) > 0:
        # Create dataset from all examples
        train_dataset = Dataset.from_list(all_train_examples)
        
        # Use validation and test from small dataset
        final_dataset = DatasetDict({
            'train': train_dataset,
            'validation': small_dataset['validation'],
            'test': small_dataset['test']
        })
        
        # Save
        output_dir = Path(output_path + "-partial")
        output_dir.mkdir(parents=True, exist_ok=True)
        
        logger.info(f"Saving partial dataset to {output_dir}")
        final_dataset.save_to_disk(str(output_dir))
        
        # Verify
        logger.info("Dataset statistics:")
        for split, data in final_dataset.items():
            logger.info(f"  {split}: {len(data)} examples")
    else:
        logger.error("No train examples found!")
        
        # Alternative: Create full dataset from scratch with all chunks
        logger.info("Will need to reprocess chunks 0-67 to create complete dataset")

if __name__ == "__main__":
    main()