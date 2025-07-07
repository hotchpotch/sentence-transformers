#!/usr/bin/env python3
"""
Create a partial dataset from available processed chunks for training.
"""

import logging
from pathlib import Path
from datasets import load_from_disk, Dataset, DatasetDict
import gc

# Setup logging
logging.basicConfig(
    format='%(asctime)s - %(levelname)s - %(name)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S',
    level=logging.INFO
)
logger = logging.getLogger(__name__)

def main():
    logger.info("Creating partial dataset from available chunks...")
    
    # Paths
    output_path = "tmp/datasets/dev-dataset/full-batched"
    
    # Load the final chunk which should have all examples from chunk 100-130
    final_chunk_path = "tmp/datasets/dev-dataset/full-final/train_chunks_100_to_130_final.arrow"
    logger.info(f"Loading final chunk: {final_chunk_path}")
    final_data = load_from_disk(final_chunk_path)
    logger.info(f"Loaded {len(final_data)} examples from chunks 100-130")
    
    # Load the chunk 68-97
    chunk_68_97_path = "tmp/datasets/dev-dataset/full-remaining/train_chunks_68_to_97.arrow"
    logger.info(f"Loading chunk 68-97: {chunk_68_97_path}")
    chunk_68_97_data = load_from_disk(chunk_68_97_path)
    logger.info(f"Loaded {len(chunk_68_97_data)} examples from chunks 68-97")
    
    # We're missing chunks 0-67, but we have chunks 68-130
    # This gives us about 630,000 samples out of 1.3M (about 48%)
    
    # Combine available data
    all_examples = []
    for example in chunk_68_97_data:
        all_examples.append(example)
    for example in final_data:
        all_examples.append(example)
    
    logger.info(f"Total train examples: {len(all_examples)}")
    
    # Create train dataset
    train_dataset = Dataset.from_list(all_examples)
    
    # Load validation and test from 100k dataset
    small_dataset_path = "tmp/datasets/dev-dataset/small-100k-batched"
    small_dataset = load_from_disk(small_dataset_path)
    
    # Create final dataset
    final_dataset = DatasetDict({
        'train': train_dataset,
        'validation': small_dataset['validation'],
        'test': small_dataset['test']
    })
    
    # Save
    output_dir = Path(output_path)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    logger.info(f"Saving partial dataset to {output_dir}")
    final_dataset.save_to_disk(str(output_dir))
    
    # Verify
    logger.info("\nDataset statistics:")
    for split, data in final_dataset.items():
        logger.info(f"  {split}: {len(data)} examples")
        
        if split == 'train' and len(data) > 0:
            # Check distribution
            all_labels = []
            all_ranking_labels = []
            sample_size = min(1000, len(data))
            
            for i in range(sample_size):
                example = data[i]
                for pruning_labels in example['pruning_labels']:
                    all_labels.extend(pruning_labels)
                all_ranking_labels.extend(example['ranking_labels'])
            
            if all_labels:
                keep_ratio = sum(all_labels) / len(all_labels)
                logger.info(f"  Pruning distribution (sample): {keep_ratio:.1%} keep, {(1-keep_ratio):.1%} prune")
            
            if all_ranking_labels:
                positive_ratio = sum(all_ranking_labels) / len(all_ranking_labels)
                logger.info(f"  Ranking distribution (sample): {positive_ratio:.1%} positive, {(1-positive_ratio):.1%} negative")
    
    logger.info("\nPartial dataset created successfully!")
    logger.info("Note: This dataset contains chunks 68-130 (about 620,000 samples)")
    logger.info("Chunks 0-67 would need to be reprocessed for the complete dataset")

if __name__ == "__main__":
    main()