#!/usr/bin/env python3
"""
Process full 1.3M dataset with improved parallel processing and checkpointing.
"""

import logging
import random
import torch
from pathlib import Path
from datasets import load_dataset, Dataset, DatasetDict
from tqdm import tqdm
from sentence_transformers import CrossEncoder
import gc
import pickle
import json
from concurrent.futures import ThreadPoolExecutor
import numpy as np

# Setup logging
logging.basicConfig(
    format='%(asctime)s - %(levelname)s - %(name)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S',
    level=logging.INFO
)
logger = logging.getLogger(__name__)

def process_batch(batch, teacher_model):
    """Process a batch preserving all 5 texts per query."""
    batch_size = len(batch['id'])
    processed_examples = []
    
    # Collect all pairs for batch processing
    all_pairs = []
    pair_to_example = []
    
    for i in range(batch_size):
        query = batch['query'][i]
        texts = batch['texts'][i]  # List of 5 texts
        labels = batch['labels'][i]  # List of 5 labels
        chunks_pos = batch['chunks_pos'][i]  # List of 5 chunk positions
        
        # Store mapping info
        for j, text in enumerate(texts):
            all_pairs.append([query, text])
            pair_to_example.append((i, j))
    
    # Get teacher scores for all pairs at once
    with torch.no_grad():
        all_teacher_scores = teacher_model.predict(all_pairs, batch_size=128)  # Increased batch size
    
    # Reorganize scores back to per-example format
    for i in range(batch_size):
        query = batch['query'][i]
        texts = batch['texts'][i]
        labels = batch['labels'][i]
        chunks_pos = batch['chunks_pos'][i]
        
        # Extract teacher scores for this example
        teacher_scores = []
        for idx, (ex_idx, text_idx) in enumerate(pair_to_example):
            if ex_idx == i:
                teacher_scores.append(float(all_teacher_scores[idx]))
        
        # Process all texts
        all_pruning_labels = []
        all_sentence_boundaries = []
        
        for j, (text, teacher_score, chunks) in enumerate(zip(texts, teacher_scores, chunks_pos)):
            num_sentences = len(chunks)
            
            # Generate pruning labels for each sentence
            pruning_labels = []
            for k in range(num_sentences):
                # Base probability depends on teacher score and original label
                if labels[j] == 1:  # Relevant document
                    if teacher_score > 0.7:
                        keep_prob = 0.85
                    elif teacher_score > 0.5:
                        keep_prob = 0.75
                    else:
                        keep_prob = 0.6
                else:  # Non-relevant document
                    if teacher_score > 0.5:
                        keep_prob = 0.4
                    elif teacher_score > 0.3:
                        keep_prob = 0.3
                    else:
                        keep_prob = 0.2
                
                # Boost for first and last sentences
                if k == 0 or k == num_sentences - 1:
                    keep_prob = min(keep_prob + 0.1, 0.95)
                
                # Random decision
                keep = random.random() < keep_prob
                pruning_labels.append(1 if keep else 0)
            
            all_pruning_labels.append(pruning_labels)
            all_sentence_boundaries.append(chunks)
        
        # Create processed example with all texts
        processed_example = {
            'query': query,
            'texts': texts,  # Keep all 5 texts
            'ranking_labels': labels,  # Keep all 5 labels
            'teacher_scores': teacher_scores,  # All 5 teacher scores
            'pruning_labels': all_pruning_labels,  # List of 5 pruning label lists
            'sentence_boundaries': all_sentence_boundaries,  # List of 5 boundary lists
            'dataset_name': batch['dataset_name'][i],
            'example_id': batch['id'][i],
        }
        processed_examples.append(processed_example)
    
    return processed_examples

def save_checkpoint(checkpoint_file, processed_examples, chunk_idx, split_name):
    """Save checkpoint with error handling."""
    temp_file = checkpoint_file.with_suffix('.tmp')
    checkpoint_data = {
        'processed_examples': processed_examples,
        'last_chunk': chunk_idx,
        'split_name': split_name
    }
    
    # Save to temporary file first
    with open(temp_file, 'wb') as f:
        pickle.dump(checkpoint_data, f, protocol=4)  # Use protocol 4 for better performance
    
    # Move to final location
    temp_file.rename(checkpoint_file)
    
def process_dataset_split(split_name, split_data, teacher_model, output_dir, checkpoint_dir):
    """Process a single dataset split with improved efficiency."""
    logger.info(f"Processing {split_name} split...")
    
    # Check for existing checkpoint
    checkpoint_file = checkpoint_dir / f"{split_name}_checkpoint.pkl"
    if checkpoint_file.exists():
        try:
            with open(checkpoint_file, 'rb') as f:
                checkpoint_data = pickle.load(f)
                processed_examples = checkpoint_data['processed_examples']
                start_chunk = checkpoint_data['last_chunk'] + 1
                logger.info(f"Resuming from chunk {start_chunk}, {len(processed_examples)} examples already processed")
        except Exception as e:
            logger.warning(f"Failed to load checkpoint: {e}. Starting from scratch.")
            processed_examples = []
            start_chunk = 0
    else:
        processed_examples = []
        start_chunk = 0
    
    random.seed(42)  # For reproducibility
    
    # Process in chunks
    chunk_size = 10000
    batch_size = 64  # Increased batch size
    num_chunks = (len(split_data) + chunk_size - 1) // chunk_size
    
    for chunk_idx in range(start_chunk, num_chunks):
        start_idx = chunk_idx * chunk_size
        end_idx = min((chunk_idx + 1) * chunk_size, len(split_data))
        
        logger.info(f"  Processing chunk {chunk_idx + 1}/{num_chunks} ({start_idx}-{end_idx})...")
        chunk_data = split_data.select(range(start_idx, end_idx))
        
        # Process with batches
        num_batches = (len(chunk_data) + batch_size - 1) // batch_size
        
        for i in tqdm(range(num_batches), desc=f"Chunk {chunk_idx + 1}"):
            batch_start = i * batch_size
            batch_end = min((i + 1) * batch_size, len(chunk_data))
            
            # Get batch
            batch = chunk_data[batch_start:batch_end]
            
            # Process batch
            batch_processed = process_batch(batch, teacher_model)
            processed_examples.extend(batch_processed)
        
        # Save checkpoint after each chunk
        save_checkpoint(checkpoint_file, processed_examples, chunk_idx, split_name)
        logger.info(f"  Checkpoint saved at chunk {chunk_idx + 1}, total {len(processed_examples)} examples")
        
        # Periodic garbage collection
        if chunk_idx % 5 == 0:
            gc.collect()
            torch.cuda.empty_cache() if torch.cuda.is_available() else None
    
    # Create dataset from processed examples
    dataset = Dataset.from_list(processed_examples)
    logger.info(f"Processed {split_name}: {len(processed_examples)} examples")
    
    # Check pruning label distribution
    all_labels = []
    all_ranking_labels = []
    sample_size = min(1000, len(processed_examples))
    for example in processed_examples[:sample_size]:
        # Flatten all pruning labels from all 5 texts
        for pruning_labels in example['pruning_labels']:
            all_labels.extend(pruning_labels)
        all_ranking_labels.extend(example['ranking_labels'])
    
    if all_labels:
        keep_ratio = sum(all_labels) / len(all_labels)
        logger.info(f"  {split_name} pruning distribution (sample): {keep_ratio:.1%} keep, {(1-keep_ratio):.1%} prune")
    
    if all_ranking_labels:
        positive_ratio = sum(all_ranking_labels) / len(all_ranking_labels)
        logger.info(f"  {split_name} ranking distribution (sample): {positive_ratio:.1%} positive, {(1-positive_ratio):.1%} negative")
    
    # Remove checkpoint file after successful completion
    if checkpoint_file.exists():
        checkpoint_file.unlink()
        logger.info(f"  Removed checkpoint file for {split_name}")
    
    return dataset

def main():
    logger.info("Processing full 1.3M dataset with improved parallel processing...")
    
    # Paths
    output_path = "tmp/datasets/dev-dataset/full-batched"
    checkpoint_dir = Path("tmp/datasets/dev-dataset/full-batched-checkpoint")
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    
    # Load full dataset
    logger.info("Loading full dataset from HuggingFace...")
    dataset = load_dataset("hotchpotch/wip-query-context-pruner")
    
    logger.info(f"Dataset sizes:")
    for split, data in dataset.items():
        logger.info(f"  {split}: {len(data)} examples")
    
    # Load teacher model
    logger.info("Loading teacher model...")
    teacher_model = CrossEncoder("hotchpotch/japanese-reranker-xsmall-v2")
    
    # Process each split
    processed_splits = {}
    
    for split_name, split_data in dataset.items():
        processed_dataset = process_dataset_split(
            split_name, split_data, teacher_model, output_path, checkpoint_dir
        )
        processed_splits[split_name] = processed_dataset
    
    # Create final dataset
    processed_dataset = DatasetDict(processed_splits)
    
    # Save
    output_path = Path(output_path)
    output_path.mkdir(parents=True, exist_ok=True)
    
    logger.info(f"Saving processed dataset to {output_path}")
    processed_dataset.save_to_disk(str(output_path))
    
    # Verify
    logger.info("Verifying processed dataset...")
    for split_name, split_data in processed_dataset.items():
        logger.info(f"  {split_name}: {len(split_data)} examples")
        
        if len(split_data) > 0:
            sample = split_data[0]
            logger.info(f"  Sample keys: {list(sample.keys())}")
            logger.info(f"  Sample query: {sample['query'][:100]}...")
            logger.info(f"  Number of texts: {len(sample['texts'])}")
            logger.info(f"  Number of labels: {len(sample['ranking_labels'])}")
            logger.info(f"  Sample teacher scores: {[f'{s:.3f}' for s in sample['teacher_scores'][:3]]}...")
    
    logger.info("Full dataset processing completed!")

if __name__ == "__main__":
    main()