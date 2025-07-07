#!/usr/bin/env python3
"""
Process final chunks of the dataset, starting from chunk 100.
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
        all_teacher_scores = teacher_model.predict(all_pairs, batch_size=64)
    
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

def main():
    logger.info("Processing final chunks of the dataset...")
    
    # Configuration - start from chunk 100
    START_CHUNK = 99  # Start from chunk 100 (0-indexed)
    chunk_size = 10000
    batch_size = 32
    
    # Paths
    output_dir = Path("tmp/datasets/dev-dataset/full-final")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load full dataset
    logger.info("Loading full dataset from HuggingFace...")
    dataset = load_dataset("hotchpotch/wip-query-context-pruner")
    
    # Load teacher model
    logger.info("Loading teacher model...")
    teacher_model = CrossEncoder("hotchpotch/japanese-reranker-xsmall-v2")
    
    # Process train split only
    split_data = dataset["train"]
    total_chunks = (len(split_data) + chunk_size - 1) // chunk_size
    
    logger.info(f"Total chunks: {total_chunks}")
    logger.info(f"Starting from chunk {START_CHUNK + 1}/{total_chunks}")
    
    processed_examples = []
    random.seed(42)  # For reproducibility
    
    # Process remaining chunks
    for chunk_idx in range(START_CHUNK, total_chunks):
        start_idx = chunk_idx * chunk_size
        end_idx = min((chunk_idx + 1) * chunk_size, len(split_data))
        
        logger.info(f"Processing chunk {chunk_idx + 1}/{total_chunks} ({start_idx}-{end_idx})...")
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
        
        # Save intermediate results every 10 chunks
        if (chunk_idx - START_CHUNK + 1) % 10 == 0:
            logger.info(f"Saving intermediate results after chunk {chunk_idx + 1}...")
            temp_dataset = Dataset.from_list(processed_examples)
            temp_path = output_dir / f"train_chunks_{START_CHUNK+1}_to_{chunk_idx+1}.arrow"
            temp_dataset.save_to_disk(str(temp_path))
            logger.info(f"Saved {len(processed_examples)} examples to {temp_path}")
        
        # Periodic garbage collection
        if chunk_idx % 5 == 0:
            gc.collect()
            torch.cuda.empty_cache() if torch.cuda.is_available() else None
    
    # Save final results
    logger.info(f"Processing complete! Total processed examples: {len(processed_examples)}")
    final_dataset = Dataset.from_list(processed_examples)
    final_path = output_dir / f"train_chunks_{START_CHUNK+1}_to_{total_chunks}_final.arrow"
    final_dataset.save_to_disk(str(final_path))
    logger.info(f"Saved final dataset to {final_path}")
    
    # Check distribution
    all_labels = []
    all_ranking_labels = []
    sample_size = min(1000, len(processed_examples))
    for example in processed_examples[:sample_size]:
        for pruning_labels in example['pruning_labels']:
            all_labels.extend(pruning_labels)
        all_ranking_labels.extend(example['ranking_labels'])
    
    if all_labels:
        keep_ratio = sum(all_labels) / len(all_labels)
        logger.info(f"Pruning distribution (sample): {keep_ratio:.1%} keep, {(1-keep_ratio):.1%} prune")
    
    if all_ranking_labels:
        positive_ratio = sum(all_ranking_labels) / len(all_ranking_labels)
        logger.info(f"Ranking distribution (sample): {positive_ratio:.1%} positive, {(1-positive_ratio):.1%} negative")
    
    logger.info("Full dataset processing completed!")

if __name__ == "__main__":
    main()