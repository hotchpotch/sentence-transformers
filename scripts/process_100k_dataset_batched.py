#!/usr/bin/env python3
"""
Process 100k dataset with proper batching for all 5 texts per query.
Full dataset processing with batched approach.
"""

import logging
import random
import torch
from pathlib import Path
from datasets import load_from_disk, Dataset, DatasetDict
from tqdm import tqdm
from sentence_transformers import CrossEncoder

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
    logger.info("Processing 100k dataset with batched approach...")
    
    # Paths
    input_path = "tmp/datasets/dev-dataset/small-100k"
    output_path = "tmp/datasets/dev-dataset/small-100k-batched"
    
    # Load dataset
    logger.info(f"Loading dataset from {input_path}")
    dataset = load_from_disk(input_path)
    
    logger.info(f"Dataset sizes:")
    for split, data in dataset.items():
        logger.info(f"  {split}: {len(data)} examples")
    
    # Load teacher model
    logger.info("Loading teacher model...")
    teacher_model = CrossEncoder("hotchpotch/japanese-reranker-xsmall-v2")
    
    # Process each split
    processed_splits = {}
    batch_size = 64  # Process 64 queries at a time
    
    for split_name, split_data in dataset.items():
        logger.info(f"Processing {split_name} split...")
        
        processed_examples = []
        random.seed(42)  # For reproducibility
        
        # Process with dataset.map for efficiency
        num_batches = (len(split_data) + batch_size - 1) // batch_size
        
        for i in tqdm(range(num_batches), desc=f"Processing {split_name}"):
            start_idx = i * batch_size
            end_idx = min((i + 1) * batch_size, len(split_data))
            
            # Get batch
            batch = split_data[start_idx:end_idx]
            
            # Process batch
            batch_processed = process_batch(batch, teacher_model)
            processed_examples.extend(batch_processed)
        
        processed_splits[split_name] = Dataset.from_list(processed_examples)
        logger.info(f"Processed {split_name}: {len(processed_examples)} examples")
        
        # Check pruning label distribution
        all_labels = []
        all_ranking_labels = []
        for example in processed_examples[:500]:  # Check first 500
            # Flatten all pruning labels from all 5 texts
            for pruning_labels in example['pruning_labels']:
                all_labels.extend(pruning_labels)
            all_ranking_labels.extend(example['ranking_labels'])
        
        if all_labels:
            keep_ratio = sum(all_labels) / len(all_labels)
            logger.info(f"  {split_name} pruning distribution: {keep_ratio:.1%} keep, {(1-keep_ratio):.1%} prune")
        
        if all_ranking_labels:
            positive_ratio = sum(all_ranking_labels) / len(all_ranking_labels)
            logger.info(f"  {split_name} ranking distribution: {positive_ratio:.1%} positive, {(1-positive_ratio):.1%} negative")
    
    # Create processed dataset
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
    
    logger.info("100k batched dataset processing completed!")

if __name__ == "__main__":
    main()