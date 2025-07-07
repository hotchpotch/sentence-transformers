#!/usr/bin/env python3
"""
Process 20k dataset with proper batching for all 5 texts per query.
Following the pattern from LambdaLoss.
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
    
    # Process each example in the batch
    for i in range(batch_size):
        query = batch['query'][i]
        texts = batch['texts'][i]  # List of 5 texts
        labels = batch['labels'][i]  # List of 5 labels
        chunks_pos = batch['chunks_pos'][i]  # List of 5 chunk positions
        
        # Prepare for teacher scoring - all 5 texts
        all_pairs = [[query, text] for text in texts]
        
        # Get teacher scores for all texts
        with torch.no_grad():
            teacher_scores = teacher_model.predict(all_pairs, batch_size=32)
        
        # Process all texts
        all_pruning_labels = []
        all_sentence_boundaries = []
        
        for j, text in enumerate(texts):
            teacher_score = float(teacher_scores[j])
            chunks = chunks_pos[j]  # Pre-computed sentence boundaries
            num_sentences = len(chunks)
            
            # Generate pruning labels for each sentence
            pruning_labels = []
            for k in range(num_sentences):
                # Base probability depends on teacher score and original label
                if labels[j] == 1:  # Relevant document
                    if teacher_score > 0.5:
                        keep_prob = 0.8
                    else:
                        keep_prob = 0.6
                else:  # Non-relevant document
                    if teacher_score > 0.5:
                        keep_prob = 0.4
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
            'teacher_scores': teacher_scores.tolist(),  # All 5 teacher scores
            'pruning_labels': all_pruning_labels,  # List of 5 pruning label lists
            'sentence_boundaries': all_sentence_boundaries,  # List of 5 boundary lists
            'dataset_name': batch['dataset_name'][i],
            'example_id': batch['id'][i],
        }
        processed_examples.append(processed_example)
    
    return processed_examples

def main():
    logger.info("Processing 20k dataset with batched approach...")
    
    # Paths
    input_path = "tmp/datasets/dev-dataset/small-100k"
    output_path = "tmp/datasets/dev-dataset/small-20k-batched"
    target_size = 20000  # 20k total
    
    # Load dataset
    logger.info(f"Loading dataset from {input_path}")
    dataset = load_from_disk(input_path)
    
    # Create 20k subset
    new_splits = {}
    
    # Take first 20k from train, maintaining split ratios
    train_size = int(0.8 * target_size)  # 16k
    val_size = int(0.1 * target_size)    # 2k  
    test_size = int(0.1 * target_size)   # 2k
    
    new_splits['train'] = dataset['train'].select(range(train_size))
    new_splits['validation'] = dataset['validation'].select(range(val_size))
    new_splits['test'] = dataset['test'].select(range(test_size))
    
    logger.info(f"Subset sizes:")
    for split, data in new_splits.items():
        logger.info(f"  {split}: {len(data)} examples")
    
    # Load teacher model
    logger.info("Loading teacher model...")
    teacher_model = CrossEncoder("hotchpotch/japanese-reranker-xsmall-v2")
    
    # Process each split
    processed_splits = {}
    batch_size = 64  # Process 64 queries at a time
    
    for split_name, split_data in new_splits.items():
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
        for example in processed_examples[:200]:  # Check first 200
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
            logger.info(f"  Sample teacher scores: {[f'{s:.3f}' for s in sample['teacher_scores']]}")
    
    logger.info("20k batched dataset processing completed!")

if __name__ == "__main__":
    main()