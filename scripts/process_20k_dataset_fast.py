#!/usr/bin/env python3
"""
Fast processing of 20k subset with teacher scores.
Uses pre-chunked data from the dataset.
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
    """Process a batch of examples efficiently."""
    batch_size = len(batch['id'])
    processed_examples = []
    
    # Prepare all query-text pairs for batch inference
    all_pairs = []
    metadata = []
    
    for i in range(batch_size):
        query = batch['query'][i]
        texts = batch['texts'][i]
        labels = batch['labels'][i]
        chunks_pos = batch['chunks_pos'][i]
        
        # Select one text randomly
        text_idx = random.randint(0, len(texts) - 1)
        selected_text = texts[text_idx]
        selected_label = labels[text_idx]
        selected_chunks = chunks_pos[text_idx]
        
        all_pairs.append([query, selected_text])
        metadata.append({
            'query': query,
            'text': selected_text,
            'label': selected_label,
            'chunks': selected_chunks,
            'dataset_name': batch['dataset_name'][i],
            'example_id': batch['id'][i],
        })
    
    # Batch inference for teacher scores
    with torch.no_grad():
        teacher_scores = teacher_model.predict(all_pairs, batch_size=32)
    
    # Process each example
    for i, meta in enumerate(metadata):
        teacher_score = float(teacher_scores[i])
        
        # Generate pruning labels for each sentence/chunk
        num_sentences = len(meta['chunks'])
        pruning_labels = []
        
        for j in range(num_sentences):
            # Base probability depends on teacher score
            if teacher_score > 0.5:
                keep_prob = 0.7  # High relevance documents
            elif teacher_score > 0.0:
                keep_prob = 0.4  # Medium relevance
            else:
                keep_prob = 0.1  # Low relevance
            
            # Boost probability for first and last sentences
            if j == 0 or j == num_sentences - 1:
                keep_prob = min(keep_prob + 0.1, 0.95)
            
            # Random decision
            keep = random.random() < keep_prob
            pruning_labels.append(1 if keep else 0)
        
        processed_example = {
            'query': meta['query'],
            'text': meta['text'],
            'ranking_label': meta['label'],
            'teacher_score': teacher_score,
            'pruning_labels': pruning_labels,
            'sentence_boundaries': meta['chunks'],  # Use pre-computed chunks
            'dataset_name': meta['dataset_name'],
            'example_id': meta['example_id'],
        }
        processed_examples.append(processed_example)
    
    return processed_examples

def main():
    logger.info("Fast processing of 20k subset...")
    
    # Paths
    input_path = "tmp/datasets/dev-dataset/small-100k"
    output_path = "tmp/datasets/dev-dataset/small-20k-processed"
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
    batch_size = 128  # Larger batch for efficiency
    
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
        for example in processed_examples[:1000]:  # Check first 1000
            all_labels.extend(example['pruning_labels'])
        
        if all_labels:
            keep_ratio = sum(all_labels) / len(all_labels)
            logger.info(f"  {split_name} pruning distribution: {keep_ratio:.1%} keep, {(1-keep_ratio):.1%} prune")
    
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
            logger.info(f"  Sample teacher score: {sample['teacher_score']:.3f}")
            logger.info(f"  Sample pruning labels: {sample['pruning_labels'][:10]}...")
    
    logger.info("20k dataset processing completed!")

if __name__ == "__main__":
    main()