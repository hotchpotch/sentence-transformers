#!/usr/bin/env python3
"""
Process 20k subset of the 100k dataset with teacher scores.
"""

import logging
import random
import torch
from pathlib import Path
from datasets import load_from_disk, Dataset, DatasetDict
from tqdm import tqdm
from sentence_transformers import CrossEncoder
from sentence_transformers.utils.text_chunking import MultilingualChunker

# Setup logging
logging.basicConfig(
    format='%(asctime)s - %(levelname)s - %(name)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S',
    level=logging.INFO
)
logger = logging.getLogger(__name__)

def load_teacher_model():
    """Load teacher reranker model."""
    logger.info("Loading teacher model: hotchpotch/japanese-reranker-xsmall-v2")
    model = CrossEncoder("hotchpotch/japanese-reranker-xsmall-v2")
    return model

def process_batch(examples, teacher_model, text_chunker, batch_size=16):
    """Process a batch of examples efficiently."""
    processed_examples = []
    
    # Collect all query-text pairs for batched inference
    all_pairs = []
    pair_to_example = []
    
    for example in examples:
        query = example['query']
        texts = example['texts']
        
        # Select one text randomly
        text_idx = random.randint(0, len(texts) - 1)
        selected_text = texts[text_idx]
        selected_label = example['labels'][text_idx]
        
        all_pairs.append([query, selected_text])
        pair_to_example.append((example, selected_text, selected_label))
    
    # Batched teacher score inference
    with torch.no_grad():
        teacher_scores = teacher_model.predict(all_pairs)
    
    # Process each example with its teacher score
    for i, (example, selected_text, selected_label) in enumerate(pair_to_example):
        teacher_score = teacher_scores[i]
        query = example['query']
        
        # Chunk text into sentences
        chunks_result = text_chunker.chunk_text(selected_text, language="auto")
        sentences = [chunk for chunk, _ in chunks_result]
        
        # Create character boundaries
        sentence_boundaries = []
        current_pos = 0
        for sentence in sentences:
            start_pos = selected_text.find(sentence, current_pos)
            if start_pos != -1:
                end_pos = start_pos + len(sentence)
                sentence_boundaries.append([start_pos, end_pos])
                current_pos = end_pos
            else:
                sentence_boundaries.append([current_pos, current_pos])
        
        # Generate pruning labels based on relevance
        # More balanced approach for better training
        pruning_labels = []
        for j, sentence in enumerate(sentences):
            # Base probability depends on teacher score
            if teacher_score > 0.5:
                keep_prob = 0.7  # High relevance documents
            elif teacher_score > 0.0:
                keep_prob = 0.4  # Medium relevance
            else:
                keep_prob = 0.1  # Low relevance
            
            # Boost if contains query words
            query_words = set(query.lower().split())
            sentence_words = set(sentence.lower().split())
            overlap_ratio = len(query_words & sentence_words) / max(len(query_words), 1)
            
            if overlap_ratio > 0.3:
                keep_prob = min(keep_prob + 0.3, 0.95)
            elif overlap_ratio > 0.1:
                keep_prob = min(keep_prob + 0.15, 0.95)
            
            # Position-based adjustment (first and last sentences are often important)
            if j == 0 or j == len(sentences) - 1:
                keep_prob = min(keep_prob + 0.1, 0.95)
            
            # Random decision with probability
            keep = random.random() < keep_prob
            pruning_labels.append(1 if keep else 0)
        
        processed_example = {
            'query': query,
            'text': selected_text,
            'ranking_label': selected_label,
            'teacher_score': float(teacher_score),
            'pruning_labels': pruning_labels,
            'sentence_boundaries': sentence_boundaries,
            'dataset_name': example.get('dataset_name', 'unknown'),
            'example_id': example.get('id', ''),
        }
        processed_examples.append(processed_example)
    
    return processed_examples

def main():
    logger.info("Processing 20k subset of 100k dataset...")
    
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
    teacher_model = load_teacher_model()
    
    # Initialize text chunker
    text_chunker = MultilingualChunker()
    
    # Process each split with batching
    processed_splits = {}
    batch_size = 64  # Larger batch size for efficiency
    
    for split_name, split_data in new_splits.items():
        logger.info(f"Processing {split_name} split ({len(split_data)} examples)...")
        
        processed_examples = []
        random.seed(42)  # For reproducible results
        
        # Process in batches
        for i in tqdm(range(0, len(split_data), batch_size), desc=f"Processing {split_name} batches"):
            batch_end = min(i + batch_size, len(split_data))
            batch_examples = [split_data[j] for j in range(i, batch_end)]
            
            try:
                batch_processed = process_batch(batch_examples, teacher_model, text_chunker)
                processed_examples.extend(batch_processed)
            except Exception as e:
                logger.warning(f"Failed to process batch {i//batch_size}: {e}")
                continue
        
        processed_splits[split_name] = Dataset.from_list(processed_examples)
        logger.info(f"Processed {split_name}: {len(processed_examples)} examples")
        
        # Check pruning label distribution
        all_labels = []
        for example in processed_examples[:1000]:  # Check first 1000
            all_labels.extend(example['pruning_labels'])
        
        if all_labels:
            keep_count = sum(all_labels)
            total_count = len(all_labels)
            keep_ratio = keep_count / total_count if total_count > 0 else 0
            
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