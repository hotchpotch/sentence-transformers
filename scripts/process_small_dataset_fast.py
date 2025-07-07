#!/usr/bin/env python3
"""
Fast processing of small dataset with batched teacher scores.
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
    logger.info(f"Computing teacher scores for {len(all_pairs)} pairs...")
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
        pruning_labels = []
        for sentence in sentences:
            # Simple heuristic: keep if contains query keywords or high teacher score
            keep_prob = 0.2  # Base probability
            
            # Boost if high teacher score
            if teacher_score > 0.3:
                keep_prob += 0.5
            elif teacher_score > 0.0:
                keep_prob += 0.3
            
            # Boost if contains query words
            query_words = set(query.lower().split())
            sentence_words = set(sentence.lower().split())
            if query_words & sentence_words:
                keep_prob += 0.3
            
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
    logger.info("Fast processing of small dataset...")
    
    # Paths
    input_path = "tmp/datasets/dev-dataset/small"
    output_path = "tmp/datasets/dev-dataset/small-processed"
    
    # Load dataset
    logger.info(f"Loading dataset from {input_path}")
    dataset = load_from_disk(input_path)
    
    # Load teacher model
    teacher_model = load_teacher_model()
    
    # Initialize text chunker
    text_chunker = MultilingualChunker()
    
    # Process each split with batching
    processed_splits = {}
    batch_size = 32  # Process in batches for efficiency
    
    for split_name, split_data in dataset.items():
        logger.info(f"Processing {split_name} split ({len(split_data)} examples)...")
        
        # Limit to smaller subset for faster processing
        if split_name == 'train':
            split_data = split_data.select(range(5000))  # Use 5k for training
        elif split_name == 'validation':
            split_data = split_data.select(range(1000))  # Use 1k for validation
        elif split_name == 'test':
            split_data = split_data.select(range(1000))   # Use 1k for test
        
        logger.info(f"Processing {len(split_data)} examples in {split_name}")
        
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
            logger.info(f"  Sample pruning labels: {sample['pruning_labels'][:5]}...")
            
            # Check pruning label distribution
            all_labels = []
            for example in split_data:
                all_labels.extend(example['pruning_labels'])
            
            keep_count = sum(all_labels)
            total_count = len(all_labels)
            keep_ratio = keep_count / total_count if total_count > 0 else 0
            
            logger.info(f"  Pruning label distribution: {keep_ratio:.1%} keep, {(1-keep_ratio):.1%} prune")
    
    logger.info("Fast small dataset processing completed!")

if __name__ == "__main__":
    main()