#!/usr/bin/env python3
"""
Process small dataset with teacher scores and pruning labels.
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

def process_example(example, teacher_model, text_chunker):
    """Process a single example."""
    query = example['query']
    texts = example['texts']
    labels = example['labels']
    chunks_pos = example['chunks_pos']
    relevant_chunks = example['relevant_chunks']
    
    # Select one text randomly
    text_idx = random.randint(0, len(texts) - 1)
    selected_text = texts[text_idx]
    selected_label = labels[text_idx]
    
    # Get teacher score
    with torch.no_grad():
        teacher_score = teacher_model.predict([[query, selected_text]])[0]
    
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
        keep_prob = 0.3  # Base probability
        
        # Boost if high teacher score
        if teacher_score > 0.5:
            keep_prob += 0.4
        elif teacher_score > 0.1:
            keep_prob += 0.2
        
        # Boost if contains query words (simple approach)
        query_words = set(query.lower().split())
        sentence_words = set(sentence.lower().split())
        if query_words & sentence_words:
            keep_prob += 0.3
        
        # Random decision with probability
        keep = random.random() < keep_prob
        pruning_labels.append(1 if keep else 0)
    
    return {
        'query': query,
        'text': selected_text,
        'ranking_label': selected_label,
        'teacher_score': float(teacher_score),
        'pruning_labels': pruning_labels,
        'sentence_boundaries': sentence_boundaries,
        'dataset_name': example.get('dataset_name', 'unknown'),
        'example_id': example.get('id', ''),
    }

def main():
    logger.info("Processing small dataset...")
    
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
    
    # Process each split
    processed_splits = {}
    
    for split_name, split_data in dataset.items():
        logger.info(f"Processing {split_name} split ({len(split_data)} examples)...")
        
        processed_examples = []
        random.seed(42)  # For reproducible results
        
        for example in tqdm(split_data, desc=f"Processing {split_name}"):
            try:
                processed = process_example(example, teacher_model, text_chunker)
                processed_examples.append(processed)
            except Exception as e:
                logger.warning(f"Failed to process example: {e}")
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
    
    logger.info("Small dataset processing completed!")

if __name__ == "__main__":
    main()