#!/usr/bin/env python3
"""
Process minimal dataset (5k) with only teacher scores.
No pruning labels are generated - they will be created dynamically during training.
"""

import logging
from pathlib import Path
from datasets import load_dataset, Dataset, DatasetDict
from sentence_transformers import CrossEncoder
import torch
from tqdm import tqdm

# Setup logging
logging.basicConfig(
    format='%(asctime)s - %(levelname)s - %(name)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S',
    level=logging.INFO
)
logger = logging.getLogger(__name__)


def process_batch(batch, teacher_model, device):
    """Process a batch to add teacher scores."""
    batch_size = len(batch['query'])
    processed_examples = []
    
    for i in range(batch_size):
        query = batch['query'][i]
        texts = batch['texts'][i]
        labels = batch['labels'][i]  # Changed from 'ranking_labels' to 'labels'
        
        # Create all query-text pairs
        pairs = [[query, text] for text in texts]
        
        # Get teacher scores
        with torch.no_grad():
            teacher_scores = teacher_model.predict(pairs, show_progress_bar=False)
        
        # Create processed example
        processed_example = {
            'query': query,
            'texts': texts,
            'ranking_labels': labels,  # Keep as 'ranking_labels' in output for consistency
            'teacher_scores': teacher_scores.tolist(),
            'chunks_pos': batch['chunks_pos'][i],  # Keep chunk position info
            'relevant_chunks': batch['relevant_chunks'][i],  # Keep relevant chunk info
            'dataset_name': batch['dataset_name'][i],
            'example_id': batch['id'][i],
        }
        processed_examples.append(processed_example)
    
    return processed_examples


def main():
    logger.info("Processing minimal (5k) dataset with teacher scores only...")
    
    # Paths
    output_path = "tmp/datasets/dev-dataset/minimal-5k-simple"
    
    # Load teacher model
    logger.info("Loading teacher model...")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    teacher_model = CrossEncoder('hotchpotch/japanese-reranker-xsmall-v2', device=device)
    logger.info(f"Teacher model loaded on {device}")
    
    # Load dataset
    logger.info("Loading dataset from HuggingFace...")
    dataset = load_dataset("hotchpotch/wip-query-context-pruner")
    
    # Take first 5k samples and split
    logger.info("Creating 5k subset...")
    train_data = dataset['train'].select(range(5000))
    
    # Split into train/val/test (80/10/10)
    train_size = len(train_data)
    train_end = int(0.8 * train_size)  # 4000
    val_end = int(0.9 * train_size)    # 4500
    
    splits = {
        'train': train_data.select(range(train_end)),
        'validation': train_data.select(range(train_end, val_end)),
        'test': train_data.select(range(val_end, train_size))
    }
    
    # Process each split
    processed_splits = {}
    
    for split_name, split_data in splits.items():
        logger.info(f"Processing {split_name} split ({len(split_data)} examples)...")
        
        processed_examples = []
        batch_size = 32
        
        for start_idx in tqdm(range(0, len(split_data), batch_size), desc=f"Processing {split_name}"):
            end_idx = min(start_idx + batch_size, len(split_data))
            batch = split_data[start_idx:end_idx]
            
            # Process batch
            batch_processed = process_batch(batch, teacher_model, device)
            processed_examples.extend(batch_processed)
            
            # Clear cache periodically
            if start_idx % 1000 == 0:
                torch.cuda.empty_cache() if torch.cuda.is_available() else None
        
        processed_splits[split_name] = Dataset.from_list(processed_examples)
        logger.info(f"Processed {split_name}: {len(processed_examples)} examples")
        
        # Check distribution
        all_ranking_labels = []
        all_teacher_scores = []
        sample_size = min(1000, len(processed_examples))
        
        for example in processed_examples[:sample_size]:
            all_ranking_labels.extend(example['ranking_labels'])
            all_teacher_scores.extend(example['teacher_scores'])
        
        if all_ranking_labels:
            positive_ratio = sum(all_ranking_labels) / len(all_ranking_labels)
            logger.info(f"  {split_name} ranking distribution: {positive_ratio:.1%} positive")
        
        if all_teacher_scores:
            avg_score = sum(all_teacher_scores) / len(all_teacher_scores)
            logger.info(f"  {split_name} average teacher score: {avg_score:.3f}")
    
    # Create final dataset
    processed_dataset = DatasetDict(processed_splits)
    
    # Save
    output_dir = Path(output_path)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    logger.info(f"Saving processed dataset to {output_dir}")
    processed_dataset.save_to_disk(str(output_dir))
    
    # Verify
    logger.info("Verifying saved dataset...")
    for split_name, split_data in processed_dataset.items():
        logger.info(f"  {split_name}: {len(split_data)} examples")
        
        if len(split_data) > 0:
            sample = split_data[0]
            logger.info(f"  Sample keys: {list(sample.keys())}")
            logger.info(f"  Number of texts: {len(sample['texts'])}")
            logger.info(f"  Number of teacher scores: {len(sample['teacher_scores'])}")
    
    logger.info("Minimal dataset processing completed!")


if __name__ == "__main__":
    main()