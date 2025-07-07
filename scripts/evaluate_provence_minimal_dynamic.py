#!/usr/bin/env python3
"""
Evaluate Provence model trained on minimal dataset with dynamic labels.
Test compression rates on test positive samples.
"""

import logging
from pathlib import Path
import torch
from torch.utils.data import DataLoader
from sentence_transformers.provence.encoder import ProvenceEncoder
from sentence_transformers.provence.data_collator_chunk_based import ProvenceChunkBasedDataCollator
from transformers import AutoTokenizer
from datasets import load_from_disk
import numpy as np
from collections import defaultdict
import json

# Setup logging
logging.basicConfig(
    format='%(asctime)s - %(levelname)s - %(name)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S',
    level=logging.INFO
)
logger = logging.getLogger(__name__)


def evaluate_model(model, dataloader, device, threshold=0.5):
    """Evaluate model on dataset."""
    model.eval()
    
    results = defaultdict(list)
    total_tokens = 0
    kept_tokens = 0
    
    with torch.no_grad():
        for batch_idx, batch in enumerate(dataloader):
            # Debug: check batch structure
            if batch_idx == 0:
                logger.info(f"Batch type: {type(batch)}")
                if isinstance(batch, list):
                    logger.info(f"Batch length: {len(batch)}")
                    for i, item in enumerate(batch):
                        logger.info(f"  Item {i} type: {type(item)}")
                        if isinstance(item, dict):
                            logger.info(f"  Item {i} keys: {list(item.keys())}")
                elif isinstance(batch, dict):
                    logger.info(f"Batch keys: {list(batch.keys())}")
            
            # The collator returns a dict with 'sentence_features' and 'labels'
            if isinstance(batch, dict) and 'sentence_features' in batch:
                features = batch['sentence_features'][0] if isinstance(batch['sentence_features'], list) else batch['sentence_features']
                labels = batch['labels']
            elif isinstance(batch, tuple) and len(batch) == 2:
                features = batch[0]
                labels = batch[1]
            else:
                raise ValueError(f"Unexpected batch format: {type(batch)}")
            
            # Move to device
            input_ids = features['input_ids'].to(device)
            attention_mask = features['attention_mask'].to(device)
            
            # Get pruning labels
            pruning_labels = labels['pruning_labels'].to(device)
            
            # Forward pass
            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask
            )
            
            # Get pruning predictions
            pruning_logits = outputs['pruning_logits']
            pruning_probs = torch.softmax(pruning_logits, dim=-1)
            keep_probs = pruning_probs[:, :, 1]  # Probability of keeping
            
            # Calculate metrics for each example
            for i in range(input_ids.shape[0]):
                mask = attention_mask[i] == 1
                active_probs = keep_probs[i][mask]
                active_labels = pruning_labels[i][mask]
                
                # Binary predictions
                predictions = (active_probs > threshold).float()
                
                # Calculate accuracy
                accuracy = (predictions == active_labels).float().mean().item()
                
                # Calculate compression ratio
                num_tokens = mask.sum().item()
                num_kept = predictions.sum().item()
                compression_ratio = 1.0 - (num_kept / num_tokens) if num_tokens > 0 else 0.0
                
                # Track totals
                total_tokens += num_tokens
                kept_tokens += num_kept
                
                # Store results
                results['accuracy'].append(accuracy)
                results['compression_ratio'].append(compression_ratio)
                results['num_tokens'].append(num_tokens)
                results['num_kept'].append(num_kept)
                
                # Log example details for first few batches
                if batch_idx < 3 and i == 0:
                    logger.info(f"Example {batch_idx}-{i}:")
                    logger.info(f"  Tokens: {num_tokens}, Kept: {num_kept}")
                    logger.info(f"  Accuracy: {accuracy:.3f}")
                    logger.info(f"  Compression: {compression_ratio:.1%}")
                    
                    # Find document boundaries
                    sep_token_id = model.tokenizer.sep_token_id
                    sep_positions = (input_ids[i] == sep_token_id).nonzero(as_tuple=True)[0]
                    if len(sep_positions) >= 3:
                        doc_start = sep_positions[1].item() + 1
                        doc_end = sep_positions[2].item()
                        doc_labels = active_labels[doc_start:doc_end]
                        doc_preds = predictions[doc_start:doc_end]
                        logger.info(f"  Document tokens: {len(doc_labels)}")
                        logger.info(f"  Document accuracy: {(doc_preds == doc_labels).float().mean().item():.3f}")
    
    return results


def main():
    logger.info("Starting evaluation of Provence model on minimal dataset...")
    
    # Paths
    model_path = "outputs/provence-minimal-dynamic/checkpoint-750-best"
    dataset_path = "tmp/datasets/dev-dataset/minimal-5k-simple"
    
    # Load dataset
    logger.info(f"Loading dataset from {dataset_path}")
    dataset = load_from_disk(dataset_path)
    test_dataset = dataset['test']
    
    # Filter for positive examples only
    logger.info("Filtering for positive examples...")
    positive_indices = []
    for idx in range(len(test_dataset)):
        example = test_dataset[idx]
        # Check if any text has positive label
        if any(example['ranking_labels']):
            positive_indices.append(idx)
    
    positive_dataset = test_dataset.select(positive_indices)
    logger.info(f"Found {len(positive_dataset)} positive examples out of {len(test_dataset)} total")
    
    # Load model
    logger.info(f"Loading model from {model_path}")
    model = ProvenceEncoder.from_pretrained(model_path)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    logger.info(f"Model loaded on {device}")
    
    # Initialize tokenizer
    tokenizer = model.tokenizer
    
    # Create data collator
    logger.info("Creating chunk-based data collator...")
    data_collator = ProvenceChunkBasedDataCollator(
        tokenizer=tokenizer,
        max_length=512,
        padding=True,
        truncation=True,
        mini_batch_size=64
    )
    
    # Create dataloader
    dataloader = DataLoader(
        positive_dataset,
        batch_size=8,
        collate_fn=data_collator,
        shuffle=False
    )
    
    # Evaluate at different thresholds
    thresholds = [0.1, 0.3, 0.5, 0.7, 0.9]
    all_results = {}
    
    for threshold in thresholds:
        logger.info(f"\nEvaluating at threshold {threshold}...")
        results = evaluate_model(model, dataloader, device, threshold)
        
        # Calculate overall metrics
        avg_accuracy = np.mean(results['accuracy'])
        avg_compression = np.mean(results['compression_ratio'])
        total_compression = 1.0 - (sum(results['num_kept']) / sum(results['num_tokens']))
        
        logger.info(f"Threshold {threshold} Results:")
        logger.info(f"  Average accuracy: {avg_accuracy:.3f}")
        logger.info(f"  Average compression ratio: {avg_compression:.1%}")
        logger.info(f"  Overall compression ratio: {total_compression:.1%}")
        
        all_results[f'threshold_{threshold}'] = {
            'avg_accuracy': float(avg_accuracy),
            'avg_compression_ratio': float(avg_compression),
            'overall_compression_ratio': float(total_compression),
            'num_examples': len(results['accuracy'])
        }
    
    # Save results
    output_path = "outputs/provence-minimal-dynamic/evaluation_results.json"
    with open(output_path, 'w') as f:
        json.dump(all_results, f, indent=2)
    logger.info(f"\nResults saved to {output_path}")
    
    # Find best threshold based on accuracy
    best_threshold = max(thresholds, key=lambda t: all_results[f'threshold_{t}']['avg_accuracy'])
    logger.info(f"\nBest threshold based on accuracy: {best_threshold}")
    logger.info(f"  Accuracy: {all_results[f'threshold_{best_threshold}']['avg_accuracy']:.3f}")
    logger.info(f"  Compression: {all_results[f'threshold_{best_threshold}']['overall_compression_ratio']:.1%}")


if __name__ == "__main__":
    main()