#!/usr/bin/env python3
"""
Evaluate Provence model with detailed metrics including precision/recall.
Also evaluate on positive texts only (texts[0]).
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
from sklearn.metrics import precision_recall_fscore_support, confusion_matrix

# Setup logging
logging.basicConfig(
    format='%(asctime)s - %(levelname)s - %(name)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S',
    level=logging.INFO
)
logger = logging.getLogger(__name__)


def evaluate_model_detailed(model, dataloader, device, threshold=0.5):
    """Evaluate model with detailed metrics."""
    model.eval()
    
    all_predictions = []
    all_labels = []
    results = defaultdict(list)
    
    with torch.no_grad():
        for batch_idx, batch in enumerate(dataloader):
            # The collator returns a dict with 'sentence_features' and 'labels'
            if isinstance(batch, dict) and 'sentence_features' in batch:
                features = batch['sentence_features'][0] if isinstance(batch['sentence_features'], list) else batch['sentence_features']
                labels = batch['labels']
            else:
                raise ValueError(f"Unexpected batch format: {type(batch)}")
            
            # Move to device
            input_ids = features['input_ids'].to(device)
            attention_mask = features['attention_mask'].to(device)
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
                
                # Store for overall metrics
                all_predictions.extend(predictions.cpu().numpy())
                all_labels.extend(active_labels.cpu().numpy())
                
                # Calculate per-example metrics
                accuracy = (predictions == active_labels).float().mean().item()
                num_tokens = mask.sum().item()
                num_kept = predictions.sum().item()
                compression_ratio = 1.0 - (num_kept / num_tokens) if num_tokens > 0 else 0.0
                
                results['accuracy'].append(accuracy)
                results['compression_ratio'].append(compression_ratio)
                results['num_tokens'].append(num_tokens)
                results['num_kept'].append(num_kept)
    
    # Calculate overall precision, recall, F1
    all_predictions = np.array(all_predictions)
    all_labels = np.array(all_labels)
    
    precision, recall, f1, support = precision_recall_fscore_support(
        all_labels, all_predictions, average='binary', pos_label=1
    )
    
    # Confusion matrix
    tn, fp, fn, tp = confusion_matrix(all_labels, all_predictions).ravel()
    
    # Calculate metrics
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    
    return {
        'accuracy': np.mean(results['accuracy']),
        'compression_ratio': np.mean(results['compression_ratio']),
        'overall_compression': 1.0 - (sum(results['num_kept']) / sum(results['num_tokens'])),
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'specificity': specificity,
        'confusion_matrix': {
            'true_negatives': int(tn),
            'false_positives': int(fp),
            'false_negatives': int(fn),
            'true_positives': int(tp)
        },
        'label_distribution': {
            'num_0s': int((all_labels == 0).sum()),
            'num_1s': int((all_labels == 1).sum()),
            'ratio_1s': float((all_labels == 1).mean())
        }
    }


def evaluate_positive_texts_only(model, dataset, tokenizer, device, threshold=0.5):
    """Evaluate only on positive texts (texts[0])."""
    model.eval()
    
    # Create data collator
    data_collator = ProvenceChunkBasedDataCollator(
        tokenizer=tokenizer,
        max_length=512,
        padding=True,
        truncation=True,
        mini_batch_size=64
    )
    
    # Process only positive texts
    positive_examples = []
    for idx in range(len(dataset)):
        example = dataset[idx]
        # Keep only the first text (positive)
        positive_example = {
            'query': example['query'],
            'texts': [example['texts'][0]],  # Only positive text
            'ranking_labels': [1],  # Always positive
            'teacher_scores': [example['teacher_scores'][0]],
            'chunks_pos': [example['chunks_pos'][0]],
            'relevant_chunks': [example['relevant_chunks'][0]],
            'dataset_name': example['dataset_name'],
            'example_id': example['example_id']
        }
        positive_examples.append(positive_example)
    
    # Create dataloader
    from datasets import Dataset
    positive_dataset = Dataset.from_list(positive_examples)
    dataloader = DataLoader(
        positive_dataset,
        batch_size=16,
        collate_fn=data_collator,
        shuffle=False
    )
    
    return evaluate_model_detailed(model, dataloader, device, threshold)


def main():
    logger.info("Starting detailed evaluation of Provence model...")
    
    # Paths
    model_path = "outputs/provence-minimal-dynamic/checkpoint-750-best"
    dataset_path = "tmp/datasets/dev-dataset/minimal-5k-simple"
    
    # Load dataset
    logger.info(f"Loading dataset from {dataset_path}")
    dataset = load_from_disk(dataset_path)
    test_dataset = dataset['test']
    
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
    
    # 1. Evaluate on all texts
    logger.info("\n=== Evaluating on ALL texts ===")
    dataloader = DataLoader(
        test_dataset,
        batch_size=8,
        collate_fn=data_collator,
        shuffle=False
    )
    
    thresholds = [0.1, 0.3, 0.5, 0.7, 0.9]
    all_results = {}
    
    for threshold in thresholds:
        logger.info(f"\nEvaluating at threshold {threshold}...")
        results = evaluate_model_detailed(model, dataloader, device, threshold)
        
        logger.info(f"Threshold {threshold} Results:")
        logger.info(f"  Accuracy: {results['accuracy']:.3f}")
        logger.info(f"  Precision: {results['precision']:.3f}")
        logger.info(f"  Recall: {results['recall']:.3f}")
        logger.info(f"  F1 Score: {results['f1']:.3f}")
        logger.info(f"  Specificity: {results['specificity']:.3f}")
        logger.info(f"  Compression ratio: {results['overall_compression']:.1%}")
        logger.info(f"  Label distribution: {results['label_distribution']['ratio_1s']:.1%} are 1s")
        logger.info(f"  Confusion matrix: TP={results['confusion_matrix']['true_positives']}, "
                   f"FP={results['confusion_matrix']['false_positives']}, "
                   f"TN={results['confusion_matrix']['true_negatives']}, "
                   f"FN={results['confusion_matrix']['false_negatives']}")
        
        all_results[f'all_texts_threshold_{threshold}'] = results
    
    # 2. Evaluate on positive texts only
    logger.info("\n=== Evaluating on POSITIVE texts only (texts[0]) ===")
    
    for threshold in thresholds:
        logger.info(f"\nEvaluating positive texts at threshold {threshold}...")
        results = evaluate_positive_texts_only(model, test_dataset, tokenizer, device, threshold)
        
        logger.info(f"Threshold {threshold} Results (Positive texts only):")
        logger.info(f"  Accuracy: {results['accuracy']:.3f}")
        logger.info(f"  Precision: {results['precision']:.3f}")
        logger.info(f"  Recall: {results['recall']:.3f}")
        logger.info(f"  F1 Score: {results['f1']:.3f}")
        logger.info(f"  Specificity: {results['specificity']:.3f}")
        logger.info(f"  Compression ratio: {results['overall_compression']:.1%}")
        logger.info(f"  Label distribution: {results['label_distribution']['ratio_1s']:.1%} are 1s")
        
        all_results[f'positive_texts_threshold_{threshold}'] = results
    
    # Save results
    output_path = "outputs/provence-minimal-dynamic/detailed_evaluation_results.json"
    with open(output_path, 'w') as f:
        json.dump(all_results, f, indent=2)
    logger.info(f"\nResults saved to {output_path}")


if __name__ == "__main__":
    main()