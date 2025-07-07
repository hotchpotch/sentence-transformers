#!/usr/bin/env python3
"""
Test trained ProvenceEncoder on test dataset.
"""

import logging
import numpy as np
from pathlib import Path
from datasets import load_from_disk
import torch

from sentence_transformers.provence import ProvenceEncoder

# Setup logging
logging.basicConfig(
    format='%(asctime)s - %(levelname)s - %(name)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S',
    level=logging.INFO
)
logger = logging.getLogger(__name__)


def evaluate_reranker(model: ProvenceEncoder, test_dataset, num_samples: int = 500):
    """Evaluate reranking performance."""
    logger.info("Evaluating reranker performance...")
    
    # Sample test data
    test_sample = test_dataset.select(range(min(num_samples, len(test_dataset))))
    
    queries = []
    documents = []
    true_labels = []
    teacher_scores = []
    
    for example in test_sample:
        queries.append(example['query'])
        documents.append(example.get('text', example.get('document', '')))
        true_labels.append(example.get('ranking_label', example.get('label', 0)))
        teacher_scores.append(example.get('teacher_score', 0.0))
    
    # Get predictions
    pairs = list(zip(queries, documents))
    predicted_scores = model.predict(pairs, batch_size=64, show_progress_bar=True)
    
    # Calculate metrics
    teacher_scores = np.array(teacher_scores)
    predicted_scores = np.array(predicted_scores)
    true_labels = np.array(true_labels)
    
    # Correlation with teacher scores
    teacher_corr = np.corrcoef(teacher_scores, predicted_scores)[0, 1]
    
    # Simple accuracy for binary classification
    if len(np.unique(true_labels)) <= 2:
        # Binary classification
        threshold = 0.5
        pred_binary = (predicted_scores > threshold).astype(int)
        accuracy = np.mean(pred_binary == true_labels)
    else:
        accuracy = None
    
    logger.info(f"Reranker Results:")
    logger.info(f"  Correlation with teacher: {teacher_corr:.4f}")
    if accuracy is not None:
        logger.info(f"  Binary accuracy: {accuracy:.4f}")
    logger.info(f"  Score range: [{predicted_scores.min():.4f}, {predicted_scores.max():.4f}]")
    logger.info(f"  Mean score: {predicted_scores.mean():.4f}")
    
    return {
        'teacher_correlation': teacher_corr,
        'accuracy': accuracy,
        'predicted_scores': predicted_scores,
        'teacher_scores': teacher_scores
    }


def evaluate_pruning(model: ProvenceEncoder, test_dataset, num_samples: int = 100):
    """Evaluate pruning performance."""
    logger.info("Evaluating pruning performance...")
    
    # Sample test data
    test_sample = test_dataset.select(range(min(num_samples, len(test_dataset))))
    
    total_compression = 0
    total_examples = 0
    
    logger.info("Sample pruning results:")
    
    for i, example in enumerate(test_sample):
        if i >= 5:  # Show only first 5 examples
            break
            
        query = example['query']
        document = example.get('text', example.get('document', ''))
        
        # Get pruning results
        result = model.predict_with_pruning(
            (query, document),
            pruning_threshold=0.5,
            return_documents=True
        )
        
        logger.info(f"\nExample {i+1}:")
        logger.info(f"  Query: {query[:80]}...")
        logger.info(f"  Original length: {len(document)} chars")
        logger.info(f"  Ranking score: {result.ranking_scores:.3f}")
        logger.info(f"  Sentences: {len(result.sentences[0])} total")
        logger.info(f"  Pruned: {result.num_pruned_sentences} sentences")
        logger.info(f"  Compression: {result.compression_ratio:.2%}")
        
        if result.pruned_documents:
            logger.info(f"  Pruned length: {len(result.pruned_documents[0])} chars")
    
    # Test various thresholds
    logger.info("\nTesting different pruning thresholds:")
    thresholds = [0.3, 0.5, 0.7]
    
    for threshold in thresholds:
        compressions = []
        for example in test_sample.select(range(min(50, len(test_sample)))):
            query = example['query']
            document = example.get('text', example.get('document', ''))
            
            result = model.predict_with_pruning(
                (query, document),
                pruning_threshold=threshold
            )
            compressions.append(result.compression_ratio)
        
        avg_compression = np.mean(compressions)
        logger.info(f"  Threshold {threshold}: {avg_compression:.2%} average compression")
    
    return {
        'avg_compressions': compressions,
        'thresholds_tested': thresholds
    }


def main():
    logger.info("=" * 70)
    logger.info("Testing Trained ProvenceEncoder")
    logger.info("=" * 70)
    
    # Load test dataset
    dataset_path = "tmp/datasets/dev-dataset/minimal"
    logger.info(f"Loading test dataset from {dataset_path}...")
    dataset = load_from_disk(dataset_path)
    test_dataset = dataset['test']
    logger.info(f"Test dataset size: {len(test_dataset)} examples")
    
    # Load trained model
    model_path = "tmp/models/provence-minimal/final"
    if not Path(model_path).exists():
        # Try best checkpoint
        model_path = "tmp/models/provence-minimal/checkpoint-400-best"
        if not Path(model_path).exists():
            # Find latest checkpoint
            checkpoints = list(Path("tmp/models/provence-minimal").glob("checkpoint-*-best"))
            if checkpoints:
                model_path = str(max(checkpoints, key=lambda x: int(x.name.split('-')[1])))
            else:
                logger.error("No trained model found!")
                return
    
    logger.info(f"Loading model from {model_path}...")
    model = ProvenceEncoder.from_pretrained(model_path)
    model.eval()
    
    logger.info(f"Model device: {model.device}")
    logger.info(f"Model loaded successfully!")
    
    # Test both functionalities
    logger.info("\n" + "="*50)
    reranker_results = evaluate_reranker(model, test_dataset)
    
    logger.info("\n" + "="*50)
    pruning_results = evaluate_pruning(model, test_dataset)
    
    # Overall assessment
    logger.info("\n" + "="*70)
    logger.info("OVERALL ASSESSMENT")
    logger.info("="*70)
    
    # Reranker assessment
    teacher_corr = reranker_results['teacher_correlation']
    if teacher_corr > 0.7:
        reranker_status = "✓ GOOD"
    elif teacher_corr > 0.5:
        reranker_status = "⚠ MODERATE"
    else:
        reranker_status = "✗ POOR"
    
    logger.info(f"Reranker: {reranker_status} (correlation: {teacher_corr:.3f})")
    
    # Pruning assessment
    if pruning_results['avg_compressions']:
        avg_compression = np.mean(pruning_results['avg_compressions'])
        if 0.1 <= avg_compression <= 0.5:  # 10-50% compression is reasonable
            pruning_status = "✓ GOOD"
        elif avg_compression > 0.8:
            pruning_status = "⚠ TOO AGGRESSIVE"
        elif avg_compression < 0.05:
            pruning_status = "⚠ TOO CONSERVATIVE"
        else:
            pruning_status = "✓ MODERATE"
        
        logger.info(f"Pruning: {pruning_status} (avg compression: {avg_compression:.2%})")
    
    logger.info("\n" + "="*70)
    logger.info("Testing completed!")
    logger.info("="*70)


if __name__ == "__main__":
    main()