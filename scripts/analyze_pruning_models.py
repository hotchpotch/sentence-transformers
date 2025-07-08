#!/usr/bin/env python
"""
Compare and analyze the results from both pruning modes.
"""

import os
import sys
from pathlib import Path
import logging
import torch
import numpy as np
from datasets import load_dataset
from datetime import datetime

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from sentence_transformers.pruning import PruningEncoder
from sentence_transformers.pruning.data_structures import RerankingPruningOutput, PruningOnlyOutput

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def analyze_outputs(outputs, mode, sample_queries, sample_texts):
    """Analyze outputs from a model."""
    logger.info(f"\n{'='*60}")
    logger.info(f"Analysis for {mode} mode")
    logger.info(f"{'='*60}\n")
    
    # Calculate statistics
    compression_ratios = []
    kept_token_counts = []
    total_token_counts = []
    ranking_scores = []
    
    for i, output in enumerate(outputs):
        compression_ratios.append(output.compression_ratio)
        
        if hasattr(output, 'num_pruned_tokens'):
            # PruningOnlyOutput
            kept_tokens = len(output.sentences[0]) - output.num_pruned_tokens
            total_tokens = len(output.sentences[0])
        else:
            # RerankingPruningOutput
            kept_tokens = len(output.sentences[0]) - output.num_pruned_sentences
            total_tokens = len(output.sentences[0])
            if output.ranking_scores is not None:
                ranking_scores.append(output.ranking_scores)
        
        kept_token_counts.append(kept_tokens)
        total_token_counts.append(total_tokens)
    
    # Print statistics
    logger.info(f"Compression Statistics:")
    logger.info(f"  Average compression ratio: {np.mean(compression_ratios):.2%}")
    logger.info(f"  Min compression ratio: {np.min(compression_ratios):.2%}")
    logger.info(f"  Max compression ratio: {np.max(compression_ratios):.2%}")
    logger.info(f"  Std compression ratio: {np.std(compression_ratios):.2%}")
    
    logger.info(f"\nToken Statistics:")
    logger.info(f"  Average tokens kept: {np.mean(kept_token_counts):.1f} / {np.mean(total_token_counts):.1f}")
    logger.info(f"  Average keep ratio: {np.mean(np.array(kept_token_counts) / np.array(total_token_counts)):.2%}")
    
    if ranking_scores:
        logger.info(f"\nRanking Statistics:")
        logger.info(f"  Average ranking score: {np.mean(ranking_scores):.4f}")
        logger.info(f"  Min ranking score: {np.min(ranking_scores):.4f}")
        logger.info(f"  Max ranking score: {np.max(ranking_scores):.4f}")
    
    # Show examples
    logger.info(f"\nExample Outputs (first 3):")
    for i in range(min(3, len(outputs))):
        logger.info(f"\n--- Example {i+1} ---")
        logger.info(f"Query: {sample_queries[i][:50]}...")
        logger.info(f"Original text: {sample_texts[i][:100]}...")
        
        if hasattr(outputs[i], 'pruned_documents') and outputs[i].pruned_documents:
            logger.info(f"Pruned text: {outputs[i].pruned_documents[0][:100]}...")
        
        logger.info(f"Compression ratio: {outputs[i].compression_ratio:.2%}")
        logger.info(f"Tokens kept: {kept_token_counts[i]} / {total_token_counts[i]}")
        
        if hasattr(outputs[i], 'ranking_scores') and outputs[i].ranking_scores is not None:
            logger.info(f"Ranking score: {outputs[i].ranking_scores:.4f}")
    
    return {
        'avg_compression': np.mean(compression_ratios),
        'avg_keep_ratio': np.mean(np.array(kept_token_counts) / np.array(total_token_counts)),
        'avg_ranking_score': np.mean(ranking_scores) if ranking_scores else None
    }


def main():
    # Load test dataset
    logger.info("Loading test dataset...")
    dataset = load_dataset(
        'hotchpotch/wip-query-context-pruner-with-teacher-scores',
        'ja-minimal'
    )
    test_data = dataset['validation'].select(range(20))  # Use first 20 samples
    
    # Prepare test samples
    test_pairs = []
    queries = []
    texts = []
    
    for sample in test_data:
        query = sample['query']
        for text in sample['texts'][:1]:  # Use only first text for each query
            test_pairs.append((query, text))
            queries.append(query)
            texts.append(text)
    
    logger.info(f"Prepared {len(test_pairs)} test pairs")
    
    # Model paths
    reranking_model_path = "outputs/pruning-ja-minimal/checkpoint-412-best"
    pruning_only_model_path = "./output/pruning_only_minimal_20250709_081603/checkpoint-1200-best"
    
    # Test reranking_pruning model
    logger.info(f"\nLoading reranking_pruning model from {reranking_model_path}")
    model1 = PruningEncoder.from_pretrained(
        reranking_model_path,
        device="cuda" if torch.cuda.is_available() else "cpu"
    )
    
    logger.info("Running inference with reranking_pruning model...")
    outputs1 = model1.predict_with_pruning(
        test_pairs,
        batch_size=8,
        pruning_threshold=0.5,
        return_documents=True,
        show_progress_bar=True
    )
    
    stats1 = analyze_outputs(outputs1, "reranking_pruning", queries, texts)
    
    # Test pruning_only model
    logger.info(f"\nLoading pruning_only model from {pruning_only_model_path}")
    model2 = PruningEncoder.from_pretrained(
        pruning_only_model_path,
        device="cuda" if torch.cuda.is_available() else "cpu"
    )
    
    logger.info("Running inference with pruning_only model...")
    outputs2 = model2.predict_with_pruning(
        test_pairs,
        batch_size=8,
        pruning_threshold=0.5,
        return_documents=True,
        show_progress_bar=True
    )
    
    stats2 = analyze_outputs(outputs2, "pruning_only", queries, texts)
    
    # Compare models
    logger.info(f"\n{'='*60}")
    logger.info("Model Comparison Summary")
    logger.info(f"{'='*60}\n")
    
    logger.info(f"Average Compression Ratio:")
    logger.info(f"  Reranking+Pruning: {stats1['avg_compression']:.2%}")
    logger.info(f"  Pruning-only: {stats2['avg_compression']:.2%}")
    logger.info(f"  Difference: {abs(stats1['avg_compression'] - stats2['avg_compression']):.2%}")
    
    logger.info(f"\nAverage Keep Ratio:")
    logger.info(f"  Reranking+Pruning: {stats1['avg_keep_ratio']:.2%}")
    logger.info(f"  Pruning-only: {stats2['avg_keep_ratio']:.2%}")
    
    if stats1['avg_ranking_score'] is not None:
        logger.info(f"\nRanking Score (only for reranking+pruning):")
        logger.info(f"  Average: {stats1['avg_ranking_score']:.4f}")
    
    # Memory usage comparison
    logger.info(f"\nModel Size Comparison:")
    model1_params = sum(p.numel() for p in model1.parameters())
    model2_params = sum(p.numel() for p in model2.parameters())
    
    logger.info(f"  Reranking+Pruning: {model1_params:,} parameters")
    logger.info(f"  Pruning-only: {model2_params:,} parameters")
    logger.info(f"  Size ratio: {model1_params / model2_params:.2f}x")
    
    # Speed comparison (rough estimate)
    import time
    
    logger.info(f"\nSpeed Test (5 samples)...")
    test_samples = test_pairs[:5]
    
    # Reranking+Pruning speed
    start = time.time()
    _ = model1.predict_with_pruning(test_samples, batch_size=5)
    time1 = time.time() - start
    
    # Pruning-only speed
    start = time.time()
    _ = model2.predict_with_pruning(test_samples, batch_size=5)
    time2 = time.time() - start
    
    logger.info(f"  Reranking+Pruning: {time1:.3f}s ({time1/5:.3f}s per sample)")
    logger.info(f"  Pruning-only: {time2:.3f}s ({time2/5:.3f}s per sample)")
    logger.info(f"  Speed ratio: {time1/time2:.2f}x")
    
    logger.info(f"\n{'='*60}")
    logger.info("Analysis Complete!")
    logger.info(f"{'='*60}")


if __name__ == "__main__":
    main()