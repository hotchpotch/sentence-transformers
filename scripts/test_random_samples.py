#!/usr/bin/env python3
"""
Test random samples from dataset with ProvenceEncoder.
"""

import logging
import random
from pathlib import Path
from datasets import load_from_disk
from sentence_transformers.provence import ProvenceEncoder

# Setup logging
logging.basicConfig(
    format='%(asctime)s - %(levelname)s - %(name)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S',
    level=logging.INFO
)
logger = logging.getLogger(__name__)

def display_sample_result(sample_idx: int, query: str, original_doc: str, result, threshold: float, teacher_score: float = None, ranking_label: int = None):
    """Display formatted results for a sample."""
    logger.info(f"\n{'='*80}")
    logger.info(f"SAMPLE #{sample_idx}")
    logger.info(f"{'='*80}")
    logger.info(f"Query: {query}")
    logger.info(f"Original Document ({len(original_doc)} chars):")
    logger.info(f"  {original_doc}")
    logger.info(f"")
    
    if teacher_score is not None:
        logger.info(f"Teacher Score: {teacher_score:.3f}")
    if ranking_label is not None:
        logger.info(f"Ground Truth Label: {ranking_label}")
    
    logger.info(f"Model Results (threshold={threshold}):")
    logger.info(f"  Reranker Score: {result.ranking_scores:.3f}")
    logger.info(f"  Compression Ratio: {result.compression_ratio:.1%}")
    logger.info(f"  Sentences: {len(result.sentences[0])} → {len(result.sentences[0]) - result.num_pruned_sentences} (pruned {result.num_pruned_sentences})")
    
    logger.info(f"")
    logger.info(f"Sentence-by-sentence Analysis:")
    for i, sentence in enumerate(result.sentences[0]):
        status = "PRUNED" if i < result.num_pruned_sentences else "KEPT"
        logger.info(f"  [{i+1}] {status}: {sentence.strip()}")
    
    logger.info(f"")
    logger.info(f"Final Pruned Document ({len(result.pruned_documents[0]) if result.pruned_documents else 0} chars):")
    if result.pruned_documents and result.pruned_documents[0]:
        logger.info(f"  {result.pruned_documents[0]}")
    else:
        logger.info(f"  [COMPLETELY PRUNED]")

def main():
    logger.info("Testing random samples from dataset")
    
    # Load model
    model_path = "tmp/models/provence-minimal-fixed/final"
    if not Path(model_path).exists():
        logger.error(f"Model not found at {model_path}")
        return
    
    model = ProvenceEncoder.from_pretrained(model_path)
    model.eval()
    logger.info(f"Model loaded successfully from {model_path}")
    
    # Load test dataset
    dataset_path = "tmp/datasets/dev-dataset/minimal"
    dataset = load_from_disk(dataset_path)
    test_dataset = dataset['test']
    logger.info(f"Test dataset loaded: {len(test_dataset)} examples")
    
    # Random sampling
    num_samples = 8
    random.seed(42)  # For reproducibility
    sample_indices = random.sample(range(len(test_dataset)), num_samples)
    
    # Test with different thresholds
    thresholds = [0.2, 0.3]
    
    for threshold in thresholds:
        logger.info(f"\n{'#'*100}")
        logger.info(f"TESTING WITH THRESHOLD = {threshold}")
        logger.info(f"{'#'*100}")
        
        for i, idx in enumerate(sample_indices):
            sample = test_dataset[idx]
            
            query = sample['query']
            document = sample.get('text', sample.get('document', ''))
            teacher_score = sample.get('teacher_score')
            ranking_label = sample.get('ranking_label', sample.get('label'))
            
            # Skip empty documents
            if not document.strip():
                continue
            
            # Get prediction
            result = model.predict_with_pruning(
                (query, document),
                pruning_threshold=threshold,
                return_documents=True
            )
            
            display_sample_result(
                sample_idx=i+1,
                query=query,
                original_doc=document,
                result=result,
                threshold=threshold,
                teacher_score=teacher_score,
                ranking_label=ranking_label
            )
            
            # Add separator between samples
            if i < len(sample_indices) - 1:
                logger.info(f"\n{'-'*60}")
    
    # Summary statistics
    logger.info(f"\n{'#'*100}")
    logger.info(f"SUMMARY STATISTICS")
    logger.info(f"{'#'*100}")
    
    for threshold in thresholds:
        compression_ratios = []
        reranker_scores = []
        
        for idx in sample_indices:
            sample = test_dataset[idx]
            query = sample['query']
            document = sample.get('text', sample.get('document', ''))
            
            if not document.strip():
                continue
                
            result = model.predict_with_pruning(
                (query, document),
                pruning_threshold=threshold
            )
            compression_ratios.append(result.compression_ratio)
            reranker_scores.append(result.ranking_scores)
        
        avg_compression = sum(compression_ratios) / len(compression_ratios) if compression_ratios else 0
        avg_reranker = sum(reranker_scores) / len(reranker_scores) if reranker_scores else 0
        
        logger.info(f"")
        logger.info(f"Threshold {threshold}:")
        logger.info(f"  Average Compression: {avg_compression:.1%}")
        logger.info(f"  Average Reranker Score: {avg_reranker:.3f}")
        logger.info(f"  Compression Range: {min(compression_ratios):.1%} - {max(compression_ratios):.1%}")
        logger.info(f"  Reranker Range: {min(reranker_scores):.3f} - {max(reranker_scores):.3f}")

if __name__ == "__main__":
    main()