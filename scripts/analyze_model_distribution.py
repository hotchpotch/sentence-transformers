#!/usr/bin/env python3
"""
Analyze sentence-level probability distribution from ProvenceEncoder.
"""

import logging
import random
import numpy as np
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

def analyze_sentence_probabilities(model: ProvenceEncoder, samples: list):
    """Analyze sentence-level keep probabilities."""
    all_probs = []
    relevant_probs = []
    irrelevant_probs = []
    
    for sample in samples:
        query = sample['query']
        document = sample.get('text', sample.get('document', ''))
        teacher_score = sample.get('teacher_score', 0.0)
        
        if not document.strip():
            continue
        
        # Get sentence probabilities
        result = model.predict_with_pruning(
            (query, document),
            pruning_threshold=0.0,  # Keep all to see probabilities
            return_documents=True
        )
        
        # Extract sentence probabilities (approximate from compression behavior)
        # We'll use multiple thresholds to estimate sentence probabilities
        thresholds = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
        sentence_probs = []
        
        for thresh in thresholds:
            test_result = model.predict_with_pruning(
                (query, document),
                pruning_threshold=thresh
            )
            kept_ratio = 1.0 - test_result.compression_ratio
            sentence_probs.append(kept_ratio)
        
        # Estimate average keep probability
        avg_prob = np.mean(sentence_probs)
        all_probs.append(avg_prob)
        
        # Categorize by teacher score
        if teacher_score > 0.1:
            relevant_probs.append(avg_prob)
        else:
            irrelevant_probs.append(avg_prob)
    
    return {
        'all_probs': all_probs,
        'relevant_probs': relevant_probs,
        'irrelevant_probs': irrelevant_probs
    }

def main():
    logger.info("Analyzing model probability distributions")
    
    # Load model
    model_path = "tmp/models/provence-minimal-fixed/final"
    if not Path(model_path).exists():
        logger.error(f"Model not found at {model_path}")
        return
    
    model = ProvenceEncoder.from_pretrained(model_path)
    model.eval()
    
    # Load test dataset
    dataset = load_from_disk("tmp/datasets/dev-dataset/minimal")
    test_dataset = dataset['test']
    
    # Sample for analysis
    random.seed(42)
    num_samples = 50
    sample_indices = random.sample(range(len(test_dataset)), num_samples)
    samples = [test_dataset[i] for i in sample_indices]
    
    logger.info(f"Analyzing {num_samples} samples...")
    
    # Analyze probabilities
    prob_data = analyze_sentence_probabilities(model, samples)
    
    # Statistics
    all_probs = prob_data['all_probs']
    relevant_probs = prob_data['relevant_probs']
    irrelevant_probs = prob_data['irrelevant_probs']
    
    logger.info("\n" + "="*60)
    logger.info("PROBABILITY DISTRIBUTION ANALYSIS")
    logger.info("="*60)
    
    logger.info(f"\nAll samples ({len(all_probs)} samples):")
    logger.info(f"  Mean keep probability: {np.mean(all_probs):.3f}")
    logger.info(f"  Median keep probability: {np.median(all_probs):.3f}")
    logger.info(f"  Std deviation: {np.std(all_probs):.3f}")
    logger.info(f"  Min: {np.min(all_probs):.3f}")
    logger.info(f"  Max: {np.max(all_probs):.3f}")
    logger.info(f"  25th percentile: {np.percentile(all_probs, 25):.3f}")
    logger.info(f"  75th percentile: {np.percentile(all_probs, 75):.3f}")
    
    if relevant_probs:
        logger.info(f"\nRelevant samples (teacher_score > 0.1, {len(relevant_probs)} samples):")
        logger.info(f"  Mean keep probability: {np.mean(relevant_probs):.3f}")
        logger.info(f"  Median: {np.median(relevant_probs):.3f}")
        logger.info(f"  Range: [{np.min(relevant_probs):.3f}, {np.max(relevant_probs):.3f}]")
    
    if irrelevant_probs:
        logger.info(f"\nIrrelevant samples (teacher_score ≤ 0.1, {len(irrelevant_probs)} samples):")
        logger.info(f"  Mean keep probability: {np.mean(irrelevant_probs):.3f}")
        logger.info(f"  Median: {np.median(irrelevant_probs):.3f}")
        logger.info(f"  Range: [{np.min(irrelevant_probs):.3f}, {np.max(irrelevant_probs):.3f}]")
    
    # Threshold recommendations
    logger.info(f"\n" + "="*60)
    logger.info("THRESHOLD RECOMMENDATIONS")
    logger.info("="*60)
    
    # Find threshold that keeps ~50% on average
    target_keep_ratio = 0.5
    closest_prob = min(all_probs, key=lambda x: abs(x - target_keep_ratio))
    recommended_threshold = 1.0 - closest_prob
    
    logger.info(f"\nFor ~50% average retention:")
    logger.info(f"  Recommended threshold: {recommended_threshold:.2f}")
    
    # Conservative threshold (keeps 70%)
    conservative_target = 0.7
    conservative_prob = min(all_probs, key=lambda x: abs(x - conservative_target))
    conservative_threshold = 1.0 - conservative_prob
    
    logger.info(f"\nFor ~70% average retention (conservative):")
    logger.info(f"  Recommended threshold: {conservative_threshold:.2f}")
    
    # Aggressive threshold (keeps 30%)
    aggressive_target = 0.3
    aggressive_prob = min(all_probs, key=lambda x: abs(x - aggressive_target))
    aggressive_threshold = 1.0 - aggressive_prob
    
    logger.info(f"\nFor ~30% average retention (aggressive):")
    logger.info(f"  Recommended threshold: {aggressive_threshold:.2f}")
    
    # Test recommended thresholds
    logger.info(f"\n" + "="*60)
    logger.info("TESTING RECOMMENDED THRESHOLDS")
    logger.info("="*60)
    
    test_thresholds = [0.1, 0.15, 0.2, 0.25, 0.3]
    
    for threshold in test_thresholds:
        compressions = []
        for sample in samples[:10]:  # Test on first 10 samples
            query = sample['query']
            document = sample.get('text', sample.get('document', ''))
            
            if not document.strip():
                continue
                
            result = model.predict_with_pruning(
                (query, document),
                pruning_threshold=threshold
            )
            compressions.append(result.compression_ratio)
        
        avg_compression = np.mean(compressions) if compressions else 0
        avg_retention = 1.0 - avg_compression
        
        logger.info(f"Threshold {threshold}: {avg_retention:.1%} average retention, {avg_compression:.1%} compression")

if __name__ == "__main__":
    main()