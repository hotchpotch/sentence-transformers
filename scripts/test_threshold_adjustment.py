#!/usr/bin/env python3
"""
Test different pruning thresholds for ProvenceEncoder.
"""

import logging
from pathlib import Path
from sentence_transformers.provence import ProvenceEncoder

# Setup logging
logging.basicConfig(
    format='%(asctime)s - %(levelname)s - %(name)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S',
    level=logging.INFO
)
logger = logging.getLogger(__name__)

def test_threshold(model: ProvenceEncoder, query: str, document: str, threshold: float):
    """Test a specific threshold."""
    result = model.predict_with_pruning(
        (query, document),
        pruning_threshold=threshold,
        return_documents=True
    )
    
    logger.info(f"Threshold {threshold}: {result.compression_ratio:.2%} compression, "
                f"{result.ranking_scores:.3f} ranking, "
                f"{result.num_pruned_sentences}/{len(result.sentences[0])} sentences pruned")
    
    if result.pruned_documents:
        logger.info(f"  Pruned doc: {result.pruned_documents[0][:100]}...")
    
    return result

def main():
    logger.info("Testing different pruning thresholds")
    
    # Load model
    model_path = "tmp/models/provence-minimal-fixed/final"
    if not Path(model_path).exists():
        logger.error(f"Model not found at {model_path}")
        return
    
    model = ProvenceEncoder.from_pretrained(model_path)
    model.eval()
    
    # Test cases
    test_cases = [
        {
            'name': 'High Relevance',
            'query': 'Python プログラミング',
            'document': 'Pythonは素晴らしいプログラミング言語です。データサイエンスに最適です。機械学習にも使われます。'
        },
        {
            'name': 'Mixed Relevance', 
            'query': '機械学習',
            'document': '機械学習は人工知能の分野です。今日は雨が降っています。アルゴリズムが重要です。'
        }
    ]
    
    # Test thresholds from paper (0.1 to 0.5)
    thresholds = [0.1, 0.2, 0.3, 0.4, 0.5]
    
    for case in test_cases:
        logger.info(f"\n{'='*60}")
        logger.info(f"Test Case: {case['name']}")
        logger.info(f"Query: {case['query']}")
        logger.info(f"Document: {case['document']}")
        logger.info(f"{'='*60}")
        
        for threshold in thresholds:
            test_threshold(model, case['query'], case['document'], threshold)
        
        logger.info("")

if __name__ == "__main__":
    main()