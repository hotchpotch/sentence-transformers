#!/usr/bin/env python
"""
Test if PruningEncoder models can be used as CrossEncoder.
"""

import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

# IMPORTANT: Import pruning module first to register models
import sentence_transformers.pruning

from sentence_transformers import CrossEncoder
import torch
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def test_crossencoder_compatibility(model_path: str):
    """Test loading PruningEncoder as CrossEncoder."""
    logger.info(f"\nTesting CrossEncoder compatibility with: {model_path}")
    
    try:
        # Try to load as CrossEncoder
        model = CrossEncoder(model_path)
        
        # If GPU available, use half precision
        if str(model.device) == "cuda" or str(model.device) == "mps":
            model.model.half()
        
        # Test data
        query = "感動的な映画について"
        passages = [
            "深いテーマを持ちながらも、観る人の心を揺さぶる名作。登場人物の心情描写が秀逸で、ラストは涙なしでは見られない。",
            "重要なメッセージ性は評価できるが、暗い話が続くので気分が落ち込んでしまった。もう少し明るい要素があればよかった。",
            "どうにもリアリティに欠ける展開が気になった。もっと深みのある人間ドラマが見たかった。",
            "アクションシーンが楽しすぎる。見ていて飽きない。ストーリーはシンプルだが、それが逆に良い。",
        ]
        
        # Predict
        scores = model.predict(
            [(query, passage) for passage in passages],
            show_progress_bar=True,
        )
        
        logger.info(f"✓ Successfully loaded as CrossEncoder!")
        logger.info(f"Scores: {scores}")
        
        # Show ranking
        sorted_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)
        logger.info("\nRanking:")
        for rank, idx in enumerate(sorted_indices, 1):
            logger.info(f"{rank}. Score: {scores[idx]:.4f} - {passages[idx][:50]}...")
        
        return True
        
    except Exception as e:
        logger.error(f"✗ Failed to load as CrossEncoder: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    # Test both minimal models
    models = {
        "reranking_pruning_minimal": "./output/transformers_compat_test/reranking_pruning_20250709_135233/final_model",
        "pruning_only_minimal": "./output/transformers_compat_test/pruning_only_20250709_135222/final_model",
    }
    
    results = {}
    
    for model_name, model_path in models.items():
        logger.info(f"\n{'='*60}")
        logger.info(f"Testing {model_name}")
        logger.info(f"{'='*60}")
        
        success = test_crossencoder_compatibility(model_path)
        results[model_name] = success
    
    # Summary
    logger.info(f"\n{'='*60}")
    logger.info("SUMMARY")
    logger.info(f"{'='*60}")
    
    for model_name, success in results.items():
        logger.info(f"{model_name}: {'✓ PASS' if success else '✗ FAIL'}")
    
    # Also test if the original model works
    logger.info(f"\n{'='*60}")
    logger.info("Testing original japanese-reranker-xsmall-v2")
    logger.info(f"{'='*60}")
    
    try:
        original_model = CrossEncoder("hotchpotch/japanese-reranker-xsmall-v2")
        logger.info("✓ Original model loads successfully as CrossEncoder")
    except Exception as e:
        logger.error(f"✗ Original model failed: {e}")


if __name__ == "__main__":
    main()