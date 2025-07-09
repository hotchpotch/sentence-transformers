#!/usr/bin/env python
"""
Export PruningEncoder reranking model as a standard CrossEncoder-compatible model.
"""

import sys
from pathlib import Path
import shutil
import json
import logging

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from sentence_transformers.pruning import PruningEncoder
from sentence_transformers import CrossEncoder

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def export_as_crossencoder(pruning_model_path: str, output_path: str):
    """Export PruningEncoder as CrossEncoder-compatible model."""
    logger.info(f"Loading PruningEncoder from {pruning_model_path}")
    
    # Load the PruningEncoder model
    pruning_encoder = PruningEncoder.from_pretrained(pruning_model_path)
    
    if pruning_encoder.mode != "reranking_pruning":
        raise ValueError("Only reranking_pruning models can be exported as CrossEncoder")
    
    # Create output directory
    output_path = Path(output_path)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Copy the base ranking model
    logger.info("Extracting base ranking model...")
    base_model = pruning_encoder.ranking_model
    tokenizer = pruning_encoder.tokenizer
    
    # Save the base model and tokenizer
    logger.info(f"Saving to {output_path}")
    base_model.save_pretrained(output_path)
    tokenizer.save_pretrained(output_path)
    
    # Update config to include sentence_transformers metadata
    config_path = output_path / "config.json"
    with open(config_path, "r") as f:
        config = json.load(f)
    
    # Add sentence_transformers metadata
    config["sentence_transformers"] = {
        "version": "3.4.0",
        "model_type": "cross-encoder",
        "note": "Exported from PruningEncoder reranking model"
    }
    
    with open(config_path, "w") as f:
        json.dump(config, f, indent=2)
    
    logger.info("✓ Export complete!")
    
    # Test loading as CrossEncoder
    logger.info("\nTesting CrossEncoder loading...")
    try:
        test_model = CrossEncoder(str(output_path))
        logger.info("✓ Successfully loaded as CrossEncoder")
        
        # Test inference
        scores = test_model.predict([
            ("テストクエリ", "テスト文書です。")
        ])
        logger.info(f"✓ Inference test passed, score: {scores[0]:.4f}")
        
    except Exception as e:
        logger.error(f"✗ Failed to load as CrossEncoder: {e}")
        raise


def main():
    # Export the reranking model
    pruning_model_path = "./output/transformers_compat_test/reranking_pruning_20250709_135233/final_model"
    output_path = "./output/crossencoder_export/reranking_pruning_crossencoder"
    
    logger.info("="*60)
    logger.info("Exporting PruningEncoder as CrossEncoder")
    logger.info("="*60)
    
    export_as_crossencoder(pruning_model_path, output_path)
    
    # Test with the example code
    logger.info("\n" + "="*60)
    logger.info("Testing with example code")
    logger.info("="*60)
    
    model = CrossEncoder(output_path)
    if str(model.device) == "cuda" or str(model.device) == "mps":
        model.model.half()
    
    query = "感動的な映画について"
    passages = [
        "深いテーマを持ちながらも、観る人の心を揺さぶる名作。登場人物の心情描写が秀逸で、ラストは涙なしでは見られない。",
        "重要なメッセージ性は評価できるが、暗い話が続くので気分が落ち込んでしまった。もう少し明るい要素があればよかった。",
        "どうにもリアリティに欠ける展開が気になった。もっと深みのある人間ドラマが見たかった。",
        "アクションシーンが楽しすぎる。見ていて飽きない。ストーリーはシンプルだが、それが逆に良い。",
    ]
    
    scores = model.predict(
        [(query, passage) for passage in passages],
        show_progress_bar=True,
    )
    
    logger.info(f"Scores: {scores}")
    
    # Show ranking
    sorted_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)
    logger.info("\nRanking:")
    for rank, idx in enumerate(sorted_indices, 1):
        logger.info(f"{rank}. Score: {scores[idx]:.4f} - {passages[idx][:50]}...")


if __name__ == "__main__":
    main()