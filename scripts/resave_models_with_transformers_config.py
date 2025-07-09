#!/usr/bin/env python
"""
Re-save existing models with Transformers-compatible config.
"""

import os
import sys
from pathlib import Path
import shutil
import logging

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from sentence_transformers.pruning import PruningEncoder

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def resave_model(original_path: str, new_path: str):
    """Load and re-save model with Transformers config."""
    logger.info(f"Loading model from {original_path}")
    
    try:
        # Load model
        model = PruningEncoder.from_pretrained(original_path)
        
        # Create new directory
        os.makedirs(new_path, exist_ok=True)
        
        # Save with new config format
        logger.info(f"Saving to {new_path} with Transformers config")
        model.save_pretrained(new_path)
        
        logger.info("✓ Successfully re-saved model")
        return True
        
    except Exception as e:
        logger.error(f"Failed to re-save model: {e}")
        return False


def main():
    # Models to re-save
    models = {
        "pruning_only_minimal": {
            "original": "./output/pruning_only_minimal_20250709_081603/checkpoint-1200-best",
            "new": "./output/pruning_only_minimal_transformers/final_model"
        },
        "reranking_pruning_minimal": {
            "original": "./output/reranking_pruning_minimal_20250709_103823/final_model",
            "new": "./output/reranking_pruning_minimal_transformers/final_model"
        }
    }
    
    for model_name, paths in models.items():
        logger.info(f"\n{'='*60}")
        logger.info(f"Processing {model_name}")
        logger.info(f"{'='*60}")
        
        if not os.path.exists(paths["original"]):
            logger.warning(f"Original model not found at {paths['original']}")
            continue
        
        success = resave_model(paths["original"], paths["new"])
        
        if success:
            logger.info(f"✓ {model_name} re-saved successfully")
        else:
            logger.error(f"✗ {model_name} failed to re-save")


if __name__ == "__main__":
    main()