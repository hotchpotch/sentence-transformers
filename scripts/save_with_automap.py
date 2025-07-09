#!/usr/bin/env python
"""
Re-save existing models with auto_map configuration.
"""

import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from sentence_transformers.pruning import PruningEncoder
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def resave_with_automap(model_path: str, output_path: str):
    """Load and re-save model with auto_map configuration."""
    logger.info(f"Loading model from {model_path}")
    
    # Load model
    model = PruningEncoder.from_pretrained(model_path)
    
    # Save with new auto_map configuration
    logger.info(f"Saving to {output_path} with auto_map")
    model.save_pretrained(output_path)
    
    logger.info("✓ Model saved with auto_map configuration")


def main():
    # Re-save one of our test models
    model_path = "./output/transformers_compat_test/reranking_pruning_20250709_135233/final_model"
    output_path = "./output/automap_test/reranking_pruning_automap"
    
    resave_with_automap(model_path, output_path)
    
    # Test with the test script
    logger.info("\nTesting AutoModel loading...")
    import subprocess
    result = subprocess.run(
        [sys.executable, "scripts/test_pruning_automodel.py", output_path],
        capture_output=True,
        text=True
    )
    
    print(result.stdout)
    if result.stderr:
        print("STDERR:", result.stderr)


if __name__ == "__main__":
    main()