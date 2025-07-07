#!/usr/bin/env python3
"""
Create a partial dataset quickly by using symlinks or direct references.
"""

import logging
from pathlib import Path
from datasets import DatasetDict
import json

# Setup logging
logging.basicConfig(
    format='%(asctime)s - %(levelname)s - %(name)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S',
    level=logging.INFO
)
logger = logging.getLogger(__name__)

def main():
    logger.info("Creating partial dataset configuration...")
    
    # Paths
    output_path = Path("tmp/datasets/dev-dataset/full-batched")
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Since we have processed data in chunks, let's create a simple config
    # that points to the available data
    
    config = {
        "train": {
            "chunks": [
                "tmp/datasets/dev-dataset/full-remaining/train_chunks_68_to_97.arrow",
                "tmp/datasets/dev-dataset/full-final/train_chunks_100_to_130_final.arrow"
            ],
            "total_examples": 600419
        },
        "validation": "tmp/datasets/dev-dataset/small-100k-batched/validation",
        "test": "tmp/datasets/dev-dataset/small-100k-batched/test"
    }
    
    # Save config
    config_path = output_path / "dataset_config.json"
    with open(config_path, 'w') as f:
        json.dump(config, f, indent=2)
    
    logger.info(f"Saved dataset configuration to {config_path}")
    
    # Create a simple dataset info file
    info = {
        "description": "Partial Provence training dataset (chunks 68-130)",
        "features": {
            "query": "string",
            "texts": ["string"],
            "ranking_labels": ["int"],
            "teacher_scores": ["float"],
            "pruning_labels": [["int"]],
            "sentence_boundaries": [[["int"]]],
            "dataset_name": "string",
            "example_id": "string"
        },
        "splits": {
            "train": 600419,
            "validation": 10000,
            "test": 10000
        }
    }
    
    info_path = output_path / "dataset_info.json"
    with open(info_path, 'w') as f:
        json.dump(info, f, indent=2)
    
    logger.info(f"Saved dataset info to {info_path}")
    
    # For now, let's use the 100k dataset to test if training works
    logger.info("\nNote: Full dataset assembly would take time due to size.")
    logger.info("For immediate training, consider using the 100k dataset at:")
    logger.info("  tmp/datasets/dev-dataset/small-100k-batched")
    logger.info("\nAlternatively, we can train on the partial 600k dataset")
    logger.info("by modifying the training script to load chunks directly.")

if __name__ == "__main__":
    main()