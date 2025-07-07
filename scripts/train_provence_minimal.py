#!/usr/bin/env python3
"""
Train ProvenceEncoder on minimal dataset.
"""

import os
import logging
from pathlib import Path
from datasets import load_from_disk

from sentence_transformers.provence import (
    ProvenceEncoder,
    ProvenceTrainer,
    ProvenceLoss,
    ProvenceDataCollator
)

# Setup logging
logging.basicConfig(
    format='%(asctime)s - %(levelname)s - %(name)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S',
    level=logging.INFO
)
logger = logging.getLogger(__name__)


def main():
    # Configuration
    base_model = "hotchpotch/japanese-reranker-xsmall-v2"  # Teacher model confirmed working
    dataset_path = "tmp/datasets/dev-dataset/minimal"
    output_dir = "tmp/models/provence-minimal"
    
    logger.info("=" * 70)
    logger.info("Training ProvenceEncoder on minimal dataset")
    logger.info("=" * 70)
    
    # Load dataset
    logger.info(f"Loading dataset from {dataset_path}...")
    dataset = load_from_disk(dataset_path)
    logger.info(f"Train: {len(dataset['train'])} examples")
    logger.info(f"Valid: {len(dataset['validation'])} examples")
    logger.info(f"Test: {len(dataset['test'])} examples")
    
    # Initialize model
    logger.info(f"Initializing ProvenceEncoder with base model: {base_model}")
    model = ProvenceEncoder(
        model_name_or_path=base_model,
        num_labels=1,
        max_length=512,
        pruning_config={
            "dropout": 0.1,
            "sentence_pooling": "mean",
            "use_weighted_pooling": False
        }
    )
    
    # Initialize trainer
    logger.info("Initializing trainer...")
    trainer = ProvenceTrainer(
        model=model,
        train_dataset=dataset['train'],
        eval_dataset=dataset['validation'],
        training_args={
            "output_dir": output_dir,
            "num_epochs": 1,  # Quick test with 1 epoch
            "batch_size": 8,  # Smaller batch size for memory
            "learning_rate": 2e-5,
            "warmup_ratio": 0.1,
            "weight_decay": 0.01,
            "gradient_accumulation_steps": 2,  # Effective batch size of 16
            "max_grad_norm": 1.0,
            "logging_steps": 10,
            "eval_steps": 50,
            "save_steps": 100,
            "save_total_limit": 2,
            "seed": 42,
            "fp16": True,  # Enable mixed precision if GPU is available
            "dataloader_num_workers": 0,  # Disable multiprocessing for debugging
        }
    )
    
    # Train
    logger.info("Starting training...")
    trainer.train()
    
    # Save final model
    final_model_path = Path(output_dir) / "final"
    logger.info(f"Saving final model to {final_model_path}")
    model.save_pretrained(final_model_path)
    
    # Evaluate on test set
    logger.info("Evaluating on test set...")
    test_dataset = dataset['test']
    
    # Sample predictions
    logger.info("\nSample predictions:")
    for i in range(min(3, len(test_dataset))):
        example = test_dataset[i]
        query = example['query']
        document = example['document']
        
        # Get predictions
        output = model.predict_with_pruning(
            (query, document),
            pruning_threshold=0.5,
            return_documents=True
        )
        
        logger.info(f"\nExample {i+1}:")
        logger.info(f"Query: {query[:100]}...")
        logger.info(f"Original doc length: {len(document)} chars")
        logger.info(f"Ranking score: {output.ranking_scores:.3f}")
        logger.info(f"Compression ratio: {output.compression_ratio:.2%}")
        logger.info(f"Sentences kept: {output.num_pruned_sentences}/{len(output.sentences[0])}")
    
    logger.info("\n" + "=" * 70)
    logger.info("Training completed!")
    logger.info("=" * 70)


if __name__ == "__main__":
    main()