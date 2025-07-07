#!/usr/bin/env python3
"""
Debug ProvenceEncoder training with micro dataset.
"""

import os
import logging
from pathlib import Path
from datasets import load_from_disk
# from transformers import AdafactorSchedule  # Not needed for debug

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
    base_model = "hotchpotch/japanese-reranker-xsmall-v2"
    dataset_path = "tmp/datasets/dev-dataset/minimal"
    output_dir = "tmp/models/provence-debug"
    
    # Debug settings
    debug_size = 50  # Very small for debugging
    
    logger.info("=" * 70)
    logger.info("DEBUG: Training ProvenceEncoder on micro dataset")
    logger.info("=" * 70)
    
    # Load and shrink dataset
    logger.info(f"Loading dataset from {dataset_path}...")
    dataset = load_from_disk(dataset_path)
    
    # Create micro datasets for debugging
    train_micro = dataset['train'].select(range(min(debug_size, len(dataset['train']))))
    valid_micro = dataset['validation'].select(range(min(10, len(dataset['validation']))))
    
    logger.info(f"Debug train: {len(train_micro)} examples")
    logger.info(f"Debug valid: {len(valid_micro)} examples")
    
    # Sample first example to check structure
    example = train_micro[0]
    logger.info("Sample data structure:")
    for key, value in example.items():
        if isinstance(value, str):
            logger.info(f"  {key}: {value[:50]}...")
        else:
            logger.info(f"  {key}: {value}")
    
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
    
    # Initialize trainer with optimized settings
    logger.info("Initializing trainer...")
    trainer = ProvenceTrainer(
        model=model,
        train_dataset=train_micro,
        eval_dataset=valid_micro,
        training_args={
            "output_dir": output_dir,
            "num_epochs": 1,
            "batch_size": 8,  # Small for debugging
            "learning_rate": 1e-5,  # Conservative learning rate
            "warmup_ratio": 0.1,
            "weight_decay": 0.01,
            "gradient_accumulation_steps": 1,
            "max_grad_norm": 1.0,
            "logging_steps": 2,  # Log frequently for debugging
            "eval_steps": 10,
            "save_steps": 20,
            "save_total_limit": 1,
            "seed": 42,
            "bf16": True,  # Use bfloat16
            "dataloader_num_workers": 0,  # Single threaded for debugging
        }
    )
    
    # Quick test: process one batch
    logger.info("Testing data collator...")
    sample_batch = [train_micro[i] for i in range(min(4, len(train_micro)))]
    collated = trainer.data_collator(sample_batch)
    
    logger.info("Batch structure:")
    logger.info(f"  sentence_features[0] keys: {collated['sentence_features'][0].keys()}")
    logger.info(f"  labels keys: {collated['labels'].keys()}")
    for key, value in collated['labels'].items():
        if hasattr(value, 'shape'):
            logger.info(f"  {key} shape: {value.shape}")
        else:
            logger.info(f"  {key}: {value}")
    
    # Test one forward pass
    logger.info("Testing forward pass...")
    model.eval()
    import torch
    with torch.no_grad():
        loss = trainer._training_step(collated)
        logger.info(f"Test loss: {loss.item():.4f}")
    
    # Run minimal training
    logger.info("Starting debug training...")
    try:
        trainer.train()
        logger.info("✓ Training completed successfully!")
        
        # Test inference
        logger.info("Testing inference...")
        query = "テスト"
        document = "これはテスト文書です。"
        
        output = model.predict_with_pruning(
            (query, document),
            pruning_threshold=0.5,
            return_documents=True
        )
        
        logger.info(f"Ranking score: {output.ranking_scores:.3f}")
        logger.info(f"Compression ratio: {output.compression_ratio:.2%}")
        
    except Exception as e:
        logger.error(f"Training failed: {e}")
        import traceback
        traceback.print_exc()
    
    logger.info("\n" + "=" * 70)
    logger.info("DEBUG session completed!")
    logger.info("=" * 70)


if __name__ == "__main__":
    main()