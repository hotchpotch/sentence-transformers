#!/usr/bin/env python3
"""
Train ProvenceEncoder with fixed pruning settings.
"""

import os
import logging
from pathlib import Path
from datasets import load_from_disk
import torch
from transformers import Adafactor, get_cosine_schedule_with_warmup

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
    output_dir = "tmp/models/provence-minimal-fixed"
    
    logger.info("=" * 70)
    logger.info("Training ProvenceEncoder with FIXED pruning settings")
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
    
    # Custom optimizer and scheduler
    logger.info("Setting up Adafactor optimizer and cosine scheduler...")
    
    # Adafactor optimizer
    optimizer = Adafactor(
        model.parameters(),
        lr=5e-5,  # Slightly lower learning rate for stability
        clip_threshold=1.0,
        decay_rate=-0.8,
        beta1=None,
        weight_decay=0.01,
        scale_parameter=True,
        relative_step=False
    )
    
    # Cosine scheduler
    num_training_steps = len(dataset['train']) // 64 * 2  # 2 epochs for quick training
    num_warmup_steps = int(num_training_steps * 0.1)
    
    scheduler = get_cosine_schedule_with_warmup(
        optimizer,
        num_warmup_steps=num_warmup_steps,
        num_training_steps=num_training_steps
    )
    
    # Custom loss with token-level pruning
    custom_loss = ProvenceLoss(
        model=model,
        ranking_weight=1.0,
        pruning_weight=1.0,  # Balanced weight for token-level
        use_teacher_scores=True,
        sentence_level_pruning=False  # KEY: Use token-level pruning
    )
    logger.info("✓ Using token-level pruning (not sentence-level)")
    
    # Initialize trainer with optimized settings
    logger.info("Initializing fixed trainer...")
    trainer = ProvenceTrainer(
        model=model,
        train_dataset=dataset['train'],
        eval_dataset=dataset['validation'],
        optimizer=optimizer,
        scheduler=scheduler,
        loss_fn=custom_loss,  # Use custom loss
        training_args={
            "output_dir": output_dir,
            "num_epochs": 2,  # Quick training for testing
            "batch_size": 64,
            "learning_rate": 5e-5,
            "warmup_ratio": 0.1,
            "weight_decay": 0.01,
            "gradient_accumulation_steps": 1,
            "max_grad_norm": 1.0,
            "logging_steps": 20,
            "eval_steps": 100,
            "save_steps": 200,
            "save_total_limit": 2,
            "seed": 42,
            "bf16": True,
            "dataloader_num_workers": 4,
        }
    )
    
    # Train
    logger.info("Starting FIXED training...")
    logger.info(f"Key changes:")
    logger.info(f"  - Token-level pruning: True (was sentence-level)")
    logger.info(f"  - Pruning loss weight: 1.0 (balanced)")
    logger.info(f"  - Learning rate: 5e-5 (was 1e-4)")
    logger.info(f"  - Epochs: 2 (for quick testing)")
    
    trainer.train()
    
    # Save final model
    final_model_path = Path(output_dir) / "final"
    logger.info(f"Saving final model to {final_model_path}")
    model.save_pretrained(final_model_path)
    
    # Quick test of fixed model
    logger.info("\n" + "="*50)
    logger.info("QUICK TEST OF FIXED MODEL")
    logger.info("="*50)
    
    test_query = "テスト"
    test_document = "これは最初の文です。これは2番目の文です。これは3番目の文です。"
    
    result = model.predict_with_pruning(
        (test_query, test_document),
        pruning_threshold=0.5,
        return_documents=True
    )
    
    logger.info(f"Test result:")
    logger.info(f"  Ranking score: {result.ranking_scores:.3f}")
    logger.info(f"  Total sentences: {len(result.sentences[0])}")
    logger.info(f"  Pruned sentences: {result.num_pruned_sentences}")
    logger.info(f"  Compression ratio: {result.compression_ratio:.2%}")
    
    if result.compression_ratio < 0.8:  # Less than 80% compression
        logger.info("✓ Pruning seems more conservative now!")
    else:
        logger.info("⚠ Still too aggressive, may need further tuning")
    
    logger.info("\n" + "=" * 70)
    logger.info("Fixed training completed!")
    logger.info("=" * 70)


if __name__ == "__main__":
    main()