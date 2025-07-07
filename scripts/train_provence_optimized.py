#!/usr/bin/env python3
"""
Train ProvenceEncoder with optimized settings.
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
    output_dir = "tmp/models/provence-minimal"
    
    logger.info("=" * 70)
    logger.info("Training ProvenceEncoder with optimized settings")
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
        lr=1e-4,  # Adafactor typically uses higher learning rates
        clip_threshold=1.0,
        decay_rate=-0.8,
        beta1=None,
        weight_decay=0.01,
        scale_parameter=True,
        relative_step=False
    )
    
    # Cosine scheduler
    num_training_steps = len(dataset['train']) // 128 * 3  # batch_size=128, epochs=3
    num_warmup_steps = int(num_training_steps * 0.1)
    
    scheduler = get_cosine_schedule_with_warmup(
        optimizer,
        num_warmup_steps=num_warmup_steps,
        num_training_steps=num_training_steps
    )
    
    # Initialize trainer with optimized settings
    logger.info("Initializing optimized trainer...")
    trainer = ProvenceTrainer(
        model=model,
        train_dataset=dataset['train'],
        eval_dataset=dataset['validation'],
        optimizer=optimizer,
        scheduler=scheduler,
        training_args={
            "output_dir": output_dir,
            "num_epochs": 3,
            "batch_size": 128,  # Larger batch size as requested
            "learning_rate": 1e-4,  # Handled by optimizer
            "warmup_ratio": 0.1,  # Handled by scheduler
            "weight_decay": 0.01,  # Handled by optimizer
            "gradient_accumulation_steps": 1,
            "max_grad_norm": 1.0,
            "logging_steps": 20,
            "eval_steps": 200,
            "save_steps": 400,
            "save_total_limit": 2,
            "seed": 42,
            "bf16": True,  # Use bfloat16 as requested
            "dataloader_num_workers": 4,  # Parallel data loading
        }
    )
    
    # Train
    logger.info("Starting optimized training...")
    logger.info(f"Using optimizer: {type(optimizer).__name__}")
    logger.info(f"Using scheduler: {type(scheduler).__name__}")
    logger.info(f"Training steps: {num_training_steps}")
    logger.info(f"Warmup steps: {num_warmup_steps}")
    
    trainer.train()
    
    # Save final model
    final_model_path = Path(output_dir) / "final"
    logger.info(f"Saving final model to {final_model_path}")
    model.save_pretrained(final_model_path)
    
    # Evaluate on test set
    logger.info("Evaluating on test set...")
    test_dataset = dataset['test'].select(range(100))  # Sample for quick eval
    
    # Sample predictions
    logger.info("\nSample predictions:")
    for i in range(min(3, len(test_dataset))):
        example = test_dataset[i]
        query = example['query']
        document = example.get('text', example.get('document', ''))
        
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
        logger.info(f"Sentences: {len(output.sentences[0])} total, {output.num_pruned_sentences} pruned")
    
    logger.info("\n" + "=" * 70)
    logger.info("Optimized training completed!")
    logger.info("=" * 70)


if __name__ == "__main__":
    main()