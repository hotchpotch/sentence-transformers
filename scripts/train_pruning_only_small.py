#!/usr/bin/env python
"""
Pruning-only model training script for small dataset.
This script trains models that focus solely on text pruning without ranking capabilities.
Uses cl-nagoya/ruri-v3-30m as the base model.
"""

import os
import sys
from pathlib import Path
import logging
from datetime import datetime
import torch

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from datasets import load_dataset
from sentence_transformers.pruning import (
    PruningEncoder, PruningTrainer, PruningLoss, PruningDataCollator
)

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(f'./log/train_pruning_only_small_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log')
    ]
)
logger = logging.getLogger(__name__)


def main():
    """Train a pruning-only model on small dataset."""
    
    dataset_name = 'small'
    logger.info(f"Starting pruning-only training with {dataset_name} dataset")
    
    # Load dataset
    logger.info(f"Loading dataset: hotchpotch/wip-query-context-pruner-with-teacher-scores/ja-small")
    dataset = load_dataset(
        'hotchpotch/wip-query-context-pruner-with-teacher-scores',
        'ja-small'
    )
    train_dataset = dataset['train']
    eval_dataset = dataset['validation']
    
    # Show dataset info
    logger.info(f"Train dataset size: {len(train_dataset)}")
    logger.info(f"Eval dataset size: {len(eval_dataset)}")
    logger.info(f"Dataset columns: {train_dataset.column_names}")
    
    # Model configuration
    model_name = "cl-nagoya/ruri-v3-30m"  # Base model for pruning-only
    max_length = 512
    
    # Initialize model in pruning_only mode
    logger.info(f"Initializing PruningEncoder in pruning_only mode with base model: {model_name}")
    model = PruningEncoder(
        model_name_or_path=model_name,
        mode="pruning_only",  # Important: pruning-only mode
        max_length=max_length,
        device="cuda" if torch.cuda.is_available() else "cpu",
        pruning_config={
            "hidden_size": 256,  # Match ruri-v3-30m actual hidden size
            "dropout": 0.1,
            "sentence_pooling": "mean",
            "use_weighted_pooling": False
        }
    )
    
    # Data collator with pruning_only mode
    data_collator = PruningDataCollator(
        tokenizer=model.tokenizer,
        max_length=max_length,
        mode="pruning_only",  # Important: pruning-only mode
        padding=True,
        truncation=True,
        query_column="query",
        texts_column="texts",
        labels_column="labels",  # Will be ignored in pruning_only mode
        chunks_pos_column="chunks_pos",
        relevant_chunks_column="relevant_chunks",
        mini_batch_size=32
    )
    
    # Loss function with pruning_only mode
    loss_fn = PruningLoss(
        model=model,
        mode="pruning_only",  # Important: pruning-only mode
        pruning_weight=1.0,  # Only pruning loss
        ranking_weight=0.0,  # No ranking loss
    )
    
    # Training configuration (optimized for small dataset)
    config = {
        'num_epochs': 2,  # Optimized based on evaluation
        'batch_size': 32,
        'learning_rate': 2e-5,
        'eval_steps': 500,
        'save_steps': 500,
        'logging_steps': 100,
    }
    
    # Output directory
    output_dir = f"./output/pruning_only_small_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    os.makedirs(output_dir, exist_ok=True)
    
    # Training arguments
    training_args = {
        "output_dir": output_dir,
        "num_epochs": config['num_epochs'],
        "batch_size": config['batch_size'],
        "learning_rate": config['learning_rate'],
        "warmup_ratio": 0.1,
        "weight_decay": 0.01,
        "gradient_accumulation_steps": 1,
        "max_grad_norm": 1.0,
        "logging_steps": config['logging_steps'],
        "eval_steps": config['eval_steps'],
        "save_steps": config['save_steps'],
        "save_total_limit": 3,
        "seed": 42,
        "fp16": torch.cuda.is_available(),
        "dataloader_num_workers": 4,
    }
    
    logger.info(f"Training arguments: {training_args}")
    
    # Initialize trainer
    trainer = PruningTrainer(
        model=model,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=data_collator,
        loss_fn=loss_fn,
        training_args=training_args
    )
    
    # Train
    logger.info("Starting training...")
    trainer.train()
    
    # Save final model
    final_model_path = os.path.join(output_dir, "final_model")
    logger.info(f"Saving final model to {final_model_path}")
    model.save_pretrained(final_model_path)
    
    logger.info("Training completed!")
    return output_dir


if __name__ == "__main__":
    # Create log directory
    os.makedirs("./log", exist_ok=True)
    
    # Train model
    output_dir = main()
    
    print(f"\nTraining completed! Model saved to: {output_dir}")