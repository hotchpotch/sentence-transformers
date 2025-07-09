#!/usr/bin/env python
"""
Reranking + Pruning model training script for minimal dataset.
This script trains models that perform both reranking and pruning.
Uses hotchpotch/japanese-reranker-xsmall-v2 as the base model.
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
        logging.FileHandler(f'./log/train_reranking_pruning_minimal_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log')
    ]
)
logger = logging.getLogger(__name__)


def main():
    """Train a reranking + pruning model on minimal dataset."""
    
    dataset_name = 'minimal'
    logger.info(f"Starting reranking + pruning training with {dataset_name} dataset")
    
    # Load dataset
    logger.info(f"Loading dataset: hotchpotch/wip-query-context-pruner-with-teacher-scores/ja-minimal")
    dataset = load_dataset(
        'hotchpotch/wip-query-context-pruner-with-teacher-scores',
        'ja-minimal'
    )
    train_dataset = dataset['train']
    eval_dataset = dataset['validation']
    
    # Show dataset info
    logger.info(f"Train dataset size: {len(train_dataset)}")
    logger.info(f"Eval dataset size: {len(eval_dataset)}")
    logger.info(f"Dataset columns: {train_dataset.column_names}")
    
    # Model configuration
    model_name = "hotchpotch/japanese-reranker-xsmall-v2"  # Base model for reranking + pruning
    max_length = 512
    
    # Initialize model in reranking_pruning mode (default)
    logger.info(f"Initializing PruningEncoder in reranking_pruning mode with base model: {model_name}")
    model = PruningEncoder(
        model_name_or_path=model_name,
        mode="reranking_pruning",  # Default mode
        max_length=max_length,
        device="cuda" if torch.cuda.is_available() else "cpu",
        pruning_config={
            "hidden_size": 256,  # Match japanese-reranker-xsmall-v2 actual hidden size
            "dropout": 0.1,
            "sentence_pooling": "mean",
            "use_weighted_pooling": False
        }
    )
    
    # Data collator with reranking_pruning mode
    data_collator = PruningDataCollator(
        tokenizer=model.tokenizer,
        max_length=max_length,
        mode="reranking_pruning",  # Default mode
        padding=True,
        truncation=True,
        query_column="query",
        texts_column="texts",
        labels_column="labels",
        chunks_pos_column="chunks_pos",
        relevant_chunks_column="relevant_chunks",
        mini_batch_size=16
    )
    
    # Loss function with both losses
    loss_fn = PruningLoss(
        model=model,
        mode="reranking_pruning",  # Default mode
        pruning_weight=1.0,
        ranking_weight=1.0,
    )
    
    # Training configuration (same as pruning-only for fair comparison)
    config = {
        'num_epochs': 5,
        'batch_size': 16,
        'learning_rate': 2e-5,
        'eval_steps': 50,
        'save_steps': 50,
        'logging_steps': 10,
    }
    
    # Output directory
    output_dir = f"./output/reranking_pruning_minimal_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
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
        "dataloader_num_workers": 2,
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