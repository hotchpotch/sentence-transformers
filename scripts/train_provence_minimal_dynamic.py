#!/usr/bin/env python3
"""
Train Provence model on minimal dataset with dynamic chunk-based pruning labels.
"""

import logging
from pathlib import Path
import torch
from torch.utils.data import DataLoader
from sentence_transformers import SentenceTransformer
from sentence_transformers.provence.encoder import ProvenceEncoder
from sentence_transformers.provence.losses_batched import ProvenceBatchedLoss
from sentence_transformers.provence.data_collator_chunk_based import ProvenceChunkBasedDataCollator
from sentence_transformers.provence.trainer import ProvenceTrainer
from transformers import AutoTokenizer
from datasets import load_from_disk
import json

# Setup logging
logging.basicConfig(
    format='%(asctime)s - %(levelname)s - %(name)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S',
    level=logging.INFO
)
logger = logging.getLogger(__name__)


def main():
    logger.info("Starting Provence training on minimal dataset with dynamic labels...")
    
    # Paths
    dataset_path = "tmp/datasets/dev-dataset/minimal-5k-simple"
    output_dir = "outputs/provence-minimal-dynamic"
    
    # Load dataset
    logger.info(f"Loading dataset from {dataset_path}")
    dataset = load_from_disk(dataset_path)
    
    # Initialize base model
    base_model_name = "intfloat/multilingual-e5-small"
    logger.info(f"Loading base model: {base_model_name}")
    
    # Create Provence model
    logger.info("Creating Provence model...")
    model = ProvenceEncoder(
        model_name_or_path=base_model_name,
        num_labels=1,  # Binary classification for ranking
        max_length=512,
        pruning_config={
            "hidden_size": 384,
            "dropout": 0.1,
            "sentence_pooling": "mean"
        }
    )
    
    # Initialize tokenizer
    tokenizer = AutoTokenizer.from_pretrained(base_model_name)
    
    # Create data collator
    logger.info("Creating chunk-based data collator...")
    data_collator = ProvenceChunkBasedDataCollator(
        tokenizer=tokenizer,
        max_length=512,
        padding=True,
        truncation=True,
        mini_batch_size=64  # Process pairs in mini-batches
    )
    
    # Create loss function
    logger.info("Creating loss function...")
    loss_fn = ProvenceBatchedLoss(
        model=model,
        ranking_weight=0.8,   # Slightly emphasize ranking
        pruning_weight=1.0,   # Also important for pruning
        use_teacher_scores=True,  # Use teacher score distillation
        mini_batch_size=64
    )
    
    # Training arguments - ProvenceTrainer expects dict format
    training_args = {
        "output_dir": output_dir,
        "num_epochs": 3,
        "batch_size": 8,  # 8 queries × 5 texts = 40 pairs per batch
        "gradient_accumulation_steps": 2,
        "warmup_ratio": 0.1,
        "learning_rate": 2e-5,
        "fp16": True,
        "logging_steps": 50,
        "save_steps": 500,
        "eval_steps": 500,
        "save_total_limit": 3,
        "dataloader_num_workers": 4,
    }
    
    # Create trainer
    logger.info("Creating Provence trainer...")
    trainer = ProvenceTrainer(
        model=model,
        training_args=training_args,  # Note: it's training_args not args
        train_dataset=dataset['train'],
        eval_dataset=dataset['validation'],
        data_collator=data_collator,
        loss_fn=loss_fn,
    )
    
    # Log dataset info
    logger.info(f"Train dataset size: {len(dataset['train'])}")
    logger.info(f"Validation dataset size: {len(dataset['validation'])}")
    logger.info(f"Test dataset size: {len(dataset['test'])}")
    
    # Check first example
    example = dataset['train'][0]
    logger.info(f"First example has {len(example['texts'])} texts")
    logger.info(f"Relevant chunks: {example['relevant_chunks']}")
    
    # Train
    logger.info("Starting training...")
    trainer.train()
    
    # Save final model - the trainer already saved best model
    logger.info("Final model already saved by trainer")
    
    # Save model using ProvenceEncoder's save method
    logger.info("Saving model in Provence format...")
    model.save_pretrained(f"{output_dir}/provence-model")
    
    # Save training info
    info = {
        "base_model": base_model_name,
        "dataset": dataset_path,
        "train_size": len(dataset['train']),
        "ranking_weight": 0.8,
        "pruning_weight": 1.0,
        "use_teacher_scores": True,
        "dynamic_labels": "chunk-based",
        "training_args": training_args
    }
    
    with open(f"{output_dir}/training_info.json", "w") as f:
        json.dump(info, f, indent=2)
    
    logger.info("Training completed!")


if __name__ == "__main__":
    main()