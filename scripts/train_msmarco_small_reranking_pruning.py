#!/usr/bin/env python
"""
Train a small reranking+pruning model with the MS MARCO dataset.
"""

import os
import sys
from pathlib import Path
import logging
from datetime import datetime
import torch
import pandas as pd
from datasets import Dataset

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from sentence_transformers.pruning import (
    PruningEncoder, PruningTrainer, PruningLoss, PruningDataCollator
)

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def load_msmarco_dataset(subset: str = "msmarco-small-ja"):
    """Load MS MARCO dataset using direct parquet loading."""
    logger.info(f"Loading {subset} dataset...")
    
    # Load parquet files directly
    base_path = f"hf://datasets/hotchpotch/wip-msmarco-context-relevance/{subset}"
    
    # Load splits
    train_df = pd.read_parquet(f"{base_path}/train-00000-of-00001.parquet")
    val_df = pd.read_parquet(f"{base_path}/validation-00000-of-00001.parquet")
    test_df = pd.read_parquet(f"{base_path}/test-00000-of-00001.parquet")
    
    # Convert to datasets
    train_dataset = Dataset.from_pandas(train_df)
    val_dataset = Dataset.from_pandas(val_df)
    test_dataset = Dataset.from_pandas(test_df)
    
    logger.info(f"Dataset loaded - Train: {len(train_dataset)}, Val: {len(val_dataset)}, Test: {len(test_dataset)}")
    
    # Show sample for verification
    sample = train_dataset[0]
    logger.info(f"Sample data structure:")
    logger.info(f"  Query: {sample['query']}")
    logger.info(f"  Texts count: {len(sample['texts'])}")
    logger.info(f"  Labels: {sample['labels']} (first=pos, rest=neg)")
    logger.info(f"  Context spans count: {len(sample['context_spans'])}")
    
    return train_dataset, val_dataset, test_dataset


def main():
    """Train small reranking+pruning model."""
    logger.info("="*60)
    logger.info("Training Small Reranking+Pruning Model (MS MARCO)")
    logger.info("="*60)
    
    # Output directory
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    output_dir = f"./output/msmarco_small_reranking_pruning_{timestamp}"
    
    # Load dataset
    train_dataset, eval_dataset, test_dataset = load_msmarco_dataset("msmarco-small-ja")
    
    # Use smaller subset for faster training (10k samples)
    train_dataset = train_dataset.select(range(min(10000, len(train_dataset))))
    eval_dataset = eval_dataset.select(range(min(500, len(eval_dataset))))
    
    logger.info(f"Using train size: {len(train_dataset)}, eval size: {len(eval_dataset)}")
    
    # Initialize model
    model = PruningEncoder(
        model_name_or_path="hotchpotch/japanese-reranker-xsmall-v2",
        mode="reranking_pruning",
        max_length=512,
        device="cuda" if torch.cuda.is_available() else "cpu",
        pruning_config={
            "hidden_size": 256,
            "dropout": 0.1,
            "sentence_pooling": "mean",
            "use_weighted_pooling": False
        }
    )
    
    # Data collator
    data_collator = PruningDataCollator(
        tokenizer=model.tokenizer,
        max_length=512,
        mode="reranking_pruning",
        padding=True,
        truncation=True,
        query_column="query",
        texts_column="texts",
        labels_column="labels",
        chunks_pos_column="context_spans",
        relevant_chunks_column="context_relevance",
        mini_batch_size=8
    )
    
    # Loss function
    loss_fn = PruningLoss(
        model=model,
        mode="reranking_pruning",
        pruning_weight=1.0,
        ranking_weight=1.0,
    )
    
    # Training configuration for small dataset
    training_args = {
        "output_dir": output_dir,
        "num_epochs": 2,  # 2 epochs for better convergence
        "batch_size": 4,
        "learning_rate": 2e-5,
        "warmup_ratio": 0.1,
        "weight_decay": 0.01,
        "gradient_accumulation_steps": 2,
        "max_grad_norm": 1.0,
        "logging_steps": 50,
        "eval_steps": 200,
        "save_steps": 500,
        "save_total_limit": 2,
        "seed": 42,
        "fp16": torch.cuda.is_available(),
        "dataloader_num_workers": 2,
    }
    
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
    
    # Quick test
    logger.info("Running quick test...")
    sample = train_dataset[0]
    test_queries_docs = [
        (sample['query'], sample['texts'][0])  # First text is positive
    ]
    
    try:
        # Test with pruning
        outputs = model.predict_with_pruning(
            test_queries_docs,
            pruning_threshold=0.5,
            return_documents=True
        )
        
        logger.info(f"Test ranking score: {outputs[0].ranking_scores:.4f}")
        logger.info(f"Test compression ratio: {outputs[0].compression_ratio:.2%}")
        
        # Test without pruning
        scores = model.predict(test_queries_docs, apply_pruning=False)
        logger.info(f"Test score without pruning: {scores[0]:.4f}")
        
    except Exception as e:
        logger.error(f"Test failed: {e}")
    
    logger.info("="*60)
    logger.info("Training completed successfully!")
    logger.info(f"Model saved at: {final_model_path}")
    logger.info("="*60)


if __name__ == "__main__":
    main()