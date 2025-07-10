#!/usr/bin/env python
"""
Unified Pruning Model Training Script

This script trains pruning models with flexible configuration options.

Recommended epochs:
- minimal datasets: --epochs 2 (for quick testing and validation)
- small datasets: --epochs 2 (for development and experiments)  
- full datasets: --epochs 1 (for production training)

Usage examples:
  # Train pruning-only model on minimal dataset
  python scripts/pruning_trainer.py --mode pruning_only --dataset msmarco-minimal-ja --epochs 2

  # Train reranking+pruning model on small dataset
  python scripts/pruning_trainer.py --mode reranking_pruning --dataset msmarco-small-ja --epochs 2

  # Train on full dataset with custom model
  python scripts/pruning_trainer.py --mode reranking_pruning --dataset msmarco-small-ja --model intfloat/multilingual-e5-small --epochs 1
"""

import argparse
import os
import sys
from pathlib import Path
import logging
from datetime import datetime
import torch
import pandas as pd
from datasets import Dataset, load_dataset as hf_load_dataset

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


def get_dataset_config(dataset_name):
    """Get dataset configuration."""
    configs = {
        "msmarco-minimal-ja": {
            "dataset_name": "hotchpotch/wip-msmarco-context-relevance",
            "subset": "msmarco-minimal-ja",
            "train_samples": 500,
            "eval_samples": 100,
            "description": "Minimal dataset for quick testing"
        },
        "msmarco-small-ja": {
            "dataset_name": "hotchpotch/wip-msmarco-context-relevance",
            "subset": "msmarco-small-ja", 
            "train_samples": None,  # Use full dataset
            "eval_samples": 1000,
            "description": "Small dataset for development"
        },
        "msmarco-full-ja": {
            "dataset_name": "hotchpotch/wip-msmarco-context-relevance",
            "subset": "msmarco-full-ja",
            "train_samples": None,  # Use full dataset
            "eval_samples": 2000,
            "description": "Full dataset for production training"
        },
        "msmarco-ja": {
            "dataset_name": "hotchpotch/wip-msmarco-context-relevance",
            "subset": "msmarco-ja",
            "train_samples": None,  # Use full dataset
            "eval_samples": 2000,
            "description": "Full Japanese MS MARCO dataset"
        }
    }
    
    if dataset_name not in configs:
        raise ValueError(f"Unknown dataset: {dataset_name}. Available: {list(configs.keys())}")
    
    return configs[dataset_name]


def get_model_config(mode):
    """Get recommended model configurations for each mode."""
    configs = {
        "pruning_only": {
            "model": "cl-nagoya/ruri-v3-30m",
            "reason": "LlamaTokenizer works well with fast tokenizers"
        },
        "reranking_pruning": {
            "model": "hotchpotch/japanese-reranker-xsmall-v2",
            "reason": "Specialized Japanese reranker (now working after fix)"
        }
    }
    return configs.get(mode, configs["pruning_only"])


def load_dataset(dataset_config):
    """Load MS MARCO dataset using direct parquet loading with authentication."""
    logger.info(f"Loading dataset: {dataset_config['dataset_name']} (subset: {dataset_config['subset']})")
    logger.info(f"Description: {dataset_config['description']}")
    
    try:
        from huggingface_hub import hf_hub_download
        
        # Special handling for msmarco-full-ja with multiple parquet files
        if dataset_config["subset"] == "msmarco-full-ja":
            logger.info("Loading msmarco-full-ja with multiple parquet files...")
            
            # Load train files (5 parts)
            train_dfs = []
            for i in range(5):
                filename = f"{dataset_config['subset']}/train-{i:05d}-of-00005.parquet"
                logger.info(f"Downloading {filename}...")
                
                file_path = hf_hub_download(
                    repo_id=dataset_config["dataset_name"],
                    filename=filename,
                    repo_type="dataset"
                )
                
                df = pd.read_parquet(file_path)
                train_dfs.append(df)
                logger.info(f"  Loaded {len(df)} samples from part {i}")
            
            train_df = pd.concat(train_dfs, ignore_index=True)
            
            # Load validation file
            val_filename = f"{dataset_config['subset']}/validation-00000-of-00001.parquet"
            val_path = hf_hub_download(
                repo_id=dataset_config["dataset_name"],
                filename=val_filename,
                repo_type="dataset"
            )
            val_df = pd.read_parquet(val_path)
            
        else:
            # Standard single file loading for other datasets
            train_filename = f"{dataset_config['subset']}/train-00000-of-00001.parquet"
            val_filename = f"{dataset_config['subset']}/validation-00000-of-00001.parquet"
            
            train_path = hf_hub_download(
                repo_id=dataset_config["dataset_name"],
                filename=train_filename,
                repo_type="dataset"
            )
            val_path = hf_hub_download(
                repo_id=dataset_config["dataset_name"],
                filename=val_filename,
                repo_type="dataset"
            )
            
            train_df = pd.read_parquet(train_path)
            val_df = pd.read_parquet(val_path)
        
        # Convert to datasets
        train_dataset = Dataset.from_pandas(train_df)
        val_dataset = Dataset.from_pandas(val_df)
        
    except Exception as e:
        logger.error(f"Failed to load dataset: {e}")
        logger.info("Attempting direct hf:// URL loading...")
        
        # Fallback to direct parquet loading
        try:
            base_path = f"hf://datasets/{dataset_config['dataset_name']}/{dataset_config['subset']}"
            train_df = pd.read_parquet(f"{base_path}/train-00000-of-00001.parquet")
            val_df = pd.read_parquet(f"{base_path}/validation-00000-of-00001.parquet")
            
            train_dataset = Dataset.from_pandas(train_df)
            val_dataset = Dataset.from_pandas(val_df)
        except Exception as e2:
            logger.error(f"All loading methods failed: {e2}")
            raise
    
    # Apply sample limits if specified
    if dataset_config["train_samples"]:
        train_dataset = train_dataset.select(range(min(dataset_config["train_samples"], len(train_dataset))))
    
    if dataset_config["eval_samples"]:
        val_dataset = val_dataset.select(range(min(dataset_config["eval_samples"], len(val_dataset))))
    
    logger.info(f"Dataset loaded - Train: {len(train_dataset)}, Val: {len(val_dataset)}")
    
    # Show sample for verification
    sample = train_dataset[0]
    logger.info(f"Sample data structure:")
    logger.info(f"  Query: {sample['query']}")
    logger.info(f"  Texts count: {len(sample['texts'])}")
    logger.info(f"  Labels: {sample['labels']} (first=pos, rest=neg)")
    
    return train_dataset, val_dataset


def create_output_dir(mode, dataset_name, model_name, epochs):
    """Create output directory with descriptive name."""
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    model_short = model_name.split('/')[-1]  # Get just the model name part
    output_dir = f"./output/{dataset_name}_{mode}_{model_short}_ep{epochs}_{timestamp}"
    return output_dir


def main():
    parser = argparse.ArgumentParser(
        description="Train pruning models with flexible configuration",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )
    
    parser.add_argument(
        "--mode", 
        choices=["pruning_only", "reranking_pruning"],
        default="pruning_only",
        help="Training mode (default: pruning_only)"
    )
    
    parser.add_argument(
        "--dataset",
        choices=["msmarco-minimal-ja", "msmarco-small-ja", "msmarco-full-ja", "msmarco-ja"],
        default="msmarco-minimal-ja", 
        help="Dataset to use (default: msmarco-minimal-ja)"
    )
    
    parser.add_argument(
        "--model",
        type=str,
        help="Base model to use (default: auto-selected based on mode)"
    )
    
    parser.add_argument(
        "--epochs",
        type=int,
        default=1,
        help="Number of training epochs (default: 1)"
    )
    
    parser.add_argument(
        "--batch-size",
        type=int,
        default=4,
        help="Training batch size (default: 4)"
    )
    
    parser.add_argument(
        "--learning-rate",
        type=float,
        default=2e-5,
        help="Learning rate (default: 2e-5)"
    )
    
    parser.add_argument(
        "--max-length",
        type=int,
        default=512,
        help="Maximum sequence length (default: 512)"
    )
    
    parser.add_argument(
        "--gradient-accumulation-steps",
        type=int,
        default=4,
        help="Gradient accumulation steps (default: 4)"
    )
    
    parser.add_argument(
        "--output-dir",
        type=str,
        help="Output directory (default: auto-generated)"
    )
    
    parser.add_argument(
        "--no-fp16",
        action="store_true",
        help="Disable FP16 training"
    )
    
    args = parser.parse_args()
    
    # Get configurations
    dataset_config = get_dataset_config(args.dataset)
    
    # Auto-select model if not specified
    if args.model is None:
        model_config = get_model_config(args.mode)
        args.model = model_config["model"]
        logger.info(f"Auto-selected model: {args.model} ({model_config['reason']})")
    
    # Create output directory
    if args.output_dir is None:
        args.output_dir = create_output_dir(args.mode, args.dataset, args.model, args.epochs)
    
    logger.info("=" * 80)
    logger.info(f"PRUNING MODEL TRAINING")
    logger.info("=" * 80)
    logger.info(f"Mode: {args.mode}")
    logger.info(f"Dataset: {args.dataset}")
    logger.info(f"Model: {args.model}")
    logger.info(f"Epochs: {args.epochs}")
    logger.info(f"Batch size: {args.batch_size}")
    logger.info(f"Learning rate: {args.learning_rate}")
    logger.info(f"Output dir: {args.output_dir}")
    logger.info("=" * 80)
    
    # Load dataset
    train_dataset, eval_dataset = load_dataset(dataset_config)
    
    # Initialize model
    logger.info(f"Initializing {args.mode} model...")
    model = PruningEncoder(
        model_name_or_path=args.model,
        mode=args.mode,
        max_length=args.max_length,
        device="cuda" if torch.cuda.is_available() else "cpu",
        pruning_config={
            "hidden_size": 256,
            "dropout": 0.1,
            "sentence_pooling": "mean",
            "use_weighted_pooling": False
        },
        tokenizer_args={"use_fast": True}  # Use fast tokenizer
    )
    
    # Data collator
    data_collator = PruningDataCollator(
        tokenizer=model.tokenizer,
        max_length=args.max_length,
        mode=args.mode,
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
    if args.mode == "pruning_only":
        loss_fn = PruningLoss(
            model=model,
            mode=args.mode,
            pruning_weight=1.0,
        )
    else:  # reranking_pruning
        loss_fn = PruningLoss(
            model=model,
            mode=args.mode,
            pruning_weight=1.0,
            ranking_weight=1.0,
        )
    
    # Training configuration
    training_args = {
        "output_dir": args.output_dir,
        "num_epochs": args.epochs,
        "batch_size": args.batch_size,
        "learning_rate": args.learning_rate,
        "warmup_ratio": 0.1,
        "weight_decay": 0.01,
        "gradient_accumulation_steps": args.gradient_accumulation_steps,
        "max_grad_norm": 1.0,
        "logging_steps": 100,
        "eval_steps": 1000,
        "save_steps": 2000,
        "save_total_limit": 2,
        "seed": 42,
        "fp16": torch.cuda.is_available() and not args.no_fp16,
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
    final_model_path = os.path.join(args.output_dir, "final_model")
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
            pruning_threshold=0.3,
            return_documents=True
        )
        
        if args.mode == "reranking_pruning":
            logger.info(f"Test ranking score: {outputs[0].ranking_scores:.4f}")
        logger.info(f"Test compression ratio: {outputs[0].compression_ratio:.2%}")
        
        # Check if num_pruned_tokens attribute exists
        if hasattr(outputs[0], 'num_pruned_tokens'):
            logger.info(f"Test pruned tokens: {outputs[0].num_pruned_tokens}")
        elif hasattr(outputs[0], 'pruned_tokens'):
            logger.info(f"Test pruned tokens: {outputs[0].pruned_tokens}")
        else:
            logger.info(f"Test pruned tokens: not available")
        
        if args.mode == "reranking_pruning":
            # Test without pruning
            scores = model.predict(test_queries_docs, apply_pruning=False)
            logger.info(f"Test score without pruning: {scores[0]:.4f}")
        
    except Exception as e:
        logger.error(f"Test failed: {e}")
    
    logger.info("=" * 80)
    logger.info("TRAINING COMPLETED SUCCESSFULLY!")
    logger.info(f"Model saved at: {final_model_path}")
    logger.info(f"Configuration: {args.mode} on {args.dataset} for {args.epochs} epochs")
    logger.info("=" * 80)
    
    # Print usage examples for evaluation
    logger.info(f"\nTo evaluate this model, run:")
    logger.info(f"  python scripts/evaluate_pruning_model.py {final_model_path}")


if __name__ == "__main__":
    main()