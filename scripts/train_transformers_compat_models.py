#!/usr/bin/env python
"""
Train minimal models to test Transformers compatibility.
"""

import os
import sys
from pathlib import Path
import logging
from datetime import datetime
import torch
import shutil

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from datasets import load_dataset
from sentence_transformers.pruning import (
    PruningEncoder, PruningTrainer, PruningLoss, PruningDataCollator
)

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def train_model(mode: str, output_base: str):
    """Train a model in the specified mode."""
    logger.info(f"\n{'='*60}")
    logger.info(f"Training {mode} model")
    logger.info(f"{'='*60}")
    
    # Load minimal dataset for quick training
    dataset = load_dataset(
        'hotchpotch/wip-query-context-pruner-with-teacher-scores',
        'ja-minimal'
    )
    # Use only first 100 samples for quick test
    train_dataset = dataset['train'].select(range(100))
    eval_dataset = dataset['validation'].select(range(20))
    
    logger.info(f"Train dataset size: {len(train_dataset)}")
    logger.info(f"Eval dataset size: {len(eval_dataset)}")
    
    # Model configuration based on mode
    if mode == "pruning_only":
        model_name = "cl-nagoya/ruri-v3-30m"
        hidden_size = 256
    else:  # reranking_pruning
        model_name = "hotchpotch/japanese-reranker-xsmall-v2"
        hidden_size = 256
    
    # Initialize model
    model = PruningEncoder(
        model_name_or_path=model_name,
        mode=mode,
        max_length=512,
        device="cuda" if torch.cuda.is_available() else "cpu",
        pruning_config={
            "hidden_size": hidden_size,
            "dropout": 0.1,
            "sentence_pooling": "mean",
            "use_weighted_pooling": False
        }
    )
    
    # Data collator
    data_collator = PruningDataCollator(
        tokenizer=model.tokenizer,
        max_length=512,
        mode=mode,
        padding=True,
        truncation=True,
        query_column="query",
        texts_column="texts",
        labels_column="labels",
        chunks_pos_column="chunks_pos",
        relevant_chunks_column="relevant_chunks",
        mini_batch_size=16
    )
    
    # Loss function
    loss_fn = PruningLoss(
        model=model,
        mode=mode,
        pruning_weight=1.0,
        ranking_weight=1.0 if mode == "reranking_pruning" else 0.0,
    )
    
    # Training configuration - very short for testing
    output_dir = f"{output_base}/{mode}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    
    training_args = {
        "output_dir": output_dir,
        "num_epochs": 1,  # Just 1 epoch for testing
        "batch_size": 8,
        "learning_rate": 2e-5,
        "warmup_ratio": 0.1,
        "weight_decay": 0.01,
        "gradient_accumulation_steps": 1,
        "max_grad_norm": 1.0,
        "logging_steps": 10,
        "eval_steps": 50,
        "save_steps": 50,
        "save_total_limit": 1,
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
    
    return final_model_path


def test_transformers_loading(model_path: str, mode: str):
    """Test loading the model with Transformers AutoModel."""
    logger.info(f"\n{'='*60}")
    logger.info(f"Testing Transformers compatibility for {mode}")
    logger.info(f"{'='*60}")
    
    from transformers import (
        AutoModelForSequenceClassification,
        AutoModelForTokenClassification,
        AutoTokenizer,
        AutoConfig
    )
    
    try:
        # Test loading config
        logger.info("Loading config...")
        config = AutoConfig.from_pretrained(model_path)
        logger.info(f"✓ Config loaded: model_type={config.model_type}, mode={config.mode}")
        
        # Test loading tokenizer
        logger.info("Loading tokenizer...")
        tokenizer = AutoTokenizer.from_pretrained(model_path)
        logger.info("✓ Tokenizer loaded")
        
        # Test loading model
        if mode == "reranking_pruning":
            logger.info("Loading with AutoModelForSequenceClassification...")
            model = AutoModelForSequenceClassification.from_pretrained(model_path)
            logger.info("✓ Model loaded with AutoModelForSequenceClassification")
        else:
            logger.info("Loading with AutoModelForTokenClassification...")
            model = AutoModelForTokenClassification.from_pretrained(model_path)
            logger.info("✓ Model loaded with AutoModelForTokenClassification")
        
        # Test inference
        query = "テストクエリ"
        document = "これはテスト文書です。"
        
        inputs = tokenizer(query, document, return_tensors="pt", truncation=True, max_length=512)
        
        # Move inputs to the same device as model
        device = next(model.parameters()).device
        inputs = {k: v.to(device) for k, v in inputs.items()}
        
        with torch.no_grad():
            outputs = model(**inputs)
            logger.info(f"✓ Inference successful, output shape: {outputs.logits.shape}")
        
        return True
        
    except Exception as e:
        logger.error(f"✗ Failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    output_base = "./output/transformers_compat_test"
    
    # Train both model types
    models = {}
    
    for mode in ["pruning_only", "reranking_pruning"]:
        model_path = train_model(mode, output_base)
        models[mode] = model_path
        
        # Test immediately after training
        success = test_transformers_loading(model_path, mode)
        
        if success:
            logger.info(f"✓ {mode} model is Transformers-compatible!")
        else:
            logger.error(f"✗ {mode} model failed Transformers compatibility test")
    
    # Summary
    logger.info(f"\n{'='*60}")
    logger.info("SUMMARY")
    logger.info(f"{'='*60}")
    logger.info(f"Models trained and saved to:")
    for mode, path in models.items():
        logger.info(f"  {mode}: {path}")


if __name__ == "__main__":
    main()