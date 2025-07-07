#!/usr/bin/env python3
"""
Train ProvenceEncoder on full 1.3M batched dataset.
Full scale production training.
"""

import logging
import os
import torch
from pathlib import Path
from sentence_transformers.provence import ProvenceEncoder
from sentence_transformers.provence.trainer import ProvenceTrainer
from datasets import load_from_disk
import numpy as np

# Import our batched components
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from sentence_transformers.provence.losses_batched import ProvenceBatchedLoss
from sentence_transformers.provence.data_collator_batched import ProvenceBatchedDataCollator

# Setup logging
logging.basicConfig(
    format='%(asctime)s - %(levelname)s - %(name)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S',
    level=logging.INFO
)
logger = logging.getLogger(__name__)

def main():
    logger.info("フルデータセット（1.3M）で学習を開始...")
    
    # Configuration
    model_name = "hotchpotch/japanese-reranker-xsmall-v2"
    dataset_path = "tmp/datasets/dev-dataset/full-batched"
    output_dir = "tmp/models/provence-full-batched"
    
    # Load dataset
    logger.info(f"データセット読み込み中: {dataset_path}")
    dataset = load_from_disk(dataset_path)
    
    logger.info(f"データセットサイズ:")
    for split, data in dataset.items():
        logger.info(f"  {split}: {len(data)} examples")
        if len(data) > 0:
            sample = data[0]
            logger.info(f"  - 各例に含まれるテキスト数: {len(sample['texts'])}")
    
    # Initialize model
    logger.info(f"モデル初期化: {model_name}")
    model = ProvenceEncoder(model_name)
    
    # Initialize batched loss function
    logger.info("バッチ対応損失関数の設定...")
    loss_fn = ProvenceBatchedLoss(
        model=model,
        ranking_weight=1.0,
        pruning_weight=1.2,  # Slightly higher for full dataset
        use_teacher_scores=True,
        mini_batch_size=16  # Smaller mini-batch for memory efficiency
    )
    
    # Batched data collator
    data_collator = ProvenceBatchedDataCollator(
        tokenizer=model.tokenizer,
        mini_batch_size=16
    )
    
    # Training arguments
    # Note: With batched approach, actual batch size is multiplied by number of texts per query
    # batch_size=32 × 5 texts = 160 text pairs per batch
    training_args = {
        "output_dir": output_dir,
        "learning_rate": 2e-5,
        "batch_size": 32,  # 32 queries × 5 texts = 160 pairs
        "num_epochs": 1,  # 1 epoch on 1.3M is substantial
        "weight_decay": 0.01,
        "logging_steps": 500,
        "eval_steps": 2000,
        "save_steps": 5000,
        "warmup_ratio": 0.05,  # Shorter warmup for large dataset
        "gradient_accumulation_steps": 4,  # Effective batch size = 128
        "max_grad_norm": 1.0,
        "fp16": torch.cuda.is_available(),  # Use mixed precision if available
    }
    
    # Initialize trainer
    trainer = ProvenceTrainer(
        model=model,
        training_args=training_args,
        loss_fn=loss_fn,
        train_dataset=dataset["train"],
        eval_dataset=dataset["validation"],
        data_collator=data_collator,
    )
    
    # Train model
    logger.info("学習開始...")
    logger.info(f"実効バッチサイズ: {training_args['batch_size']} queries × 5 texts = {training_args['batch_size'] * 5} pairs")
    logger.info(f"Gradient accumulation: {training_args['gradient_accumulation_steps']} steps")
    logger.info(f"Total effective batch size: {training_args['batch_size'] * training_args['gradient_accumulation_steps']} queries")
    
    # Log estimated training time
    total_steps = len(dataset["train"]) // (training_args['batch_size'] * training_args['gradient_accumulation_steps'])
    logger.info(f"Estimated total steps: {total_steps}")
    logger.info(f"Estimated time (at ~3 steps/sec): {total_steps / 3 / 3600:.1f} hours")
    
    trainer.train()
    
    # Save final model
    final_model_path = Path(output_dir) / "final"
    final_model_path.mkdir(parents=True, exist_ok=True)
    
    logger.info(f"最終モデルを保存: {final_model_path}")
    model.save_pretrained(final_model_path)
    
    logger.info("学習完了!")
    
    # Quick test on pruning behavior
    logger.info("\nプルーニング動作のクイックテスト...")
    test_data = dataset["test"]
    test_samples = [test_data[i] for i in range(min(5, len(test_data)))]
    
    thresholds = [0.01, 0.05, 0.1, 0.15, 0.2, 0.3]
    
    for i, sample in enumerate(test_samples):
        query = sample['query']
        # Test on the first text (highest ranking)
        document = sample['texts'][0]
        teacher_score = sample['teacher_scores'][0]
        
        logger.info(f"\nサンプル {i+1}:")
        logger.info(f"  クエリ: {query[:80]}...")
        logger.info(f"  教師スコア: {teacher_score:.3f}")
        
        # Test with various thresholds
        for threshold in thresholds:
            try:
                result = model.predict_with_pruning(
                    (query, document),
                    pruning_threshold=threshold,
                    return_documents=True
                )
                compression = result.compression_ratio
                logger.info(f"  閾値 {threshold}: 圧縮率 {compression:.1%}")
            except Exception as e:
                logger.warning(f"  閾値 {threshold}: エラー {e}")
    
    return model

if __name__ == "__main__":
    main()