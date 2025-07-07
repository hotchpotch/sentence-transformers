#!/usr/bin/env python3
"""
Train ProvenceEncoder on 5k batched dataset with proper handling of multiple texts per query.
Based on LambdaLoss pattern.
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
    logger.info("5kバッチデータセットで学習を開始...")
    
    # Configuration
    model_name = "hotchpotch/japanese-reranker-xsmall-v2"
    dataset_path = "tmp/datasets/dev-dataset/small-5k-batched"
    output_dir = "tmp/models/provence-5k-batched"
    
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
        pruning_weight=0.8,
        use_teacher_scores=True,
        mini_batch_size=32  # Process pairs in chunks
    )
    
    # Batched data collator
    data_collator = ProvenceBatchedDataCollator(
        tokenizer=model.tokenizer,
        mini_batch_size=32
    )
    
    # Training arguments
    # Note: With batched approach, actual batch size is multiplied by number of texts per query
    # batch_size=48 × 5 texts = 240 text pairs per batch
    training_args = {
        "output_dir": output_dir,
        "learning_rate": 3e-5,
        "batch_size": 48,  # 48 queries × 5 texts = 240 pairs
        "num_epochs": 2,
        "weight_decay": 0.01,
        "logging_steps": 20,
        "eval_steps": 100,
        "save_steps": 200,
        "warmup_ratio": 0.1,
        "gradient_accumulation_steps": 1,
        "max_grad_norm": 1.0,
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
    trainer.train()
    
    # Save final model
    final_model_path = Path(output_dir) / "final"
    final_model_path.mkdir(parents=True, exist_ok=True)
    
    logger.info(f"最終モデルを保存: {final_model_path}")
    model.save_pretrained(final_model_path)
    
    logger.info("学習完了!")
    
    # Test pruning behavior
    logger.info("\nプルーニング動作のテスト...")
    test_data = dataset["test"]
    test_samples = [test_data[i] for i in range(min(5, len(test_data)))]
    
    thresholds = [0.001, 0.005, 0.01, 0.05, 0.1]
    
    for i, sample in enumerate(test_samples):
        query = sample['query']
        # Test on the first text (highest ranking)
        document = sample['texts'][0]
        teacher_score = sample['teacher_scores'][0]
        
        logger.info(f"\nサンプル {i+1}:")
        logger.info(f"  クエリ: {query[:80]}...")
        logger.info(f"  教師スコア: {teacher_score:.3f}")
        logger.info(f"  元のラベル: {sample['ranking_labels'][0]}")
        
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