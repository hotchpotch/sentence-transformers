#!/usr/bin/env python3
"""
Train ProvenceEncoder on minimal dataset with adjusted loss weights.
Aims to achieve better pruning balance.
"""

import logging
import os
import torch
from pathlib import Path
from sentence_transformers.provence import ProvenceEncoder
from sentence_transformers.provence.trainer import ProvenceTrainer
from sentence_transformers.provence.losses import ProvenceLoss
from sentence_transformers.provence.data_collator import ProvenceDataCollator
from datasets import load_from_disk
from transformers import Adafactor
import numpy as np

# Setup logging
logging.basicConfig(
    format='%(asctime)s - %(levelname)s - %(name)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S',
    level=logging.INFO
)
logger = logging.getLogger(__name__)

def main():
    logger.info("Minimalデータセットで損失重みを調整した再学習を開始...")
    
    # Configuration
    model_name = "hotchpotch/japanese-reranker-xsmall-v2"
    dataset_path = "tmp/datasets/dev-dataset/minimal-fixed"  # 既存のminimalデータ
    output_dir = "tmp/models/provence-minimal-adjusted"
    
    # Load dataset
    logger.info(f"データセット読み込み中: {dataset_path}")
    dataset = load_from_disk(dataset_path)
    
    logger.info(f"データセットサイズ:")
    for split, data in dataset.items():
        logger.info(f"  {split}: {len(data)} examples")
    
    # Initialize model
    logger.info(f"モデル初期化: {model_name}")
    model = ProvenceEncoder(model_name)
    
    # Initialize loss function with adjusted weights
    logger.info("損失関数の設定（調整版）...")
    loss_fn = ProvenceLoss(
        model=model,
        ranking_weight=0.5,   # ランキング損失の重みを下げる
        pruning_weight=2.0,   # プルーニング損失の重みを上げる
        use_teacher_scores=True,
        sentence_level_pruning=False  # トークンレベルプルーニング
    )
    
    # Data collator
    from sentence_transformers.utils.text_chunking import MultilingualChunker
    text_chunker = MultilingualChunker()
    
    data_collator = ProvenceDataCollator(
        tokenizer=model.tokenizer,
        text_chunker=text_chunker,
        sentence_level_pruning=False
    )
    
    # Training arguments
    training_args = {
        "output_dir": output_dir,
        "learning_rate": 5e-5,
        "batch_size": 64,
        "num_epochs": 3,
        "weight_decay": 0.01,
        "logging_steps": 10,
        "eval_steps": 50,
        "save_steps": 100,
        "warmup_ratio": 0.1,
        "gradient_accumulation_steps": 2,
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
    trainer.train()
    
    # Save final model
    final_model_path = Path(output_dir) / "final"
    final_model_path.mkdir(parents=True, exist_ok=True)
    
    logger.info(f"最終モデルを保存: {final_model_path}")
    model.save_pretrained(final_model_path)
    
    logger.info("学習完了!")
    
    # Test pruning behavior
    logger.info("\nプルーニング動作のテスト...")
    test_sample = dataset["test"][0]
    query = test_sample['query']
    document = test_sample['text']
    teacher_score = test_sample.get('teacher_score', 0.0)
    
    logger.info(f"テストサンプル:")
    logger.info(f"  クエリ: {query}")
    logger.info(f"  教師スコア: {teacher_score:.3f}")
    
    # Test with various thresholds
    thresholds = [0.001, 0.01, 0.05, 0.1, 0.15, 0.2]
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