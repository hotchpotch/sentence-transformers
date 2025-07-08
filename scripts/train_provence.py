#!/usr/bin/env python3
"""
統合されたProvenceモデル学習スクリプト
Usage: python train_provence.py --target {minimal|small|full}
"""

import argparse
import os
import logging
from pathlib import Path
from datasets import load_dataset

from sentence_transformers.provence import (
    ProvenceEncoder,
    ProvenceTrainer,
    ProvenceDataCollator,
    ProvenceLoss
)

# ログ設定
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


# データセット設定
DATASET_CONFIGS = {
    'minimal': {
        'name': 'ja-minimal',
        'train_samples': None,  # 全て使用
        'validation_samples': None
    },
    'small': {
        'name': 'ja-small',
        'train_samples': None,
        'validation_samples': None
    },
    'full': {
        'name': 'ja-full',
        'train_samples': None,
        'validation_samples': None
    }
}

# 学習設定
TRAINING_CONFIGS = {
    'minimal': {
        'num_epochs': 2,
        'batch_size': 48,
        'learning_rate': 2e-5,
        'warmup_ratio': 0.1,
        'weight_decay': 0.01,
        'gradient_accumulation_steps': 1,
        'max_grad_norm': 1.0,
        'logging_steps': 20,
        'eval_steps': 200,
        'save_steps': 200,
        'save_total_limit': 3,
        'fp16': True,
        'bf16': True,
        'dataloader_num_workers': 4,
        'seed': 42
    },
    'small': {
        'num_epochs': 3,
        'batch_size': 32,
        'learning_rate': 2e-5,
        'warmup_ratio': 0.1,
        'weight_decay': 0.01,
        'gradient_accumulation_steps': 1,
        'max_grad_norm': 1.0,
        'logging_steps': 50,
        'eval_steps': 500,
        'save_steps': 500,
        'save_total_limit': 2,
        'fp16': True,
        'bf16': True,
        'dataloader_num_workers': 4,
        'seed': 42
    },
    'full': {
        'num_epochs': 1,
        'batch_size': 24,
        'learning_rate': 2e-5,
        'warmup_ratio': 0.05,
        'weight_decay': 0.01,
        'gradient_accumulation_steps': 2,
        'max_grad_norm': 1.0,
        'logging_steps': 100,
        'eval_steps': 1000,
        'save_steps': 1000,
        'save_total_limit': 3,
        'fp16': True,
        'bf16': True,
        'dataloader_num_workers': 4,
        'seed': 42
    }
}

# 損失関数設定
LOSS_CONFIGS = {
    'minimal': {
        'ranking_weight': 1.0,
        'pruning_weight': 0.8,
        'is_regression': True
    },
    'small': {
        'ranking_weight': 1.0,
        'pruning_weight': 0.8,
        'is_regression': True
    },
    'full': {
        'ranking_weight': 1.0,
        'pruning_weight': 0.8,
        'is_regression': True
    }
}


def main():
    parser = argparse.ArgumentParser(description='Train Provence model')
    parser.add_argument(
        '--target', 
        type=str, 
        required=True,
        choices=['minimal', 'small', 'full'],
        help='Target dataset size'
    )
    parser.add_argument(
        '--model_name',
        type=str,
        default='hotchpotch/japanese-reranker-xsmall-v2',
        help='Base model name or path'
    )
    parser.add_argument(
        '--output_dir',
        type=str,
        default=None,
        help='Output directory (default: outputs/provence-ja-{target}/)'
    )
    parser.add_argument(
        '--max_length',
        type=int,
        default=512,
        help='Maximum sequence length'
    )
    parser.add_argument(
        '--resume_from_checkpoint',
        type=str,
        default=None,
        help='Resume training from checkpoint'
    )
    parser.add_argument(
        '--logging_dir',
        type=str,
        default='./logs',
        help='TensorBoard logging directory'
    )
    
    args = parser.parse_args()
    
    # 出力ディレクトリの設定
    if args.output_dir is None:
        args.output_dir = f'outputs/provence-ja-{args.target}'
    
    # データセット設定の取得
    dataset_config = DATASET_CONFIGS[args.target]
    training_config = TRAINING_CONFIGS[args.target]
    loss_config = LOSS_CONFIGS[args.target]
    
    print(f"=== Provence {args.target.upper()} Training ===")
    print(f"Model: {args.model_name}")
    print(f"Dataset: {dataset_config['name']}")
    print(f"Output: {args.output_dir}")
    print(f"Max length: {args.max_length}")
    print("="*50)
    
    # モデルの初期化
    print("🤖 モデル初期化中...")
    model = ProvenceEncoder(
        model_name_or_path=args.model_name,
        max_length=args.max_length,
        pruning_config={
            'num_labels': 2,
            'classifier_dropout': 0.1,
            'sentence_pooling': 'mean',
            'use_weighted_pooling': False,
        }
    )
    
    # データセットの読み込み
    print("📚 データセット読み込み中...")
    dataset = load_dataset(
        'hotchpotch/wip-query-context-pruner-with-teacher-scores',
        dataset_config['name']
    )
    
    # データ数の制限（必要な場合）
    if dataset_config['train_samples']:
        dataset['train'] = dataset['train'].select(range(dataset_config['train_samples']))
    if dataset_config['validation_samples']:
        dataset['validation'] = dataset['validation'].select(range(dataset_config['validation_samples']))
    
    print(f"Training samples: {len(dataset['train'])}")
    print(f"Validation samples: {len(dataset['validation'])}")
    
    # データコレーターの初期化
    data_collator = ProvenceDataCollator(
        tokenizer=model.tokenizer,
        query_column="query",
        texts_column="texts",
        labels_column="labels",
        scores_column="teacher_scores_japanese-reranker-xsmall-v2",
        chunks_pos_column="chunks_pos",
        relevant_chunks_column="relevant_chunks"
    )
    
    # 損失関数
    loss_fn = ProvenceLoss(
        model=model,
        ranking_weight=loss_config['ranking_weight'],
        pruning_weight=loss_config['pruning_weight'],
        is_regression=loss_config['is_regression']
    )
    
    # 学習設定
    training_args = {
        "output_dir": args.output_dir,
        "num_epochs": training_config['num_epochs'],
        "batch_size": training_config['batch_size'],
        "learning_rate": training_config['learning_rate'],
        "warmup_ratio": training_config['warmup_ratio'],
        "weight_decay": training_config['weight_decay'],
        "gradient_accumulation_steps": training_config['gradient_accumulation_steps'],
        "max_grad_norm": training_config['max_grad_norm'],
        "logging_steps": training_config['logging_steps'],
        "eval_steps": training_config['eval_steps'],
        "save_steps": training_config['save_steps'],
        "save_total_limit": training_config['save_total_limit'],
        "fp16": training_config['fp16'],
        "bf16": training_config['bf16'],
        "dataloader_num_workers": training_config['dataloader_num_workers'],
        "seed": training_config['seed'],
        "logging_dir": args.logging_dir,
        "resume_from_checkpoint": args.resume_from_checkpoint
    }
    
    # トレーナー初期化
    print("🚀 トレーナー初期化中...")
    trainer = ProvenceTrainer(
        model=model,
        train_dataset=dataset['train'],
        eval_dataset=dataset['validation'],
        data_collator=data_collator,
        loss_fn=loss_fn,
        training_args=training_args
    )
    
    # 学習開始
    print("🏃 学習開始...")
    trainer.train()
    
    # 最良モデルの保存
    if hasattr(trainer, 'best_model_path') and trainer.best_model_path:
        # 最良モデルをfinal-modelとしてコピー
        import shutil
        print(f"💾 最良モデルを保存中: {args.output_dir}/final-model")
        shutil.copytree(trainer.best_model_path, f"{args.output_dir}/final-model")
    else:
        # 現在のモデルを保存
        print(f"💾 モデルを保存中: {args.output_dir}/final-model")
        model.save_pretrained(f"{args.output_dir}/final-model")
    
    print("✅ 学習完了！")


if __name__ == "__main__":
    main()