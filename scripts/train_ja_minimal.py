#!/usr/bin/env python3
"""
ja-minimal データセットでの Provence モデル学習スクリプト
"""

import os
import logging
from pathlib import Path
from datasets import load_dataset

from sentence_transformers.provence import (
    ProvenceEncoder,
    ProvenceTrainer,
    ProvenceChunkBasedDataCollator
)
from sentence_transformers.provence.losses_chunk_based import ProvenceChunkBasedLoss

# ログ設定
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def convert_dataset_format(hf_dataset):
    """
    HuggingFace データセットを Provence 学習形式に変換
    """
    converted_data = []
    
    for item in hf_dataset:
        # teacher_scores のキー名を変更
        teacher_scores = item['teacher_scores_japanese-reranker-xsmall-v2']
        
        converted_item = {
            'query': item['query'],
            'texts': item['texts'],
            'chunks_pos': item['chunks_pos'],
            'relevant_chunks': item['relevant_chunks'],
            'ranking_labels': item['labels'],  # labels -> ranking_labels
            'teacher_scores': teacher_scores,
            'dataset_name': item['dataset_name'],
            'id': item['id']
        }
        converted_data.append(converted_item)
    
    return converted_data

def main():
    print("=== ja-minimal での Provence モデル学習 ===")
    
    # 出力ディレクトリ
    output_dir = "./outputs/provence-ja-minimal"
    os.makedirs(output_dir, exist_ok=True)
    
    # データセットロード
    print("📥 データセットロード中...")
    dataset = load_dataset('hotchpotch/wip-query-context-pruner-with-teacher-scores', 'ja-minimal')
    
    # データ変換
    print("🔄 データ形式変換中...")
    train_data = convert_dataset_format(dataset['train'])
    eval_data = convert_dataset_format(dataset['validation'])
    
    print(f"✅ 学習データ: {len(train_data):,} 件")
    print(f"✅ 評価データ: {len(eval_data):,} 件")
    
    # モデル初期化
    print("🤖 モデル初期化中...")
    model = ProvenceEncoder(
        model_name_or_path="hotchpotch/japanese-reranker-xsmall-v2",
        num_labels=1,
        max_length=512,
        pruning_config={
            "dropout": 0.1,
            "sentence_pooling": "mean"
        }
    )
    
    # データコレクター
    data_collator = ProvenceChunkBasedDataCollator(
        tokenizer=model.tokenizer,
        max_length=512,
        padding=True,
        truncation=True
    )
    
    # 損失関数（chunk-based専用）
    loss_fn = ProvenceChunkBasedLoss(
        model=model,
        ranking_weight=1.0,
        pruning_weight=0.8,  # プルーニング重視
        use_teacher_scores=True
    )
    
    # 学習設定（大容量GPU用）
    training_args = {
        "output_dir": output_dir,
        "num_epochs": 2,  # 検証用に短縮
        "batch_size": 48,  # 大容量GPUメモリ利用
        "learning_rate": 2e-5,
        "warmup_ratio": 0.1,
        "weight_decay": 0.01,
        "gradient_accumulation_steps": 1,  # 実効バッチサイズ48
        "max_grad_norm": 1.0,
        "logging_steps": 20,  # より頻繁にログ出力
        "eval_steps": 200,  # より頻繁に評価
        "save_steps": 200,  # より頻繁に保存
        "save_total_limit": 3,
        "fp16": True,
        "dataloader_num_workers": 4,  # データロード高速化
        "seed": 42
    }
    
    # トレーナー初期化
    print("🚀 トレーナー初期化中...")
    trainer = ProvenceTrainer(
        model=model,
        train_dataset=train_data,
        eval_dataset=eval_data,
        data_collator=data_collator,
        loss_fn=loss_fn,
        training_args=training_args
    )
    
    # 学習開始
    print(f"🎯 学習開始 - ja-minimal ({len(train_data):,} 件)")
    print(f"📁 出力先: {output_dir}")
    print(f"⚙️  設定: エポック数={training_args['num_epochs']}, バッチサイズ={training_args['batch_size']}, 実効BS={training_args['batch_size'] * training_args['gradient_accumulation_steps']}")
    
    try:
        trainer.train()
        print("✅ 学習完了!")
        
        # 最終モデル保存
        final_model_path = os.path.join(output_dir, "final-model")
        model.save_pretrained(final_model_path)
        print(f"💾 最終モデル保存: {final_model_path}")
        
    except Exception as e:
        logger.error(f"❌ 学習中にエラー: {e}")
        raise

if __name__ == "__main__":
    main()