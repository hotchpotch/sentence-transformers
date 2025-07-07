#!/usr/bin/env python3
"""
Evaluate the 20k trained model's pruning behavior.
"""

import logging
import torch
from pathlib import Path
from sentence_transformers.provence import ProvenceEncoder
from datasets import load_from_disk
import numpy as np

# Setup logging
logging.basicConfig(
    format='%(asctime)s - %(levelname)s - %(name)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S',
    level=logging.INFO
)
logger = logging.getLogger(__name__)

def evaluate_pruning_behavior(model, dataset, num_samples=20):
    """Evaluate pruning behavior on test samples."""
    logger.info(f"\n{'='*60}")
    logger.info("プルーニング動作の評価")
    logger.info(f"{'='*60}\n")
    
    # Test thresholds
    thresholds = [0.001, 0.005, 0.01, 0.05, 0.1, 0.15, 0.2, 0.3]
    
    # Collect statistics
    compression_stats = {t: [] for t in thresholds}
    
    for i in range(min(num_samples, len(dataset))):
        sample = dataset[i]
        query = sample['query']
        document = sample['text']
        teacher_score = sample.get('teacher_score', 0.0)
        
        if i < 5:  # 最初の5つだけ詳細表示
            logger.info(f"\nサンプル {i+1}:")
            logger.info(f"  クエリ: {query[:100]}...")
            logger.info(f"  教師スコア: {teacher_score:.3f}")
            logger.info(f"  文書長: {len(document)} 文字")
        
        # Test different thresholds
        if i < 5:
            logger.info("  閾値別圧縮率:")
        
        for threshold in thresholds:
            try:
                result = model.predict_with_pruning(
                    (query, document),
                    pruning_threshold=threshold,
                    return_documents=True
                )
                compression = result.compression_ratio
                compression_stats[threshold].append(compression)
                
                if i < 5:
                    logger.info(f"    閾値 {threshold:.3f}: {compression:.1%}")
            except Exception as e:
                if i < 5:
                    logger.warning(f"    閾値 {threshold:.3f}: エラー {e}")
    
    # Summary statistics
    logger.info(f"\n{'='*60}")
    logger.info("全体統計")
    logger.info(f"{'='*60}\n")
    
    # Compression statistics by threshold
    logger.info(f"閾値別圧縮率統計（{num_samples}サンプル）:")
    for threshold in thresholds:
        if compression_stats[threshold]:
            rates = np.array(compression_stats[threshold])
            logger.info(f"\n  閾値 {threshold:.3f}:")
            logger.info(f"    平均圧縮率: {rates.mean():.1%}")
            logger.info(f"    標準偏差: {rates.std():.1%}")
            logger.info(f"    最小: {rates.min():.1%}")
            logger.info(f"    最大: {rates.max():.1%}")
            
            # Count of full pruning
            full_prune_count = (rates == 1.0).sum()
            no_prune_count = (rates == 0.0).sum()
            logger.info(f"    100%プルーニング: {full_prune_count}/{len(rates)} ({full_prune_count/len(rates)*100:.1f}%)")
            logger.info(f"    0%プルーニング: {no_prune_count}/{len(rates)} ({no_prune_count/len(rates)*100:.1f}%)")

def compare_models():
    """Compare different model versions."""
    logger.info(f"\n{'='*60}")
    logger.info("モデル比較")
    logger.info(f"{'='*60}\n")
    
    models = {
        "オリジナル（minimal）": "tmp/models/provence-minimal/final",
        "調整版（minimal）": "tmp/models/provence-minimal-adjusted/final",
        "20kバランス版": "tmp/models/provence-20k-balanced/final"
    }
    
    # Load test dataset
    dataset_path = "tmp/datasets/dev-dataset/small-20k-processed"
    dataset = load_from_disk(dataset_path)
    test_data = dataset["test"]
    
    # Test on same samples
    test_samples = 5
    thresholds = [0.01, 0.05, 0.1]
    
    for model_name, model_path in models.items():
        if not Path(model_path).exists():
            logger.warning(f"{model_name} が見つかりません: {model_path}")
            continue
            
        logger.info(f"\n{model_name}:")
        model = ProvenceEncoder.from_pretrained(model_path)
        model.eval()
        
        compression_results = {t: [] for t in thresholds}
        
        for i in range(min(test_samples, len(test_data))):
            sample = test_data[i]
            query = sample['query']
            document = sample['text']
            
            for threshold in thresholds:
                try:
                    result = model.predict_with_pruning(
                        (query, document),
                        pruning_threshold=threshold,
                        return_documents=True
                    )
                    compression_results[threshold].append(result.compression_ratio)
                except:
                    compression_results[threshold].append(0.0)
        
        # Report averages
        for threshold in thresholds:
            if compression_results[threshold]:
                avg_comp = np.mean(compression_results[threshold])
                logger.info(f"  閾値 {threshold}: 平均圧縮率 {avg_comp:.1%}")

def main():
    logger.info("20k学習済みモデルの評価開始...")
    
    # Paths
    model_path = "tmp/models/provence-20k-balanced/final"
    dataset_path = "tmp/datasets/dev-dataset/small-20k-processed"
    
    # Load model
    logger.info(f"モデル読み込み: {model_path}")
    model = ProvenceEncoder.from_pretrained(model_path)
    model.eval()
    
    # Load test dataset
    logger.info(f"テストデータ読み込み: {dataset_path}")
    dataset = load_from_disk(dataset_path)
    test_data = dataset["test"]
    
    logger.info(f"テストデータサイズ: {len(test_data)} samples")
    
    # Evaluate
    evaluate_pruning_behavior(model, test_data, num_samples=20)
    
    # Compare models
    compare_models()

if __name__ == "__main__":
    main()