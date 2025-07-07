#!/usr/bin/env python3
"""
Evaluate the adjusted minimal model's pruning behavior.
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

def evaluate_pruning_behavior(model, dataset, num_samples=10):
    """Evaluate pruning behavior on test samples."""
    logger.info(f"\n{'='*60}")
    logger.info("プルーニング動作の評価")
    logger.info(f"{'='*60}\n")
    
    # Test thresholds
    thresholds = [0.001, 0.01, 0.05, 0.1, 0.15, 0.2, 0.3, 0.4, 0.5]
    
    # Collect statistics
    compression_stats = {t: [] for t in thresholds}
    token_prob_stats = []
    
    for i in range(min(num_samples, len(dataset))):
        sample = dataset[i]
        query = sample['query']
        document = sample['text']
        teacher_score = sample.get('teacher_score', 0.0)
        
        logger.info(f"\nサンプル {i+1}:")
        logger.info(f"  クエリ: {query[:100]}...")
        logger.info(f"  教師スコア: {teacher_score:.3f}")
        logger.info(f"  文書長: {len(document)} 文字")
        
        # Skip token probability analysis for now
        # Since the API doesn't provide token scores directly
        
        # Test different thresholds
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
                logger.info(f"    閾値 {threshold:.3f}: {compression:.1%}")
            except Exception as e:
                logger.warning(f"    閾値 {threshold:.3f}: エラー {e}")
    
    # Summary statistics
    logger.info(f"\n{'='*60}")
    logger.info("全体統計")
    logger.info(f"{'='*60}\n")
    
    # Skip token probability statistics since API doesn't provide them
    
    # Compression statistics by threshold
    logger.info("\n閾値別圧縮率統計:")
    for threshold in thresholds:
        if compression_stats[threshold]:
            rates = np.array(compression_stats[threshold])
            logger.info(f"\n  閾値 {threshold:.3f}:")
            logger.info(f"    平均圧縮率: {rates.mean():.1%}")
            logger.info(f"    標準偏差: {rates.std():.1%}")
            logger.info(f"    最小: {rates.min():.1%}")
            logger.info(f"    最大: {rates.max():.1%}")

def main():
    logger.info("調整済みモデルの評価開始...")
    
    # Paths
    model_path = "tmp/models/provence-minimal-adjusted/final"
    dataset_path = "tmp/datasets/dev-dataset/minimal-fixed"
    
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
    
    # Compare with original model if available
    original_model_path = "tmp/models/provence-minimal/final"
    if Path(original_model_path).exists():
        logger.info(f"\n\n{'='*60}")
        logger.info("オリジナルモデルとの比較")
        logger.info(f"{'='*60}")
        
        original_model = ProvenceEncoder.from_pretrained(original_model_path)
        original_model.eval()
        
        # Test on same samples
        test_samples = 5
        thresholds = [0.05, 0.1, 0.15]
        
        for i in range(min(test_samples, len(test_data))):
            sample = test_data[i]
            query = sample['query']
            document = sample['text']
            
            logger.info(f"\nサンプル {i+1}:")
            logger.info(f"  クエリ: {query[:80]}...")
            
            for threshold in thresholds:
                logger.info(f"\n  閾値 {threshold}:")
                
                # Original model
                try:
                    orig_result = original_model.predict_with_pruning(
                        (query, document),
                        pruning_threshold=threshold,
                        return_documents=True
                    )
                    orig_comp = orig_result.compression_ratio
                except:
                    orig_comp = 0.0
                
                # Adjusted model
                try:
                    adj_result = model.predict_with_pruning(
                        (query, document),
                        pruning_threshold=threshold,
                        return_documents=True
                    )
                    adj_comp = adj_result.compression_ratio
                except:
                    adj_comp = 0.0
                
                logger.info(f"    オリジナル: {orig_comp:.1%}")
                logger.info(f"    調整版: {adj_comp:.1%}")
                logger.info(f"    差分: {(adj_comp - orig_comp):.1%}")

if __name__ == "__main__":
    main()