#!/usr/bin/env python3
"""
Evaluate Provence model trained with batched approach.
"""

import logging
import torch
from pathlib import Path
from sentence_transformers.provence import ProvenceEncoder
from datasets import load_from_disk
import numpy as np
from tqdm import tqdm

# Setup logging
logging.basicConfig(
    format='%(asctime)s - %(levelname)s - %(name)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S',
    level=logging.INFO
)
logger = logging.getLogger(__name__)

def main():
    logger.info("バッチ学習モデルの評価を開始...")
    
    # Configuration
    model_path = "tmp/models/provence-5k-batched/final"
    dataset_path = "tmp/datasets/dev-dataset/small-5k-batched"
    
    # Load model
    logger.info(f"モデル読み込み: {model_path}")
    model = ProvenceEncoder.from_pretrained(model_path)
    model.eval()
    
    # Load test dataset
    logger.info(f"テストデータセット読み込み: {dataset_path}")
    dataset = load_from_disk(dataset_path)
    test_data = dataset["test"]
    
    # Evaluate pruning behavior across different thresholds
    thresholds = [0.001, 0.005, 0.01, 0.05, 0.1, 0.2, 0.3, 0.5]
    
    logger.info(f"\nテストサンプル数: {len(test_data)}")
    logger.info("各閾値での圧縮率を評価中...")
    
    # Sample 100 examples for evaluation
    num_samples = min(100, len(test_data))
    compression_stats = {t: [] for t in thresholds}
    
    for i in tqdm(range(num_samples), desc="評価中"):
        sample = test_data[i]
        query = sample['query']
        
        # Evaluate on all 5 texts
        for j, text in enumerate(sample['texts']):
            teacher_score = sample['teacher_scores'][j]
            label = sample['ranking_labels'][j]
            
            for threshold in thresholds:
                try:
                    result = model.predict_with_pruning(
                        (query, text),
                        pruning_threshold=threshold,
                        return_documents=True
                    )
                    compression = result.compression_ratio
                    compression_stats[threshold].append({
                        'compression': compression,
                        'teacher_score': teacher_score,
                        'label': label,
                        'text_idx': j  # Position in ranking
                    })
                except Exception as e:
                    logger.warning(f"エラー (sample {i}, text {j}, threshold {threshold}): {e}")
    
    # Analyze results
    logger.info("\n=== 圧縮率統計 ===")
    for threshold in thresholds:
        stats = compression_stats[threshold]
        if stats:
            compressions = [s['compression'] for s in stats]
            mean_comp = np.mean(compressions)
            std_comp = np.std(compressions)
            median_comp = np.median(compressions)
            
            # Separate by relevance
            relevant = [s['compression'] for s in stats if s['label'] == 1]
            non_relevant = [s['compression'] for s in stats if s['label'] == 0]
            
            logger.info(f"\n閾値 {threshold}:")
            logger.info(f"  全体: 平均 {mean_comp:.1%} ± {std_comp:.1%}, 中央値 {median_comp:.1%}")
            if relevant:
                logger.info(f"  関連文書: 平均 {np.mean(relevant):.1%}")
            if non_relevant:
                logger.info(f"  非関連文書: 平均 {np.mean(non_relevant):.1%}")
            
            # Count how many have non-zero compression
            non_zero = sum(1 for c in compressions if c > 0)
            logger.info(f"  圧縮された文書: {non_zero}/{len(compressions)} ({non_zero/len(compressions):.1%})")
    
    # Analyze by teacher score ranges
    logger.info("\n=== 教師スコア別分析 (閾値0.1) ===")
    stats_01 = compression_stats[0.1]
    if stats_01:
        score_ranges = [(0, 0.2), (0.2, 0.4), (0.4, 0.6), (0.6, 0.8), (0.8, 1.0)]
        for low, high in score_ranges:
            in_range = [s['compression'] for s in stats_01 
                       if low <= s['teacher_score'] < high]
            if in_range:
                logger.info(f"  スコア {low:.1f}-{high:.1f}: 平均圧縮率 {np.mean(in_range):.1%} (n={len(in_range)})")
    
    # Analyze by text position
    logger.info("\n=== テキスト位置別分析 (閾値0.1) ===")
    for idx in range(5):
        at_position = [s['compression'] for s in stats_01 if s['text_idx'] == idx]
        if at_position:
            logger.info(f"  位置 {idx+1}: 平均圧縮率 {np.mean(at_position):.1%}")
    
    # Test on specific examples with high pruning potential
    logger.info("\n=== 高圧縮可能性サンプルのテスト ===")
    high_prune_candidates = []
    
    for i in range(min(20, len(test_data))):
        sample = test_data[i]
        for j, (score, label) in enumerate(zip(sample['teacher_scores'], sample['ranking_labels'])):
            # Low relevance documents should have high pruning
            if score < 0.3 and label == 0:
                high_prune_candidates.append((i, j, score))
    
    for i, j, score in high_prune_candidates[:5]:
        sample = test_data[i]
        query = sample['query']
        text = sample['texts'][j]
        
        logger.info(f"\n低関連性サンプル (教師スコア: {score:.3f}):")
        logger.info(f"  クエリ: {query[:80]}...")
        
        for threshold in [0.05, 0.1, 0.2, 0.3]:
            try:
                result = model.predict_with_pruning(
                    (query, text),
                    pruning_threshold=threshold,
                    return_documents=True
                )
                logger.info(f"  閾値 {threshold}: 圧縮率 {result.compression_ratio:.1%}")
            except Exception as e:
                logger.warning(f"  閾値 {threshold}: エラー {e}")
    
    logger.info("\n評価完了!")

if __name__ == "__main__":
    main()