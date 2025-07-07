#!/usr/bin/env python3
"""
Evaluate Provence model trained on 100k batched dataset.
"""

import logging
import warnings
import torch
import os
from pathlib import Path
from sentence_transformers.provence import ProvenceEncoder
from datasets import load_from_disk
import numpy as np
from tqdm import tqdm

# Suppress NLTK warnings
warnings.filterwarnings("ignore", message="Failed to load NLTK tokenizer")
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

# Setup logging
logging.basicConfig(
    format='%(asctime)s - %(levelname)s - %(name)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S',
    level=logging.INFO
)
logger = logging.getLogger(__name__)

def main():
    logger.info("100kバッチ学習モデルの評価を開始...")
    
    # Configuration
    model_path = "tmp/models/provence-100k-batched/checkpoint-500-best"
    dataset_path = "tmp/datasets/dev-dataset/small-100k-batched"
    
    # Load model
    logger.info(f"モデル読み込み: {model_path}")
    model = ProvenceEncoder.from_pretrained(model_path)
    model.eval()
    
    # Load test dataset
    logger.info(f"テストデータセット読み込み: {dataset_path}")
    dataset = load_from_disk(dataset_path)
    test_data = dataset["test"]
    
    # Evaluation on larger subset
    num_samples = 200
    thresholds = [0.001, 0.01, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.4, 0.5]
    
    logger.info(f"\n評価サンプル数: {num_samples}")
    logger.info("各閾値での圧縮率を評価中...")
    
    results = {t: [] for t in thresholds}
    
    for i in tqdm(range(min(num_samples, len(test_data))), desc="評価中"):
        sample = test_data[i]
        query = sample['query']
        
        # Test on all 5 texts
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
                    results[threshold].append({
                        'compression': result.compression_ratio,
                        'teacher_score': teacher_score,
                        'label': label,
                        'position': j
                    })
                except Exception as e:
                    pass
    
    # Detailed analysis
    logger.info("\n=== 圧縮率統計（100kモデル） ===")
    for threshold in thresholds:
        data = results[threshold]
        if data:
            compressions = [d['compression'] for d in data]
            mean_comp = np.mean(compressions)
            std_comp = np.std(compressions)
            median_comp = np.median(compressions)
            
            # Separate by relevance
            relevant = [d['compression'] for d in data if d['label'] == 1]
            non_relevant = [d['compression'] for d in data if d['label'] == 0]
            
            logger.info(f"\n閾値 {threshold}:")
            logger.info(f"  全体: 平均 {mean_comp:.1%} ± {std_comp:.1%}, 中央値 {median_comp:.1%}")
            if relevant:
                logger.info(f"  関連文書: 平均 {np.mean(relevant):.1%}")
            if non_relevant:
                logger.info(f"  非関連文書: 平均 {np.mean(non_relevant):.1%}")
            
            # Count distribution
            non_zero = sum(1 for c in compressions if c > 0)
            over_25 = sum(1 for c in compressions if c > 0.25)
            over_50 = sum(1 for c in compressions if c > 0.5)
            over_75 = sum(1 for c in compressions if c > 0.75)
            
            logger.info(f"  圧縮分布:")
            logger.info(f"    > 0%: {non_zero}/{len(compressions)} ({non_zero/len(compressions):.1%})")
            logger.info(f"    > 25%: {over_25}/{len(compressions)} ({over_25/len(compressions):.1%})")
            logger.info(f"    > 50%: {over_50}/{len(compressions)} ({over_50/len(compressions):.1%})")
            logger.info(f"    > 75%: {over_75}/{len(compressions)} ({over_75/len(compressions):.1%})")
    
    # Analyze by teacher score ranges
    logger.info("\n=== 教師スコア別分析（閾値0.1, 0.2） ===")
    for t in [0.1, 0.2]:
        stats = results[t]
        if stats:
            logger.info(f"\n閾値 {t}:")
            score_ranges = [(0, 0.2), (0.2, 0.4), (0.4, 0.6), (0.6, 0.8), (0.8, 1.0)]
            for low, high in score_ranges:
                in_range = [s['compression'] for s in stats 
                           if low <= s['teacher_score'] < high]
                if in_range:
                    logger.info(f"  スコア {low:.1f}-{high:.1f}: 平均圧縮率 {np.mean(in_range):.1%} (n={len(in_range)})")
    
    # Analyze by text position
    logger.info("\n=== テキスト位置別分析（閾値0.2） ===")
    stats_02 = results[0.2]
    if stats_02:
        for idx in range(5):
            at_position = [s['compression'] for s in stats_02 if s['position'] == idx]
            if at_position:
                logger.info(f"  位置 {idx+1}: 平均圧縮率 {np.mean(at_position):.1%} (n={len(at_position)})")
    
    # Test specific examples
    logger.info("\n=== 具体例テスト ===")
    
    # Find examples with different characteristics
    test_cases = []
    
    # High relevance example
    for i in range(min(50, len(test_data))):
        sample = test_data[i]
        for j, (score, label) in enumerate(zip(sample['teacher_scores'], sample['ranking_labels'])):
            if score > 0.8 and label == 1:
                test_cases.append((i, j, "高関連性", score))
                break
        if len(test_cases) >= 1:
            break
    
    # Low relevance example
    for i in range(min(50, len(test_data))):
        sample = test_data[i]
        for j, (score, label) in enumerate(zip(sample['teacher_scores'], sample['ranking_labels'])):
            if score < 0.2 and label == 0:
                test_cases.append((i, j, "低関連性", score))
                break
        if len(test_cases) >= 2:
            break
    
    # Medium relevance example
    for i in range(min(50, len(test_data))):
        sample = test_data[i]
        for j, (score, label) in enumerate(zip(sample['teacher_scores'], sample['ranking_labels'])):
            if 0.4 < score < 0.6:
                test_cases.append((i, j, "中関連性", score))
                break
        if len(test_cases) >= 3:
            break
    
    for i, j, case_type, score in test_cases:
        sample = test_data[i]
        query = sample['query']
        text = sample['texts'][j]
        
        logger.info(f"\n{case_type}例（教師スコア: {score:.3f}）:")
        logger.info(f"  クエリ: {query[:80]}...")
        logger.info(f"  圧縮率:")
        
        for threshold in [0.05, 0.1, 0.15, 0.2, 0.25, 0.3]:
            try:
                result = model.predict_with_pruning(
                    (query, text),
                    pruning_threshold=threshold,
                    return_documents=True
                )
                logger.info(f"    閾値 {threshold}: {result.compression_ratio:.1%}")
            except Exception as e:
                logger.warning(f"    閾値 {threshold}: エラー")
    
    logger.info("\n100kモデル評価完了!")

if __name__ == "__main__":
    main()