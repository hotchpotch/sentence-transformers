#!/usr/bin/env python3
"""
Quick evaluation of 100k trained model.
"""

import logging
import warnings
import torch
import os
from pathlib import Path
from sentence_transformers.provence import ProvenceEncoder
from datasets import load_from_disk
import numpy as np

# Suppress warnings
warnings.filterwarnings("ignore")
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
logging.getLogger("sentence_transformers.utils.text_chunking").setLevel(logging.ERROR)

# Setup logging
logging.basicConfig(
    format='%(asctime)s - %(message)s',
    datefmt='%H:%M:%S',
    level=logging.INFO
)
logger = logging.getLogger(__name__)

def main():
    logger.info("100kモデルのクイック評価...")
    
    # Configuration
    model_path = "tmp/models/provence-100k-batched/checkpoint-500-best"
    dataset_path = "tmp/datasets/dev-dataset/small-100k-batched"
    
    # Load model
    logger.info(f"モデル読み込み中...")
    model = ProvenceEncoder.from_pretrained(model_path)
    model.eval()
    
    # Load test dataset
    dataset = load_from_disk(dataset_path)
    test_data = dataset["test"]
    
    # Quick test on 50 samples
    num_samples = 50
    thresholds = [0.01, 0.1, 0.2, 0.3]
    
    results = {t: [] for t in thresholds}
    
    logger.info(f"テスト開始 ({num_samples}サンプル)...")
    
    for i in range(num_samples):
        if i % 10 == 0:
            logger.info(f"  進捗: {i}/{num_samples}")
        
        sample = test_data[i]
        query = sample['query']
        
        # Test only first and last text
        for j in [0, 4]:
            text = sample['texts'][j]
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
                except:
                    pass
    
    # Results summary
    logger.info("\n=== 100kモデル結果サマリー ===")
    
    for threshold in thresholds:
        data = results[threshold]
        if data:
            compressions = [d['compression'] for d in data]
            mean_comp = np.mean(compressions)
            
            # By position
            first = [d['compression'] for d in data if d['position'] == 0]
            last = [d['compression'] for d in data if d['position'] == 4]
            
            # By relevance
            relevant = [d['compression'] for d in data if d['label'] == 1]
            non_relevant = [d['compression'] for d in data if d['label'] == 0]
            
            logger.info(f"\n閾値 {threshold}:")
            logger.info(f"  全体平均: {mean_comp:.1%}")
            logger.info(f"  1位テキスト: {np.mean(first):.1%}")
            logger.info(f"  5位テキスト: {np.mean(last):.1%}")
            if relevant:
                logger.info(f"  関連文書: {np.mean(relevant):.1%}")
            if non_relevant:
                logger.info(f"  非関連文書: {np.mean(non_relevant):.1%}")
            
            # Distribution
            non_zero = sum(1 for c in compressions if c > 0)
            over_50 = sum(1 for c in compressions if c > 0.5)
            logger.info(f"  圧縮あり: {non_zero}/{len(compressions)} ({non_zero/len(compressions):.1%})")
            logger.info(f"  50%以上圧縮: {over_50}/{len(compressions)} ({over_50/len(compressions):.1%})")
    
    # Score range analysis for threshold 0.2
    logger.info("\n=== 教師スコア別分析（閾値0.2） ===")
    data_02 = results[0.2]
    if data_02:
        ranges = [(0, 0.3), (0.3, 0.7), (0.7, 1.0)]
        for low, high in ranges:
            in_range = [d['compression'] for d in data_02 
                       if low <= d['teacher_score'] < high]
            if in_range:
                logger.info(f"スコア {low:.1f}-{high:.1f}: 平均 {np.mean(in_range):.1%}")
    
    # Test examples
    logger.info("\n=== 具体例 ===")
    
    # Find high/low relevance examples
    for i in range(20):
        sample = test_data[i]
        
        # High relevance
        if sample['teacher_scores'][0] > 0.8:
            logger.info(f"\n高関連性例（スコア: {sample['teacher_scores'][0]:.2f}）:")
            query = sample['query']
            text = sample['texts'][0]
            
            for t in [0.1, 0.2, 0.3]:
                try:
                    result = model.predict_with_pruning(
                        (query, text),
                        pruning_threshold=t,
                        return_documents=True
                    )
                    logger.info(f"  閾値 {t}: {result.compression_ratio:.1%}")
                except:
                    pass
            break
    
    for i in range(20):
        sample = test_data[i]
        
        # Low relevance
        if sample['teacher_scores'][4] < 0.2:
            logger.info(f"\n低関連性例（スコア: {sample['teacher_scores'][4]:.2f}）:")
            query = sample['query']
            text = sample['texts'][4]
            
            for t in [0.1, 0.2, 0.3]:
                try:
                    result = model.predict_with_pruning(
                        (query, text),
                        pruning_threshold=t,
                        return_documents=True
                    )
                    logger.info(f"  閾値 {t}: {result.compression_ratio:.1%}")
                except:
                    pass
            break
    
    logger.info("\n評価完了!")

if __name__ == "__main__":
    main()