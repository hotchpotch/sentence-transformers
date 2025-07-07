#!/usr/bin/env python3
"""
Evaluate compression rates specifically on test positive samples.
Focus on relevant documents to understand pruning behavior.
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
    logger.info("テストポジティブサンプルの圧縮率評価...")
    
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
    
    # Evaluate only positive samples
    thresholds = [0.001, 0.01, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.4, 0.5]
    
    # Collect all positive samples
    positive_samples = []
    for i in range(len(test_data)):
        sample = test_data[i]
        query = sample['query']
        
        # Check each text for positive label
        for j, (text, label, teacher_score) in enumerate(
            zip(sample['texts'], sample['ranking_labels'], sample['teacher_scores'])
        ):
            if label == 1:  # Positive sample
                positive_samples.append({
                    'query': query,
                    'text': text,
                    'teacher_score': teacher_score,
                    'position': j,
                    'sample_idx': i
                })
    
    logger.info(f"ポジティブサンプル数: {len(positive_samples)}")
    
    # Evaluate compression rates
    results = {t: [] for t in thresholds}
    
    logger.info("圧縮率を評価中...")
    for sample in tqdm(positive_samples[:500], desc="評価中"):  # Evaluate up to 500 samples
        query = sample['query']
        text = sample['text']
        
        for threshold in thresholds:
            try:
                result = model.predict_with_pruning(
                    (query, text),
                    pruning_threshold=threshold,
                    return_documents=True
                )
                results[threshold].append({
                    'compression': result.compression_ratio,
                    'teacher_score': sample['teacher_score'],
                    'position': sample['position']
                })
            except:
                pass
    
    # Detailed analysis
    logger.info("\n=== ポジティブサンプルの圧縮率統計 ===")
    for threshold in thresholds:
        data = results[threshold]
        if data:
            compressions = [d['compression'] for d in data]
            mean_comp = np.mean(compressions)
            std_comp = np.std(compressions)
            median_comp = np.median(compressions)
            
            logger.info(f"\n閾値 {threshold}:")
            logger.info(f"  平均: {mean_comp:.1%} ± {std_comp:.1%}")
            logger.info(f"  中央値: {median_comp:.1%}")
            logger.info(f"  最小値: {np.min(compressions):.1%}")
            logger.info(f"  最大値: {np.max(compressions):.1%}")
            
            # Distribution
            zero_comp = sum(1 for c in compressions if c == 0)
            low_comp = sum(1 for c in compressions if 0 < c <= 0.25)
            med_comp = sum(1 for c in compressions if 0.25 < c <= 0.5)
            high_comp = sum(1 for c in compressions if c > 0.5)
            
            logger.info(f"  分布:")
            logger.info(f"    圧縮なし: {zero_comp} ({zero_comp/len(compressions):.1%})")
            logger.info(f"    低圧縮 (0-25%): {low_comp} ({low_comp/len(compressions):.1%})")
            logger.info(f"    中圧縮 (25-50%): {med_comp} ({med_comp/len(compressions):.1%})")
            logger.info(f"    高圧縮 (>50%): {high_comp} ({high_comp/len(compressions):.1%})")
    
    # Analyze by teacher score for threshold 0.1 and 0.2
    logger.info("\n=== 教師スコア別分析（ポジティブサンプルのみ） ===")
    for t in [0.1, 0.2]:
        stats = results[t]
        if stats:
            logger.info(f"\n閾値 {t}:")
            score_ranges = [(0.7, 0.8), (0.8, 0.9), (0.9, 1.0)]
            for low, high in score_ranges:
                in_range = [s['compression'] for s in stats 
                           if low <= s['teacher_score'] < high]
                if in_range:
                    logger.info(f"  スコア {low:.1f}-{high:.1f}: 平均 {np.mean(in_range):.1%} (n={len(in_range)})")
    
    # Analyze by position (ranking)
    logger.info("\n=== ランキング位置別分析（閾値0.2） ===")
    stats_02 = results[0.2]
    if stats_02:
        for pos in range(5):
            at_position = [s['compression'] for s in stats_02 if s['position'] == pos]
            if at_position:
                logger.info(f"  位置 {pos+1}: 平均 {np.mean(at_position):.1%} (n={len(at_position)})")
    
    # Show specific examples
    logger.info("\n=== 具体例（ポジティブサンプル） ===")
    
    # High score positive
    high_score_samples = [s for s in positive_samples if s['teacher_score'] > 0.9]
    if high_score_samples:
        sample = high_score_samples[0]
        logger.info(f"\n高スコアポジティブ（教師スコア: {sample['teacher_score']:.3f}）:")
        logger.info(f"  クエリ: {sample['query'][:80]}...")
        
        for t in [0.05, 0.1, 0.15, 0.2, 0.3]:
            try:
                result = model.predict_with_pruning(
                    (sample['query'], sample['text']),
                    pruning_threshold=t,
                    return_documents=True
                )
                logger.info(f"  閾値 {t}: {result.compression_ratio:.1%}")
                
                # Show pruned content for one threshold
                if t == 0.2 and result.compression_ratio > 0:
                    logger.info(f"  元の長さ: {len(sample['text'])} 文字")
                    logger.info(f"  圧縮後: {len(result.pruned_document)} 文字")
            except:
                pass
    
    # Medium score positive
    med_score_samples = [s for s in positive_samples if 0.5 < s['teacher_score'] < 0.7]
    if med_score_samples:
        sample = med_score_samples[0]
        logger.info(f"\n中スコアポジティブ（教師スコア: {sample['teacher_score']:.3f}）:")
        logger.info(f"  クエリ: {sample['query'][:80]}...")
        
        for t in [0.05, 0.1, 0.15, 0.2, 0.3]:
            try:
                result = model.predict_with_pruning(
                    (sample['query'], sample['text']),
                    pruning_threshold=t,
                    return_documents=True
                )
                logger.info(f"  閾値 {t}: {result.compression_ratio:.1%}")
            except:
                pass
    
    logger.info("\nポジティブサンプル評価完了!")

if __name__ == "__main__":
    main()