#!/usr/bin/env python3
"""
Simple evaluation of batched Provence model.
"""

import logging
import warnings
import torch
import os
from pathlib import Path
from sentence_transformers.provence import ProvenceEncoder
from datasets import load_from_disk
import numpy as np

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
    logger.info("バッチ学習モデルの簡易評価...")
    
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
    
    # Quick evaluation on subset
    num_samples = 20
    thresholds = [0.01, 0.05, 0.1, 0.2, 0.3]
    
    logger.info(f"\nクイック評価（{num_samples}サンプル）")
    
    results = {t: [] for t in thresholds}
    
    for i in range(min(num_samples, len(test_data))):
        sample = test_data[i]
        query = sample['query']
        
        # Test on first (highest ranking) and last (lowest ranking) texts
        for j in [0, 4]:  # First and last
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
                except Exception as e:
                    pass
    
    # Summary statistics
    logger.info("\n=== 圧縮率サマリー ===")
    for threshold in thresholds:
        data = results[threshold]
        if data:
            compressions = [d['compression'] for d in data]
            mean_comp = np.mean(compressions)
            
            # By position
            first_comp = [d['compression'] for d in data if d['position'] == 0]
            last_comp = [d['compression'] for d in data if d['position'] == 4]
            
            logger.info(f"\n閾値 {threshold}:")
            logger.info(f"  全体平均: {mean_comp:.1%}")
            if first_comp:
                logger.info(f"  1位テキスト: {np.mean(first_comp):.1%}")
            if last_comp:
                logger.info(f"  5位テキスト: {np.mean(last_comp):.1%}")
            
            # Count non-zero
            non_zero = sum(1 for c in compressions if c > 0)
            logger.info(f"  圧縮あり: {non_zero}/{len(compressions)} ({non_zero/len(compressions):.1%})")
    
    # Test specific low-relevance examples
    logger.info("\n=== 低関連性サンプルテスト ===")
    for i in range(min(10, len(test_data))):
        sample = test_data[i]
        
        # Find low relevance text
        for j, (score, label) in enumerate(zip(sample['teacher_scores'], sample['ranking_labels'])):
            if score < 0.3 and label == 0:
                query = sample['query']
                text = sample['texts'][j]
                
                logger.info(f"\n教師スコア {score:.3f} のサンプル:")
                
                for threshold in [0.1, 0.2, 0.3]:
                    try:
                        result = model.predict_with_pruning(
                            (query, text),
                            pruning_threshold=threshold,
                            return_documents=True
                        )
                        logger.info(f"  閾値 {threshold}: {result.compression_ratio:.1%}")
                    except:
                        pass
                break
    
    logger.info("\n評価完了!")

if __name__ == "__main__":
    main()