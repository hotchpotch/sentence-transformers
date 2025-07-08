#!/usr/bin/env python3
"""
ja-smallとja-fullモデルのPOSサンプル圧縮率を比較
"""

import os
import torch
import numpy as np
from datasets import load_dataset
from tqdm import tqdm

from sentence_transformers.provence import ProvenceEncoder


def evaluate_pos_compression(model_path: str, model_name: str, threshold: float = 0.3):
    """POSサンプルのみの圧縮率を評価"""
    
    print(f"\n=== {model_name} POSサンプル圧縮率評価 ===")
    print(f"モデル: {model_path}")
    print(f"閾値: {threshold}")
    
    # データセットロード
    dataset = load_dataset('hotchpotch/wip-query-context-pruner-with-teacher-scores', 'ja-full')
    test_data = dataset['test']
    
    # モデルロード
    model = ProvenceEncoder.from_pretrained(model_path)
    model.eval()
    
    # POSサンプルのみ抽出
    pos_pairs = []
    for item in test_data:
        query = item['query']
        texts = item['texts']
        labels = item['labels']
        
        for text, label in zip(texts, labels):
            if label == 1:  # POSサンプルのみ
                pos_pairs.append((query, text))
    
    print(f"POSサンプル数: {len(pos_pairs):,}")
    
    # バッチサイズ512で推論
    print(f"推論実行中 (batch_size=512)...")
    outputs = model.predict_with_pruning(
        pos_pairs,
        batch_size=512,
        pruning_threshold=threshold,
        return_documents=True,
        show_progress_bar=True
    )
    
    # 圧縮率を計算
    compression_ratios = []
    original_lengths = []
    pruned_lengths = []
    
    for i, output in enumerate(outputs):
        original_text = pos_pairs[i][1]
        pruned_text = output.pruned_documents[0] if output.pruned_documents else ""
        
        original_len = len(original_text)
        pruned_len = len(pruned_text)
        
        original_lengths.append(original_len)
        pruned_lengths.append(pruned_len)
        compression_ratios.append(output.compression_ratio)
    
    # 統計計算
    avg_compression = np.mean(compression_ratios)
    std_compression = np.std(compression_ratios)
    total_original = sum(original_lengths)
    total_pruned = sum(pruned_lengths)
    overall_compression = 1 - (total_pruned / total_original) if total_original > 0 else 0
    
    print(f"\n📊 結果:")
    print(f"  平均圧縮率: {avg_compression:.1%} ± {std_compression:.1%}")
    print(f"  全体圧縮率: {overall_compression:.1%}")
    print(f"  元の総文字数: {total_original:,}")
    print(f"  圧縮後総文字数: {total_pruned:,}")
    print(f"  削減文字数: {total_original - total_pruned:,}")
    
    # 圧縮率の分布
    print(f"\n圧縮率分布:")
    bins = [0, 0.1, 0.3, 0.5, 0.7, 0.9, 1.0]
    hist, _ = np.histogram(compression_ratios, bins=bins)
    for i, (low, high) in enumerate(zip(bins[:-1], bins[1:])):
        count = hist[i]
        pct = count / len(compression_ratios) * 100
        print(f"  {low:.0%}-{high:.0%}: {count:,} ({pct:.1f}%)")
    
    return {
        'model': model_name,
        'avg_compression': avg_compression,
        'std_compression': std_compression,
        'overall_compression': overall_compression,
        'total_samples': len(pos_pairs)
    }


def main():
    """両モデルの比較を実行"""
    
    # 各閾値で比較
    thresholds = [0.1, 0.3, 0.5]
    
    for threshold in thresholds:
        print(f"\n{'='*60}")
        print(f"閾値 {threshold} での比較")
        print(f"{'='*60}")
        
        # ja-small
        small_path = "./outputs/provence-ja-small/final-model"
        if os.path.exists(small_path):
            small_results = evaluate_pos_compression(small_path, "ja-small", threshold)
        
        # ja-full
        full_path = "./outputs/provence-ja-full/checkpoint-10423-best"
        if os.path.exists(full_path):
            full_results = evaluate_pos_compression(full_path, "ja-full", threshold)
        
        # 比較表示
        if 'small_results' in locals() and 'full_results' in locals():
            print(f"\n📊 比較結果 (閾値={threshold}):")
            print(f"  ja-small: {small_results['avg_compression']:.1%} (全体: {small_results['overall_compression']:.1%})")
            print(f"  ja-full:  {full_results['avg_compression']:.1%} (全体: {full_results['overall_compression']:.1%})")
            print(f"  差分: {full_results['avg_compression'] - small_results['avg_compression']:.1%}")


if __name__ == "__main__":
    main()