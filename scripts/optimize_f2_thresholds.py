#!/usr/bin/env python3
"""
SmallとFullモデルのF2最適化のための閾値調整
"""

import numpy as np
from datasets import load_dataset
from sentence_transformers.provence import ProvenceEncoder
from sklearn.metrics import precision_recall_fscore_support
import json
from itertools import product

def calculate_f2(precision, recall):
    """F2スコアを計算"""
    if precision + recall == 0:
        return 0
    return 5 * precision * recall / (4 * precision + recall)

def evaluate_thresholds(model, test_dataset, token_thresholds, chunk_thresholds):
    """閾値組み合わせでF2最適化"""
    
    # データの準備
    queries = test_dataset['query']
    texts = test_dataset['texts']
    chunk_positions = test_dataset['chunks_pos']
    relevant_chunks = test_dataset['relevant_chunks']
    
    sentences = [(str(q), str(t)) for q, t in zip(queries, texts)]
    simplified_chunks = [chunks[0] if chunks else [] for chunks in chunk_positions]
    
    best_f2 = 0
    best_config = None
    all_results = []
    
    total_combinations = len(token_thresholds) * len(chunk_thresholds)
    current = 0
    
    for token_th, chunk_th in product(token_thresholds, chunk_thresholds):
        current += 1
        print(f"  進捗: {current}/{total_combinations} - token={token_th}, chunk={chunk_th}")
        
        # 推論実行
        outputs = model.predict_context(
            sentences=sentences,
            chunk_positions=simplified_chunks,
            batch_size=32,  # バッチサイズを小さくして安定性向上
            token_threshold=token_th,
            chunk_threshold=chunk_th,
            show_progress_bar=True
        )
        
        # ラベル準備
        all_true_labels = []
        all_pred_labels = []
        
        for i, output in enumerate(outputs):
            relevant_indices = set(relevant_chunks[i][0] if relevant_chunks[i] and relevant_chunks[i][0] else [])
            true_labels = [1 if j in relevant_indices else 0 for j in range(len(simplified_chunks[i]))]
            
            all_true_labels.extend(true_labels)
            all_pred_labels.extend(output.chunk_predictions)
        
        # メトリクス計算
        precision, recall, f1, _ = precision_recall_fscore_support(
            all_true_labels, all_pred_labels, average='binary', zero_division=0
        )
        f2 = calculate_f2(precision, recall)
        
        total_chunks = len(all_true_labels)
        kept_chunks = sum(all_pred_labels)
        compression_ratio = 1.0 - (kept_chunks / total_chunks) if total_chunks > 0 else 0.0
        
        result = {
            'token_threshold': token_th,
            'chunk_threshold': chunk_th,
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'f2': f2,
            'compression_ratio': compression_ratio,
            'total_chunks': total_chunks,
            'kept_chunks': kept_chunks
        }
        all_results.append(result)
        
        print(f"    F2: {f2:.4f}, P: {precision:.3f}, R: {recall:.3f}, 圧縮: {compression_ratio:.1%}")
        
        # 最良スコア更新
        if f2 > best_f2:
            best_f2 = f2
            best_config = result
    
    return best_config, all_results

def main():
    print("=== F2最適化のための閾値調整 ===")
    
    # モデル読み込み
    print("モデル読み込み中...")
    small_model = ProvenceEncoder.from_pretrained("outputs/provence-ja-small/final-model")
    full_model = ProvenceEncoder.from_pretrained("outputs/provence-ja-full/final-model")
    
    # データセット読み込み
    print("データセット読み込み中...")
    dataset = load_dataset('hotchpotch/wip-query-context-pruner-with-teacher-scores', 'ja-full')
    test_dataset = dataset['test']
    
    print(f"テストサンプル数: {len(test_dataset)}")
    
    # 閾値範囲の設定（細かく設定）
    token_thresholds = [0.05, 0.1, 0.15, 0.2, 0.25, 0.3]
    chunk_thresholds = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7]
    
    print(f"Token thresholds: {token_thresholds}")
    print(f"Chunk thresholds: {chunk_thresholds}")
    print(f"総組み合わせ数: {len(token_thresholds) * len(chunk_thresholds)}")
    print()
    
    results = {}
    
    # Smallモデルの最適化
    print("=== Smallモデル F2最適化 ===")
    small_best, small_all = evaluate_thresholds(small_model, test_dataset, token_thresholds, chunk_thresholds)
    results['small'] = {
        'best_config': small_best,
        'all_results': small_all
    }
    
    print(f"\nSmallモデル最適設定:")
    print(f"  Token threshold: {small_best['token_threshold']}")
    print(f"  Chunk threshold: {small_best['chunk_threshold']}")
    print(f"  F2 score: {small_best['f2']:.4f}")
    print(f"  Precision: {small_best['precision']:.3f}")
    print(f"  Recall: {small_best['recall']:.3f}")
    print(f"  圧縮率: {small_best['compression_ratio']:.1%}")
    print()
    
    # Fullモデルの最適化
    print("=== Fullモデル F2最適化 ===")
    full_best, full_all = evaluate_thresholds(full_model, test_dataset, token_thresholds, chunk_thresholds)
    results['full'] = {
        'best_config': full_best,
        'all_results': full_all
    }
    
    print(f"\nFullモデル最適設定:")
    print(f"  Token threshold: {full_best['token_threshold']}")
    print(f"  Chunk threshold: {full_best['chunk_threshold']}")
    print(f"  F2 score: {full_best['f2']:.4f}")
    print(f"  Precision: {full_best['precision']:.3f}")
    print(f"  Recall: {full_best['recall']:.3f}")
    print(f"  圧縮率: {full_best['compression_ratio']:.1%}")
    print()
    
    # 結果保存
    with open('results/f2_optimization_results.json', 'w') as f:
        json.dump(results, f, indent=2)
    
    print("結果を保存: results/f2_optimization_results.json")
    
    # 比較表示
    print("\n=== モデル比較 ===")
    print(f"Small vs Full F2差: {abs(small_best['f2'] - full_best['f2']):.4f}")
    print(f"Small vs Full 圧縮率差: {abs(small_best['compression_ratio'] - full_best['compression_ratio']):.1%}")

if __name__ == "__main__":
    main()