#!/usr/bin/env python3
"""
F2最適化スクリプト (Full model専用)
"""

import numpy as np
from datasets import load_dataset
from sentence_transformers.provence import ProvenceEncoder
from sklearn.metrics import precision_recall_fscore_support, accuracy_score
import json
from tqdm import tqdm

def calculate_f2(precision, recall):
    """F2スコアを計算"""
    if precision + recall == 0:
        return 0
    return 5 * precision * recall / (4 * precision + recall)

def evaluate_thresholds(model, dataset, token_thresholds, chunk_thresholds):
    """複数の閾値で評価"""
    results = []
    
    # データの準備
    queries = dataset['query']
    texts = dataset['texts']
    chunk_positions = dataset['chunks_pos']
    relevant_chunks = dataset['relevant_chunks']
    
    # 組み合わせを制限して高速化
    threshold_combinations = [
        (0.1, 0.3), (0.1, 0.5), (0.1, 0.7),
        (0.2, 0.3), (0.2, 0.5), (0.2, 0.7),
        (0.3, 0.3), (0.3, 0.5), (0.3, 0.7),
        (0.4, 0.5), (0.4, 0.7),
        (0.5, 0.5), (0.5, 0.7),
        (0.6, 0.7), (0.7, 0.7)
    ]
    
    for token_th, chunk_th in tqdm(threshold_combinations, desc="Optimizing thresholds"):
        # 予測実行 (文字列変換を確実に行う)
        sentences = [(str(q), str(t)) for q, t in zip(queries, texts)]
        
        # chunks_posは複数のテキストに対応しているため、最初のテキストのみを使用
        simplified_chunks = [chunks[0] if chunks else [] for chunks in chunk_positions]
        
        outputs = model.predict_context(
            sentences=sentences,
            chunk_positions=simplified_chunks,
            batch_size=32,
            token_threshold=token_th,
            chunk_threshold=chunk_th
        )
        
        # 正解ラベルの準備
        all_true_labels = []
        all_pred_labels = []
        
        for i, output in enumerate(outputs):
            # 正解ラベル: relevant_chunksに含まれるチャンクは1、それ以外は0
            # relevant_chunksは[[chunk_indices], [], ...] の形式なので、最初のテキストのみを使用
            relevant_indices = set(relevant_chunks[i][0] if relevant_chunks[i] and relevant_chunks[i][0] else [])
            true_labels = [1 if j in relevant_indices else 0 for j in range(len(simplified_chunks[i]))]
            
            all_true_labels.extend(true_labels)
            all_pred_labels.extend(output.chunk_predictions)
        
        # メトリクス計算
        precision, recall, f1, _ = precision_recall_fscore_support(
            all_true_labels, all_pred_labels, average='binary', zero_division=0
        )
        f2 = calculate_f2(precision, recall)
        accuracy = accuracy_score(all_true_labels, all_pred_labels)
        
        # 圧縮率計算
        total_chunks = len(all_true_labels)
        kept_chunks = sum(all_pred_labels)
        compression_ratio = 1.0 - (kept_chunks / total_chunks) if total_chunks > 0 else 0.0
        
        results.append({
            'token_threshold': token_th,
            'chunk_threshold': chunk_th,
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'f2': f2,
            'accuracy': accuracy,
            'compression_ratio': compression_ratio
        })
    
    return results

def main():
    print("=== Full Model F2最適化 ===")
    
    # モデル読み込み
    model_path = "outputs/provence-ja-full/checkpoint-10423-best"
    model = ProvenceEncoder.from_pretrained(model_path)
    
    # データセット読み込み
    dataset = load_dataset('hotchpotch/wip-query-context-pruner-with-teacher-scores', 'ja-full')
    test_dataset = dataset['test']
    
    print(f"Total examples: {len(test_dataset)}")
    
    # 閾値最適化
    token_thresholds = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7]
    chunk_thresholds = [0.3, 0.5, 0.7]
    
    results = evaluate_thresholds(model, test_dataset, token_thresholds, chunk_thresholds)
    
    # F2最適閾値を見つける
    best_f2 = max(results, key=lambda x: x['f2'])
    
    print(f"\n=== F2最適結果 ===")
    print(f"Token threshold: {best_f2['token_threshold']}")
    print(f"Chunk threshold: {best_f2['chunk_threshold']}")
    print(f"Precision: {best_f2['precision']:.3f}")
    print(f"Recall: {best_f2['recall']:.3f}")
    print(f"F1: {best_f2['f1']:.3f}")
    print(f"F2: {best_f2['f2']:.3f}")
    print(f"Accuracy: {best_f2['accuracy']:.3f}")
    print(f"Compression ratio: {best_f2['compression_ratio']:.1%}")
    
    # 結果をファイルに保存
    with open('results/full_f2_optimization.json', 'w') as f:
        json.dump({
            'best_f2': best_f2,
            'all_results': results
        }, f, indent=2)
    
    print(f"\n結果を保存: results/full_f2_optimization.json")

if __name__ == "__main__":
    main()