#!/usr/bin/env python3
"""
効率的にF2最適設定を探索
"""

import numpy as np
from datasets import load_dataset
from sentence_transformers.provence import ProvenceEncoder
from sklearn.metrics import precision_recall_fscore_support
import json

def calculate_f2(precision, recall):
    """F2スコアを計算"""
    if precision + recall == 0:
        return 0
    return 5 * precision * recall / (4 * precision + recall)

def find_optimal_f2(model_name, model, test_dataset):
    """特定モデルのF2最適設定を探索"""
    
    print(f"\n=== {model_name}モデル F2最適化 ===")
    
    # データの準備
    queries = test_dataset['query']
    texts = test_dataset['texts']
    chunk_positions = test_dataset['chunks_pos']
    relevant_chunks = test_dataset['relevant_chunks']
    
    sentences = [(str(q), str(t)) for q, t in zip(queries, texts)]
    simplified_chunks = [chunks[0] if chunks else [] for chunks in chunk_positions]
    
    # 段階的探索（粗い→細かい）
    print("段階1: 粗い探索")
    coarse_token_ths = [0.05, 0.1, 0.2, 0.3]
    coarse_chunk_ths = [0.1, 0.3, 0.5, 0.7]
    
    best_f2 = 0
    best_token_th = 0.1
    best_chunk_th = 0.5
    
    for token_th in coarse_token_ths:
        for chunk_th in coarse_chunk_ths:
            print(f"  測定中: token={token_th}, chunk={chunk_th}")
            
            outputs = model.predict_context(
                sentences=sentences,
                chunk_positions=simplified_chunks,
                batch_size=64,
                token_threshold=token_th,
                chunk_threshold=chunk_th,
                show_progress_bar=True
            )
            
            # メトリクス計算
            all_true_labels = []
            all_pred_labels = []
            
            for i, output in enumerate(outputs):
                relevant_indices = set(relevant_chunks[i][0] if relevant_chunks[i] and relevant_chunks[i][0] else [])
                true_labels = [1 if j in relevant_indices else 0 for j in range(len(simplified_chunks[i]))]
                
                all_true_labels.extend(true_labels)
                all_pred_labels.extend(output.chunk_predictions)
            
            precision, recall, f1, _ = precision_recall_fscore_support(
                all_true_labels, all_pred_labels, average='binary', zero_division=0
            )
            f2 = calculate_f2(precision, recall)
            
            total_chunks = len(all_true_labels)
            kept_chunks = sum(all_pred_labels)
            compression_ratio = 1.0 - (kept_chunks / total_chunks) if total_chunks > 0 else 0.0
            
            print(f"    F2: {f2:.4f}, P: {precision:.3f}, R: {recall:.3f}, 圧縮: {compression_ratio:.1%}")
            
            if f2 > best_f2:
                best_f2 = f2
                best_token_th = token_th
                best_chunk_th = chunk_th
    
    print(f"\n段階1最良: token={best_token_th}, chunk={best_chunk_th}, F2={best_f2:.4f}")
    
    # 段階2: 最良設定周辺の細かい探索
    print("\n段階2: 細かい探索")
    
    # 最良設定の周辺を細かく探索
    token_range = [best_token_th - 0.05, best_token_th, best_token_th + 0.05]
    chunk_range = [best_chunk_th - 0.1, best_chunk_th, best_chunk_th + 0.1]
    
    # 範囲を調整
    token_range = [max(0.01, t) for t in token_range if t <= 0.5]
    chunk_range = [max(0.05, c) for c in chunk_range if c <= 0.9]
    
    for token_th in token_range:
        for chunk_th in chunk_range:
            if token_th == best_token_th and chunk_th == best_chunk_th:
                continue  # 既に測定済み
                
            print(f"  測定中: token={token_th:.2f}, chunk={chunk_th:.1f}")
            
            outputs = model.predict_context(
                sentences=sentences,
                chunk_positions=simplified_chunks,
                batch_size=64,
                token_threshold=token_th,
                chunk_threshold=chunk_th,
                show_progress_bar=True
            )
            
            # メトリクス計算
            all_true_labels = []
            all_pred_labels = []
            
            for i, output in enumerate(outputs):
                relevant_indices = set(relevant_chunks[i][0] if relevant_chunks[i] and relevant_chunks[i][0] else [])
                true_labels = [1 if j in relevant_indices else 0 for j in range(len(simplified_chunks[i]))]
                
                all_true_labels.extend(true_labels)
                all_pred_labels.extend(output.chunk_predictions)
            
            precision, recall, f1, _ = precision_recall_fscore_support(
                all_true_labels, all_pred_labels, average='binary', zero_division=0
            )
            f2 = calculate_f2(precision, recall)
            
            total_chunks = len(all_true_labels)
            kept_chunks = sum(all_pred_labels)
            compression_ratio = 1.0 - (kept_chunks / total_chunks) if total_chunks > 0 else 0.0
            
            print(f"    F2: {f2:.4f}, P: {precision:.3f}, R: {recall:.3f}, 圧縮: {compression_ratio:.1%}")
            
            if f2 > best_f2:
                best_f2 = f2
                best_token_th = token_th
                best_chunk_th = chunk_th
    
    # 最終結果
    print(f"\n{model_name}モデル最終結果:")
    print(f"  Token threshold: {best_token_th}")
    print(f"  Chunk threshold: {best_chunk_th}")
    print(f"  F2 score: {best_f2:.4f}")
    
    return {
        'token_threshold': best_token_th,
        'chunk_threshold': best_chunk_th,
        'f2_score': best_f2
    }

def main():
    print("=== 効率的F2最適化 ===")
    
    # データセット読み込み
    print("データセット読み込み中...")
    dataset = load_dataset('hotchpotch/wip-query-context-pruner-with-teacher-scores', 'ja-full')
    test_dataset = dataset['test']
    print(f"テストサンプル数: {len(test_dataset)}")
    
    # モデル読み込みと最適化
    results = {}
    
    # Smallモデル
    print("Smallモデル読み込み中...")
    small_model = ProvenceEncoder.from_pretrained("outputs/provence-ja-small/final-model")
    small_result = find_optimal_f2("Small", small_model, test_dataset)
    results['small'] = small_result
    
    # Fullモデル
    print("Fullモデル読み込み中...")
    full_model = ProvenceEncoder.from_pretrained("outputs/provence-ja-full/final-model")
    full_result = find_optimal_f2("Full", full_model, test_dataset)
    results['full'] = full_result
    
    # 結果比較
    print("\n=== 最終比較 ===")
    print(f"Small: token={small_result['token_threshold']}, chunk={small_result['chunk_threshold']}, F2={small_result['f2_score']:.4f}")
    print(f"Full:  token={full_result['token_threshold']}, chunk={full_result['chunk_threshold']}, F2={full_result['f2_score']:.4f}")
    print(f"F2差分: {abs(small_result['f2_score'] - full_result['f2_score']):.4f}")
    
    # 結果保存
    with open('results/f2_optimal_settings.json', 'w') as f:
        json.dump(results, f, indent=2)
    
    print("\n結果を保存: results/f2_optimal_settings.json")

if __name__ == "__main__":
    main()