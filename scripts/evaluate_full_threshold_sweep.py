#!/usr/bin/env python3
"""
Fullモデルの閾値を変えながらNEGサンプルの性能を調査
"""

import numpy as np
from datasets import load_dataset
from sentence_transformers.provence import ProvenceEncoder
from sklearn.metrics import precision_recall_fscore_support, accuracy_score
import json

def calculate_f2(precision, recall):
    """F2スコアを計算"""
    if precision + recall == 0:
        return 0
    return 5 * precision * recall / (4 * precision + recall)

def evaluate_with_thresholds(model, test_dataset, token_thresholds, chunk_thresholds):
    """複数の閾値設定で評価"""
    
    # データの準備
    queries = test_dataset['query']
    texts = test_dataset['texts']
    chunk_positions = test_dataset['chunks_pos']
    relevant_chunks = test_dataset['relevant_chunks']
    
    # POS/NEG分離
    pos_indices = []
    neg_indices = []
    
    for i, rel_chunks in enumerate(relevant_chunks):
        if rel_chunks and rel_chunks[0]:
            pos_indices.append(i)
        else:
            neg_indices.append(i)
    
    sentences = [(str(q), str(t)) for q, t in zip(queries, texts)]
    simplified_chunks = [chunks[0] if chunks else [] for chunks in chunk_positions]
    
    results = {}
    
    for token_th in token_thresholds:
        for chunk_th in chunk_thresholds:
            print(f"評価中: token_th={token_th}, chunk_th={chunk_th}")
            
            outputs = model.predict_context(
                sentences=sentences,
                chunk_positions=simplified_chunks,
                batch_size=64,
                token_threshold=token_th,
                chunk_threshold=chunk_th,
                show_progress_bar=False
            )
            
            # 全体結果
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
            accuracy = accuracy_score(all_true_labels, all_pred_labels)
            
            # NEG専用結果
            neg_true = []
            neg_pred = []
            
            for i, output in enumerate(outputs):
                if i in neg_indices:
                    relevant_indices = set(relevant_chunks[i][0] if relevant_chunks[i] and relevant_chunks[i][0] else [])
                    true_labels = [1 if j in relevant_indices else 0 for j in range(len(simplified_chunks[i]))]
                    
                    neg_true.extend(true_labels)
                    neg_pred.extend(output.chunk_predictions)
            
            neg_precision, neg_recall, neg_f1, _ = precision_recall_fscore_support(
                neg_true, neg_pred, average='binary', zero_division=0
            )
            neg_f2 = calculate_f2(neg_precision, neg_recall)
            neg_accuracy = accuracy_score(neg_true, neg_pred)
            
            total_chunks = len(neg_true)
            kept_chunks = sum(neg_pred)
            compression_ratio = 1.0 - (kept_chunks / total_chunks) if total_chunks > 0 else 0.0
            
            results[f"token_{token_th}_chunk_{chunk_th}"] = {
                'token_threshold': token_th,
                'chunk_threshold': chunk_th,
                'overall': {
                    'precision': precision,
                    'recall': recall,
                    'f1': f1,
                    'f2': f2,
                    'accuracy': accuracy
                },
                'neg_only': {
                    'precision': neg_precision,
                    'recall': neg_recall,
                    'f1': neg_f1,
                    'f2': neg_f2,
                    'accuracy': neg_accuracy,
                    'compression_ratio': compression_ratio,
                    'total_chunks': total_chunks,
                    'kept_chunks': kept_chunks
                }
            }
    
    return results

def main():
    print("=== Fullモデル閾値スイープ評価 ===")
    
    # モデル読み込み
    model = ProvenceEncoder.from_pretrained("outputs/provence-ja-full/final-model")
    
    # データセット読み込み
    dataset = load_dataset('hotchpotch/wip-query-context-pruner-with-teacher-scores', 'ja-full')
    test_dataset = dataset['test']
    
    # 閾値設定
    token_thresholds = [0.05, 0.1, 0.15, 0.2, 0.3, 0.5]
    chunk_thresholds = [0.1, 0.2, 0.3, 0.5, 0.7, 0.9]
    
    print(f"Token thresholds: {token_thresholds}")
    print(f"Chunk thresholds: {chunk_thresholds}")
    print(f"Total combinations: {len(token_thresholds) * len(chunk_thresholds)}")
    print()
    
    # 評価実行
    results = evaluate_with_thresholds(model, test_dataset, token_thresholds, chunk_thresholds)
    
    # 結果分析
    print("=== NEGサンプル結果分析 ===")
    best_neg_f2 = 0
    best_neg_config = None
    
    for config, result in results.items():
        neg_result = result['neg_only']
        if neg_result['f2'] > best_neg_f2:
            best_neg_f2 = neg_result['f2']
            best_neg_config = config
        
        print(f"{config}: NEG F2={neg_result['f2']:.3f}, P={neg_result['precision']:.3f}, R={neg_result['recall']:.3f}, Acc={neg_result['accuracy']:.3f}, Comp={neg_result['compression_ratio']:.1%}")
    
    print(f"\nNEGサンプル最良設定: {best_neg_config}")
    print(f"NEGサンプル最良F2: {best_neg_f2:.3f}")
    
    # F2最適設定の問題を確認
    f2_optimal_result = results.get('token_0.2_chunk_0.5')
    if f2_optimal_result:
        print(f"\nF2最適設定(0.2, 0.5)のNEG結果:")
        neg = f2_optimal_result['neg_only']
        print(f"  Precision: {neg['precision']:.3f}")
        print(f"  Recall: {neg['recall']:.3f}")
        print(f"  F1: {neg['f1']:.3f}")
        print(f"  F2: {neg['f2']:.3f}")
        print(f"  Accuracy: {neg['accuracy']:.3f}")
        print(f"  Compression: {neg['compression_ratio']:.1%}")
        print(f"  Total chunks: {neg['total_chunks']}")
        print(f"  Kept chunks: {neg['kept_chunks']}")
    
    # 結果保存
    with open('results/full_threshold_sweep.json', 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\n結果を保存: results/full_threshold_sweep.json")

if __name__ == "__main__":
    main()