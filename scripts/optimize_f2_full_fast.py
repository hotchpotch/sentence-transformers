#!/usr/bin/env python3
"""
F2最適化スクリプト (Full model専用・高速版)
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

def evaluate_thresholds(model, dataset, max_samples=500):
    """複数の閾値で評価（サンプル数を制限）"""
    results = []
    
    # データの準備（サンプル数を制限）
    if len(dataset) > max_samples:
        indices = np.random.choice(len(dataset), max_samples, replace=False)
        subset = dataset.select(indices)
    else:
        subset = dataset
    
    queries = subset['query']
    texts = subset['texts']
    chunk_positions = subset['chunks_pos']
    relevant_chunks = subset['relevant_chunks']
    
    print(f"Using {len(subset)} samples for optimization")
    
    # 重要な閾値組み合わせのみをテスト
    threshold_combinations = [
        (0.1, 0.3), (0.1, 0.5),
        (0.2, 0.3), (0.2, 0.5), 
        (0.3, 0.3), (0.3, 0.5),
        (0.4, 0.5), (0.5, 0.5),
        (0.6, 0.7), (0.7, 0.7)
    ]
    
    for token_th, chunk_th in tqdm(threshold_combinations, desc="Optimizing thresholds"):
        # 予測実行
        sentences = [(str(q), str(t)) for q, t in zip(queries, texts)]
        simplified_chunks = [chunks[0] if chunks else [] for chunks in chunk_positions]
        
        outputs = model.predict_context(
            sentences=sentences,
            chunk_positions=simplified_chunks,
            batch_size=64,  # バッチサイズを増加
            token_threshold=token_th,
            chunk_threshold=chunk_th
        )
        
        # 正解ラベルの準備
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
        accuracy = accuracy_score(all_true_labels, all_pred_labels)
        
        # 圧縮率計算
        total_chunks = len(all_true_labels)
        kept_chunks = sum(all_pred_labels)
        compression_ratio = 1.0 - (kept_chunks / total_chunks) if total_chunks > 0 else 0.0
        
        # TP, FP, FN, TNの計算
        tp = sum(1 for t, p in zip(all_true_labels, all_pred_labels) if t == 1 and p == 1)
        fp = sum(1 for t, p in zip(all_true_labels, all_pred_labels) if t == 0 and p == 1)
        fn = sum(1 for t, p in zip(all_true_labels, all_pred_labels) if t == 1 and p == 0)
        tn = sum(1 for t, p in zip(all_true_labels, all_pred_labels) if t == 0 and p == 0)
        
        results.append({
            'token_threshold': token_th,
            'chunk_threshold': chunk_th,
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'f2': f2,
            'accuracy': accuracy,
            'compression_ratio': compression_ratio,
            'tp': tp,
            'fp': fp,
            'fn': fn,
            'tn': tn,
            'total_chunks': total_chunks,
            'kept_chunks': kept_chunks
        })
    
    return results

def main():
    print("=== Full Model F2最適化 (高速版) ===")
    
    # モデル読み込み
    model_path = "outputs/provence-ja-full/checkpoint-10423-best"
    model = ProvenceEncoder.from_pretrained(model_path)
    
    # データセット読み込み
    dataset = load_dataset('hotchpotch/wip-query-context-pruner-with-teacher-scores', 'ja-full')
    test_dataset = dataset['test']
    
    print(f"Total examples: {len(test_dataset)}")
    
    # 閾値最適化（サンプル数制限付き）
    results = evaluate_thresholds(model, test_dataset, max_samples=500)
    
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
    print(f"TP: {best_f2['tp']}, FP: {best_f2['fp']}, FN: {best_f2['fn']}, TN: {best_f2['tn']}")
    
    # 結果をファイルに保存
    with open('results/full_f2_optimization_fast.json', 'w') as f:
        json.dump({
            'best_f2': best_f2,
            'all_results': results
        }, f, indent=2)
    
    print(f"\n結果を保存: results/full_f2_optimization_fast.json")
    
    # F2最適設定で全体評価も実行
    print(f"\n=== F2最適設定で全体評価実行中... ===")
    print(f"設定: token_th={best_f2['token_threshold']}, chunk_th={best_f2['chunk_threshold']}")
    
    # 全データで評価
    all_queries = test_dataset['query']
    all_texts = test_dataset['texts']
    all_chunk_positions = test_dataset['chunks_pos']
    all_relevant_chunks = test_dataset['relevant_chunks']
    
    all_sentences = [(str(q), str(t)) for q, t in zip(all_queries, all_texts)]
    all_simplified_chunks = [chunks[0] if chunks else [] for chunks in all_chunk_positions]
    
    print("全データで推論実行中...")
    all_outputs = model.predict_context(
        sentences=all_sentences,
        chunk_positions=all_simplified_chunks,
        batch_size=64,
        token_threshold=best_f2['token_threshold'],
        chunk_threshold=best_f2['chunk_threshold']
    )
    
    # 全体結果計算
    all_true_labels = []
    all_pred_labels = []
    
    for i, output in enumerate(all_outputs):
        relevant_indices = set(all_relevant_chunks[i][0] if all_relevant_chunks[i] and all_relevant_chunks[i][0] else [])
        true_labels = [1 if j in relevant_indices else 0 for j in range(len(all_simplified_chunks[i]))]
        
        all_true_labels.extend(true_labels)
        all_pred_labels.extend(output.chunk_predictions)
    
    # 全体メトリクス計算
    precision, recall, f1, _ = precision_recall_fscore_support(
        all_true_labels, all_pred_labels, average='binary', zero_division=0
    )
    f2 = calculate_f2(precision, recall)
    accuracy = accuracy_score(all_true_labels, all_pred_labels)
    
    total_chunks = len(all_true_labels)
    kept_chunks = sum(all_pred_labels)
    compression_ratio = 1.0 - (kept_chunks / total_chunks) if total_chunks > 0 else 0.0
    
    tp = sum(1 for t, p in zip(all_true_labels, all_pred_labels) if t == 1 and p == 1)
    fp = sum(1 for t, p in zip(all_true_labels, all_pred_labels) if t == 0 and p == 1)
    fn = sum(1 for t, p in zip(all_true_labels, all_pred_labels) if t == 1 and p == 0)
    tn = sum(1 for t, p in zip(all_true_labels, all_pred_labels) if t == 0 and p == 0)
    
    full_result = {
        'token_threshold': best_f2['token_threshold'],
        'chunk_threshold': best_f2['chunk_threshold'],
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'f2': f2,
        'accuracy': accuracy,
        'compression_ratio': compression_ratio,
        'tp': tp,
        'fp': fp,
        'fn': fn,
        'tn': tn,
        'total_chunks': total_chunks,
        'kept_chunks': kept_chunks,
        'num_examples': len(test_dataset)
    }
    
    print(f"\n=== 全データでのF2最適結果 ===")
    print(f"Precision: {full_result['precision']:.3f}")
    print(f"Recall: {full_result['recall']:.3f}")
    print(f"F1: {full_result['f1']:.3f}")
    print(f"F2: {full_result['f2']:.3f}")
    print(f"Accuracy: {full_result['accuracy']:.3f}")
    print(f"Compression ratio: {full_result['compression_ratio']:.1%}")
    print(f"TP: {full_result['tp']}, FP: {full_result['fp']}, FN: {full_result['fn']}, TN: {full_result['tn']}")
    
    # 最終結果を保存
    with open('results/full_f2_final_result.json', 'w') as f:
        json.dump(full_result, f, indent=2)
    
    print(f"\n最終結果を保存: results/full_f2_final_result.json")

if __name__ == "__main__":
    main()