#!/usr/bin/env python3
"""
F2最適設定でFullモデルを評価
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

def main():
    print("=== Full Model F2最適設定評価 ===")
    
    # F2最適パラメータ
    token_threshold = 0.2
    chunk_threshold = 0.5
    
    print(f"Token threshold: {token_threshold}")
    print(f"Chunk threshold: {chunk_threshold}")
    print()
    
    # モデル読み込み
    model_path = "outputs/provence-ja-full/checkpoint-10423-best"
    model = ProvenceEncoder.from_pretrained(model_path)
    
    # データセット読み込み
    dataset = load_dataset('hotchpotch/wip-query-context-pruner-with-teacher-scores', 'ja-full')
    test_dataset = dataset['test']
    
    print(f"Total examples: {len(test_dataset)}")
    
    # データの準備
    queries = test_dataset['query']
    texts = test_dataset['texts']
    chunk_positions = test_dataset['chunks_pos']
    relevant_chunks = test_dataset['relevant_chunks']
    
    # POS/NEG分離
    pos_indices = []
    neg_indices = []
    
    for i, rel_chunks in enumerate(relevant_chunks):
        # 最初のテキストにrelevant chunkがあるかチェック
        if rel_chunks and rel_chunks[0]:
            pos_indices.append(i)
        else:
            neg_indices.append(i)
    
    print(f"POS examples: {len(pos_indices)}")
    print(f"NEG examples: {len(neg_indices)}")
    print()
    
    # 全体評価
    sentences = [(str(q), str(t)) for q, t in zip(queries, texts)]
    simplified_chunks = [chunks[0] if chunks else [] for chunks in chunk_positions]
    
    print("推論実行中...")
    outputs = model.predict_context(
        sentences=sentences,
        chunk_positions=simplified_chunks,
        batch_size=64,
        token_threshold=token_threshold,
        chunk_threshold=chunk_threshold,
        show_progress_bar=True
    )
    
    # 全体結果計算
    all_true_labels = []
    all_pred_labels = []
    
    for i, output in enumerate(outputs):
        relevant_indices = set(relevant_chunks[i][0] if relevant_chunks[i] and relevant_chunks[i][0] else [])
        true_labels = [1 if j in relevant_indices else 0 for j in range(len(simplified_chunks[i]))]
        
        all_true_labels.extend(true_labels)
        all_pred_labels.extend(output.chunk_predictions)
    
    # 全体メトリクス
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
    
    print("=== 全体結果 ===")
    print(f"Precision: {precision:.3f}")
    print(f"Recall: {recall:.3f}")
    print(f"F1: {f1:.3f}")
    print(f"F2: {f2:.3f}")
    print(f"Accuracy: {accuracy:.3f}")
    print(f"Compression ratio: {compression_ratio:.1%}")
    print(f"TP: {tp}, FP: {fp}, FN: {fn}, TN: {tn}")
    print()
    
    # POS/NEG分離評価
    def evaluate_subset(indices, subset_name):
        subset_true = []
        subset_pred = []
        
        chunk_offset = 0
        for i, output in enumerate(outputs):
            chunk_count = len(simplified_chunks[i])
            if i in indices:
                relevant_indices = set(relevant_chunks[i][0] if relevant_chunks[i] and relevant_chunks[i][0] else [])
                true_labels = [1 if j in relevant_indices else 0 for j in range(chunk_count)]
                
                subset_true.extend(true_labels)
                subset_pred.extend(output.chunk_predictions)
            
            chunk_offset += chunk_count
        
        if not subset_true:
            return None
        
        precision, recall, f1, _ = precision_recall_fscore_support(
            subset_true, subset_pred, average='binary', zero_division=0
        )
        f2 = calculate_f2(precision, recall)
        accuracy = accuracy_score(subset_true, subset_pred)
        
        total_chunks = len(subset_true)
        kept_chunks = sum(subset_pred)
        compression_ratio = 1.0 - (kept_chunks / total_chunks) if total_chunks > 0 else 0.0
        
        tp = sum(1 for t, p in zip(subset_true, subset_pred) if t == 1 and p == 1)
        fp = sum(1 for t, p in zip(subset_true, subset_pred) if t == 0 and p == 1)
        fn = sum(1 for t, p in zip(subset_true, subset_pred) if t == 1 and p == 0)
        tn = sum(1 for t, p in zip(subset_true, subset_pred) if t == 0 and p == 0)
        
        return {
            'subset_type': subset_name,
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
            'num_examples': len(indices)
        }
    
    pos_result = evaluate_subset(pos_indices, 'POS')
    neg_result = evaluate_subset(neg_indices, 'NEG')
    
    if pos_result:
        print("=== POSサンプル結果 ===")
        print(f"Precision: {pos_result['precision']:.3f}")
        print(f"Recall: {pos_result['recall']:.3f}")
        print(f"F1: {pos_result['f1']:.3f}")
        print(f"F2: {pos_result['f2']:.3f}")
        print(f"Accuracy: {pos_result['accuracy']:.3f}")
        print(f"Compression ratio: {pos_result['compression_ratio']:.1%}")
        print(f"TP: {pos_result['tp']}, FP: {pos_result['fp']}, FN: {pos_result['fn']}, TN: {pos_result['tn']}")
        print()
    
    if neg_result:
        print("=== NEGサンプル結果 ===")
        print(f"Precision: {neg_result['precision']:.3f}")
        print(f"Recall: {neg_result['recall']:.3f}")
        print(f"F1: {neg_result['f1']:.3f}")
        print(f"F2: {neg_result['f2']:.3f}")
        print(f"Accuracy: {neg_result['accuracy']:.3f}")
        print(f"Compression ratio: {neg_result['compression_ratio']:.1%}")
        print(f"TP: {neg_result['tp']}, FP: {neg_result['fp']}, FN: {neg_result['fn']}, TN: {neg_result['tn']}")
        print()
    
    # 結果を保存
    result = {
        'f2_optimal_settings': {
            'token_threshold': token_threshold,
            'chunk_threshold': chunk_threshold
        },
        'overall': {
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'f2': f2,
            'accuracy': accuracy,
            'compression_ratio': compression_ratio,
            'tp': int(tp),
            'fp': int(fp),
            'fn': int(fn),
            'tn': int(tn),
            'total_chunks': int(total_chunks),
            'kept_chunks': int(kept_chunks),
            'num_examples': len(test_dataset)
        }
    }
    
    if pos_result:
        result['POS'] = {k: (int(v) if isinstance(v, np.integer) else v) for k, v in pos_result.items()}
    if neg_result:
        result['NEG'] = {k: (int(v) if isinstance(v, np.integer) else v) for k, v in neg_result.items()}
    
    with open('results/full_f2_optimal_evaluation.json', 'w') as f:
        json.dump(result, f, indent=2)
    
    print(f"結果を保存: results/full_f2_optimal_evaluation.json")

if __name__ == "__main__":
    main()