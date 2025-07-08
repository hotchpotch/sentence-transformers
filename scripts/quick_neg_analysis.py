#!/usr/bin/env python3
"""
NEGサンプルでの問題を素早く分析
"""

import numpy as np
from datasets import load_dataset
from sentence_transformers.provence import ProvenceEncoder
from sklearn.metrics import precision_recall_fscore_support, accuracy_score
import json

def quick_analysis():
    print("=== NEGサンプル問題の素早い分析 ===")
    
    # モデル読み込み
    small_model = ProvenceEncoder.from_pretrained("outputs/provence-ja-small/final-model")
    full_model = ProvenceEncoder.from_pretrained("outputs/provence-ja-full/final-model")
    
    # データセット読み込み（少数サンプルのみ）
    dataset = load_dataset('hotchpotch/wip-query-context-pruner-with-teacher-scores', 'ja-full')
    test_dataset = dataset['test'].select(range(100))  # 最初の100例のみ
    
    # NEGサンプル抽出
    neg_indices = []
    for i, rel_chunks in enumerate(test_dataset['relevant_chunks']):
        if not rel_chunks or not rel_chunks[0]:
            neg_indices.append(i)
    
    print(f"分析対象: {len(test_dataset)}例中 {len(neg_indices)}個のNEGサンプル")
    
    if len(neg_indices) == 0:
        print("NEGサンプルが見つかりませんでした")
        return
    
    # 複数の閾値設定で比較
    thresholds = [
        (0.1, 0.1),  # 低閾値
        (0.2, 0.5),  # F2最適
        (0.5, 0.8),  # 高閾値
    ]
    
    for token_th, chunk_th in thresholds:
        print(f"\n=== 閾値設定: token={token_th}, chunk={chunk_th} ===")
        
        for model_name, model in [("Small", small_model), ("Full", full_model)]:
            # NEGサンプルのみで評価
            neg_true_labels = []
            neg_pred_labels = []
            neg_kept_counts = []
            
            for idx in neg_indices:
                example = test_dataset[idx]
                query = example['query']
                texts = example['texts']
                chunks_pos = example['chunks_pos']
                relevant_chunks = example['relevant_chunks']
                
                sentence = (str(query), str(texts[0]))
                chunk_positions = chunks_pos[0] if chunks_pos else []
                
                if not chunk_positions:
                    continue
                
                # 予測
                outputs = model.predict_context(
                    sentences=[sentence],
                    chunk_positions=[chunk_positions],
                    batch_size=1,
                    token_threshold=token_th,
                    chunk_threshold=chunk_th,
                    show_progress_bar=False
                )
                
                # 真の値（NEGなので全て0）
                true_labels = [0] * len(chunk_positions)
                pred_labels = outputs[0].chunk_predictions
                
                neg_true_labels.extend(true_labels)
                neg_pred_labels.extend(pred_labels)
                neg_kept_counts.append(sum(pred_labels))
            
            if neg_true_labels:
                precision, recall, f1, _ = precision_recall_fscore_support(
                    neg_true_labels, neg_pred_labels, average='binary', zero_division=0
                )
                accuracy = accuracy_score(neg_true_labels, neg_pred_labels)
                
                total_chunks = len(neg_true_labels)
                kept_chunks = sum(neg_pred_labels)
                compression_ratio = 1.0 - (kept_chunks / total_chunks) if total_chunks > 0 else 0.0
                
                print(f"  {model_name}モデル:")
                print(f"    Precision: {precision:.3f}")
                print(f"    Recall: {recall:.3f}")
                print(f"    F1: {f1:.3f}")
                print(f"    Accuracy: {accuracy:.3f}")
                print(f"    圧縮率: {compression_ratio:.1%}")
                print(f"    総チャンク数: {total_chunks}")
                print(f"    保持チャンク数: {kept_chunks}")
                print(f"    NEGサンプル平均保持: {np.mean(neg_kept_counts):.1f}")

if __name__ == "__main__":
    quick_analysis()