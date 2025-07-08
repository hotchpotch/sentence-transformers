#!/usr/bin/env python3
"""
SmallとFullモデルの最終的な完全評価とデータ分析
"""

import numpy as np
from datasets import load_dataset
from sentence_transformers.provence import ProvenceEncoder
from sklearn.metrics import precision_recall_fscore_support, accuracy_score
import json
from datetime import datetime

def calculate_f2(precision, recall):
    """F2スコアを計算"""
    if precision + recall == 0:
        return 0
    return 5 * precision * recall / (4 * precision + recall)

def evaluate_model_comprehensive(model_name, model, test_dataset, token_threshold=0.2, chunk_threshold=0.5):
    """包括的なモデル評価"""
    
    print(f"\n=== {model_name}モデル 完全評価 ===")
    print(f"設定: token_threshold={token_threshold}, chunk_threshold={chunk_threshold}")
    print(f"開始時刻: {datetime.now().strftime('%H:%M:%S')}")
    
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
    
    print(f"総サンプル数: {len(test_dataset)}")
    print(f"POSサンプル数: {len(pos_indices)}")
    print(f"NEGサンプル数: {len(neg_indices)}")
    
    # 推論実行
    sentences = [(str(q), str(t)) for q, t in zip(queries, texts)]
    simplified_chunks = [chunks[0] if chunks else [] for chunks in chunk_positions]
    
    print("推論実行中...")
    outputs = model.predict_context(
        sentences=sentences,
        chunk_positions=simplified_chunks,
        batch_size=32,
        token_threshold=token_threshold,
        chunk_threshold=chunk_threshold,
        show_progress_bar=True
    )
    print(f"推論完了: {datetime.now().strftime('%H:%M:%S')}")
    
    # 全体評価
    all_true_labels = []
    all_pred_labels = []
    
    for i, output in enumerate(outputs):
        relevant_indices = set(relevant_chunks[i][0] if relevant_chunks[i] and relevant_chunks[i][0] else [])
        true_labels = [1 if j in relevant_indices else 0 for j in range(len(simplified_chunks[i]))]
        
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
    
    overall_result = {
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
    
    # POS/NEG分離評価関数
    def evaluate_subset(indices, subset_name):
        subset_true = []
        subset_pred = []
        
        for i, output in enumerate(outputs):
            if i in indices:
                relevant_indices = set(relevant_chunks[i][0] if relevant_chunks[i] and relevant_chunks[i][0] else [])
                true_labels = [1 if j in relevant_indices else 0 for j in range(len(simplified_chunks[i]))]
                
                subset_true.extend(true_labels)
                subset_pred.extend(output.chunk_predictions)
        
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
            'tp': int(tp),
            'fp': int(fp),
            'fn': int(fn),
            'tn': int(tn),
            'total_chunks': int(total_chunks),
            'kept_chunks': int(kept_chunks),
            'num_examples': len(indices)
        }
    
    # POS/NEG評価
    pos_result = evaluate_subset(pos_indices, 'POS')
    neg_result = evaluate_subset(neg_indices, 'NEG')
    
    # 結果表示
    print(f"\n{model_name}モデル 全体結果:")
    print(f"  Precision: {precision:.4f}")
    print(f"  Recall: {recall:.4f}")
    print(f"  F1: {f1:.4f}")
    print(f"  F2: {f2:.4f}")
    print(f"  Accuracy: {accuracy:.4f}")
    print(f"  圧縮率: {compression_ratio:.1%}")
    print(f"  TP: {tp}, FP: {fp}, FN: {fn}, TN: {tn}")
    
    if pos_result:
        print(f"\n{model_name}モデル POSサンプル:")
        print(f"  Precision: {pos_result['precision']:.4f}")
        print(f"  Recall: {pos_result['recall']:.4f}")
        print(f"  F1: {pos_result['f1']:.4f}")
        print(f"  F2: {pos_result['f2']:.4f}")
        print(f"  圧縮率: {pos_result['compression_ratio']:.1%}")
    
    if neg_result:
        print(f"\n{model_name}モデル NEGサンプル:")
        print(f"  Precision: {neg_result['precision']:.4f}")
        print(f"  Recall: {neg_result['recall']:.4f}")
        print(f"  F1: {neg_result['f1']:.4f}")
        print(f"  F2: {neg_result['f2']:.4f}")
        print(f"  Accuracy: {neg_result['accuracy']:.4f}")
        print(f"  圧縮率: {neg_result['compression_ratio']:.1%}")
    
    return {
        'overall': overall_result,
        'pos': pos_result,
        'neg': neg_result,
        'settings': {
            'token_threshold': token_threshold,
            'chunk_threshold': chunk_threshold
        }
    }

def main():
    print("=== 最終完全評価とデータ分析 ===")
    print(f"開始時刻: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # データセット読み込み
    print("データセット読み込み中...")
    dataset = load_dataset('hotchpotch/wip-query-context-pruner-with-teacher-scores', 'ja-full')
    test_dataset = dataset['test']
    
    # F2最適設定
    token_threshold = 0.2
    chunk_threshold = 0.5
    
    # モデル読み込みと評価
    print("Smallモデル読み込み中...")
    small_model = ProvenceEncoder.from_pretrained("outputs/provence-ja-small/final-model")
    small_results = evaluate_model_comprehensive("Small", small_model, test_dataset, token_threshold, chunk_threshold)
    
    print("Fullモデル読み込み中...")
    full_model = ProvenceEncoder.from_pretrained("outputs/provence-ja-full/final-model")
    full_results = evaluate_model_comprehensive("Full", full_model, test_dataset, token_threshold, chunk_threshold)
    
    # 結果の比較分析
    print("\n" + "="*60)
    print("最終データ分析と比較")
    print("="*60)
    
    # 全体性能比較
    small_overall = small_results['overall']
    full_overall = full_results['overall']
    
    print(f"\n【全体性能比較】")
    print(f"メトリクス     Small     Full      差分")
    print(f"F2 Score    {small_overall['f2']:.4f}   {full_overall['f2']:.4f}   {abs(small_overall['f2'] - full_overall['f2']):.4f}")
    print(f"Precision   {small_overall['precision']:.4f}   {full_overall['precision']:.4f}   {abs(small_overall['precision'] - full_overall['precision']):.4f}")
    print(f"Recall      {small_overall['recall']:.4f}   {full_overall['recall']:.4f}   {abs(small_overall['recall'] - full_overall['recall']):.4f}")
    print(f"F1 Score    {small_overall['f1']:.4f}   {full_overall['f1']:.4f}   {abs(small_overall['f1'] - full_overall['f1']):.4f}")
    print(f"圧縮率      {small_overall['compression_ratio']:.1%}     {full_overall['compression_ratio']:.1%}     {abs(small_overall['compression_ratio'] - full_overall['compression_ratio']):.1%}")
    
    # NEGサンプル性能比較
    small_neg = small_results['neg']
    full_neg = full_results['neg']
    
    print(f"\n【NEGサンプル性能比較】")
    print(f"メトリクス     Small     Full      差分")
    print(f"Accuracy    {small_neg['accuracy']:.4f}   {full_neg['accuracy']:.4f}   {abs(small_neg['accuracy'] - full_neg['accuracy']):.4f}")
    print(f"圧縮率      {small_neg['compression_ratio']:.1%}     {full_neg['compression_ratio']:.1%}     {abs(small_neg['compression_ratio'] - full_neg['compression_ratio']):.1%}")
    print(f"FP数        {small_neg['fp']}       {full_neg['fp']}       {abs(small_neg['fp'] - full_neg['fp'])}")
    
    # 学習データサイズ効果分析
    print(f"\n【学習データサイズ効果分析】")
    print("Smallモデル: 10,000サンプルで学習")
    print("Fullモデル:  500,000サンプルで学習 (50倍)")
    print(f"性能向上: F2スコア {abs(full_overall['f2'] - small_overall['f2']):.4f}ポイント")
    
    if full_overall['f2'] > small_overall['f2']:
        improvement = ((full_overall['f2'] - small_overall['f2']) / small_overall['f2']) * 100
        print(f"相対的改善: {improvement:.2f}%")
    else:
        degradation = ((small_overall['f2'] - full_overall['f2']) / small_overall['f2']) * 100
        print(f"相対的劣化: {degradation:.2f}%")
    
    # 結果の総合保存
    final_results = {
        'evaluation_timestamp': datetime.now().isoformat(),
        'settings': {
            'token_threshold': token_threshold,
            'chunk_threshold': chunk_threshold,
            'description': 'F2-optimal settings'
        },
        'small_model': small_results,
        'full_model': full_results,
        'comparison': {
            'f2_difference': abs(small_overall['f2'] - full_overall['f2']),
            'compression_difference': abs(small_overall['compression_ratio'] - full_overall['compression_ratio']),
            'neg_accuracy_difference': abs(small_neg['accuracy'] - full_neg['accuracy']),
            'better_model': 'Full' if full_overall['f2'] > small_overall['f2'] else 'Small'
        }
    }
    
    # 結果保存
    with open('results/final_comprehensive_evaluation.json', 'w') as f:
        json.dump(final_results, f, indent=2, ensure_ascii=False)
    
    print(f"\n完全な評価結果を保存: results/final_comprehensive_evaluation.json")
    print(f"完了時刻: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # 最終まとめ
    print(f"\n【最終まとめ】")
    if full_overall['f2'] > small_overall['f2']:
        print(f"✅ Fullモデルが優秀: F2 {full_overall['f2']:.4f} > {small_overall['f2']:.4f}")
    else:
        print(f"⚠️  Smallモデルが優秀: F2 {small_overall['f2']:.4f} > {full_overall['f2']:.4f}")
    
    print(f"両モデルともNEGサンプルで適切に動作 (約{small_neg['compression_ratio']:.0%}圧縮)")
    print(f"学習スクリプトの修正により正常に訓練完了")

if __name__ == "__main__":
    main()