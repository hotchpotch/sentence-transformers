#!/usr/bin/env python3
"""
Optimize thresholds for F0.5 score (Precision-focused)
"""

import argparse
import json
import numpy as np
from typing import Dict, List, Tuple
import pandas as pd
from itertools import product

from datasets import load_dataset
from sentence_transformers.provence import ProvenceEncoder

def f_beta_score(precision: float, recall: float, beta: float = 2.0) -> float:
    """Calculate F-beta score with recall emphasis (beta=2.0)"""
    if precision + recall == 0:
        return 0.0
    return (1 + beta**2) * (precision * recall) / (beta**2 * precision + recall)

def evaluate_with_f05(model, examples, token_threshold, chunk_threshold, batch_size=32):
    """Evaluate model with F0.5 score focus"""
    
    # Prepare data
    sentences = [(ex['query'], ex['text']) for ex in examples]
    chunk_positions = [ex['chunks_pos'] for ex in examples]
    true_chunks = [ex['relevant_chunks'] for ex in examples]
    
    # Get predictions
    try:
        outputs = model.predict_context(
            sentences,
            chunk_positions,
            batch_size=batch_size,
            token_threshold=token_threshold,
            chunk_threshold=chunk_threshold,
            show_progress_bar=False
        )
    except Exception as e:
        print(f"Error in prediction: {e}")
        return None
    
    # Convert to binary labels and calculate metrics
    all_true = []
    all_pred = []
    
    for true_chunk_indices, output, chunk_pos in zip(true_chunks, outputs, chunk_positions):
        # Convert indices to binary labels
        num_chunks = len(chunk_pos)
        true_binary = [0] * num_chunks
        for idx in true_chunk_indices:
            if idx < num_chunks:
                true_binary[idx] = 1
        
        # Get predictions (ensure same length)
        pred_binary = output.chunk_predictions.tolist()[:num_chunks]
        pred_binary += [0] * max(0, num_chunks - len(pred_binary))
        
        all_true.extend(true_binary)
        all_pred.extend(pred_binary)
    
    # Calculate metrics
    if not all_true or not all_pred:
        return None
        
    tp = sum(1 for t, p in zip(all_true, all_pred) if t == 1 and p == 1)
    fp = sum(1 for t, p in zip(all_true, all_pred) if t == 0 and p == 1)
    fn = sum(1 for t, p in zip(all_true, all_pred) if t == 1 and p == 0)
    tn = sum(1 for t, p in zip(all_true, all_pred) if t == 0 and p == 0)
    
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    accuracy = (tp + tn) / (tp + fp + fn + tn) if (tp + fp + fn + tn) > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
    f2 = f_beta_score(precision, recall, beta=2.0)
    
    # Calculate compression metrics
    total_chunks = len(all_true)
    kept_chunks = sum(all_pred)
    compression_ratio = 1.0 - (kept_chunks / total_chunks) if total_chunks > 0 else 0.0
    
    return {
        'token_threshold': token_threshold,
        'chunk_threshold': chunk_threshold,
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
    }

def load_test_data(dataset_name: str, max_samples: int = None):
    """Load test data for optimization"""
    dataset = load_dataset('hotchpotch/wip-query-context-pruner-with-teacher-scores', dataset_name)
    data = dataset['train']
    
    if max_samples:
        data = data.select(range(min(max_samples, len(data))))
    
    examples = []
    for example in data:
        for i, text in enumerate(example['texts']):
            if (example['labels'][i] == 1 and  # Only POS samples
                i < len(example['chunks_pos']) and 
                i < len(example['relevant_chunks'])):
                
                chunks_pos = example['chunks_pos'][i]
                relevant_chunks = example['relevant_chunks'][i]
                
                if chunks_pos and relevant_chunks:
                    examples.append({
                        'query': example['query'],
                        'text': text,
                        'chunks_pos': chunks_pos,
                        'relevant_chunks': relevant_chunks
                    })
    
    return examples

def optimize_thresholds(model_path: str, dataset_name: str, max_samples: int = 200):
    """Optimize thresholds for best F2 score (Recall emphasis)"""
    
    print(f"=== F2スコア最適化（Recall重視） ===")
    print(f"モデル: {model_path}")
    print(f"データセット: {dataset_name}")
    print(f"最大サンプル数: {max_samples}")
    
    # Load model
    print("モデル読み込み中...")
    model = ProvenceEncoder.from_pretrained(model_path)
    model.eval()
    
    # Load test data
    print("テストデータ読み込み中...")
    examples = load_test_data(dataset_name, max_samples)
    print(f"読み込んだPOSサンプル数: {len(examples)}")
    
    # Define threshold ranges
    token_thresholds = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7]
    chunk_thresholds = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7]
    
    total_combinations = len(token_thresholds) * len(chunk_thresholds)
    print(f"評価する閾値組み合わせ数: {total_combinations}")
    
    results = []
    
    # Test all combinations
    for i, (token_th, chunk_th) in enumerate(product(token_thresholds, chunk_thresholds)):
        print(f"進捗: {i+1}/{total_combinations} - トークン={token_th}, チャンク={chunk_th}")
        
        result = evaluate_with_f05(model, examples, token_th, chunk_th)
        if result:
            results.append(result)
            print(f"  F2: {result['f2']:.3f}, P: {result['precision']:.3f}, R: {result['recall']:.3f}, 圧縮: {result['compression_ratio']:.1%}")
    
    if not results:
        print("有効な結果が得られませんでした")
        return
    
    # Sort by F2 score
    results.sort(key=lambda x: x['f2'], reverse=True)
    
    print(f"\n=== F2スコア最適化結果 ===")
    print(f"最適な設定上位10個:")
    print("順位 | トークン | チャンク | F2     | Precision | Recall | F1     | 圧縮率  | 保持チャンク")
    print("-" * 80)
    
    for i, result in enumerate(results[:10]):
        print(f"{i+1:2d}   | {result['token_threshold']:6.1f}   | {result['chunk_threshold']:6.1f}   | "
              f"{result['f2']:6.3f} | {result['precision']:9.3f} | {result['recall']:6.3f} | "
              f"{result['f1']:6.3f} | {result['compression_ratio']:6.1%} | "
              f"{result['kept_chunks']}/{result['total_chunks']}")
    
    # Best result analysis
    best = results[0]
    print(f"\n=== 最適設定詳細 ===")
    print(f"トークン閾値: {best['token_threshold']}")
    print(f"チャンク閾値: {best['chunk_threshold']}")
    print(f"F2スコア: {best['f2']:.3f}")
    print(f"Precision: {best['precision']:.3f}")
    print(f"Recall: {best['recall']:.3f}")
    print(f"F1スコア: {best['f1']:.3f}")
    print(f"圧縮率: {best['compression_ratio']:.1%}")
    print(f"Confusion Matrix:")
    print(f"  TP: {best['tp']}, FP: {best['fp']}")
    print(f"  FN: {best['fn']}, TN: {best['tn']}")
    
    # Compare with current best F1
    f1_best = max(results, key=lambda x: x['f1'])
    if f1_best != best:
        print(f"\n=== F1最適設定との比較 ===")
        print(f"F1最適: トークン={f1_best['token_threshold']}, チャンク={f1_best['chunk_threshold']}")
        print(f"  F1: {f1_best['f1']:.3f}, F2: {f1_best['f2']:.3f}, 圧縮: {f1_best['compression_ratio']:.1%}")
        print(f"F2最適: トークン={best['token_threshold']}, チャンク={best['chunk_threshold']}")
        print(f"  F1: {best['f1']:.3f}, F2: {best['f2']:.3f}, 圧縮: {best['compression_ratio']:.1%}")
        
        print(f"\nF2最適化により:")
        print(f"  Precision: {f1_best['precision']:.3f} → {best['precision']:.3f} ({best['precision']-f1_best['precision']:+.3f})")
        print(f"  Recall: {f1_best['recall']:.3f} → {best['recall']:.3f} ({best['recall']-f1_best['recall']:+.3f})")
        print(f"  圧縮率: {f1_best['compression_ratio']:.1%} → {best['compression_ratio']:.1%}")
    
    # Save results
    output_file = f"results/f2_optimization_{dataset_name}.json"
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\n結果を保存: {output_file}")
    
    return results

def main():
    parser = argparse.ArgumentParser(description="Optimize thresholds for F2 score (Recall emphasis)")
    parser.add_argument("--model_path", type=str, default="outputs/provence-ja-small/final-model",
                       help="Path to trained model")
    parser.add_argument("--dataset", type=str, default="ja-small",
                       choices=["ja-minimal", "ja-small", "ja-full"],
                       help="Dataset to use for optimization")
    parser.add_argument("--max_samples", type=int, default=200,
                       help="Maximum number of samples to use")
    
    args = parser.parse_args()
    
    optimize_thresholds(args.model_path, args.dataset, args.max_samples)

if __name__ == "__main__":
    main()