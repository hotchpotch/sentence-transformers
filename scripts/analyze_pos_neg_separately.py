#!/usr/bin/env python3
"""
Analyze POS and NEG samples separately for threshold optimization
"""

import argparse
import json
import numpy as np
from typing import Dict, List, Tuple
from itertools import product

from datasets import load_dataset
from sentence_transformers.provence import ProvenceEncoder

def f_beta_score(precision: float, recall: float, beta: float = 2.0) -> float:
    """Calculate F-beta score"""
    if precision + recall == 0:
        return 0.0
    return (1 + beta**2) * (precision * recall) / (beta**2 * precision + recall)

def evaluate_threshold_on_subset(model, examples, token_threshold, chunk_threshold, subset_type="POS"):
    """Evaluate model on POS or NEG subset"""
    
    # Prepare data
    sentences = [(ex['query'], ex['text']) for ex in examples]
    chunk_positions = [ex['chunks_pos'] for ex in examples]
    true_chunks = [ex['relevant_chunks'] for ex in examples]
    
    # Get predictions
    try:
        outputs = model.predict_context(
            sentences,
            chunk_positions,
            batch_size=32,
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
        'subset_type': subset_type,
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
        'kept_chunks': kept_chunks,
        'num_examples': len(examples)
    }

def load_data_by_type(dataset_name: str, max_samples: int = None):
    """Load POS and NEG data separately"""
    dataset = load_dataset('hotchpotch/wip-query-context-pruner-with-teacher-scores', dataset_name)
    data = dataset['train']
    
    if max_samples:
        data = data.select(range(min(max_samples, len(data))))
    
    pos_examples = []
    neg_examples = []
    
    for example in data:
        for i, text in enumerate(example['texts']):
            if (i < len(example['chunks_pos']) and 
                i < len(example['relevant_chunks']) and
                i < len(example['labels'])):
                
                chunks_pos = example['chunks_pos'][i]
                relevant_chunks = example['relevant_chunks'][i]
                label = example['labels'][i]
                
                if chunks_pos and relevant_chunks:
                    example_data = {
                        'query': example['query'],
                        'text': text,
                        'chunks_pos': chunks_pos,
                        'relevant_chunks': relevant_chunks,
                        'label': label
                    }
                    
                    if label == 1:  # POS
                        pos_examples.append(example_data)
                    else:  # NEG
                        neg_examples.append(example_data)
    
    return pos_examples, neg_examples

def analyze_pos_neg_separately(model_path: str, dataset_name: str, max_samples: int = 200):
    """Analyze POS and NEG samples separately"""
    
    print(f"=== POS/NEG分離分析 ===")
    print(f"モデル: {model_path}")
    print(f"データセット: {dataset_name}")
    print(f"最大サンプル数: {max_samples}")
    
    # Load model
    print("モデル読み込み中...")
    model = ProvenceEncoder.from_pretrained(model_path)
    model.eval()
    
    # Load data separately
    print("データ読み込み中...")
    pos_examples, neg_examples = load_data_by_type(dataset_name, max_samples)
    print(f"POSサンプル数: {len(pos_examples)}")
    print(f"NEGサンプル数: {len(neg_examples)}")
    
    # Define threshold ranges (focus on key values)
    key_thresholds = [
        (0.3, 0.5, "F2最適"),  # F2 optimal
        (0.4, 0.7, "F1最適"),  # F1 optimal  
        (0.7, 0.6, "F0.5最適"), # F0.5 optimal
        (0.5, 0.5, "標準設定")   # Standard
    ]
    
    results = {}
    
    for token_th, chunk_th, desc in key_thresholds:
        print(f"\n=== {desc} (トークン={token_th}, チャンク={chunk_th}) ===")
        
        # Evaluate on POS samples
        pos_result = evaluate_threshold_on_subset(
            model, pos_examples, token_th, chunk_th, "POS"
        )
        
        # Evaluate on NEG samples  
        neg_result = evaluate_threshold_on_subset(
            model, neg_examples, token_th, chunk_th, "NEG"
        )
        
        if pos_result and neg_result:
            results[desc] = {
                'POS': pos_result,
                'NEG': neg_result,
                'thresholds': (token_th, chunk_th)
            }
            
            # Print results
            print(f"POS結果 (関連文書):")
            print(f"  サンプル数: {pos_result['num_examples']}")
            print(f"  Precision: {pos_result['precision']:.3f}")
            print(f"  Recall: {pos_result['recall']:.3f}")
            print(f"  F1: {pos_result['f1']:.3f}")
            print(f"  F2: {pos_result['f2']:.3f}")
            print(f"  圧縮率: {pos_result['compression_ratio']:.1%}")
            print(f"  TP: {pos_result['tp']}, FP: {pos_result['fp']}, FN: {pos_result['fn']}, TN: {pos_result['tn']}")
            
            print(f"NEG結果 (非関連文書):")
            print(f"  サンプル数: {neg_result['num_examples']}")
            print(f"  Precision: {neg_result['precision']:.3f}")
            print(f"  Recall: {neg_result['recall']:.3f}")  
            print(f"  F1: {neg_result['f1']:.3f}")
            print(f"  F2: {neg_result['f2']:.3f}")
            print(f"  圧縮率: {neg_result['compression_ratio']:.1%}")
            print(f"  TP: {neg_result['tp']}, FP: {neg_result['fp']}, FN: {neg_result['fn']}, TN: {neg_result['tn']}")
    
    # Analysis and comparison
    print(f"\n{'='*60}")
    print("=== 比較分析 ===")
    print(f"{'='*60}")
    
    if results:
        # Create comparison table
        print("\n📊 設定別性能比較:")
        print("設定       | サブセット | Precision | Recall | F1     | F2     | 圧縮率  | FN")
        print("-" * 80)
        
        for desc, result in results.items():
            for subset_type in ['POS', 'NEG']:
                data = result[subset_type]
                print(f"{desc:10} | {subset_type:8} | {data['precision']:9.3f} | "
                      f"{data['recall']:6.3f} | {data['f1']:6.3f} | {data['f2']:6.3f} | "
                      f"{data['compression_ratio']:6.1%} | {data['fn']:2d}")
        
        # Key insights
        print(f"\n🔍 重要な洞察:")
        
        # Compare F1 vs F2 on POS
        if "F1最適" in results and "F2最適" in results:
            f1_pos = results["F1最適"]["POS"]
            f2_pos = results["F2最適"]["POS"]
            
            print(f"\n1. POS（関連文書）での比較:")
            print(f"   F1最適 → F2最適:")
            print(f"   - Precision: {f1_pos['precision']:.3f} → {f2_pos['precision']:.3f} ({f2_pos['precision']-f1_pos['precision']:+.3f})")
            print(f"   - Recall: {f1_pos['recall']:.3f} → {f2_pos['recall']:.3f} ({f2_pos['recall']-f1_pos['recall']:+.3f})")
            print(f"   - FN: {f1_pos['fn']} → {f2_pos['fn']} ({f2_pos['fn']-f1_pos['fn']:+d})")
            print(f"   - 圧縮率: {f1_pos['compression_ratio']:.1%} → {f2_pos['compression_ratio']:.1%}")
        
        # Compare F1 vs F2 on NEG
        if "F1最適" in results and "F2最適" in results:
            f1_neg = results["F1最適"]["NEG"]
            f2_neg = results["F2最適"]["NEG"]
            
            print(f"\n2. NEG（非関連文書）での比較:")
            print(f"   F1最適 → F2最適:")
            print(f"   - Precision: {f1_neg['precision']:.3f} → {f2_neg['precision']:.3f} ({f2_neg['precision']-f1_neg['precision']:+.3f})")
            print(f"   - Recall: {f1_neg['recall']:.3f} → {f2_neg['recall']:.3f} ({f2_neg['recall']-f1_neg['recall']:+.3f})")
            print(f"   - FP: {f1_neg['fp']} → {f2_neg['fp']} ({f2_neg['fp']-f1_neg['fp']:+d})")
            print(f"   - 圧縮率: {f1_neg['compression_ratio']:.1%} → {f2_neg['compression_ratio']:.1%}")
        
        # POS vs NEG performance differences
        print(f"\n3. POS vs NEG の性能差:")
        for desc in ["F2最適", "F1最適"]:
            if desc in results:
                pos_data = results[desc]["POS"]
                neg_data = results[desc]["NEG"]
                print(f"   {desc}:")
                print(f"   - Recall: POS {pos_data['recall']:.3f} vs NEG {neg_data['recall']:.3f}")
                print(f"   - 圧縮率: POS {pos_data['compression_ratio']:.1%} vs NEG {neg_data['compression_ratio']:.1%}")
    
    # Save results
    output_file = f"results/pos_neg_analysis_{dataset_name}.json"
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\n結果を保存: {output_file}")
    
    return results

def main():
    parser = argparse.ArgumentParser(description="Analyze POS and NEG samples separately")
    parser.add_argument("--model_path", type=str, default="outputs/provence-ja-small/final-model",
                       help="Path to trained model")
    parser.add_argument("--dataset", type=str, default="ja-small",
                       choices=["ja-minimal", "ja-small", "ja-full"],
                       help="Dataset to use for analysis")
    parser.add_argument("--max_samples", type=int, default=200,
                       help="Maximum number of samples to use")
    
    args = parser.parse_args()
    
    analyze_pos_neg_separately(args.model_path, args.dataset, args.max_samples)

if __name__ == "__main__":
    main()