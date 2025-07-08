#!/usr/bin/env python3
"""
統合されたPruningモデル評価スクリプト
Usage: python evaluate_pruning.py --model_path path/to/model --target {minimal|small|full}
"""

import argparse
import json
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from itertools import product
from tqdm import tqdm

from datasets import load_dataset
from sentence_transformers.pruning import PruningEncoder
from sentence_transformers.pruning.evaluation import (
    comprehensive_evaluation,
    print_evaluation_report,
    evaluate_multiple_thresholds
)


# データセット設定
DATASET_CONFIGS = {
    'minimal': {
        'name': 'ja-minimal',
        'max_samples': 200
    },
    'small': {
        'name': 'ja-small',
        'max_samples': 200
    },
    'full': {
        'name': 'ja-full',
        'max_samples': 200
    }
}

# 評価モード設定
EVALUATION_MODES = {
    'standard': {
        'token_threshold': 0.5,
        'chunk_threshold': 0.5,
        'description': '標準設定'
    },
    'f2_optimal': {
        'token_threshold': 0.3,
        'chunk_threshold': 0.5,
        'description': 'F2最適（Recall重視）'
    },
    'f1_optimal': {
        'token_threshold': 0.4,
        'chunk_threshold': 0.7,
        'description': 'F1最適（バランス）'
    },
    'f05_optimal': {
        'token_threshold': 0.7,
        'chunk_threshold': 0.6,
        'description': 'F0.5最適（Precision重視）'
    }
}


def f_beta_score(precision: float, recall: float, beta: float = 2.0) -> float:
    """Calculate F-beta score"""
    if precision + recall == 0:
        return 0.0
    return (1 + beta**2) * (precision * recall) / (beta**2 * precision + recall)


def evaluate_threshold_on_subset(
    model: PruningEncoder,
    examples: List[Dict],
    token_threshold: float,
    chunk_threshold: float,
    subset_type: str = "ALL"
) -> Optional[Dict]:
    """Evaluate model on a subset with specific thresholds"""
    
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
    
    # Calculate metrics
    all_true = []
    all_pred = []
    
    for true_chunk_indices, output, chunk_pos in zip(true_chunks, outputs, chunk_positions):
        # Convert indices to binary labels
        num_chunks = len(chunk_pos)
        true_binary = [0] * num_chunks
        for idx in true_chunk_indices:
            if idx < num_chunks:
                true_binary[idx] = 1
        
        # Get predictions
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
    all_examples = []
    
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
                    
                    all_examples.append(example_data)
                    if label == 1:  # POS
                        pos_examples.append(example_data)
                    else:  # NEG
                        neg_examples.append(example_data)
    
    return all_examples, pos_examples, neg_examples


def optimize_thresholds(
    model: PruningEncoder,
    examples: List[Dict],
    metric: str = 'f2',
    subset_type: str = 'ALL'
) -> Dict:
    """Optimize thresholds for best metric score"""
    
    # Define threshold ranges
    token_thresholds = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
    chunk_thresholds = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
    
    best_score = -1
    best_result = None
    
    print(f"Optimizing for {metric.upper()} on {subset_type} data...")
    
    for token_th, chunk_th in tqdm(
        product(token_thresholds, chunk_thresholds),
        total=len(token_thresholds) * len(chunk_thresholds)
    ):
        result = evaluate_threshold_on_subset(
            model, examples, token_th, chunk_th, subset_type
        )
        
        if result and result[metric] > best_score:
            best_score = result[metric]
            best_result = result
    
    return best_result


def main():
    parser = argparse.ArgumentParser(description='Evaluate Pruning model')
    parser.add_argument(
        '--model_path',
        type=str,
        required=True,
        help='Path to trained model'
    )
    parser.add_argument(
        '--target',
        type=str,
        required=True,
        choices=['minimal', 'small', 'full'],
        help='Target dataset'
    )
    parser.add_argument(
        '--mode',
        type=str,
        default='all',
        choices=['all', 'standard', 'optimize', 'pos_neg'],
        help='Evaluation mode'
    )
    parser.add_argument(
        '--output_dir',
        type=str,
        default='results',
        help='Output directory for results'
    )
    parser.add_argument(
        '--max_samples',
        type=int,
        default=None,
        help='Maximum number of samples to evaluate'
    )
    
    args = parser.parse_args()
    
    # データセット設定の取得
    dataset_config = DATASET_CONFIGS[args.target]
    if args.max_samples:
        dataset_config['max_samples'] = args.max_samples
    
    print(f"=== Pruning Evaluation ===")
    print(f"Model: {args.model_path}")
    print(f"Dataset: {dataset_config['name']}")
    print(f"Mode: {args.mode}")
    print("="*50)
    
    # モデルの読み込み
    print("モデル読み込み中...")
    model = PruningEncoder.from_pretrained(args.model_path)
    model.eval()
    
    # データの読み込み
    print("データ読み込み中...")
    all_examples, pos_examples, neg_examples = load_data_by_type(
        dataset_config['name'],
        dataset_config['max_samples']
    )
    print(f"Total examples: {len(all_examples)}")
    print(f"POS examples: {len(pos_examples)}")
    print(f"NEG examples: {len(neg_examples)}")
    
    # 出力ディレクトリの作成
    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True)
    
    results = {}
    
    # 標準評価
    if args.mode in ['all', 'standard']:
        print("\n=== 標準評価 ===")
        for mode_name, mode_config in EVALUATION_MODES.items():
            print(f"\n{mode_config['description']}:")
            result = evaluate_threshold_on_subset(
                model,
                all_examples,
                mode_config['token_threshold'],
                mode_config['chunk_threshold'],
                'ALL'
            )
            if result:
                results[mode_name] = result
                print(f"  Precision: {result['precision']:.3f}")
                print(f"  Recall: {result['recall']:.3f}")
                print(f"  F1: {result['f1']:.3f}")
                print(f"  F2: {result['f2']:.3f}")
                print(f"  圧縮率: {result['compression_ratio']:.1%}")
    
    # 閾値最適化
    if args.mode in ['all', 'optimize']:
        print("\n=== 閾値最適化 ===")
        for metric in ['f1', 'f2', 'precision', 'recall']:
            best = optimize_thresholds(model, all_examples, metric)
            if best:
                results[f'optimal_{metric}'] = best
                print(f"\n{metric.upper()}最適:")
                print(f"  閾値: token={best['token_threshold']}, chunk={best['chunk_threshold']}")
                print(f"  {metric.upper()}: {best[metric]:.3f}")
                print(f"  圧縮率: {best['compression_ratio']:.1%}")
    
    # POS/NEG分離評価
    if args.mode in ['all', 'pos_neg']:
        print("\n=== POS/NEG分離評価 ===")
        for mode_name, mode_config in EVALUATION_MODES.items():
            print(f"\n{mode_config['description']}:")
            
            # POS評価
            pos_result = evaluate_threshold_on_subset(
                model,
                pos_examples,
                mode_config['token_threshold'],
                mode_config['chunk_threshold'],
                'POS'
            )
            
            # NEG評価
            neg_result = evaluate_threshold_on_subset(
                model,
                neg_examples,
                mode_config['token_threshold'],
                mode_config['chunk_threshold'],
                'NEG'
            )
            
            if pos_result and neg_result:
                results[f'{mode_name}_pos_neg'] = {
                    'POS': pos_result,
                    'NEG': neg_result
                }
                
                print(f"  POS: P={pos_result['precision']:.3f}, R={pos_result['recall']:.3f}, "
                      f"F1={pos_result['f1']:.3f}, 圧縮={pos_result['compression_ratio']:.1%}")
                print(f"  NEG: P={neg_result['precision']:.3f}, R={neg_result['recall']:.3f}, "
                      f"F1={neg_result['f1']:.3f}, 圧縮={neg_result['compression_ratio']:.1%}")
    
    # 結果の保存
    output_file = output_dir / f"evaluation_{args.target}_{args.mode}.json"
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"\n結果を保存: {output_file}")
    
    print("\n✅ 評価完了！")


if __name__ == "__main__":
    main()