#!/usr/bin/env python3
"""
Show actual predict_context() compression results for POS samples
"""

from datasets import load_dataset
import numpy as np
from sentence_transformers.provence import ProvenceEncoder

def show_compression_examples(
    model_path: str, 
    dataset_name: str = "ja-minimal", 
    num_samples: int = 10,
    token_thresholds: list = None,
    chunk_thresholds: list = None
):
    """Show actual compression results for POS samples"""
    
    # Default thresholds if not provided
    if token_thresholds is None:
        token_thresholds = [0.3, 0.5, 0.7]  # Include optimized threshold
    if chunk_thresholds is None:
        chunk_thresholds = [0.3, 0.5, 0.6]  # Include optimized threshold
    
    print(f"=== predict_context() 圧縮結果の確認 ===")
    print(f"モデル: {model_path}")
    print(f"データセット: {dataset_name}")
    print(f"サンプル数: {num_samples}")
    print(f"評価する閾値: トークン{token_thresholds}, チャンク{chunk_thresholds}\n")
    
    # Load model
    model = ProvenceEncoder.from_pretrained(model_path)
    model.eval()
    
    # Load dataset
    dataset = load_dataset('hotchpotch/wip-query-context-pruner-with-teacher-scores', dataset_name)
    data = dataset['train']
    
    # Find POS samples (relevant documents)
    pos_examples = []
    for example in data:
        for i, (text, label) in enumerate(zip(example['texts'], example['labels'])):
            if label == 1 and i < len(example['chunks_pos']) and i < len(example['relevant_chunks']):
                chunks_pos = example['chunks_pos'][i]
                relevant_chunks = example['relevant_chunks'][i]
                
                if chunks_pos and relevant_chunks:  # チャンクが存在する場合のみ
                    pos_examples.append({
                        'query': example['query'],
                        'text': text,
                        'chunks_pos': chunks_pos,
                        'relevant_chunks': relevant_chunks,
                        'teacher_score': example['teacher_scores_japanese-reranker-xsmall-v2'][i]
                    })
    
    print(f"見つかったPOSサンプル数: {len(pos_examples)}")
    
    # Sample random examples
    if len(pos_examples) > num_samples:
        import random
        random.seed(42)
        selected_examples = random.sample(pos_examples, num_samples)
    else:
        selected_examples = pos_examples[:num_samples]
    
    # Create threshold combinations
    threshold_configs = []
    for token_th in token_thresholds:
        for chunk_th in chunk_thresholds:
            if token_th == 0.3 and chunk_th == 0.3:
                desc = "適度な設定"
            elif token_th == 0.5 and chunk_th == 0.5:
                desc = "標準設定"
            elif token_th == 0.3 and chunk_th == 0.5:
                desc = "旧最適設定（F1重視）"
            elif token_th == 0.7 and chunk_th == 0.6:
                desc = "F0.5最適設定（推奨）"
            else:
                desc = f"トークン{token_th}_チャンク{chunk_th}"
            threshold_configs.append((token_th, chunk_th, desc))
    
    for token_th, chunk_th, desc in threshold_configs:
        print(f"\n{'='*60}")
        print(f"閾値設定: トークン={token_th}, チャンク={chunk_th} ({desc})")
        print(f"{'='*60}")
        
        for i, example in enumerate(selected_examples):
            print(f"\n--- サンプル {i+1} ---")
            print(f"クエリ: {example['query'][:100]}...")
            print(f"元テキスト長: {len(example['text'])} 文字")
            print(f"教師スコア: {example['teacher_score']:.3f}")
            print(f"総チャンク数: {len(example['chunks_pos'])}")
            print(f"関連チャンク数: {len(example['relevant_chunks'])}")
            print(f"関連チャンクインデックス: {example['relevant_chunks']}")
            
            # Apply predict_context
            output = model.predict_context(
                (example['query'], example['text']),
                example['chunks_pos'],
                token_threshold=token_th,
                chunk_threshold=chunk_th
            )
            
            print(f"ランキングスコア: {output.ranking_scores:.3f}")
            print(f"チャンク予測: {output.chunk_predictions}")
            print(f"チャンクスコア: {[f'{score:.3f}' for score in output.chunk_scores]}")
            print(f"圧縮率: {output.compression_ratio:.1%}")
            
            # Show detailed chunk analysis
            print(f"\nチャンク詳細分析:")
            kept_chunks = []
            for j, (chunk_pos, pred, score, is_relevant) in enumerate(zip(
                example['chunks_pos'], 
                output.chunk_predictions, 
                output.chunk_scores,
                [1 if j in example['relevant_chunks'] else 0 for j in range(len(example['chunks_pos']))]
            )):
                start, end = chunk_pos
                chunk_text = example['text'][start:end]
                status = "✓保持" if pred == 1 else "✗削除"
                relevance = "関連" if is_relevant else "非関連"
                
                print(f"  チャンク{j}: {status} ({relevance}, スコア:{score:.3f})")
                print(f"    位置: {start}-{end}")
                print(f"    内容: {chunk_text[:80]}{'...' if len(chunk_text) > 80 else ''}")
                
                if pred == 1:
                    kept_chunks.append(chunk_text)
            
            # Show compressed result
            if kept_chunks:
                compressed_text = " ".join(kept_chunks)
                print(f"\n圧縮後テキスト:")
                print(f"  長さ: {len(compressed_text)} 文字 (元: {len(example['text'])} 文字)")
                print(f"  内容: {compressed_text[:200]}{'...' if len(compressed_text) > 200 else ''}")
            else:
                print(f"\n圧縮後テキスト: （全て削除）")
            
            # Evaluation against ground truth
            true_relevant = [1 if j in example['relevant_chunks'] else 0 for j in range(len(example['chunks_pos']))]
            pred_relevant = output.chunk_predictions.tolist()
            
            # Calculate metrics for this example
            tp = sum(1 for t, p in zip(true_relevant, pred_relevant) if t == 1 and p == 1)
            fp = sum(1 for t, p in zip(true_relevant, pred_relevant) if t == 0 and p == 1)
            fn = sum(1 for t, p in zip(true_relevant, pred_relevant) if t == 1 and p == 0)
            tn = sum(1 for t, p in zip(true_relevant, pred_relevant) if t == 0 and p == 0)
            
            precision = tp / (tp + fp) if (tp + fp) > 0 else 0
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0
            f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
            
            print(f"\nこのサンプルの性能:")
            print(f"  適合率: {precision:.3f}, 再現率: {recall:.3f}, F1: {f1:.3f}")
            print(f"  TP: {tp}, FP: {fp}, FN: {fn}, TN: {tn}")


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description="Show predict_context compression results")
    parser.add_argument("--model_path", type=str, default="outputs/provence-ja-small/final-model",
                       help="Path to trained model")
    parser.add_argument("--dataset", type=str, default="ja-small",
                       choices=["ja-minimal", "ja-small", "ja-full"],
                       help="Dataset to use")
    parser.add_argument("--num_samples", type=int, default=10,
                       help="Number of samples to show")
    parser.add_argument("--token_thresholds", type=float, nargs="+", 
                       default=[0.3, 0.5, 0.7],
                       help="Token thresholds to test")
    parser.add_argument("--chunk_thresholds", type=float, nargs="+",
                       default=[0.3, 0.5, 0.6], 
                       help="Chunk thresholds to test")
    
    args = parser.parse_args()
    
    show_compression_examples(
        model_path=args.model_path,
        dataset_name=args.dataset,
        num_samples=args.num_samples,
        token_thresholds=args.token_thresholds,
        chunk_thresholds=args.chunk_thresholds
    )

if __name__ == "__main__":
    main()