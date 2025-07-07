#!/usr/bin/env python3
"""
Test inference with teacher model: hotchpotch/japanese-reranker-xsmall-v2
"""

import torch
import time
import random
from sentence_transformers import CrossEncoder
from tqdm import tqdm
from typing import List, Tuple


def generate_japanese_text(length: int) -> str:
    """Generate Japanese text of approximately specified length."""
    base_sentences = [
        "これは非常に興味深い技術的な話題です。",
        "人工知能の発展により、私たちの生活は大きく変わりました。",
        "機械学習アルゴリズムは日々進歩し続けています。",
        "データサイエンスの分野では新しい発見が頻繁に行われています。",
        "自然言語処理技術は多くの応用分野で活用されています。",
        "コンピュータビジョンの研究は画像認識の精度を向上させました。",
        "深層学習モデルは複雑なパターンを学習することができます。",
        "ニューラルネットワークの構造は脳の働きからヒントを得ています。",
        "強化学習は環境との相互作用を通じて学習を行います。",
        "トランスフォーマーモデルは言語理解において大きな進歩をもたらしました。",
        "生成AIは創造性の領域にも影響を与えています。",
        "エッジコンピューティングにより、リアルタイム処理が可能になりました。",
        "クラウドコンピューティングはスケーラブルなソリューションを提供します。",
        "ビッグデータ分析により、新しい洞察を得ることができます。",
        "ブロックチェーン技術は分散システムに革命をもたらしました。",
        "インターネットオブシングスは身の回りの機器を接続します。",
        "サイバーセキュリティの重要性は年々高まっています。",
        "量子コンピューティングは計算パラダイムを変える可能性があります。",
        "バイオインフォマティクスは生命科学と情報技術を結びつけます。",
        "ロボット工学の進歩により、自動化が進んでいます。",
    ]
    
    result = ""
    while len(result) < length:
        sentence = random.choice(base_sentences)
        if len(result) + len(sentence) <= length:
            result += sentence
        else:
            # Truncate the last sentence to fit exactly
            remaining = length - len(result)
            result += sentence[:remaining]
            break
    
    return result


def generate_query_document_pairs(num_pairs: int, doc_length: int = 512) -> List[Tuple[str, str]]:
    """Generate query-document pairs for testing."""
    query_templates = [
        "人工知能について",
        "機械学習の応用",
        "データサイエンスの手法",
        "自然言語処理技術",
        "コンピュータビジョン",
        "深層学習モデル",
        "ニューラルネットワーク",
        "強化学習アルゴリズム",
        "トランスフォーマー",
        "生成AI技術",
        "エッジコンピューティング",
        "クラウドサービス",
        "ビッグデータ分析",
        "ブロックチェーン",
        "IoTシステム",
        "サイバーセキュリティ",
        "量子コンピューティング",
        "バイオインフォマティクス",
        "ロボット技術",
        "自動化システム",
    ]
    
    pairs = []
    for i in range(num_pairs):
        query = random.choice(query_templates)
        document = generate_japanese_text(doc_length)
        pairs.append((query, document))
    
    return pairs


def test_batch_inference(model: CrossEncoder, pairs: List[Tuple[str, str]], batch_size: int) -> List[float]:
    """Test batch inference with specified batch size."""
    print(f"Testing batch inference with batch_size={batch_size}")
    
    all_scores = []
    start_time = time.time()
    
    for i in tqdm(range(0, len(pairs), batch_size), desc=f"Batch size {batch_size}"):
        batch = pairs[i:i + batch_size]
        
        with torch.no_grad():
            scores = model.predict(batch, show_progress_bar=False)
            all_scores.extend(scores)
    
    end_time = time.time()
    elapsed = end_time - start_time
    
    print(f"Processed {len(pairs)} pairs in {elapsed:.2f} seconds")
    print(f"Average time per pair: {elapsed/len(pairs)*1000:.2f}ms")
    print(f"Throughput: {len(pairs)/elapsed:.2f} pairs/second")
    
    return all_scores


def main():
    print("=" * 70)
    print("Testing Teacher Model Inference")
    print("Model: hotchpotch/japanese-reranker-xsmall-v2")
    print("=" * 70)
    
    # Load model
    print("Loading model...")
    model = CrossEncoder("hotchpotch/japanese-reranker-xsmall-v2")
    
    # Enable half precision if GPU available
    if model.device == "cuda" or model.device == "mps":
        model.model.half()
        print("Enabled half precision")
    
    print(f"Model device: {model.device}")
    print(f"Model dtype: {next(model.model.parameters()).dtype}")
    
    # Generate test data
    print("\nGenerating test data...")
    test_pairs = generate_query_document_pairs(
        num_pairs=5000,  # Large number for testing
        doc_length=512   # 512 tokens approximately
    )
    
    print(f"Generated {len(test_pairs)} query-document pairs")
    
    # Sample a few pairs for inspection
    print("\nSample pairs:")
    for i, (query, doc) in enumerate(test_pairs[:3]):
        print(f"Pair {i+1}:")
        print(f"  Query: {query}")
        print(f"  Document length: {len(doc)} chars")
        print(f"  Document preview: {doc[:100]}...")
        print()
    
    # Test different batch sizes
    batch_sizes = [1, 16, 64, 256, 512]
    results = {}
    
    for batch_size in batch_sizes:
        print(f"\n{'='*50}")
        try:
            scores = test_batch_inference(model, test_pairs, batch_size)
            results[batch_size] = {
                'success': True,
                'scores': scores,
                'avg_score': sum(scores) / len(scores),
                'min_score': min(scores),
                'max_score': max(scores)
            }
            print(f"✓ Batch size {batch_size}: SUCCESS")
            print(f"  Average score: {results[batch_size]['avg_score']:.4f}")
            print(f"  Score range: [{results[batch_size]['min_score']:.4f}, {results[batch_size]['max_score']:.4f}]")
        except Exception as e:
            results[batch_size] = {
                'success': False,
                'error': str(e)
            }
            print(f"✗ Batch size {batch_size}: FAILED")
            print(f"  Error: {e}")
    
    # Summary
    print("\n" + "=" * 70)
    print("INFERENCE TEST SUMMARY")
    print("=" * 70)
    
    successful_batches = [bs for bs, result in results.items() if result['success']]
    if successful_batches:
        max_batch_size = max(successful_batches)
        print(f"✓ Maximum working batch size: {max_batch_size}")
        print(f"✓ Model can handle large batch inference")
        
        if max_batch_size >= 512:
            print("✓ Confirmed: Model supports batch_size=512 as expected")
        else:
            print(f"⚠ Warning: Model only supports batch_size up to {max_batch_size}")
    else:
        print("✗ No batch sizes worked - there may be an issue with the model")
    
    print("\nReady to proceed with dataset creation using this model.")


if __name__ == "__main__":
    main()