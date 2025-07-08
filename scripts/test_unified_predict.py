#!/usr/bin/env python3
"""
Test unified predict method with trained models
"""

import os
from sentence_transformers.provence import ProvenceEncoder


def test_unified_predict():
    """Test unified predict with a trained model"""
    
    # Load trained model
    model_path = "./outputs/provence-ja-small/final-model"
    if not os.path.exists(model_path):
        print(f"Model not found at {model_path}")
        return
    
    model = ProvenceEncoder.from_pretrained(model_path)
    model.eval()
    
    # Test cases
    test_pairs = [
        ("Pythonとは何ですか？", "Pythonはプログラミング言語です。Web開発やデータ分析に広く使用されています。"),
        ("機械学習について教えて", "機械学習は人工知能の一分野です。データから学習してパターンを見つけます。"),
        ("東京の天気は？", "今日の東京は晴れです。気温は25度で快適な一日になるでしょう。"),
    ]
    
    print("=== Testing Unified Predict Method ===\n")
    
    # Test 1: Ranking only
    print("1. Ranking Only (apply_pruning=False)")
    scores = model.predict(
        test_pairs,
        apply_pruning=False,
        show_progress_bar=True
    )
    
    for i, (query, doc) in enumerate(test_pairs):
        print(f"  Query: {query[:30]}...")
        print(f"  Score: {scores[i]:.3f}")
    
    # Test 2: Full functionality (ranking + pruning)
    print("\n2. Full Functionality (apply_pruning=True, threshold=0.3)")
    outputs = model.predict(
        test_pairs,
        apply_pruning=True,
        pruning_threshold=0.3,
        return_documents=True,
        show_progress_bar=True
    )
    
    for i, output in enumerate(outputs):
        query, doc = test_pairs[i]
        print(f"\n  Query: {query}")
        print(f"  Original: {doc}")
        print(f"  Score: {output.ranking_scores:.3f}")
        print(f"  Compression: {output.compression_ratio:.1%}")
        if output.pruned_documents:
            print(f"  Pruned: {output.pruned_documents[0]}")
    
    # Test 3: Different thresholds
    print("\n3. Different Thresholds")
    query, doc = test_pairs[0]
    
    for threshold in [0.1, 0.3, 0.5, 0.7]:
        output = model.predict(
            (query, doc),
            apply_pruning=True,
            pruning_threshold=threshold,
            return_documents=True
        )
        print(f"  Threshold {threshold}: Compression {output.compression_ratio:.1%}, Score {output.ranking_scores:.3f}")
    
    # Test 4: Batch processing
    print("\n4. Batch Processing")
    many_pairs = test_pairs * 5  # 15 pairs
    
    outputs = model.predict(
        many_pairs,
        batch_size=4,
        apply_pruning=True,
        pruning_threshold=0.3,
        show_progress_bar=True
    )
    
    print(f"  Processed {len(outputs)} pairs")
    avg_compression = sum(o.compression_ratio for o in outputs) / len(outputs)
    avg_score = sum(o.ranking_scores for o in outputs) / len(outputs)
    print(f"  Average compression: {avg_compression:.1%}")
    print(f"  Average score: {avg_score:.3f}")


def test_backward_compatibility():
    """Test that old methods still work"""
    
    model_path = "./outputs/provence-ja-small/final-model"
    if not os.path.exists(model_path):
        return
    
    model = ProvenceEncoder.from_pretrained(model_path)
    model.eval()
    
    pair = ("テストクエリ", "これはテスト文書です。")
    
    print("\n=== Testing Backward Compatibility ===")
    
    # Old predict_with_pruning should still work
    try:
        output = model.predict_with_pruning(
            pair,
            pruning_threshold=0.3,
            return_documents=True
        )
        print("✓ predict_with_pruning still works")
        print(f"  Compression: {output.compression_ratio:.1%}")
    except Exception as e:
        print(f"✗ predict_with_pruning failed: {e}")


if __name__ == "__main__":
    test_unified_predict()
    test_backward_compatibility()