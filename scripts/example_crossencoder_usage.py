#!/usr/bin/env python
"""
Example: Using PruningEncoder reranking model as CrossEncoder.
"""

import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

# IMPORTANT: Import pruning module first to register models
import sentence_transformers.pruning
from sentence_transformers import CrossEncoder
import torch

# モデルパス
MODEL_PATH = "./output/transformers_compat_test/reranking_pruning_20250709_135233/final_model"

# CrossEncoderとして読み込み
model = CrossEncoder(MODEL_PATH)
if str(model.device) in ["cuda", "mps"]:
    model.model.half()

# クエリと文書
query = "感動的な映画について"
passages = [
    "深いテーマを持ちながらも、観る人の心を揺さぶる名作。登場人物の心情描写が秀逸で、ラストは涙なしでは見られない。",
    "重要なメッセージ性は評価できるが、暗い話が続くので気分が落ち込んでしまった。もう少し明るい要素があればよかった。",
    "どうにもリアリティに欠ける展開が気になった。もっと深みのある人間ドラマが見たかった。",
    "アクションシーンが楽しすぎる。見ていて飽きない。ストーリーはシンプルだが、それが逆に良い。",
]

# スコア予測
scores = model.predict(
    [(query, passage) for passage in passages],
    show_progress_bar=True,
)

print(f"Query: {query}")
print("\nScores:")
for passage, score in zip(passages, scores):
    print(f"  {score:.4f}: {passage[:50]}...")

# ランキング
print("\nRanking:")
results = model.rank(query, passages, return_documents=True)
for i, result in enumerate(results, 1):
    print(f"{i}. Score: {result['score']:.4f} - {result['text'][:50]}...")

print("\n✓ PruningEncoder reranking models can be used as CrossEncoder!")
print("  Just import sentence_transformers.pruning before loading the model.")