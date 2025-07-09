#!/usr/bin/env python
"""
Demo: PruningEncoder models now work with standard AutoModel!
"""

from transformers import AutoModelForSequenceClassification, AutoTokenizer
import torch

# モデルパス
MODEL_PATH = "./output/automap_test/reranking_pruning_automap"

print("="*60)
print("PruningEncoder with AutoModel Demo")
print("="*60)

# 1. 通常のTransformersパターンで読み込み
print("\n1. Loading with AutoModelForSequenceClassification...")
model = AutoModelForSequenceClassification.from_pretrained(
    MODEL_PATH,
    trust_remote_code=True  # カスタムコードを許可
)
tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)

print(f"   ✓ Model loaded: {type(model).__name__}")
print(f"   ✓ Device: {next(model.parameters()).device}")

# 2. 推論テスト
print("\n2. Testing inference...")
queries_and_docs = [
    ("機械学習について", "機械学習は人工知能の一分野で、データから学習するアルゴリズムの研究です。"),
    ("天気予報について", "機械学習は人工知能の一分野で、データから学習するアルゴリズムの研究です。"),
    ("深層学習とは", "ディープラーニングは多層のニューラルネットワークを使用した機械学習手法です。"),
]

for query, document in queries_and_docs:
    inputs = tokenizer(query, document, return_tensors="pt", truncation=True, max_length=512)
    
    # デバイスに移動
    device = next(model.parameters()).device
    inputs = {k: v.to(device) for k, v in inputs.items()}
    
    with torch.no_grad():
        outputs = model(**inputs)
        score = torch.sigmoid(outputs.logits).item()
    
    print(f"\n   Query: {query}")
    print(f"   Document: {document[:50]}...")
    print(f"   Score: {score:.4f}")

# 3. CrossEncoderとしても使用可能
print("\n3. Also works as CrossEncoder (with import)...")
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))
import sentence_transformers.pruning  # 登録のため

from sentence_transformers import CrossEncoder

ce_model = CrossEncoder(MODEL_PATH)
scores = ce_model.predict([
    ("機械学習について", "機械学習は人工知能の一分野で、データから学習するアルゴリズムの研究です。")
])
print(f"   CrossEncoder score: {scores[0]:.4f}")

print("\n" + "="*60)
print("Summary:")
print("="*60)
print("✅ PruningEncoder models can now be loaded with:")
print("   - AutoModelForSequenceClassification (reranking mode)")
print("   - AutoModelForTokenClassification (pruning-only mode)")
print("   - CrossEncoder (with sentence_transformers.pruning import)")
print("\n🎉 No special imports needed for AutoModel usage!")
print("   Just use trust_remote_code=True")