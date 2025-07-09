# PruningEncoder Loading Guide

PruningEncoderは複数の方法でロードでき、用途に応じて最適な方法を選択できます。

## 🚀 クイックスタート

### ベースのランキングモデルのみ使用（特別なインポート不要）

```python
from transformers import AutoModelForSequenceClassification, AutoTokenizer

# ranking_modelサブディレクトリを指定
model = AutoModelForSequenceClassification.from_pretrained("path/to/saved_model/ranking_model")
tokenizer = AutoTokenizer.from_pretrained("path/to/saved_model/ranking_model")

# 通常の推論
inputs = tokenizer("クエリ", "文書", return_tensors="pt")
outputs = model(**inputs)
score = torch.sigmoid(outputs.logits).item()
```

## 📊 全ての読み込み方法

### 1. フルPruningEncoder（プルーニング機能付き）

```python
from sentence_transformers.pruning import PruningEncoder

model = PruningEncoder.from_pretrained("path/to/saved_model")

# プルーニング付き推論
outputs = model.predict_with_pruning([("クエリ", "文書")])
print(f"スコア: {outputs[0].ranking_scores}")
print(f"圧縮率: {outputs[0].compression_ratio}")
```

### 2. ベースランキングモデル（標準Transformers）

```python
from transformers import AutoModelForSequenceClassification, AutoTokenizer

# 方法A: サブディレクトリを直接指定
model = AutoModelForSequenceClassification.from_pretrained("path/to/saved_model/ranking_model")

# 方法B: エクスポート機能を使用
pruning_model = PruningEncoder.from_pretrained("path/to/saved_model")
pruning_model.export_ranking_model("./exported_model")
model = AutoModelForSequenceClassification.from_pretrained("./exported_model")
```

### 3. AutoModel統合（自動登録）

```python
import sentence_transformers  # 自動登録

from transformers import AutoModelForSequenceClassification
model = AutoModelForSequenceClassification.from_pretrained("path/to/saved_model")
# trust_remote_code=True 不要！
```

### 4. CrossEncoder互換

```python
import sentence_transformers
from sentence_transformers import CrossEncoder

model = CrossEncoder("path/to/saved_model")
scores = model.predict([("クエリ", "文書")])
```

## 🎯 使い分けガイド

| 用途 | 推奨方法 | インポート | メリット |
|------|----------|------------|----------|
| ランキングのみ | `/ranking_model` | transformersのみ | 最小限、高速 |
| プルーニング必要 | PruningEncoder | sentence_transformers | フル機能 |
| 既存システム統合 | CrossEncoder | sentence_transformers | API互換性 |
| 柔軟な使用 | AutoModel + 登録 | sentence_transformers | 標準パターン |

## 💾 保存されるファイル構造

```
saved_model/
├── config.json                    # PruningEncoder設定
├── pruning_encoder_config.json    # 詳細設定
├── modeling_pruning_encoder.py    # カスタムコード（auto_map用）
├── tokenizer files               # トークナイザー
├── README.md                     # 使用ガイド（自動生成）
├── ranking_model/                # ⭐ ベースモデル（完全なTransformersモデル）
│   ├── config.json              # ModernBertConfig等
│   ├── model.safetensors        # モデル重み
│   └── tokenizer files          # トークナイザー
└── pruning_head/                 # プルーニングヘッド
    └── pytorch_model.bin
```

## 🔑 重要ポイント

1. **ベースモデルは既に利用可能**: `/ranking_model`サブディレクトリに完全なTransformersモデルとして保存
2. **特別なインポート不要**: ベースモデルのみ使用する場合
3. **同じ重み、異なるインターフェース**: 用途に応じて選択可能
4. **後方互換性維持**: 既存のコードは全て動作

## 📝 実装例

### シンプルなランキングタスク
```python
# 最小限のコード - sentence_transformersインポート不要！
from transformers import AutoModelForSequenceClassification, AutoTokenizer

model = AutoModelForSequenceClassification.from_pretrained("model_path/ranking_model")
tokenizer = AutoTokenizer.from_pretrained("model_path/ranking_model")
```

### RAGパイプラインでのプルーニング
```python
from sentence_transformers.pruning import PruningEncoder

model = PruningEncoder.from_pretrained("model_path")
outputs = model.predict_with_pruning(query_doc_pairs, pruning_threshold=0.5)

for output in outputs:
    print(f"スコア: {output.ranking_scores}")
    print(f"圧縮率: {output.compression_ratio}%")
    print(f"プルーニング後: {output.pruned_documents[0]}")
```