# Text-Pruner実装仕様 (最新版)

## 更新履歴
- 2025-01-07: 初版作成

## 概要

Sentence TransformersにProvence論文ベースのtext-pruner機能を実装する。query-dependentな文書プルーニングとrerankingを統合した効率的なRAGパイプラインを実現。

## 実装ステータス

- [x] 仕様策定
- [ ] ProvenceCrossEncoderクラス実装
- [ ] データローダー実装  
- [ ] 損失関数実装
- [ ] 評価メトリクス実装
- [ ] サンプルコード作成
- [ ] ドキュメント作成

## アーキテクチャ

### モジュール構成

```
sentence_transformers/
├── provence/                    # Provence実装
│   ├── modeling.py             # ProvenceCrossEncoder
│   ├── data_collator.py        # データ処理
│   ├── losses.py               # 損失関数
│   └── evaluation.py           # 評価器
```

### 主要クラス

1. **ProvenceCrossEncoder**: CrossEncoderを継承
   - reranking機能（継承）
   - pruning機能（新規追加）
   - DeBERTa-v3ベースモデル使用

2. **ProvenceLoss**: 複合損失関数
   - Reranking損失
   - Pruning損失（BCE）
   - 重み付き結合

## データフォーマット

### 学習データ
```json
{
    "query": "検索クエリ",
    "document": "対象文書",
    "label": 1.0,
    "pruning_labels": [1, 1, 0, 1]
}
```

### 推論データ  
```json
{
    "query": "検索クエリ",
    "documents": ["文書1", "文書2"],
    "pruning_config": {
        "mode": "dynamic",
        "ratio": 0.5
    }
}
```

## API設計

### 基本的な使用方法

```python
from sentence_transformers.provence import ProvenceCrossEncoder

# モデル初期化
model = ProvenceCrossEncoder("microsoft/deberta-v3-base")

# Reranking + Pruning
results = model.rank_and_prune(
    query="質問文",
    documents=documents,
    pruning_ratio=0.5
)

# Pruningのみ
pruned = model.prune(query, document)
```

### 学習

```python
from sentence_transformers.provence import ProvenceLoss

# 損失関数
loss = ProvenceLoss(
    reranking_weight=1.0,
    pruning_weight=0.5
)

# 学習
model.fit(
    train_dataloader=train_dataloader,
    loss=loss,
    epochs=3
)
```

## 評価メトリクス

- **Reranking**: MAP@k, MRR@k, NDCG@k
- **Pruning**: Precision, Recall, F1, Compression Ratio
- **統合**: QA精度維持率、レイテンシ改善率

## 実装優先度

1. **Phase 1**: 基本実装
   - ProvenceCrossEncoderクラス
   - 基本的なpredict/rank機能

2. **Phase 2**: Pruning機能
   - Pruningヘッドの追加
   - 文分割処理
   - マスク生成

3. **Phase 3**: 学習・評価
   - 損失関数
   - データローダー
   - 評価メトリクス

## 参考資料

- [Provence論文](../provence_paper.md)
- [実装詳細仕様](./provence-implementation-spec.md)
- [データフォーマット仕様](./data-format-spec.md)