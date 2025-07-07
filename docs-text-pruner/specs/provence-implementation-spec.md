# Provence実装仕様書

## 概要

ProvenceをSentence Transformersに統合するための設計仕様とキーワード定義です。既存のCrossEncoderアーキテクチャを拡張し、reranking（再ランキング）とpruning（枝刈り）の両機能を実現します。

## アーキテクチャ設計

### クラス階層

```
CrossEncoder (既存)
    └── ProvenceCrossEncoder (新規)
         ├── Reranking機能（継承）
         └── Pruning機能（追加）
```

### 主要コンポーネント

1. **ProvenceCrossEncoder**: CrossEncoderを継承し、pruning機能を追加
2. **ProvenceDataCollator**: pruning用のラベルを含むデータ処理
3. **ProvenceLoss**: reranking + pruningの複合損失関数
4. **ProvenceEvaluator**: pruning精度とreranking性能の評価

## キーワード定義

### 基本用語（Sentence Transformers標準）

| キーワード | 意味 | 使用例 |
|---------|------|--------|
| `query` | 検索クエリまたは質問 | ユーザーの入力テキスト |
| `document` / `passage` | ランキング/プルーニング対象のテキスト | 検索結果の文書 |
| `positive` | 関連性の高い文書 | クエリに対する正解文書 |
| `negative` | 関連性の低い文書 | クエリに無関係な文書 |
| `hard_negative` | 紛らわしい非関連文書 | 似ているが不正解の文書 |

### Provence固有用語

| キーワード | 意味 | 使用例 |
|---------|------|--------|
| `context` | documentの別名（RAG文脈で使用） | 複数のpassageを連結したもの |
| `chunks` | contextを分割した単位 | 文または段落レベルの分割 |
| `pruning_labels` | 各chunk/文の保持/削除ラベル | 0: 削除, 1: 保持 |
| `reranking_score` | 文書全体の関連性スコア | CrossEncoderの標準出力 |
| `pruning_mask` | pruning後の保持マスク | バイナリマスク配列 |
| `pruning_ratio` | 削除する割合 | 0.0〜1.0の実数 |

### データ構造

#### 学習データフォーマット

```python
{
    # 基本フィールド（CrossEncoder互換）
    "query": str,                    # クエリテキスト
    "document": str,                 # 文書テキスト
    "label": float,                  # reranking用ラベル（0-1）
    
    # Provence拡張フィールド
    "pruning_labels": List[int],    # 各文のバイナリラベル
    "sentences": List[str],          # 文分割済みテキスト（オプション）
}
```

#### 評価データフォーマット

```python
{
    "query": str,
    "positive": List[str],           # 関連文書リスト
    "negative": List[str],           # 非関連文書リスト
    # または
    "documents": List[str],          # 全文書（positiveを含む）
    "relevance_scores": List[float], # 各文書の関連度（オプション）
}
```

### メソッド名規約

| メソッド | 機能 | 戻り値 |
|---------|------|--------|
| `predict()` | reranking scoreの予測 | scores: Tensor |
| `predict_with_pruning()` | score + pruning maskの予測 | (scores, masks) |
| `rank()` | 文書のランキング | List[Dict] |
| `rank_and_prune()` | ランキング + プルーニング | List[Dict] |
| `prune()` | 単一文書のプルーニング | pruned_text: str |

### 損失関数

| 損失関数 | 用途 | 入力 |
|---------|------|------|
| `ProvenceLoss` | 複合損失（ranking + pruning） | scores, masks, labels |
| `PruningBCELoss` | pruningのみの学習 | masks, pruning_labels |
| `RerankingMarginMSELoss` | 教師モデルからの蒸留 | scores, teacher_scores |

### 評価メトリクス

| メトリクス | 説明 | 対象 |
|-----------|------|------|
| `MAP@k` | Mean Average Precision | Reranking |
| `MRR@k` | Mean Reciprocal Rank | Reranking |
| `NDCG@k` | Normalized DCG | Reranking |
| `pruning_precision` | 保持すべき文の精度 | Pruning |
| `pruning_recall` | 保持すべき文の再現率 | Pruning |
| `compression_ratio` | 圧縮率（削除された割合） | Pruning |

## 実装パス

```
sentence_transformers/
├── provence/                        # 新規ディレクトリ
│   ├── __init__.py
│   ├── modeling.py                  # ProvenceCrossEncoder
│   ├── data_collator.py            # データ処理
│   ├── losses.py                   # 損失関数
│   └── evaluation.py               # 評価器
├── cross_encoder/                   # 既存（継承元）
└── examples/
    └── training/
        └── provence/                # サンプルコード
```

## 使用例

```python
from sentence_transformers.provence import ProvenceCrossEncoder

# モデルの初期化
model = ProvenceCrossEncoder("microsoft/deberta-v3-base")

# Reranking + Pruning
results = model.rank_and_prune(
    query="What is machine learning?",
    documents=documents,
    pruning_ratio=0.5  # 50%削減
)

# Pruningのみ
pruned_text = model.prune(
    query=query,
    document=long_document,
    min_sentences=3  # 最低3文は保持
)
```

## 設計原則

1. **互換性**: 既存のCrossEncoderコードとの完全な互換性を維持
2. **拡張性**: pruning機能は追加モジュールとして実装
3. **柔軟性**: reranking単独、pruning単独、両方の組み合わせに対応
4. **効率性**: バッチ処理と最適化されたトークナイゼーション

## 次のステップ

1. ProvenceCrossEncoderクラスの実装
2. 学習用データローダーの作成
3. 損失関数の実装
4. 評価メトリクスの実装
5. サンプルコードとドキュメントの作成