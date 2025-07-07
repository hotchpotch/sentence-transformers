# Text-Pruner実装仕様 (最新版)

## 更新履歴
- 2025-01-07: 初版作成
- 2025-01-07: Sentence Transformers統合設計追加

## 概要

Sentence TransformersにProvence論文ベースのtext-pruner機能を実装する。query-dependentな文書プルーニングとrerankingを統合した効率的なRAGパイプラインを実現。既存のCrossEncoderアーキテクチャを拡張し、OSSコミュニティへのPR/マージを目標とする。

## 実装ステータス

- [x] 仕様策定
- [x] アーキテクチャ設計
- [ ] ProvenceCrossEncoderクラス実装
- [ ] データローダー実装  
- [ ] 損失関数実装
- [ ] 評価メトリクス実装
- [ ] サンプルコード作成
- [ ] ドキュメント作成
- [ ] テスト実装
- [ ] PR作成

## アーキテクチャ設計

### 設計原則

1. **既存CrossEncoderとの互換性維持**
   - 基本的なCrossEncoder機能は完全に保持
   - 追加機能は拡張として実装
   - 既存のAPIを破壊しない

2. **モジュラー設計**
   - Reranking単独、Pruning単独、両方の組み合わせに対応
   - 損失関数は柔軟に組み合わせ可能
   - 既存のSentence Transformersパターンに従う

3. **効率的な実装**
   - バッチ処理の最適化
   - メモリ効率を考慮した設計
   - 推論時のパフォーマンス重視

### ディレクトリ構成

```
sentence_transformers/
├── cross_encoder/
│   ├── __init__.py             # ProvenceCrossEncoderをエクスポート
│   ├── ProvenceCrossEncoder.py # メインクラス
│   ├── data_collator.py        # 既存を拡張
│   ├── losses/
│   │   ├── ProvenceLoss.py     # 統合損失関数
│   │   ├── PruningBCELoss.py   # Pruning専用損失
│   │   └── PruningMSELoss.py   # スコアベース損失
│   └── evaluation/
│       └── ProvenceEvaluator.py # 評価メトリクス
├── utils/
│   └── multilingual_chunker.py # 言語別文分割
```

### 主要コンポーネント

#### 1. ProvenceCrossEncoder

```python
class ProvenceCrossEncoder(CrossEncoder):
    """
    Provenceアーキテクチャに基づくCrossEncoder拡張
    - Reranking: クエリ・文書ペアの関連性スコア
    - Pruning: 文レベルの保持/削除マスク
    """
    
    def __init__(self, 
                 model_name: str,
                 num_labels: int = 1,
                 enable_pruning: bool = True,
                 pruning_head_config: Optional[Dict] = None,
                 **kwargs):
        # 親クラスの初期化
        super().__init__(model_name, num_labels, **kwargs)
        
        # Pruning head追加
        if enable_pruning:
            self._add_pruning_head(pruning_head_config)
    
    def predict(self, sentences, **kwargs) -> Union[np.ndarray, Tuple[np.ndarray, np.ndarray]]:
        """予測: ranking scoreとoptionalなpruning mask"""
        
    def predict_with_pruning(self, queries, documents, **kwargs) -> ProvenceOutput:
        """Provence特有の出力形式"""
        
    def prune(self, query: str, document: str, threshold: float = 0.5) -> str:
        """文書のプルーニング実行"""
```

#### 2. データ構造

```python
@dataclass
class ProvenceOutput:
    """Provence出力のデータクラス"""
    ranking_scores: np.ndarray      # [batch_size]
    pruning_masks: np.ndarray       # [batch_size, max_sentences]
    sentences: List[List[str]]      # 分割された文リスト
    sentence_positions: List[List[Tuple[int, int]]]  # 文の位置情報
```

#### 3. 損失関数設計

```python
class ProvenceLoss(nn.Module):
    """統合損失関数: Reranking + Pruning"""
    
    def __init__(self,
                 ranking_loss_fn: nn.Module = BinaryCrossEntropyLoss(),
                 pruning_loss_fn: nn.Module = nn.BCEWithLogitsLoss(),
                 ranking_weight: float = 1.0,
                 pruning_weight: float = 0.5,
                 use_teacher_scores: bool = False):
        """
        Args:
            ranking_loss_fn: Reranking用損失関数
            pruning_loss_fn: Pruning用損失関数
            ranking_weight: Ranking損失の重み
            pruning_weight: Pruning損失の重み
            use_teacher_scores: 教師モデルのスコアを使用するか
        """
```

#### 4. データフォーマット拡張

```python
# 学習データフォーマット
{
    "query": str,                    # クエリ
    "document": str,                 # 文書
    "label": Union[int, float],      # Reranking label (0/1 or score)
    "pruning_labels": List[int],     # 各文のバイナリラベル
    "teacher_score": Optional[float], # 教師モデルのスコア
    "sentences": Optional[List[str]], # 事前分割された文（オプション）
}
```

#### 5. 言語対応チャンカー

```python
class MultilingualChunker:
    """言語別の文分割器"""
    
    @staticmethod
    def chunk_text(text: str, language: str = "auto") -> List[Tuple[str, Tuple[int, int]]]:
        """テキストを文に分割し、位置情報も返す"""
        
    @staticmethod
    def reconstruct_text(sentences: List[str], 
                        masks: List[bool], 
                        positions: List[Tuple[int, int]]) -> str:
        """マスクに基づいてテキストを再構築"""
```

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

### 利用可能なデータセット
- **hotchpotch/wip-query-context-pruner**: 日本語Wikipedia QAベース（約130万サンプル）
  - 現在は開発中フォーマット、変換が必要
  - 詳細は[text-pruner-dataset.md](./text-pruner-dataset.md)参照

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

## 実装詳細

### 学習フロー

#### 1. データローダー拡張

```python
class ProvenceCrossEncoderDataCollator(CrossEncoderDataCollator):
    """Provence用データコレーター"""
    
    def __init__(self, 
                 tokenizer,
                 chunker: Optional[MultilingualChunker] = None,
                 max_sentences: int = 64,
                 **kwargs):
        super().__init__(tokenizer, **kwargs)
        self.chunker = chunker or MultilingualChunker()
        self.max_sentences = max_sentences
    
    def __call__(self, features):
        # 文分割とpruning_labelsの処理を追加
        # パディングとマスク生成
```

#### 2. トレーナー統合

```python
# 既存のCrossEncoderTrainerを使用
from sentence_transformers import CrossEncoderTrainer

# Provence用の設定
trainer = CrossEncoderTrainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    loss=ProvenceLoss(),
    data_collator=ProvenceCrossEncoderDataCollator(tokenizer),
    evaluator=ProvenceEvaluator()
)
```

### API使用例

#### 基本的な使用方法

```python
from sentence_transformers import ProvenceCrossEncoder

# 1. モデルの初期化
model = ProvenceCrossEncoder("microsoft/deberta-v3-base")

# 2. Reranking + Pruning（統合モード）
results = model.predict_with_pruning(
    queries=["What is machine learning?"],
    documents=["Machine learning is a subset of AI..."]
)

# 3. Pruningのみ（文書圧縮）
pruned_text = model.prune(
    query="What is machine learning?",
    document=long_document,
    threshold=0.5  # 保持閾値
)

# 4. Rankingのみ（従来のCrossEncoder互換）
scores = model.predict([
    ["query1", "document1"],
    ["query2", "document2"]
])
```

#### 学習例

```python
from sentence_transformers import CrossEncoderTrainer
from sentence_transformers.cross_encoder.losses import ProvenceLoss
from datasets import load_dataset

# データセット準備
dataset = load_dataset("hotchpotch/wip-query-context-pruner")
train_dataset = dataset["train"].map(convert_to_provence_format)

# 損失関数設定
loss = ProvenceLoss(
    ranking_loss_fn=BinaryCrossEntropyLoss(),
    pruning_loss_fn=BCEWithLogitsLoss(),
    ranking_weight=1.0,
    pruning_weight=0.5,
    use_teacher_scores=True  # 教師スコア使用
)

# 学習
trainer = CrossEncoderTrainer(
    model=model,
    args=CrossEncoderTrainingArguments(
        output_dir="./provence-model",
        num_epochs=3,
        per_device_train_batch_size=8,
        warmup_ratio=0.1,
    ),
    train_dataset=train_dataset,
    loss=loss,
    evaluator=ProvenceEvaluator()
)

trainer.train()
```

#### 推論時の詳細制御

```python
# 詳細な制御オプション
output = model.predict_with_pruning(
    queries=queries,
    documents=documents,
    # Pruning設定
    pruning_mode="dynamic",      # "fixed", "dynamic", "threshold"
    pruning_ratio=0.5,           # 固定比率モード時
    pruning_threshold=0.7,       # 閾値モード時
    min_sentences=2,             # 最小保持文数
    # チャンキング設定
    language="auto",             # "ja", "en", "auto"
    preserve_structure=True,     # 段落構造を保持
    # バッチ処理
    batch_size=32,
    show_progress_bar=True
)
```

### 評価メトリクス

```python
class ProvenceEvaluator:
    """Provence用評価器"""
    
    def __init__(self,
                 queries: List[str],
                 documents: List[List[str]],
                 relevant_docs: List[List[int]],
                 pruning_labels: Optional[List[List[List[int]]]] = None):
        """
        評価メトリクス:
        - Reranking: MAP@k, MRR@k, NDCG@k
        - Pruning: Precision, Recall, F1
        - Efficiency: Compression Ratio, Speed
        - QA Integration: Answer Extraction Accuracy
        """
```

## 実装上の注意点

### 1. 後方互換性

- 既存のCrossEncoderAPIを完全に維持
- `enable_pruning=False`で従来のCrossEncoderとして動作
- 既存の損失関数やevaluatorとの互換性確保

### 2. パフォーマンス最適化

```python
# バッチ処理の最適化
- 文分割の並列化
- Attention maskの効率的な生成
- GPUメモリの効率的な使用

# 推論時の最適化
- 文分割結果のキャッシュ
- バッチサイズの動的調整
- Mixed precision対応
```

### 3. エラーハンドリング

```python
# 入力検証
- pruning_labelsと文数の整合性チェック
- 空文書や極端に長い文書への対応
- 言語検出の失敗時のフォールバック

# 学習時の安定性
- Gradient clippingの適用
- 損失値の監視とNaN検出
- 学習率スケジューリング
```

### 4. テスト戦略

```python
# ユニットテスト
tests/test_provence_cross_encoder.py
- 基本機能のテスト
- エッジケースのテスト
- 後方互換性のテスト

# 統合テスト
- 学習フローの完全テスト
- 評価メトリクスの検証
- マルチGPU対応のテスト
```

## 実装ステップ

### Phase 1: 基礎実装（1-2週間）

1. **ProvenceCrossEncoder基本クラス**
   - CrossEncoderを継承
   - Pruning headの追加
   - 基本的なforward pass

2. **データ構造の定義**
   - ProvenceOutput dataclass
   - 拡張データフォーマット

3. **基本的な推論機能**
   - predict_with_pruning
   - prune メソッド

### Phase 2: 学習機能（1-2週間）

1. **損失関数実装**
   - ProvenceLoss（統合損失）
   - PruningBCELoss（バイナリ）
   - PruningMSELoss（スコアベース）

2. **データローダー拡張**
   - ProvenceCrossEncoderDataCollator
   - 文分割とラベル処理

3. **MultilingualChunker**
   - 言語別の文分割実装
   - 位置情報の保持

### Phase 3: 評価・最適化（1週間）

1. **評価メトリクス**
   - ProvenceEvaluator実装
   - 統合評価指標

2. **最適化**
   - バッチ処理の高速化
   - メモリ使用量の削減

3. **ドキュメント・テスト**
   - APIドキュメント
   - 使用例の作成
   - 包括的なテスト

### Phase 4: PR準備（3-5日）

1. **コードレビュー準備**
   - コードスタイルの統一
   - Docstringの完成
   - Type hintsの追加

2. **PR作成**
   - 詳細な説明文
   - ベンチマーク結果
   - 使用例の提供

## 成功指標

1. **機能要件**
   - 既存APIとの完全な互換性
   - Reranking精度の維持
   - 50%以上の圧縮率達成

2. **非機能要件**
   - 推論速度: 既存CrossEncoderの1.5倍以内
   - メモリ使用量: 2倍以内
   - 学習の安定性

3. **コミュニティ受容**
   - PRのマージ
   - ドキュメントの充実
   - サンプルコードの提供

## 参考資料

- [Provence論文](../provence/provence_paper.md)
- [実装詳細仕様](./provence-implementation-spec.md)
- [データフォーマット仕様](./data-format-spec.md)
- [データセット仕様](./text-pruner-dataset.md)

## 次のステップ

1. この仕様書のレビューと改善（2回以上）
2. Phase 1の実装開始
3. 定期的な進捗確認とフィードバック