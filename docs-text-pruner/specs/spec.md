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
│   ├── CrossEncoder.py         # 既存クラスにProvence機能統合
│   ├── data_collator.py        # ProvenceDataCollator追加
│   ├── losses/
│   │   ├── __init__.py         # 新規損失関数をエクスポート
│   │   ├── ProvenceLoss.py     # 統合損失関数
│   │   ├── PruningBCELoss.py   # Pruning専用損失
│   │   └── PruningMSELoss.py   # スコアベース損失
│   └── evaluation/
│       └── reranking.py        # 既存にProvence評価追加
├── utils/
│   └── text_chunking.py        # 言語別文分割（新規）
└── models/                      # Provence固有のモデル部品
    └── ProvencePruningHead.py  # Pruning head実装
```

**設計方針の変更**：
- 既存のCrossEncoderクラスを拡張（別クラスではなく）
- 既存ファイルへの最小限の変更で機能追加
- models/にProvence固有のコンポーネントを配置

### 主要コンポーネント

#### 1. CrossEncoder拡張

```python
from typing import Union, Optional, Dict, List, Tuple
import numpy as np
from sentence_transformers.models import ProvencePruningHead

class CrossEncoder:
    """既存のCrossEncoderクラスにProvence機能を統合"""
    
    def __init__(self, 
                 model_name: str,
                 num_labels: int = 1,
                 max_length: int = 512,
                 device: Optional[str] = None,
                 tokenizer_args: Optional[Dict] = None,
                 automodel_args: Optional[Dict] = None,
                 default_activation_function=None,
                 classifier_dropout: Optional[float] = None,
                 # Provence拡張パラメータ
                 enable_pruning: bool = False,
                 pruning_head_config: Optional[Dict] = None,
                 **kwargs):
        """
        Args:
            enable_pruning: Pruning機能を有効化
            pruning_head_config: Pruning headの設定
        """
        # 既存の初期化処理...
        
        # Provence拡張
        self.enable_pruning = enable_pruning
        if enable_pruning and self.model:
            self._add_pruning_head(pruning_head_config)
    
    def _add_pruning_head(self, config: Optional[Dict] = None):
        """Pruning headを追加"""
        from sentence_transformers.models import ProvencePruningHead
        config = config or {}
        self.pruning_head = ProvencePruningHead(
            hidden_size=self.model.config.hidden_size,
            **config
        )
    
    def predict_with_pruning(self, 
                           queries: List[str], 
                           documents: List[str], 
                           **kwargs) -> 'ProvenceOutput':
        """Provence特有の出力形式"""
        if not self.enable_pruning:
            raise ValueError("Pruning is not enabled. Initialize with enable_pruning=True")
        
    def prune(self, 
              query: str, 
              document: str, 
              threshold: float = 0.5,
              min_sentences: int = 1) -> str:
        """文書のプルーニング実行"""
        
    def save_pretrained(self, path: str):
        """Pruning headも含めて保存"""
        # 既存の保存処理...
        if self.enable_pruning:
            # Pruning headの保存処理
            pass
            
    @classmethod
    def from_pretrained(cls, model_name_or_path: str, **kwargs):
        """Pruning head対応のロード"""
        # Provence設定の自動検出
        # 既存のロード処理を拡張
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
import torch
import torch.nn as nn
from sentence_transformers.cross_encoder.losses import BinaryCrossEntropyLoss

class ProvenceLoss(nn.Module):
    """統合損失関数: Reranking + Pruning"""
    
    def __init__(self,
                 ranking_loss_fn: Optional[nn.Module] = None,
                 pruning_loss_fn: Optional[nn.Module] = None,
                 ranking_weight: float = 1.0,
                 pruning_weight: float = 0.5,
                 use_teacher_scores: bool = False):
        """
        Args:
            ranking_loss_fn: Reranking用損失関数（デフォルト: BinaryCrossEntropyLoss）
            pruning_loss_fn: Pruning用損失関数（デフォルト: BCEWithLogitsLoss）
            ranking_weight: Ranking損失の重み
            pruning_weight: Pruning損失の重み
            use_teacher_scores: 教師モデルのスコアを使用するか
        """
        super().__init__()
        self.ranking_loss_fn = ranking_loss_fn or BinaryCrossEntropyLoss()
        self.pruning_loss_fn = pruning_loss_fn or nn.BCEWithLogitsLoss()
        self.ranking_weight = ranking_weight
        self.pruning_weight = pruning_weight
        self.use_teacher_scores = use_teacher_scores
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
from typing import List, Tuple, Optional
import langdetect

class MultilingualChunker:
    """言語別の文分割器（参考: lm-trainers/pruning/scripts/multilingual_chunkers.py）"""
    
    def __init__(self):
        self._chunkers = {}  # 言語別チャンカーのキャッシュ
    
    def chunk_text(self, 
                   text: str, 
                   language: str = "auto",
                   preserve_whitespace: bool = True) -> List[Tuple[str, Tuple[int, int]]]:
        """
        テキストを文に分割し、位置情報も返す
        
        Args:
            text: 分割対象のテキスト
            language: 言語コード（"ja", "en", "auto"）
            preserve_whitespace: 空白文字を保持するか
            
        Returns:
            List[(文, (開始位置, 終了位置))]
        """
        if language == "auto":
            language = self._detect_language(text)
        
        chunker = self._get_chunker(language)
        return chunker.chunk(text, preserve_whitespace)
    
    def _detect_language(self, text: str) -> str:
        """言語を自動検出"""
        try:
            return langdetect.detect(text)
        except:
            return "en"  # フォールバック
    
    def _get_chunker(self, language: str):
        """言語別のチャンカーを取得（遅延ロード）"""
        if language not in self._chunkers:
            if language == "ja":
                from sentence_transformers.utils.japanese_chunker import JapaneseChunker
                self._chunkers[language] = JapaneseChunker()
            elif language == "zh":
                from sentence_transformers.utils.chinese_chunker import ChineseChunker
                self._chunkers[language] = ChineseChunker()
            else:
                from sentence_transformers.utils.default_chunker import DefaultChunker
                self._chunkers[language] = DefaultChunker()
        return self._chunkers[language]
    
    @staticmethod
    def reconstruct_text(sentences: List[str], 
                        masks: List[bool], 
                        positions: List[Tuple[int, int]],
                        original_text: Optional[str] = None) -> str:
        """マスクに基づいてテキストを再構築"""
        if original_text and positions:
            # 位置情報を使用して正確に再構築
            result = []
            for sent, mask, (start, end) in zip(sentences, masks, positions):
                if mask:
                    # 元のテキストから空白も含めて抽出
                    result.append(original_text[start:end])
            return "".join(result)
        else:
            # 位置情報なしの場合は簡易的に結合
            return " ".join(sent for sent, mask in zip(sentences, masks) if mask)
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
- **hotchpotch/wip-query-context-pruner**: バイリンガルQAデータセット（約130万サンプル）
  - 日本語（約61%）と英語（約39%）の混在データ
  - MS MARCO（英語/日本語翻訳版）が全体の約78%
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

# バックエンドサポート
- PyTorch（フル機能）
- ONNX（Reranking機能のみ、Pruningは未対応）
- OpenVINO（Reranking機能のみ、Pruningは未対応）
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