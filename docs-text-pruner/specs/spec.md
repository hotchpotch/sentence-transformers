# Text-Pruner実装仕様 (最新版)

## 更新履歴
- 2025-01-07: 初版作成
- 2025-01-07: Sentence Transformers統合設計追加
- 2025-01-07: Provence実装完了、バッチ学習アプローチ確立
- 2025-01-08: トークンレベルプルーニング実装完了、ja-minimal評価済み
- 2025-01-08: チャンクベース評価システム実装、3モデル（minimal, small, full）学習完了

## 概要

Sentence TransformersにProvence論文ベースのtext-pruner機能を実装する。query-dependentな文書プルーニングとrerankingを統合した効率的なRAGパイプラインを実現。既存のCrossEncoderアーキテクチャを拡張し、OSSコミュニティへのPR/マージを目標とする。

## 実装ステータス

- [x] 仕様策定
- [x] アーキテクチャ設計
- [x] ProvenceEncoderクラス実装（sentence_transformers/provence/）
- [x] データローダー実装（チャンクベースのダイナミックラベル生成）
- [x] 損失関数実装（ProvenceChunkBasedLoss）
- [x] トークンレベルプルーニング実装
- [x] チャンクベース評価（predict_context()メソッド）
- [x] 評価メトリクス実装（圧縮率、ランキング性能）
- [x] 学習・評価スクリプト作成（minimal, small, full対応）
- [x] 閾値最適化（F2スコアベース）
- [ ] ドキュメント作成（API仕様等）
- [ ] テスト実装
- [ ] PR作成

## 現在の実装成果

### バッチ学習アプローチ
- **Hard Negative学習**: 各クエリに対して5つのテキスト（1つの正例＋4つの負例）を同時処理
- **バッチサイズ**: 48クエリ × 5テキスト = 240ペア/バッチ
- **教師モデル**: hotchpotch/japanese-reranker-xsmall-v2による蒸留

### 学習結果サマリー

#### モデル性能比較（ja-fullデータセット、F2最適: トークン0.3/チャンク0.5）
| モデル | POS Recall | POS FN | NEG Precision | NEG FP | 総合評価 |
|--------|-----------|--------|---------------|--------|----------|
| ja-small | 89.85% | 27 | 75.41% | 30 | 汎化性能高 |
| ja-full | 94.36% | 15 | 89.13% | 10 | 最高性能 |

#### 学習設定比較
| パラメータ | ja-minimal | ja-small | ja-full |
|-----------|------------|----------|---------|
| エポック数 | 2 | 3 | 1 |
| バッチサイズ | 48 | 32 | 24 |
| 実効バッチサイズ | 48 | 32 | 48 |
| Gradient Accumulation | 1 | 1 | 2 |

### 主な特徴
- チャンクベース評価: predict_context()によるチャンク単位の性能評価
- F2最適化: Recall重視で誤削除（FN）を最小化
- 多段階閾値: トークンレベルとチャンクレベルの2段階制御
- POS/NEG分離評価: 関連/非関連文書で異なる最適化戦略

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
├── provence/                    # Provence実装（実装済み）
│   ├── __init__.py
│   ├── encoder.py              # ProvenceEncoderクラス（トークンレベル対応）
│   ├── losses_chunk_based.py   # チャンクベース損失関数
│   ├── data_collator_chunk_based.py # ダイナミックラベル生成
│   ├── trainer.py              # ProvenceTrainer
│   ├── data_structures.py      # データ構造定義
│   └── models/
│       └── pruning_head.py     # プルーニングヘッド実装
├── utils/
│   └── text_chunking.py        # 言語別文分割（実装済み）

scripts/                         # 学習・評価スクリプト（実装済み）
├── train_ja_minimal.py         # ja-minimal学習
├── evaluate_ja_minimal.py      # ja-minimal評価
└── check_ja_dataset.py         # データセット確認
```

**実装アプローチ**：
- 独立したprovence/モジュールとして実装
- CrossEncoderパターンを参考に、バッチ処理に最適化
- LambdaLossパターンを活用したhard negative学習

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
from sentence_transformers import CrossEncoder

# 1. モデルの初期化（Provence機能を有効化）
model = CrossEncoder(
    "microsoft/deberta-v3-base",
    enable_pruning=True,  # Provence機能を有効化
    pruning_head_config={
        "hidden_size": 768,
        "dropout": 0.1
    }
)

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
# enable_pruning=Trueでも従来のAPIは完全に動作
scores = model.predict([
    ["query1", "document1"],
    ["query2", "document2"]
])
```

#### データ変換

```python
def convert_to_provence_format(example):
    """WIPデータセットをProvence形式に変換"""
    converted = []
    
    for idx, (text, label, relevant_chunks, chunks_pos) in enumerate(
        zip(example["texts"], example["labels"], 
            example["relevant_chunks"], example["chunks_pos"])
    ):
        # 文の抽出
        sentences = []
        for start, end in chunks_pos:
            sentences.append(text[start:end])
        
        # pruning_labelsの生成
        pruning_labels = [
            1 if i in relevant_chunks else 0 
            for i in range(len(sentences))
        ]
        
        # 少なくとも1つの文は保持
        if sum(pruning_labels) == 0 and len(pruning_labels) > 0:
            pruning_labels[0] = 1
        
        converted.append({
            "query": example["query"],
            "document": text,
            "label": float(label),
            "pruning_labels": pruning_labels,
            "sentences": sentences,  # 事前分割済み
            "metadata": {
                "dataset_name": example["dataset_name"],
                "document_id": f"{example['id']}_{idx}"
            }
        })
    
    return {"examples": converted}
```

#### 学習例

```python
from sentence_transformers import CrossEncoderTrainer
from sentence_transformers.cross_encoder.losses import ProvenceLoss, BinaryCrossEntropyLoss
from datasets import load_dataset
import torch.nn as nn

# データセット準備
dataset = load_dataset("hotchpotch/wip-query-context-pruner")
# フラット化（1クエリ-1文書ペアに変換）
train_dataset = dataset["train"].map(
    convert_to_provence_format,
    batched=True,
    remove_columns=dataset["train"].column_names
).map(lambda x: x["examples"]).flatten()

# 損失関数設定
loss = ProvenceLoss(
    ranking_loss_fn=BinaryCrossEntropyLoss(),
    pruning_loss_fn=nn.BCEWithLogitsLoss(reduction='none'),  # 文ごとの損失
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
def validate_inputs(self, queries, documents, pruning_labels=None):
    """入力データの検証"""
    # 空文書チェック
    for doc in documents:
        if not doc or len(doc.strip()) == 0:
            raise ValueError("Empty document detected")
    
    # 極端に長い文書の処理
    MAX_SENTENCES = 512
    for doc in documents:
        sentences = self.chunker.chunk_text(doc)
        if len(sentences) > MAX_SENTENCES:
            logger.warning(f"Document has {len(sentences)} sentences, truncating to {MAX_SENTENCES}")
    
    # pruning_labelsの整合性チェック
    if pruning_labels:
        for doc, labels in zip(documents, pruning_labels):
            sentences = self.chunker.chunk_text(doc)
            if len(labels) != len(sentences):
                raise ValueError(f"Mismatch: {len(sentences)} sentences but {len(labels)} labels")

# エッジケース処理
class RobustMultilingualChunker(MultilingualChunker):
    def chunk_text(self, text, **kwargs):
        try:
            return super().chunk_text(text, **kwargs)
        except Exception as e:
            logger.warning(f"Chunking failed: {e}, falling back to simple split")
            # フォールバック: 改行と句点で簡易分割
            sentences = []
            for line in text.split('\n'):
                if line.strip():
                    # 句点で分割（日本語対応）
                    parts = line.replace('。', '。\n').replace('. ', '.\n').split('\n')
                    sentences.extend([s.strip() for s in parts if s.strip()])
            return [(s, (0, 0)) for s in sentences]  # 位置情報なし

# 学習時の安定性
training_args = CrossEncoderTrainingArguments(
    gradient_checkpointing=True,      # メモリ効率化
    gradient_accumulation_steps=4,    # 大きな実効バッチサイズ
    max_grad_norm=1.0,               # Gradient clipping
    eval_strategy="steps",
    eval_steps=500,
    metric_for_best_model="eval_f1",
    early_stopping_patience=3,
)

# カスタム損失関数でのNaN検出
class SafeProvenceLoss(ProvenceLoss):
    def forward(self, logits, labels):
        loss = super().forward(logits, labels)
        if torch.isnan(loss) or torch.isinf(loss):
            logger.error("NaN/Inf detected in loss")
            return torch.tensor(0.0, requires_grad=True)
        return loss
```

### 4. テスト戦略

```python
# ユニットテスト: tests/test_provence_cross_encoder.py
class TestProvenceCrossEncoder(unittest.TestCase):
    def test_backward_compatibility(self):
        """既存のCrossEncoder APIが動作することを確認"""
        model = CrossEncoder("bert-base-uncased", enable_pruning=False)
        scores = model.predict([["query", "doc"]])
        self.assertIsInstance(scores, np.ndarray)
    
    def test_pruning_initialization(self):
        """Pruning headが正しく初期化されることを確認"""
        model = CrossEncoder("bert-base-uncased", enable_pruning=True)
        self.assertTrue(hasattr(model, 'pruning_head'))
    
    def test_empty_document_handling(self):
        """空文書の適切なエラーハンドリング"""
        model = CrossEncoder("bert-base-uncased", enable_pruning=True)
        with self.assertRaises(ValueError):
            model.prune("query", "")
    
    def test_multilingual_chunking(self):
        """多言語文分割のテスト"""
        chunker = MultilingualChunker()
        
        # 日本語
        ja_text = "これは日本語です。文分割のテストです。"
        ja_chunks = chunker.chunk_text(ja_text, language="ja")
        self.assertEqual(len(ja_chunks), 2)
        
        # 英語
        en_text = "This is English. Testing sentence splitting."
        en_chunks = chunker.chunk_text(en_text, language="en")
        self.assertEqual(len(en_chunks), 2)
    
    def test_pruning_preservation(self):
        """最小文数が保持されることを確認"""
        model = CrossEncoder("bert-base-uncased", enable_pruning=True)
        result = model.prune(
            "query",
            "Sentence 1. Sentence 2. Sentence 3.",
            threshold=0.99,  # 非常に高い閾値
            min_sentences=1
        )
        self.assertGreater(len(result.strip()), 0)

# 統合テスト: tests/test_provence_integration.py
def test_full_training_pipeline():
    """学習パイプライン全体のテスト"""
    # 小さなダミーデータセット
    dataset = Dataset.from_dict({
        "query": ["q1", "q2"],
        "document": ["doc1", "doc2"],
        "label": [1, 0],
        "pruning_labels": [[1, 0], [0, 1]],
        "sentences": [["s1", "s2"], ["s3", "s4"]]
    })
    
    model = CrossEncoder("bert-base-uncased", enable_pruning=True)
    trainer = CrossEncoderTrainer(
        model=model,
        train_dataset=dataset,
        loss=ProvenceLoss(),
        args=CrossEncoderTrainingArguments(
            output_dir="./test_output",
            num_epochs=1,
            per_device_train_batch_size=2
        )
    )
    trainer.train()
    
    # モデルが学習されたことを確認
    assert os.path.exists("./test_output/model.safetensors")

# パフォーマンステスト
def test_inference_performance():
    """推論速度のベンチマーク"""
    model = CrossEncoder("bert-base-uncased", enable_pruning=True)
    queries = ["query"] * 100
    documents = ["Long document..." * 50] * 100
    
    # Provence推論
    start = time.time()
    provence_results = model.predict_with_pruning(queries, documents)
    provence_time = time.time() - start
    
    # 通常のreranking
    start = time.time()
    ranking_scores = model.predict(list(zip(queries, documents)))
    ranking_time = time.time() - start
    
    # Provence推論が1.5倍以内であることを確認
    assert provence_time < ranking_time * 1.5
```

## 実装ステップ（詳細版）

### Phase 1: 基礎実装（Week 1）

1. **CrossEncoderクラスの拡張**
   ```python
   # enable_pruning パラメータ追加
   # AutoModelForTokenClassification互換のpruning head
   # save/load メソッドの拡張
   ```

2. **データ構造とモジュール設計**
   - ProvenceOutput dataclass（ranking + pruning出力）
   - ProvencePruningHead（PreTrainedModel継承）
   - AutoModelへの登録

3. **基本的な推論機能**
   - predict_with_pruning（バッチ処理対応）
   - prune メソッド（単一文書処理）
   - 文境界の効率的な計算

### Phase 2: 学習機能（Week 2）

1. **損失関数実装**
   - ProvenceLoss（統合損失、教師スコア対応）
   - 文レベル/トークンレベルの切り替え
   - Gradient accumulation対応

2. **データコレーター拡張**
   - ProvenceCrossEncoderDataCollator
   - 動的文分割とキャッシング
   - offset_mappingを使用した境界計算

3. **MultilingualChunker完成**
   - 日本語（BudouX）、英語（NLTK）対応
   - フォールバック処理
   - 位置情報の正確な保持

### Phase 3: 推論最適化とAutoModel統合（Week 3）

1. **AutoModelForTokenClassification互換性**
   - ProvencePruningConfig/ProvencePruningHead
   - 単独使用可能な設計
   - HuggingFace Hubへのアップロード対応

2. **推論の最適化**
   - バッチ推論の並列化
   - Mixed precision推論
   - 動的バッチサイズ調整

3. **エラーハンドリング強化**
   - RobustMultilingualChunker
   - 空文書・長文書対応
   - メモリ不足時のフォールバック

### Phase 4: 評価とベンチマーク（Week 4）

1. **ProvenceEvaluator実装**
   - Reranking評価（MAP, MRR, NDCG）
   - Pruning評価（Precision, Recall, F1）
   - 統合評価（QA精度、圧縮率）

2. **ベンチマークスクリプト**
   - 速度比較（vs 標準CrossEncoder）
   - メモリ使用量測定
   - 多言語性能評価

3. **ドキュメント・サンプル**
   - APIリファレンス生成
   - Jupyterノートブック作成
   - 学習済みモデルの公開準備

### Phase 4: PR準備（3-5日）

1. **コードレビュー準備**
   ```bash
   # コードスタイルのチェック
   make check  # pre-commitフックの実行
   
   # Type hintsの完全性確認
   mypy sentence_transformers/
   
   # Docstringの確認
   pydocstyle sentence_transformers/
   ```

2. **PR作成ガイドライン**
   ```markdown
   ## Title: Add Provence (Query-dependent Text Pruning) support to CrossEncoder
   
   ### Description
   This PR adds query-dependent text pruning capabilities to CrossEncoder based on the Provence paper.
   The implementation extends the existing CrossEncoder while maintaining full backward compatibility.
   
   ### Key Features
   - ✅ Query-dependent document pruning for efficient RAG pipelines
   - ✅ Joint training of reranking and pruning objectives
   - ✅ Multilingual support (tested on English and Japanese)
   - ✅ Full backward compatibility with existing CrossEncoder API
   - ✅ Comprehensive test coverage
   
   ### Benchmarks
   | Model | Reranking (NDCG@10) | Compression Ratio | Speed (docs/sec) |
   |-------|---------------------|-------------------|------------------|
   | CrossEncoder | 0.687 | - | 120 |
   | CrossEncoder + Provence | 0.685 | 52.3% | 95 |
   
   ### Usage Example
   ```python
   from sentence_transformers import CrossEncoder
   
   model = CrossEncoder("microsoft/deberta-v3-base", enable_pruning=True)
   pruned_text = model.prune(query, document, threshold=0.5)
   ```
   
   ### Testing
   - Added 15 unit tests covering all new functionality
   - Integration tests for training pipeline
   - Performance benchmarks included
   
   ### Documentation
   - Updated API documentation
   - Added example notebook
   - Included in model card template
   
   Closes #[issue_number]
   ```

3. **チェックリスト**
   - [ ] `CONTRIBUTING.md`のガイドラインに従った
   - [ ] すべてのテストがパス
   - [ ] ドキュメントを更新
   - [ ] CHANGELOGにエントリを追加
   - [ ] ベンチマーク結果を含めた
   - [ ] サンプルコードが動作することを確認

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
- [詳細実装計画](./detailed-implementation-plan.md) - AutoModelForTokenClassification統合を含む実装手順

## 技術的な特徴

### AutoModelForTokenClassification互換性

```python
# 単独でToken Classificationモデルとして使用可能
from transformers import AutoModelForTokenClassification

model = AutoModelForTokenClassification.from_pretrained(
    "your-org/provence-pruner-deberta-v3"
)

# または、CrossEncoderの一部として
model = CrossEncoder(
    "microsoft/deberta-v3-base",
    enable_pruning=True  # 内部でtoken classification headを追加
)
```

### モジュラー設計の利点

1. **柔軟な使用方法**
   - Reranking only（従来のCrossEncoder）
   - Pruning only（Token Classifier）
   - Joint（Reranking + Pruning）

2. **既存エコシステムとの統合**
   - HuggingFace Transformersとの完全互換
   - Sentence Transformersのパターンに準拠
   - AutoModelレジストリへの登録

3. **拡張性**
   - カスタムチャンカーの追加が容易
   - 新しい言語サポートの追加
   - 異なるプーリング戦略の実装

## 次のステップ

1. この仕様書のレビューと改善（2回以上）
2. Phase 1の実装開始
3. 定期的な進捗確認とフィードバック