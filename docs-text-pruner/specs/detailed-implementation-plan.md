# Provence詳細実装計画

## 概要
Sentence TransformersにProvence（Query-dependent Text Pruning）を統合する詳細な実装計画。AutoModelForTokenClassificationとの互換性も考慮した設計。

## アーキテクチャ設計の詳細

### 1. モジュラー設計による柔軟性

```python
# 基本的なアーキテクチャ
from transformers import AutoModelForSequenceClassification, AutoModelForTokenClassification
from sentence_transformers import SentenceTransformer, models
import torch.nn as nn

class ProvenceModel(nn.Module):
    """
    Provenceモデル：3つのモードをサポート
    1. Reranking only (既存のCrossEncoder互換)
    2. Pruning only (AutoModelForTokenClassification互換)
    3. Joint (Reranking + Pruning)
    """
    
    def __init__(self, model_name: str, enable_reranking=True, enable_pruning=True):
        super().__init__()
        self.enable_reranking = enable_reranking
        self.enable_pruning = enable_pruning
        
        # ベースモデル
        self.encoder = AutoModel.from_pretrained(model_name)
        hidden_size = self.encoder.config.hidden_size
        
        # Reranking head (AutoModelForSequenceClassification互換)
        if self.enable_reranking:
            self.ranking_classifier = nn.Linear(hidden_size, 1)
            
        # Pruning head (AutoModelForTokenClassification互換)
        if self.enable_pruning:
            self.pruning_classifier = nn.Linear(hidden_size, 2)  # keep/prune
            
    def forward(self, input_ids, attention_mask, token_type_ids=None):
        # エンコーダー出力
        outputs = self.encoder(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids
        )
        
        sequence_output = outputs.last_hidden_state  # [batch, seq_len, hidden]
        pooled_output = outputs.pooler_output if hasattr(outputs, 'pooler_output') else sequence_output[:, 0]
        
        results = {}
        
        # Reranking
        if self.enable_reranking:
            results['ranking_logits'] = self.ranking_classifier(pooled_output)
            
        # Pruning (token classification)
        if self.enable_pruning:
            results['pruning_logits'] = self.pruning_classifier(sequence_output)
            
        return results
```

### 2. Sentence Transformersとの統合

```python
from sentence_transformers.models import Transformer

class ProvenceTransformer(Transformer):
    """Sentence Transformers用のProvenceモジュール"""
    
    def __init__(self, model_name_or_path: str, enable_pruning: bool = True, **kwargs):
        # 通常のTransformer初期化
        super().__init__(model_name_or_path, **kwargs)
        
        self.enable_pruning = enable_pruning
        if enable_pruning:
            # AutoModelForTokenClassificationのヘッドを追加
            hidden_size = self.auto_model.config.hidden_size
            self.pruning_head = nn.Linear(hidden_size, 2)
            
    def forward(self, features):
        # 通常のforward処理
        trans_features = super().forward(features)
        
        if self.enable_pruning and self.training:
            # Token classificationの出力も追加
            outputs = self.auto_model(**features, return_dict=True)
            token_logits = self.pruning_head(outputs.last_hidden_state)
            trans_features['token_logits'] = token_logits
            
        return trans_features
```

## 実装手順の詳細

### Phase 1: 基礎実装（Week 1）

#### 1.1 CrossEncoderの拡張

```python
# sentence_transformers/cross_encoder/CrossEncoder.py の修正

class CrossEncoder:
    def __init__(self, 
                 model_name: str,
                 num_labels: int = 1,
                 max_length: int = 512,
                 device: Optional[str] = None,
                 tokenizer_args: Optional[Dict] = None,
                 automodel_args: Optional[Dict] = None,
                 # Provence追加パラメータ
                 enable_pruning: bool = False,
                 pruning_mode: str = "sentence",  # "token" or "sentence"
                 pruning_head_config: Optional[Dict] = None,
                 **kwargs):
        
        # 既存の初期化処理...
        
        # Provence拡張
        self.enable_pruning = enable_pruning
        self.pruning_mode = pruning_mode
        
        if enable_pruning:
            # AutoModelForTokenClassificationとしても使えるように設計
            if hasattr(self.model, 'classifier'):
                # 既存のclassifierをrankingに変更
                self.model.ranking_classifier = self.model.classifier
                
            # Pruning head追加
            config = pruning_head_config or {}
            self.model.pruning_classifier = nn.Linear(
                self.model.config.hidden_size,
                config.get('num_labels', 2),  # binary: keep/prune
                bias=config.get('bias', True)
            )
            
            # Dropout層
            if 'dropout' in config:
                self.model.pruning_dropout = nn.Dropout(config['dropout'])
```

#### 1.2 データ構造の定義

```python
# sentence_transformers/cross_encoder/data_structures.py (新規)

from dataclasses import dataclass
from typing import List, Tuple, Optional, Union
import numpy as np
import torch

@dataclass
class ProvenceOutput:
    """Provence出力のデータクラス"""
    # Reranking outputs
    ranking_scores: Optional[np.ndarray] = None      # [batch_size]
    ranking_logits: Optional[torch.Tensor] = None    # [batch_size, 1]
    
    # Pruning outputs  
    pruning_masks: Optional[np.ndarray] = None       # [batch_size, max_sentences/tokens]
    pruning_logits: Optional[torch.Tensor] = None   # [batch_size, seq_len, 2]
    pruning_probs: Optional[np.ndarray] = None      # [batch_size, seq_len, 2]
    
    # Chunking information
    sentences: Optional[List[List[str]]] = None      # 分割された文リスト
    sentence_boundaries: Optional[List[List[Tuple[int, int]]]] = None  # トークン境界
    original_positions: Optional[List[List[Tuple[int, int]]]] = None   # 元テキストでの位置
    
    # Metadata
    compression_ratio: Optional[float] = None
    num_pruned_sentences: Optional[int] = None
    
    def to_dict(self) -> dict:
        """辞書形式に変換（シリアライズ用）"""
        return {
            k: v.tolist() if isinstance(v, (np.ndarray, torch.Tensor)) else v
            for k, v in self.__dict__.items()
            if v is not None
        }
```

#### 1.3 文分割器の実装

```python
# sentence_transformers/utils/text_chunking.py (新規)

import re
from typing import List, Tuple, Optional
import langdetect
from abc import ABC, abstractmethod

class BaseChunker(ABC):
    """基底チャンカークラス"""
    
    @abstractmethod
    def chunk(self, text: str) -> List[Tuple[str, Tuple[int, int]]]:
        """テキストを文に分割し、位置情報を返す"""
        pass

class EnglishChunker(BaseChunker):
    """英語用チャンカー"""
    
    def __init__(self):
        # nltk punkt tokenizerの遅延ロード
        self._tokenizer = None
        
    @property
    def tokenizer(self):
        if self._tokenizer is None:
            import nltk
            try:
                self._tokenizer = nltk.data.load('tokenizers/punkt/english.pickle')
            except LookupError:
                nltk.download('punkt', quiet=True)
                self._tokenizer = nltk.data.load('tokenizers/punkt/english.pickle')
        return self._tokenizer
        
    def chunk(self, text: str) -> List[Tuple[str, Tuple[int, int]]]:
        # Punktを使用した文分割
        spans = list(self.tokenizer.span_tokenize(text))
        return [(text[start:end], (start, end)) for start, end in spans]

class JapaneseChunker(BaseChunker):
    """日本語用チャンカー"""
    
    def __init__(self):
        self._segmenter = None
        
    @property 
    def segmenter(self):
        if self._segmenter is None:
            try:
                import budoux
                self._segmenter = budoux.load_default_japanese_parser()
            except ImportError:
                # フォールバック: 句点での分割
                self._segmenter = None
        return self._segmenter
        
    def chunk(self, text: str) -> List[Tuple[str, Tuple[int, int]]]:
        if self.segmenter:
            # BudouXを使用
            sentences = self.segmenter.parse(text)
            result = []
            pos = 0
            for sent in sentences:
                start = text.find(sent, pos)
                if start != -1:
                    end = start + len(sent)
                    result.append((sent, (start, end)))
                    pos = end
            return result
        else:
            # フォールバック: 句点での分割
            pattern = r'([^。！？\n]+[。！？]?)'
            sentences = []
            for match in re.finditer(pattern, text):
                sentences.append((match.group(0), match.span()))
            return sentences if sentences else [(text, (0, len(text)))]

class MultilingualChunker:
    """統合チャンカー"""
    
    def __init__(self):
        self._chunkers = {}
        
    def chunk_text(self, 
                   text: str, 
                   language: str = "auto",
                   tokenizer = None,
                   max_length: int = None) -> List[Tuple[str, Tuple[int, int]]]:
        """
        テキストを文に分割
        
        Args:
            text: 分割対象のテキスト
            language: 言語コード
            tokenizer: トークナイザー（トークン境界の計算用）
            max_length: 最大トークン長（長い文の分割用）
        """
        if language == "auto":
            language = self._detect_language(text)
            
        chunker = self._get_chunker(language)
        sentences = chunker.chunk(text)
        
        # トークン境界の計算（必要な場合）
        if tokenizer and max_length:
            sentences = self._split_long_sentences(sentences, tokenizer, max_length)
            
        return sentences
```

### Phase 2: 学習機能実装（Week 2）

#### 2.1 損失関数の実装

```python
# sentence_transformers/cross_encoder/losses/ProvenceLoss.py

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Optional, Union

class ProvenceLoss(nn.Module):
    """
    Provence統合損失関数
    Reranking損失とPruning損失を組み合わせる
    """
    
    def __init__(self,
                 model,
                 ranking_loss_fn: Optional[nn.Module] = None,
                 pruning_loss_fn: Optional[nn.Module] = None,
                 ranking_weight: float = 1.0,
                 pruning_weight: float = 0.5,
                 use_teacher_scores: bool = False,
                 sentence_level_pruning: bool = True):
        super().__init__()
        
        self.model = model
        self.ranking_loss_fn = ranking_loss_fn or nn.BCEWithLogitsLoss()
        self.pruning_loss_fn = pruning_loss_fn or nn.CrossEntropyLoss(reduction='none')
        self.ranking_weight = ranking_weight
        self.pruning_weight = pruning_weight
        self.use_teacher_scores = use_teacher_scores
        self.sentence_level_pruning = sentence_level_pruning
        
    def forward(self, sentence_features: List[Dict[str, torch.Tensor]], labels: torch.Tensor):
        """
        Args:
            sentence_features: トークナイズされた入力
            labels: 辞書形式のラベル
                - 'ranking_labels': [batch_size] (0/1 or scores)
                - 'pruning_labels': [batch_size, max_sentences] or [batch_size, seq_len]
                - 'sentence_boundaries': [batch_size, max_sentences, 2] (オプション)
                - 'teacher_scores': [batch_size] (オプション)
        """
        # モデルのforward
        outputs = self.model(sentence_features[0])
        
        total_loss = 0.0
        losses = {}
        
        # Ranking loss
        if 'ranking_logits' in outputs and 'ranking_labels' in labels:
            ranking_labels = labels['ranking_labels']
            
            if self.use_teacher_scores and 'teacher_scores' in labels:
                # 教師スコアを使用（蒸留）
                ranking_loss = F.mse_loss(
                    outputs['ranking_logits'].squeeze(),
                    labels['teacher_scores']
                )
            else:
                # バイナリ分類
                ranking_loss = self.ranking_loss_fn(
                    outputs['ranking_logits'].squeeze(),
                    ranking_labels.float()
                )
                
            total_loss += self.ranking_weight * ranking_loss
            losses['ranking_loss'] = ranking_loss
            
        # Pruning loss
        if 'pruning_logits' in outputs and 'pruning_labels' in labels:
            pruning_logits = outputs['pruning_logits']  # [batch, seq_len, 2]
            pruning_labels = labels['pruning_labels']   # [batch, max_sentences] or [batch, seq_len]
            
            if self.sentence_level_pruning and 'sentence_boundaries' in labels:
                # 文レベルのpruning
                sentence_loss = self._compute_sentence_level_loss(
                    pruning_logits,
                    pruning_labels,
                    labels['sentence_boundaries'],
                    sentence_features[0]['attention_mask']
                )
                total_loss += self.pruning_weight * sentence_loss
                losses['pruning_loss'] = sentence_loss
            else:
                # トークンレベルのpruning
                # attention_maskを考慮
                attention_mask = sentence_features[0]['attention_mask']
                active_loss = attention_mask.view(-1) == 1
                active_logits = pruning_logits.view(-1, 2)[active_loss]
                active_labels = pruning_labels.view(-1)[active_loss]
                
                token_loss = self.pruning_loss_fn(active_logits, active_labels)
                token_loss = token_loss.mean()
                
                total_loss += self.pruning_weight * token_loss
                losses['pruning_loss'] = token_loss
        
        # ログ用
        self.last_losses = losses
        
        return total_loss
        
    def _compute_sentence_level_loss(self, 
                                   token_logits: torch.Tensor,
                                   sentence_labels: torch.Tensor,
                                   sentence_boundaries: torch.Tensor,
                                   attention_mask: torch.Tensor) -> torch.Tensor:
        """文レベルの損失を計算"""
        batch_size = token_logits.size(0)
        total_loss = 0.0
        num_sentences = 0
        
        for i in range(batch_size):
            for j, (start, end) in enumerate(sentence_boundaries[i]):
                if start == -1:  # パディング
                    break
                    
                # 文内のトークンのlogitsを集約
                sentence_logits = token_logits[i, start:end, :]  # [num_tokens, 2]
                
                # 平均プーリング
                sentence_score = sentence_logits.mean(dim=0)  # [2]
                
                # 損失計算
                label = sentence_labels[i, j].long()
                loss = F.cross_entropy(sentence_score.unsqueeze(0), label.unsqueeze(0))
                
                total_loss += loss
                num_sentences += 1
                
        return total_loss / max(num_sentences, 1)
```

#### 2.2 データコレーターの実装

```python
# sentence_transformers/cross_encoder/data_collator.py への追加

class ProvenceCrossEncoderDataCollator(CrossEncoderDataCollator):
    """Provence用のデータコレーター"""
    
    def __init__(self,
                 tokenizer,
                 chunker: Optional[MultilingualChunker] = None,
                 max_sentences: int = 64,
                 sentence_level: bool = True,
                 include_positions: bool = True,
                 **kwargs):
        super().__init__(tokenizer, **kwargs)
        self.chunker = chunker or MultilingualChunker()
        self.max_sentences = max_sentences
        self.sentence_level = sentence_level
        self.include_positions = include_positions
        
    def __call__(self, features: List[Dict]) -> Dict[str, torch.Tensor]:
        """
        期待される入力フォーマット:
        {
            'query': str,
            'document': str,
            'label': float/int (ranking label),
            'pruning_labels': List[int] (optional),
            'sentences': List[str] (optional, pre-chunked),
            'teacher_score': float (optional)
        }
        """
        batch = super().__call__(features)
        
        # Provence固有の処理
        if any('pruning_labels' in f for f in features):
            batch_pruning_labels = []
            batch_sentence_boundaries = []
            
            for feature in features:
                # 文分割
                if 'sentences' in feature:
                    sentences = feature['sentences']
                else:
                    # 動的に文分割
                    doc = feature['document']
                    sentences_with_pos = self.chunker.chunk_text(doc)
                    sentences = [s for s, _ in sentences_with_pos]
                
                # トークナイズして文境界を特定
                query_doc = feature['query'] + ' ' + feature['document']
                tokenized = self.tokenizer(
                    query_doc,
                    truncation=True,
                    max_length=self.max_length,
                    return_offsets_mapping=True
                )
                
                # 文境界のトークン位置を計算
                if self.sentence_level:
                    boundaries = self._compute_sentence_boundaries(
                        feature['document'],
                        sentences,
                        tokenized['offset_mapping']
                    )
                    
                    # pruning_labelsをパディング
                    pruning_labels = feature.get('pruning_labels', [1] * len(sentences))
                    pruning_labels = pruning_labels[:self.max_sentences]
                    pruning_labels += [-100] * (self.max_sentences - len(pruning_labels))
                    
                    batch_pruning_labels.append(pruning_labels)
                    batch_sentence_boundaries.append(boundaries)
                else:
                    # トークンレベルのラベル
                    # TODO: 実装
                    pass
            
            batch['labels'] = {
                'ranking_labels': torch.tensor([f['label'] for f in features]),
                'pruning_labels': torch.tensor(batch_pruning_labels),
                'sentence_boundaries': torch.tensor(batch_sentence_boundaries)
            }
            
            if any('teacher_score' in f for f in features):
                batch['labels']['teacher_scores'] = torch.tensor(
                    [f.get('teacher_score', 0.0) for f in features]
                )
                
        return batch
        
    def _compute_sentence_boundaries(self, 
                                   document: str,
                                   sentences: List[str],
                                   offset_mapping: List[Tuple[int, int]]) -> List[Tuple[int, int]]:
        """文のトークン境界を計算"""
        boundaries = []
        doc_start = offset_mapping[0][0]  # クエリの後のドキュメント開始位置を探す
        
        for sent in sentences[:self.max_sentences]:
            # 文の開始・終了位置を特定
            sent_start = document.find(sent)
            sent_end = sent_start + len(sent)
            
            # トークンインデックスに変換
            start_token = None
            end_token = None
            
            for i, (token_start, token_end) in enumerate(offset_mapping):
                if token_start >= doc_start + sent_start and start_token is None:
                    start_token = i
                if token_end >= doc_start + sent_end:
                    end_token = i + 1
                    break
                    
            if start_token is not None and end_token is not None:
                boundaries.append((start_token, end_token))
            else:
                boundaries.append((-1, -1))  # 見つからない場合
                
        # パディング
        boundaries += [(-1, -1)] * (self.max_sentences - len(boundaries))
        
        return boundaries
```

### Phase 3: 推論機能の実装（Week 3）

#### 3.1 予測メソッドの実装

```python
# CrossEncoder.pyへの追加メソッド

def predict_with_pruning(self,
                        queries: List[str],
                        documents: List[str],
                        batch_size: int = 32,
                        show_progress_bar: bool = False,
                        pruning_mode: str = "dynamic",
                        pruning_threshold: float = 0.5,
                        pruning_ratio: Optional[float] = None,
                        min_sentences: int = 1,
                        return_documents: bool = False,
                        **kwargs) -> ProvenceOutput:
    """
    Reranking + Pruning の統合予測
    
    Args:
        queries: クエリのリスト
        documents: 文書のリスト
        pruning_mode: "dynamic", "threshold", "ratio"
        pruning_threshold: 閾値モード時の保持閾値
        pruning_ratio: 比率モード時の削減率
        min_sentences: 最小保持文数
        return_documents: プルーニング後の文書を返すか
    """
    if not self.enable_pruning:
        raise ValueError("Pruning is not enabled. Initialize with enable_pruning=True")
        
    # バッチ処理の準備
    results = ProvenceOutput(
        ranking_scores=[],
        pruning_masks=[],
        sentences=[],
        sentence_boundaries=[]
    )
    
    # チャンカーの初期化
    chunker = MultilingualChunker()
    
    # バッチ処理
    for batch_start in tqdm(range(0, len(queries), batch_size), 
                           disable=not show_progress_bar,
                           desc="Provence Inference"):
        batch_queries = queries[batch_start:batch_start + batch_size]
        batch_documents = documents[batch_start:batch_start + batch_size]
        
        # 文分割
        batch_sentences = []
        for doc in batch_documents:
            sentences_with_pos = chunker.chunk_text(doc, **kwargs)
            sentences = [s for s, _ in sentences_with_pos]
            batch_sentences.append(sentences)
            
        # トークナイズ
        inputs = []
        for q, d in zip(batch_queries, batch_documents):
            inputs.append(q + ' [SEP] ' + d)
            
        tokenized = self.tokenizer(
            inputs,
            padding=True,
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt",
            return_offsets_mapping=True
        )
        
        # GPU転送
        tokenized = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v 
                    for k, v in tokenized.items()}
        
        # 推論
        with torch.no_grad():
            outputs = self.model(tokenized)
            
        # Ranking scores
        if 'ranking_logits' in outputs:
            scores = torch.sigmoid(outputs['ranking_logits']).cpu().numpy().squeeze()
            results.ranking_scores.extend(scores)
            
        # Pruning predictions
        if 'pruning_logits' in outputs:
            # 文境界の計算
            boundaries = self._compute_batch_boundaries(
                batch_documents,
                batch_sentences,
                tokenized['offset_mapping']
            )
            
            # 文レベルのスコア集約
            sentence_scores = self._aggregate_token_scores(
                outputs['pruning_logits'],
                boundaries
            )
            
            # Pruning決定
            masks = self._decide_pruning(
                sentence_scores,
                mode=pruning_mode,
                threshold=pruning_threshold,
                ratio=pruning_ratio,
                min_sentences=min_sentences
            )
            
            results.pruning_masks.extend(masks)
            results.sentences.extend(batch_sentences)
            results.sentence_boundaries.extend(boundaries)
    
    # 圧縮率の計算
    if results.pruning_masks:
        total_sentences = sum(len(s) for s in results.sentences)
        kept_sentences = sum(sum(m) for m in results.pruning_masks)
        results.compression_ratio = 1.0 - (kept_sentences / total_sentences)
        
    # プルーニング後の文書生成（オプション）
    if return_documents:
        pruned_docs = []
        for sentences, mask in zip(results.sentences, results.pruning_masks):
            pruned = ' '.join(s for s, m in zip(sentences, mask) if m)
            pruned_docs.append(pruned)
        results.pruned_documents = pruned_docs
        
    return results

def prune(self,
          query: str,
          document: str,
          threshold: float = 0.5,
          min_sentences: int = 1,
          return_scores: bool = False) -> Union[str, Tuple[str, np.ndarray]]:
    """
    単一文書のプルーニング
    
    Args:
        query: クエリ
        document: プルーニング対象の文書
        threshold: 保持閾値
        min_sentences: 最小保持文数
        return_scores: 各文のスコアも返すか
    """
    result = self.predict_with_pruning(
        [query],
        [document],
        pruning_mode="threshold",
        pruning_threshold=threshold,
        min_sentences=min_sentences,
        return_documents=True
    )
    
    pruned_text = result.pruned_documents[0]
    
    if return_scores:
        # 各文のスコアを返す
        scores = result.pruning_probs[0] if result.pruning_probs else None
        return pruned_text, scores
    else:
        return pruned_text
```

#### 3.2 AutoModelForTokenClassification互換性

```python
# sentence_transformers/models/ProvencePruningHead.py

import torch
import torch.nn as nn
from transformers import PreTrainedModel, PretrainedConfig
from transformers.modeling_outputs import TokenClassifierOutput

class ProvencePruningConfig(PretrainedConfig):
    """Provence Pruning Headの設定クラス"""
    model_type = "provence_pruning"
    
    def __init__(self,
                 hidden_size=768,
                 num_labels=2,
                 classifier_dropout=0.1,
                 sentence_pooling="mean",
                 **kwargs):
        super().__init__(**kwargs)
        self.hidden_size = hidden_size
        self.num_labels = num_labels
        self.classifier_dropout = classifier_dropout
        self.sentence_pooling = sentence_pooling

class ProvencePruningHead(PreTrainedModel):
    """
    AutoModelForTokenClassification互換のPruning Head
    単独でも使用可能
    """
    config_class = ProvencePruningConfig
    
    def __init__(self, config):
        super().__init__(config)
        
        self.dropout = nn.Dropout(config.classifier_dropout)
        self.classifier = nn.Linear(config.hidden_size, config.num_labels)
        self.sentence_pooling = config.sentence_pooling
        
        # 重み初期化
        self.init_weights()
        
    def forward(self,
                hidden_states: torch.Tensor,
                attention_mask: Optional[torch.Tensor] = None,
                sentence_boundaries: Optional[torch.Tensor] = None,
                labels: Optional[torch.Tensor] = None,
                **kwargs) -> TokenClassifierOutput:
        """
        Args:
            hidden_states: [batch_size, seq_len, hidden_size]
            attention_mask: [batch_size, seq_len]
            sentence_boundaries: [batch_size, max_sentences, 2]
            labels: [batch_size, seq_len] or [batch_size, max_sentences]
        """
        # Dropout
        hidden_states = self.dropout(hidden_states)
        
        # Classification
        logits = self.classifier(hidden_states)  # [batch_size, seq_len, num_labels]
        
        loss = None
        if labels is not None:
            loss_fct = nn.CrossEntropyLoss()
            
            if sentence_boundaries is not None:
                # 文レベルの損失
                loss = self._compute_sentence_loss(
                    logits, labels, sentence_boundaries, attention_mask
                )
            else:
                # トークンレベルの損失
                active_loss = attention_mask.view(-1) == 1
                active_logits = logits.view(-1, self.config.num_labels)[active_loss]
                active_labels = labels.view(-1)[active_loss]
                loss = loss_fct(active_logits, active_labels)
                
        return TokenClassifierOutput(
            loss=loss,
            logits=logits,
            hidden_states=kwargs.get('hidden_states'),
            attentions=kwargs.get('attentions')
        )
        
    def _compute_sentence_loss(self, logits, labels, boundaries, attention_mask):
        """文レベルの損失計算"""
        # Phase 2の実装を参照
        pass

# AutoModelへの登録
from transformers import AutoConfig, AutoModelForTokenClassification

AutoConfig.register("provence_pruning", ProvencePruningConfig)
AutoModelForTokenClassification.register(ProvencePruningConfig, ProvencePruningHead)
```

### Phase 4: 評価とベンチマーク（Week 4）

#### 4.1 評価メトリクスの実装

```python
# sentence_transformers/cross_encoder/evaluation/ProvenceEvaluator.py

from typing import List, Dict, Tuple
import numpy as np
from sklearn.metrics import precision_recall_fscore_support
import logging

logger = logging.getLogger(__name__)

class ProvenceEvaluator:
    """
    Provence用の統合評価器
    Reranking性能とPruning性能の両方を評価
    """
    
    def __init__(self,
                 queries: List[str],
                 documents: List[List[str]],
                 relevant_docs: List[List[int]],
                 pruning_labels: Optional[List[List[List[int]]]] = None,
                 k_values: List[int] = [1, 3, 5, 10],
                 name: str = "provence"):
        """
        Args:
            queries: クエリのリスト
            documents: 各クエリに対する文書リスト
            relevant_docs: 関連文書のインデックス
            pruning_labels: 各文書の文レベルのpruningラベル
        """
        self.queries = queries
        self.documents = documents
        self.relevant_docs = relevant_docs
        self.pruning_labels = pruning_labels
        self.k_values = k_values
        self.name = name
        
    def __call__(self, model, output_path: str = None, epoch: int = -1, steps: int = -1) -> Dict[str, float]:
        """評価を実行"""
        metrics = {}
        
        # Reranking評価
        reranking_metrics = self._evaluate_reranking(model)
        metrics.update(reranking_metrics)
        
        # Pruning評価（ラベルがある場合）
        if self.pruning_labels and model.enable_pruning:
            pruning_metrics = self._evaluate_pruning(model)
            metrics.update(pruning_metrics)
            
            # 統合評価
            integrated_metrics = self._evaluate_integrated(model)
            metrics.update(integrated_metrics)
        
        # ログ出力
        logger.info(f"Provence Evaluation Results (epoch {epoch}, steps {steps}):")
        for k, v in metrics.items():
            logger.info(f"  {k}: {v:.4f}")
            
        return metrics
        
    def _evaluate_reranking(self, model) -> Dict[str, float]:
        """Reranking性能の評価"""
        all_scores = []
        
        # 各クエリに対してスコアを計算
        for query, docs in zip(self.queries, self.documents):
            pairs = [[query, doc] for doc in docs]
            scores = model.predict(pairs, show_progress_bar=False)
            all_scores.append(scores)
            
        # メトリクス計算
        metrics = {}
        for k in self.k_values:
            map_scores = []
            mrr_scores = []
            ndcg_scores = []
            
            for scores, relevant in zip(all_scores, self.relevant_docs):
                # ランキング
                ranked_indices = np.argsort(scores)[::-1]
                
                # MAP@k
                ap = self._average_precision_at_k(ranked_indices, relevant, k)
                map_scores.append(ap)
                
                # MRR@k
                mrr = self._reciprocal_rank_at_k(ranked_indices, relevant, k)
                mrr_scores.append(mrr)
                
                # NDCG@k
                ndcg = self._ndcg_at_k(ranked_indices, relevant, k)
                ndcg_scores.append(ndcg)
                
            metrics[f'reranking_map@{k}'] = np.mean(map_scores)
            metrics[f'reranking_mrr@{k}'] = np.mean(mrr_scores)
            metrics[f'reranking_ndcg@{k}'] = np.mean(ndcg_scores)
            
        return metrics
        
    def _evaluate_pruning(self, model) -> Dict[str, float]:
        """Pruning性能の評価"""
        all_predictions = []
        all_labels = []
        compression_ratios = []
        
        for query, docs, labels_per_doc in zip(self.queries, self.documents, self.pruning_labels):
            # Pruning予測
            result = model.predict_with_pruning(
                [query] * len(docs),
                docs,
                show_progress_bar=False
            )
            
            # 予測とラベルの収集
            for pred_mask, true_labels in zip(result.pruning_masks, labels_per_doc):
                # 長さを合わせる
                min_len = min(len(pred_mask), len(true_labels))
                all_predictions.extend(pred_mask[:min_len])
                all_labels.extend(true_labels[:min_len])
                
            # 圧縮率
            if result.compression_ratio:
                compression_ratios.append(result.compression_ratio)
                
        # メトリクス計算
        precision, recall, f1, _ = precision_recall_fscore_support(
            all_labels, all_predictions, average='binary', pos_label=1
        )
        
        metrics = {
            'pruning_precision': precision,
            'pruning_recall': recall,
            'pruning_f1': f1,
            'avg_compression_ratio': np.mean(compression_ratios) if compression_ratios else 0.0
        }
        
        return metrics
        
    def _evaluate_integrated(self, model) -> Dict[str, float]:
        """統合評価: QAタスクでの性能など"""
        # TODO: 実装
        # - プルーニング後の文書でQAモデルを実行
        # - 回答の正確性を評価
        # - レイテンシの測定
        return {}
```

#### 4.2 ベンチマークスクリプト

```python
# examples/training/provence/benchmark_provence.py

import time
import torch
from sentence_transformers import CrossEncoder
from datasets import load_dataset
import numpy as np

def benchmark_provence(model_name: str = "microsoft/deberta-v3-base"):
    """Provenceモデルのベンチマーク"""
    
    # モデルの初期化
    print("Loading models...")
    standard_model = CrossEncoder(model_name, enable_pruning=False)
    provence_model = CrossEncoder(model_name, enable_pruning=True)
    
    # テストデータの準備
    print("Loading test data...")
    dataset = load_dataset("hotchpotch/wip-query-context-pruner", split="train[:100]")
    
    queries = dataset["query"]
    documents = [doc[0] for doc in dataset["texts"]]  # 最初の文書のみ
    
    # ベンチマーク結果
    results = {
        'model': model_name,
        'num_samples': len(queries)
    }
    
    # 1. Reranking速度
    print("Benchmarking standard reranking...")
    start = time.time()
    standard_scores = standard_model.predict(
        list(zip(queries, documents)),
        batch_size=32
    )
    standard_time = time.time() - start
    results['standard_time'] = standard_time
    results['standard_throughput'] = len(queries) / standard_time
    
    # 2. Provence速度
    print("Benchmarking Provence (reranking + pruning)...")
    start = time.time()
    provence_results = provence_model.predict_with_pruning(
        queries,
        documents,
        batch_size=32,
        pruning_mode="dynamic"
    )
    provence_time = time.time() - start
    results['provence_time'] = provence_time
    results['provence_throughput'] = len(queries) / provence_time
    results['slowdown_factor'] = provence_time / standard_time
    
    # 3. 圧縮率
    results['avg_compression_ratio'] = provence_results.compression_ratio
    
    # 4. メモリ使用量
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
        
        # Standard model
        _ = standard_model.predict(list(zip(queries[:10], documents[:10])))
        standard_memory = torch.cuda.max_memory_allocated() / 1024**2  # MB
        
        torch.cuda.reset_peak_memory_stats()
        
        # Provence model
        _ = provence_model.predict_with_pruning(queries[:10], documents[:10])
        provence_memory = torch.cuda.max_memory_allocated() / 1024**2  # MB
        
        results['standard_memory_mb'] = standard_memory
        results['provence_memory_mb'] = provence_memory
        results['memory_overhead'] = provence_memory / standard_memory
    
    # 結果の表示
    print("\n=== Benchmark Results ===")
    for k, v in results.items():
        if isinstance(v, float):
            print(f"{k}: {v:.2f}")
        else:
            print(f"{k}: {v}")
            
    return results

if __name__ == "__main__":
    benchmark_provence()
```

## 実装の優先順位とマイルストーン

### Week 1: 基礎実装
- [ ] CrossEncoderクラスの拡張
- [ ] ProvenceOutputデータ構造
- [ ] 基本的なpredict_with_pruningメソッド
- [ ] テストケースの作成

### Week 2: 学習機能
- [ ] ProvenceLoss実装
- [ ] データコレーター実装
- [ ] 学習スクリプトの作成
- [ ] 小規模データでの動作確認

### Week 3: 推論最適化
- [ ] MultilingualChunker完成
- [ ] バッチ推論の最適化
- [ ] AutoModelForTokenClassification互換性
- [ ] pruneメソッドの完成

### Week 4: 評価とドキュメント
- [ ] ProvenceEvaluator実装
- [ ] ベンチマークスクリプト
- [ ] APIドキュメント作成
- [ ] サンプルノートブック作成

### Week 5: PR準備
- [ ] コードレビューと修正
- [ ] 包括的なテスト追加
- [ ] PR作成とコミュニティフィードバック対応

## 技術的な考慮事項

1. **メモリ効率**
   - 長い文書の効率的な処理
   - バッチサイズの動的調整
   - Gradient checkpointing

2. **互換性**
   - 既存のCrossEncoder APIとの完全互換
   - HuggingFace Transformersとの統合
   - ONNX/OpenVINOサポート（将来）

3. **拡張性**
   - カスタムチャンカーのサポート
   - 異なるプーリング戦略
   - マルチタスク学習への対応

4. **パフォーマンス**
   - Mixed precision training
   - データ並列処理
   - 推論時のキャッシング