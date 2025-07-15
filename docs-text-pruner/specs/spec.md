# Text-Pruner実装仕様 (最新版)

## 更新履歴
- 2025-01-07: 初版作成
- 2025-01-07: Sentence Transformers統合設計追加
- 2025-01-07: Provence実装完了、バッチ学習アプローチ確立
- 2025-01-08: トークンレベルプルーニング実装完了、ja-minimal評価済み
- 2025-01-08: チャンクベース評価システム実装、3モデル（minimal, small, full）学習完了
- 2025-01-09: Provence → Pruning リネーム、pruning-onlyモード実装
- 2025-01-09: デュアルモードアーキテクチャ完成、F2評価結果更新、両モード対応のデータ構造実装
- 2025-01-09: 全6モデル学習完了（pruning-only×3、reranking+pruning×3）、包括的な性能比較実施
- 2025-01-09: Transformers互換性実装完了、5つの読み込み方法対応
- 2025-01-15: 評価スクリプトcheck_pruning.py追加、プルーニング効果の可視化機能実装

## 概要

Sentence TransformersにProvence論文ベースのtext-pruner機能を実装する。query-dependentな文書プルーニングとrerankingを統合した効率的なRAGパイプラインを実現。既存のCrossEncoderアーキテクチャを拡張し、OSSコミュニティへのPR/マージを目標とする。

### 動作モード

1. **reranking_pruning** (デフォルト): ランキングとプルーニングの両方を行う統合モード
   - ベースモデル: `hotchpotch/japanese-reranker-xsmall-v2`
   - 出力: `RerankingPruningOutput`（ランキングスコア + プルーニングマスク）
   - 用途: 高性能なRAGパイプライン、完全な機能が必要な場合

2. **pruning_only**: プルーニングのみを行う軽量モード
   - ベースモデル: `cl-nagoya/ruri-v3-30m`等の小型モデル
   - 出力: `PruningOnlyOutput`（プルーニングマスクのみ）
   - 用途: 計算コストを抑えたテキスト圧縮、プルーニング専用タスク

## 実装ステータス

- [x] 仕様策定
- [x] アーキテクチャ設計
- [x] PruningEncoderクラス実装（sentence_transformers/pruning/）
- [x] データローダー実装（チャンクベースのダイナミックラベル生成）
- [x] 損失関数実装（PruningLoss - モード自動判定）
- [x] トークンレベルプルーニング実装
- [x] チャンクベース評価（predict_context()メソッド）
- [x] 評価メトリクス実装（圧縮率、ランキング性能）
- [x] 学習・評価スクリプト作成（minimal, small, full対応）
- [x] 閾値最適化（F2スコアベース）
- [x] pruning-onlyモード実装（軽量モデル対応）
- [x] 包括的なモードテスト実装
- [x] デュアルモードアーキテクチャ（reranking_pruning + pruning_only）
- [x] モード専用データ構造（RerankingPruningOutput, PruningOnlyOutput）
- [x] 全テストセット完全評価（F2スコア最適化）
- [x] 6モデル性能比較（pruning-only vs reranking+pruning）
- [x] gradient_accumulation_stepsのバグ発見・修正
- [x] Transformers互換性実装（AutoModel対応）
- [x] CrossEncoder互換性確認
- [x] trust_remote_code不要の自動登録機能
- [ ] ドキュメント作成（API仕様等）
- [ ] PR作成

### 実装上の注意点

1. **学習スクリプト**: 本番用は`scripts/pruning_train.py`を使用。評価・実験用スクリプトは`tmp/old_scripts/`に保存
2. **設定ファイル**: `pruning-config/train-models/`にYAML形式で保存
3. **gradient_accumulation_steps**: HuggingFace TrainingArgumentsを使用することで正しく反映される

## 現在の実装成果

### 学習済みモデル

#### Pruning-Onlyモデル（cl-nagoya/ruri-v3-30m ベース）
- **Minimal**: エポック5、F2=0.7204
- **Small**: エポック2、F2=0.7204
- **Full**: エポック1、F2=0.7516

#### Reranking+Pruningモデル（japanese-reranker-xsmall-v2 ベース）
- **Minimal**: エポック5、F2=0.7187
- **Small**: エポック2、F2=0.7823（最良のバランス）
- **Full**: エポック1、F2=0.7647

### 性能比較（閾値0.5）

| モデル | データセット | POS F2 | NEG F2 | ALL F2 |
|--------|-------------|--------|--------|--------|
| Pruning-Only | Small | 0.7277 | 0.6645 | 0.6869 |
| **Reranking+Pruning** | **Small** | **0.8202** | **0.6815** | **0.7326** |
| Pruning-Only | Full | 0.7551 | 0.6579 | 0.6934 |
| **Reranking+Pruning** | **Full** | **0.8259** | 0.6458 | **0.7131** |

### 重要な発見

1. **最適閾値**: すべてのモデルで0.3が最高のF2スコア
2. **Reranking+Pruningの優位性**: 特にPOSサンプルで+9-13%の改善
3. **推奨モデル**:
   - 高精度: Reranking+Pruning Small (F2=0.7823)
   - 効率重視: Pruning-Only Full (F2=0.7516)

### バッチ学習アプローチ

複数のquery-textペアを効率的に処理する新設計：
1. **PruningDataCollator**: 柔軟なバッチ処理とチャンクベースのラベル生成
2. **動的な教師ラベル生成**: ペアごとにプルーニングラベルを自動生成
3. **効率的なメモリ使用**: mini-batch処理で大規模データセットに対応

### チャンクベース評価システム

1. **文単位の評価**: 日本語文分割による正確な評価
2. **F2スコア最適化**: Recall重視で誤削除を最小化
3. **閾値調整**: 最適閾値0.3で高いRecall維持

## 設計原則

1. **OSSコミュニティへの統合を前提**
   - Sentence Transformersの既存パターンに従う
   - CrossEncoderアーキテクチャを拡張
   - 独立したモジュールとして実装

2. **既存の設計パターンの活用**
   - CrossEncoderのpredict()パターン
   - 損失関数は柔軟に組み合わせ可能
   - 既存のSentence Transformersパターンに従う

3. **効率的な実装**
   - バッチ処理の最適化
   - メモリ効率を考慮した設計
   - 推論時のパフォーマンス重視

### ディレクトリ構成

```
sentence_transformers/
├── pruning/                     # Pruning実装（旧provence/、実装済み）
│   ├── __init__.py
│   ├── encoder.py              # PruningEncoderクラス（デュアルモード対応）
│   ├── losses.py               # PruningLoss（モード自動判定）
│   ├── data_collator.py        # PruningDataCollator（モード自動判定）
│   ├── trainer.py              # PruningTrainer（gradient_accumulation修正済み）
│   ├── data_structures.py      # データ構造定義（RerankingPruningOutput, PruningOnlyOutput）
│   ├── evaluation.py           # 評価メトリクス実装
│   ├── modeling_pruning_encoder.py  # Transformers互換モデル
│   ├── transformers_compat.py  # Transformers互換性ラッパー
│   ├── crossencoder_wrapper.py # CrossEncoder互換ラッパー
│   └── models/
│       └── pruning_head.py     # プルーニングヘッド実装
├── utils/
│   └── text_chunking.py        # 言語別文分割（実装済み）

scripts/                         # 学習・評価スクリプト
├── pruning_train.py            # 統合学習スクリプト（YAML/JSON設定ファイル対応）
├── check_pruning.py            # プルーニング効果可視化（削除/保持の正誤表示、F2スコア）
└── pruning_exec.py             # プルーニング実行とJSON評価（混合行列、F1/F2スコア）

tmp/old_scripts/                 # 評価・実験スクリプト（参考実装）
├── train_pruning_only_*.py     # pruning-onlyモード学習（minimal/small/full）
├── train_reranking_pruning_*.py # reranking+pruningモード学習
├── evaluate_pruning_f2*.py     # F2スコア評価
├── compare_all_models.py       # 全モデル比較評価
└── compare_models_full_test.py # 詳細な性能分析

tests/pruning/                   # テスト（実装済み）
└── test_pruning_modes.py       # デュアルモード包括テスト（save/load含む）
```

**実装アプローチ**：
- 独立したpruning/モジュールとして実装
- CrossEncoderパターンを参考に、バッチ処理に最適化
- LambdaLossパターンを活用したhard negative学習

### 主要コンポーネント

#### 1. PruningEncoder

```python
from typing import Union, Optional, Dict, List, Tuple
import numpy as np
from sentence_transformers.pruning import PruningEncoder

class PruningEncoder:
    """デュアルモード対応のPruningEncoder"""
    
    def __init__(self,
                 model_name_or_path: str,
                 mode: str = "reranking_pruning",  # or "pruning_only"
                 device: Optional[str] = None,
                 max_length: int = 512,
                 pruning_config: Optional[Dict] = None):
        """
        Args:
            mode: "reranking_pruning" or "pruning_only"
            pruning_config: プルーニングヘッドの設定
        """
        
    def predict_with_pruning(self,
                           sentences: List[Tuple[str, str]],
                           batch_size: int = 32,
                           pruning_threshold: float = 0.3,
                           return_documents: bool = True) -> List[Union[RerankingPruningOutput, PruningOnlyOutput]]:
        """統合予測メソッド"""
        
    def predict_context(self,
                       query: str,
                       contexts: List[str],
                       batch_size: int = 32,
                       pruning_threshold: float = 0.3) -> Dict:
        """チャンクベース評価用メソッド"""
```

#### 2. データ構造

```python
from dataclasses import dataclass
from typing import Optional, List
import numpy as np

@dataclass
class RerankingPruningOutput:
    """Reranking+Pruningモードの出力"""
    score: float
    pruning_masks: Optional[np.ndarray] = None
    sentences: Optional[List[str]] = None
    compression_ratio: Optional[float] = None
    pruned_document: Optional[str] = None

@dataclass  
class PruningOnlyOutput:
    """Pruning-Onlyモードの出力"""
    pruning_masks: Optional[np.ndarray] = None
    sentences: Optional[List[List[str]]] = None
    compression_ratio: Optional[float] = None
    num_pruned_tokens: Optional[int] = None
    pruned_documents: Optional[List[str]] = None
```

#### 3. 損失関数設計

```python
import torch
import torch.nn as nn

class PruningLoss(nn.Module):
    """モード自動判定統合損失関数"""
    
    def __init__(self,
                 model: PruningEncoder,
                 mode: Optional[str] = None,  # 自動検出
                 ranking_weight: float = 1.0,
                 pruning_weight: float = 1.0):
        """
        モードはmodelから自動検出、または明示的に指定
        """
```

## Transformers互換性（実装済み）

### 5つの読み込み方法

1. **フルPruningEncoder（完全機能）**
```python
from sentence_transformers.pruning import PruningEncoder
model = PruningEncoder.from_pretrained("path/to/model")
```

2. **ベースモデルのみ（特別なインポート不要）**
```python
from transformers import AutoModelForSequenceClassification
model = AutoModelForSequenceClassification.from_pretrained("path/to/model/ranking_model")
```

3. **AutoModel + 自動登録**
```python
import sentence_transformers  # 自動登録
from transformers import AutoModelForSequenceClassification
model = AutoModelForSequenceClassification.from_pretrained("path/to/model")
```

4. **CrossEncoder互換**
```python
from sentence_transformers import CrossEncoder
model = CrossEncoder("path/to/model")
```

5. **trust_remote_code**
```python
from transformers import AutoModelForSequenceClassification
model = AutoModelForSequenceClassification.from_pretrained("path/to/model", trust_remote_code=True)
```

### 実装詳細

- `transformers_compat.py`: Transformers互換性ラッパークラス
- `modeling_pruning_encoder.py`: auto_map用スタンドアロン実装
- 自動登録: `sentence_transformers.__init__.py`で自動インポート
- ベースモデル: `/ranking_model`サブディレクトリに完全なTransformersモデルとして保存

## 評価スクリプト

### check_pruning.py
プルーニング効果を視覚的に確認するためのスクリプト。削除されるコンテキストを`<del correct/incorrect>`タグで表示し、モデルの判定精度を評価。

```bash
# 使用例
python scripts/check_pruning.py -m output/model_path/final_model  # 日本語データ（デフォルト）
python scripts/check_pruning.py -j pruning-config/pruning_data_en.json -m output/model_path/final_model  # 英語データ
python scripts/check_pruning.py -j pruning-config/pruning_data_easy_ja.json -m output/model_path/final_model -s 100 -t 0.5  # 簡易評価用データ
```

**主な機能：**
- 削除/保持の正誤をビジュアル表示
- 混合行列とF1/F2スコア計算
- FN（重要文書の誤削除）の検出と警告
- 任意のJSONデータセットに対応

### 評価結果の例（閾値0.5）
- **削除率**: 17.9%（保守的な戦略）
- **FN = 0**: 重要な文書を誤って削除していない
- **F2 = 0.8929**: 高い再現率重視のスコア
- **精度**: 69.2%

## 今後の課題

1. **ドキュメント作成**: API仕様、使用例、ベンチマーク結果
2. **PR準備**: コミュニティへの貢献準備
3. **追加評価**: より大規模なデータセットでの評価

## 参考

- Provence論文: `@docs-text-pruner/provence/provence_paper.md`
- 実装仕様: `@docs-text-pruner/specs/provence-implementation-spec.md`
- データフォーマット: `@docs-text-pruner/specs/data-format-spec.md`