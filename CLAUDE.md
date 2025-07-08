# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## プロジェクト概要

Sentence Transformersは、最先端の文章・画像埋め込みモデルを提供するPythonフレームワークです。このブランチ（text-pruner）では、Provence論文に基づくquery-dependent text prunerの実装を進めています。

## 開発環境設定

### パッケージマネージャ
このプロジェクトでは**UV**を使用します：
- `uv sync` - 依存関係のインストール
- `uv run python` - Pythonスクリプトの実行
- `uv add <package>` - パッケージの追加
- `uv pip install -e .` - 開発モードでのインストール

### よく使うコマンド

```bash
# テスト実行
make test                     # 単体テストの実行
pytest tests/test_XXX.py      # 特定のテストファイルの実行
pytest -k "test_name"         # 特定のテスト関数の実行
make test-cov                 # カバレッジレポート付きテスト

# コード品質チェック
make check                    # pre-commitフックの実行（linting）
pre-commit install            # gitフックのインストール

# 開発用サンプル実行
uv run python tmp/sample.py   # tmpディレクトリのサンプルコード実行
```

## アーキテクチャ概要

### 主要コンポーネント

1. **SentenceTransformer**: メインクラス。モデルのロード、エンコーディング、学習を管理
2. **models/**: モジュラーなコンポーネント（Transformer、Pooling、Dense等）をパイプライン形式で組み合わせ
3. **losses/**: 20以上の損失関数（MSE、Triplet、Contrastive等）
4. **evaluation/**: 各種評価メトリクス
5. **cross_encoder/**: リランカー実装（Provenceアプローチとの統合予定）

### Provenceアプローチの実装状況

`@docs-text-pruner/provence/provence_paper.md`に基づく実装：

1. **目的**: RAGパイプラインでquery依存のcontext pruningを実現
2. **特徴**:
   - トークン/文レベルのバイナリ分類として定式化
   - Rerankerとの統合学習が可能
   - 動的なpruning比率の決定（0-100%）
   - hotchpotch/japanese-reranker-xsmall-v2ベースで実装

3. **実装済みコンポーネント**:
   - `sentence_transformers/provence/` - Provence実装
     - `encoder.py` - ProvenceEncoderクラス（トークンレベルプルーニング対応）
     - `losses_chunk_based.py` - チャンクベース損失関数
     - `data_collator_chunk_based.py` - 動的ラベル生成コレーター
     - `trainer.py` - ProvenceTrainer
     - `models/pruning_head.py` - プルーニングヘッド実装
   - `scripts/` - 学習・評価スクリプト群
     - `train_ja_minimal.py` - ja-minimalデータセット学習
     - `evaluate_ja_minimal.py` - 評価スクリプト
   - ja-minimalデータセットでの学習済み（閾値0.3で約30-60%圧縮）

### 参考ドキュメント

`@docs-text-pruner/`以下に関連ドキュメントが整理されています。実装時は適宜参照してください：

- **provence/**: Provence論文関連
  - `provence_paper.md`: Provence論文の詳細解説
  - `provence_blog.md`: Provence実装のブログ記事
- **sentence_transformer_blogs/**: Sentence Transformersの関連技術
  - `train-reranker.md`: Rerankerの学習方法
  - `train-sparse-encoder.md`: Sparse Encoderの学習方法
  - `static-embeddings.md`: Static Embeddingsの解説
- **specs/**: 実装仕様書（前述の通り）

### コーディング規約

- **Linting**: Ruffを使用（line-length: 119）
- **テスト**: 新機能には必ずテストを追加
- **ドキュメント**: docstringはGoogle/NumPyスタイル
- **命名規則**: 既存コードのパターンに従う
- **実装時の重要事項**: 新しい機能を実装する際は、必ずSentence Transformers内の類似実装を探して参考にする。特にhard negative学習、multiple texts処理、batch処理の最適化については既存の実装パターンを活用すること。

### 開発時の注意事項

1. **サンプルコード**: `tmp/`ディレクトリに作成し、`uv run python`で実行
2. **ログ出力**: `./log/`ディレクトリに保存
3. **既存コードの参照**: 新しいコンポーネント作成時は、類似の既存実装を参考に
4. **OSSプロジェクト**: コードスタイルや設計パターンは既存コードに合わせる

### text-prunerブランチの作業内容

1. Provenceペーパーの実装 ✅
2. Query-dependent text prunerの開発 ✅
3. Rerankerとの統合機能 ✅
4. 評価メトリクスの実装 ✅
5. サンプルコードとドキュメントの作成 ✅

#### 実装済み機能
- ProvenceEncoder: デュアルヘッド（ランキング＋プルーニング）アーキテクチャ
- トークンレベルプルーニング: チャンクベースの動的ラベル生成
- バッチ学習: Hard Negative学習による効率的な学習
- 教師蒸留: 既存リランカーからの知識蒸留
- 動的プルーニング: クエリに応じた適応的な圧縮（閾値により0-100%）

#### 最新の実装状況（2025年1月）
- チャンクベースのプルーニングラベル生成が完成
- トークンレベルでの正確なプルーニングが動作
- ja-minimalデータセットでの学習・評価が完了
- 閾値0.3で実用的な圧縮率（ポジティブ: 30-60%、ネガティブ: 80-90%）

### 仕様書管理

- **仕様書の場所**: `@docs-text-pruner/specs/`
- **主要仕様書**:
  - `spec.md`: 実装の最新仕様（更新時は必ず反映）
  - `provence-implementation-spec.md`: Provence実装の詳細設計
  - `data-format-spec.md`: データフォーマット仕様
- **更新ルール**: 実装やスペック変更時は必ず`git add`と`git commit`を実行

### 実装の差分確認

text-prunerブランチとmasterブランチの差分を確認することで、Provence実装の全体像を把握できます：

```bash
# 変更されたファイルのリスト
git diff --name-only master...text-pruner

# 詳細な差分
git diff master...text-pruner

# 新規追加ファイルのみ
git diff --name-only --diff-filter=A master...text-pruner
```

この差分を追跡することで：
- OSS本体の実装パターン
- 今回の拡張による変更点
- 新規追加されたProvence関連機能

を理解できます。