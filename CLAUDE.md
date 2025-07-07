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

### Provenceアプローチの実装方針

`@docs-text-pruner/provence_paper.md`に基づく実装：

1. **目的**: RAGパイプラインでquery依存のcontext pruningを実現
2. **特徴**:
   - トークン/文レベルのバイナリ分類として定式化
   - Rerankerとの統合学習が可能
   - 動的なpruning比率の決定（0-100%）
   - DeBERTa-v3ベースで高効率

3. **実装場所**:
   - `sentence_transformers/text_pruner/`に新規モジュール作成
   - 既存のcross_encoderインフラを活用
   - 新しい損失関数の追加が必要

### コーディング規約

- **Linting**: Ruffを使用（line-length: 119）
- **テスト**: 新機能には必ずテストを追加
- **ドキュメント**: docstringはGoogle/NumPyスタイル
- **命名規則**: 既存コードのパターンに従う

### 開発時の注意事項

1. **サンプルコード**: `tmp/`ディレクトリに作成し、`uv run python`で実行
2. **ログ出力**: `./log/`ディレクトリに保存
3. **既存コードの参照**: 新しいコンポーネント作成時は、類似の既存実装を参考に
4. **OSSプロジェクト**: コードスタイルや設計パターンは既存コードに合わせる

### text-prunerブランチの作業内容

1. Provenceペーパーの実装
2. Query-dependent text prunerの開発
3. Rerankerとの統合機能
4. 評価メトリクスの実装
5. サンプルコードとドキュメントの作成

### 仕様書管理

- **仕様書の場所**: `@docs-text-pruner/specs/`
- **主要仕様書**:
  - `spec.md`: 実装の最新仕様（更新時は必ず反映）
  - `provence-implementation-spec.md`: Provence実装の詳細設計
  - `data-format-spec.md`: データフォーマット仕様
- **更新ルール**: 実装やスペック変更時は必ず`git add`と`git commit`を実行