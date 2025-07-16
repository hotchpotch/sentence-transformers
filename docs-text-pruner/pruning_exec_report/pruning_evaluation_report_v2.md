# PruningEncoder性能評価レポート v2.0: 修正版アルゴリズムでの包括的評価

## 概要

本レポートは、修正されたpruning評価アルゴリズム（softmax適用による確率正規化）を用いて、11個のPruningEncoderモデルを`pruning-config/pruning_data_ja.json`データセット（188クエリ、1396コンテキスト）で再評価した結果です。Query-dependent text pruning（クエリ依存テキストプルーニング）の実用性と効率性を最新のアルゴリズムで検証しました。

### 評価日時
- 2025年1月16日（修正版アルゴリズムによる再評価）

### 評価環境
- **評価スクリプト**: 修正版 `scripts/check_pruning.py`（softmax正規化実装）
- **データセット**: `pruning-config/pruning_data_ja.json`
  - 総クエリ数: 188
  - 総コンテキスト数: 1396
  - 保持対象: 691コンテキスト（49.5%）
  - 削除対象: 705コンテキスト（50.5%）

### 評価対象モデル
1. **ruri-re310m-full-all** - cl-nagoya/ruri-v3-30mベース（全データ学習）
2. **ruri-re310m-msmarco-ja-small** - cl-nagoya/ruri-v3-30mベース（MSMARCO-JA small）
3. **jpre-xs-full-all** - japanese-reranker-xsmall-v2ベース（全データ学習）
4. **jpre-xs-msmarco-ja-minimal** - xsmall-v2ベース（MSMARCO-JA minimal）
5. **jpre-xs-msmarco-ja-small** - xsmall-v2ベース（MSMARCO-JA small）
6. **jpre-base-msmarco-ja-small** - japanese-reranker-base-v1ベース（MSMARCO-JA small）
7. **jpre-base-full-all** - japanese-reranker-base-v1ベース（全データ学習）
8. **jpre-base-full-all-fixed** - japanese-reranker-base-v1ベース（改良版）
9. **jpre-base-full-all-fixed-v2** - japanese-reranker-base-v1ベース（最新改良版）
10. **jpre-base-full-small** - japanese-reranker-base-v1ベース（smallデータセット）
11. **jpre-base-full-small-en** - japanese-reranker-base-v1ベース（英語+日本語混合）

## 総合性能比較表

| モデル | 最適閾値 | 圧縮率 | F2スコア | 精度 | 再現率 | 実用性評価 |
|--------|----------|--------|----------|------|--------|------------|
| **jpre-base-full-small-en** | **0.2** | **20.8%** | **0.8783** | 61.4% | 98.4% | **最高F2** |
| jpre-base-full-small | 0.1 | 17.3% | 0.8758 | 60.0% | 99.0% | 最高再現率 |
| jpre-base-msmarco-small | 0.3 | 24.9% | 0.8749 | 63.7% | 96.5% | バランス型 |
| jpre-base-full-fixed-v2 | 0.1 | 18.0% | 0.8723 | 59.6% | 98.7% | 高再現率 |
| ruri-re310m-msmarco-small | 0.4 | 26.9% | 0.8673 | 64.4% | 94.9% | バランス型 |
| ruri-re310m | 0.7 | 37.3% | 0.8601 | 71.5% | 90.6% | 高圧縮 |
| jpre-xs-msmarco-small | 0.4 | 20.5% | 0.8570 | 59.8% | 96.1% | 安定型 |
| jpre-base-full-all | 0.3 | 27.7% | 0.8495 | 63.5% | 92.8% | 高効率 |
| jpre-base-full-fixed | 0.4 | 31.5% | 0.8416 | 65.6% | 90.6% | 改良版 |
| jpre-xs-full-all | 0.4 | 26.4% | 0.8215 | 60.6% | 90.2% | 標準型 |
| jpre-xs-minimal | 0.4 | 24.4% | 0.8194 | 59.3% | 90.6% | 基本型 |

## 主要な発見と技術的洞察

### 1. jpre-base-full-small-enが新たな最高F2スコアを達成

修正されたアルゴリズムでの評価により、3つのモデルが傑出した性能を示しました：

#### jpre-base-full-small-en（最高F2スコア）
- **最適閾値**: 0.2
- **圧縮率**: 20.8%（実用的）
- **F2スコア**: 0.8783（全モデル中最高）
- **再現率**: 98.4%（極めて高い）
- **特徴**: 英語データも含む学習により汎用性向上

#### jpre-base-full-small（超高再現率）
- **最適閾値**: 0.1
- **圧縮率**: 17.3%（最も控えめ）
- **F2スコア**: 0.8758（第3位）
- **再現率**: 99.0%（ほぼ完璧）
- **特徴**: 情報損失を極限まで抑制

#### jpre-base-msmarco-ja-small（バランス型）
- **最適閾値**: 0.3（低閾値での高性能）
- **圧縮率**: 24.9%（実用的な範囲）
- **再現率**: 96.5%（重要情報の保持に優れる）
- **F2スコア**: 0.8749（第4位）
- **特徴**: 低い閾値で安定した性能を発揮

#### jpre-base-full-all-fixed-v2（最高再現率）
- **最適閾値**: 0.1（最も低い閾値）
- **圧縮率**: 18.0%（控えめだが実用的）
- **F2スコア**: 0.8723（第5位）
- **再現率**: 98.7%（全モデル中最高）
- **特徴**: 重要情報の見逃しを最小限に抑える

### 2. ruri-re310m-full-allの高圧縮性能

**ruri-re310m-full-all**は最も高い圧縮率を実現：

- **最適閾値**: 0.7（高閾値でも安定）
- **圧縮率**: 37.3%（全モデル中最高）
- **F2スコア**: 0.8601（高圧縮でも高性能維持）
- **特徴**: 積極的な圧縮が必要な場合に最適

### 3. モデルアーキテクチャによる性能差

#### ベースモデルサイズの影響
- **jpre-base系**: 高いF2スコアと効率的な圧縮を実現
- **jpre-xs系**: やや保守的だが安定した性能
- **ruri系**: 高閾値でも性能を維持する堅牢性

#### 学習データセットの影響
- **MSMARCO-JA small**: 最もバランスの取れた性能
- **full-all**: 高圧縮率を実現
- **minimal**: 最も保守的で安定

### 4. 閾値戦略の最適化

モデルタイプ別の最適閾値パターン：
- **jpre-base系**: 0.3-0.4（低閾値で最適）
- **jpre-xs系**: 0.4（中間閾値で安定）
- **ruri系**: 0.4-0.7（幅広い閾値で対応可能）

## 実用導入シナリオ別推奨

### シナリオ1: 最高精度重視（医療・金融・法務）
**推奨**: jpre-base-full-small（閾値0.1）またはjpre-base-full-small-en（閾値0.2）

#### jpre-base-full-small（最高再現率）
- F2スコア: 0.8758（第3位）
- 再現率: 99.0%（ほぼ完璧、情報損失リスク最小）
- 圧縮率: 17.3%（最も控えめ）

#### jpre-base-full-small-en（最高F2スコア）
- F2スコア: 0.8783（最高）
- 再現率: 98.4%（極めて高い）
- 圧縮率: 20.8%（実用的）

```python
config = {
    'model': 'jpre-base-msmarco-ja-small',
    'threshold': 0.3,
    'expected_compression': 0.249,
    'expected_f2_score': 0.8749
}

# 最高F2スコアの場合
config_best_f2 = {
    'model': 'jpre-base-full-small-en',
    'threshold': 0.2,
    'expected_compression': 0.208,
    'expected_f2_score': 0.8783,
    'expected_recall': 0.984
}

# 最高再現率の場合
config_best_recall = {
    'model': 'jpre-base-full-small',
    'threshold': 0.1,
    'expected_compression': 0.173,
    'expected_f2_score': 0.8758,
    'expected_recall': 0.990
}
```

### シナリオ2: 高圧縮重視（大規模データ処理）
**推奨**: ruri-re310m-full-all（閾値0.7）
- 圧縮率: 37.3%（最高）
- F2スコア: 0.8601（高圧縮でも高性能）
- 用途: ストレージ削減、高速処理が必要な場合

### シナリオ3: バランス重視（一般ビジネス）
**推奨**: ruri-re310m-msmarco-ja-small（閾値0.4）
- F2スコア: 0.8673（第2位）
- 圧縮率: 26.9%（適度な効率）
- 安定性: 幅広い用途に対応

### シナリオ4: 軽量・安定重視（組み込みシステム）
**推奨**: jpre-xs-msmarco-ja-small（閾値0.4）
- モデルサイズ: xsmall（軽量）
- F2スコア: 0.8570
- 圧縮率: 20.5%（控えめだが安定）

## 技術的推奨事項

### 1. 実装時の注意点
- **確率正規化**: softmaxを適用して適切な確率分布を確保
- **閾値調整**: モデルタイプに応じた最適閾値の使用
- **評価指標**: F2スコアを重視（再現率重視の評価）

### 2. システム統合ガイドライン
```python
# 推奨実装パターン
def apply_pruning(model, texts, queries, model_type):
    if model_type == "jpre-base":
        threshold = 0.3
    elif model_type == "jpre-xs":
        threshold = 0.4
    elif model_type == "ruri":
        threshold = 0.4  # または0.7（圧縮率重視の場合）
    
    # softmax正規化を含む推論
    outputs = model(texts, queries)
    probs = torch.softmax(outputs.pruning_logits, dim=-1)
    keep_mask = probs[:, :, 1] > threshold
    
    return keep_mask
```

### 3. モニタリング指標
- **圧縮率**: 20-40%の範囲を目標
- **F2スコア**: 0.82以上を維持
- **再現率**: 90%以上を確保
- **処理速度**: 圧縮による高速化を測定

## 結論

修正されたアルゴリズムによる再評価により、PruningEncoderモデルの真の性能が明らかになりました：

1. **jpre-base-full-small-en**が最高のF2スコア（0.8783）を達成
2. **jpre-base-full-small**がほぼ完璧な再現率（99.0%）を実現
3. **jpre-base-full-all-fixed-v2**も極めて高い再現率（98.7%）を維持
4. **ruri-re310m-full-all**が最高圧縮率（37.3%）を実現
5. すべてのモデルが90%以上の高い再現率を維持
6. モデルタイプごとに明確な最適閾値パターンが存在

日本語RAGシステムにおけるquery-dependent text pruningは、適切なモデルと閾値の選択により、情報の品質を保ちながら20-40%のテキスト圧縮を実現できることが実証されました。

## 再現方法

```bash
# 1. 環境準備
git clone https://github.com/hotchpotch/sentence-transformers.git
cd sentence-transformers
git checkout text-pruner
uv sync

# 2. 最高F2スコアモデルの評価
python scripts/pruning_exec.py \
    -m output/jpre-base-msmarco-ja-small_20250715_095147/final_model \
    --thresholds 0.3

# 3. 最高圧縮率モデルの評価
python scripts/pruning_exec.py \
    -m output/ruri-re310m-full-all_20250713_143249/final_model \
    --thresholds 0.7

# 4. 全モデルの包括的評価
for model in jpre-base-msmarco-ja-small jpre-base-full-all-fixed-v2 \
             ruri-re310m-msmarco-ja-small ruri-re310m-full-all \
             jpre-xs-msmarco-ja-small jpre-base-full-all \
             jpre-base-full-all-fixed jpre-xs-full-all \
             jpre-xs-msmarco-ja-minimal jpre-base-full-small \
             jpre-base-full-small-en; do
    echo "Evaluating $model..."
    # 各モデルの最適閾値で評価
done
```

## 付録: アルゴリズム修正の詳細

### 修正前の問題点
- pruning確率が適切に正規化されていなかった
- バイナリ分類の確率分布が不適切

### 修正内容
```python
# 修正後のコード（scripts/check_pruning.py）
pruning_logits = outputs["pruning_logits"]
# softmaxで適切な確率分布に変換
pruning_probs = torch.softmax(pruning_logits, dim=-1).cpu().numpy()
# "keep"クラスの確率を使用（index 1）
if pruning_probs.ndim == 3 and pruning_probs.shape[-1] == 2:
    pruning_probs = pruning_probs[0, :, 1]
```

この修正により、すべてのモデルで一貫性のある評価が可能となりました。