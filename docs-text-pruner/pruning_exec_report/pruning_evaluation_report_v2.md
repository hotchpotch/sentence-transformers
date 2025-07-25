# PruningEncoder性能評価レポート v2.1: 統一評価スクリプトによる最終検証

## 概要

本レポートは、最新の`scripts/pruning_exec.py`スクリプトを用いて、11個のPruningEncoderモデルを`pruning-config/pruning_data_ja.json`データセット（188クエリ、1396コンテキスト）で統一的に評価した最終結果です。混同行列の詳細表示とF2スコア最適化により、Query-dependent text pruning（クエリ依存テキストプルーニング）の実用性を総合的に検証しました。

### 評価日時
- 2025年7月17日（統一評価スクリプトによる最終検証）

### 評価環境
- **評価スクリプト**: 最新版 `scripts/pruning_exec.py`（混同行列・詳細メトリクス対応）
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
| **jpre-base-full-small-en** | **0.2** | **30.7%** | **0.8779** | 61.3% | 98.4% | **最高F2** |
| **jpre-base-full-small** | **0.1** | **33.7%** | **0.8739** | 59.3% | **99.1%** | **最高再現率** |
| **jpre-base-msmarco-small** | **0.3** | **26.9%** | **0.8739** | 63.9% | 96.2% | **バランス型** |
| jpre-base-full-fixed-v2 | 0.1 | 34.3% | 0.8688 | 58.7% | 98.7% | 高再現率 |
| ruri-re310m-msmarco-small | 0.4 | 24.9% | 0.8693 | 64.7% | 95.1% | バランス型 |
| ruri-re310m-full-all | 0.7 | 32.9% | 0.8620 | 71.8% | 90.7% | 高圧縮 |
| jpre-xs-msmarco-small | 0.4 | 18.8% | 0.8598 | 60.1% | 96.4% | 安定型 |
| jpre-base-full-all | 0.3 | 25.3% | 0.8506 | 64.4% | 92.5% | 高効率 |
| jpre-base-full-fixed | 0.4 | 23.1% | 0.8421 | 66.0% | 90.5% | 改良版 |
| jpre-xs-minimal | 0.4 | 19.4% | 0.8251 | 59.3% | 91.5% | 基本型 |
| jpre-xs-full-all | 0.4 | 21.2% | 0.8228 | 60.4% | 90.5% | 標準型 |

## 主要な発見と技術的洞察

### 1. 最新評価での上位3モデルの詳細分析

統一評価スクリプトでの最終検証により、以下の3モデルが傑出した性能を示しました：

#### jpre-base-full-small-en（最高F2スコア）
- **最適閾値**: 0.2
- **圧縮率**: 30.7%（実用的な高圧縮）
- **F2スコア**: 0.8779（全モデル中最高）
- **精度**: 61.3% / **再現率**: 98.4%（極めて高い）
- **混同行列**: TP=680, FP=429, TN=276, FN=11
- **特徴**: 英語データも含む学習により汎用性向上、最適なバランス

#### jpre-base-full-small（最高再現率）
- **最適閾値**: 0.1
- **圧縮率**: 33.7%（最も控えめな削除）
- **F2スコア**: 0.8739（第2位）
- **精度**: 59.3% / **再現率**: 99.1%（ほぼ完璧）
- **混同行列**: TP=685, FP=470, TN=235, FN=6
- **特徴**: 情報損失を極限まで抑制、重要情報の保持に最適

#### jpre-base-msmarco-small（同率2位バランス型）
- **最適閾値**: 0.3
- **圧縮率**: 26.9%（バランス良好）
- **F2スコア**: 0.8739（同率第2位）
- **精度**: 63.9% / **再現率**: 96.2%（高精度・高再現率）
- **混同行列**: TP=665, FP=376, TN=329, FN=26
- **特徴**: 精度と再現率の最適バランス、実用性が高い

### 2. 高精度型とバランス型モデルの性能

#### jpre-base-full-fixed-v2（高再現率特化）
- **最適閾値**: 0.1（低閾値での最適化）
- **圧縮率**: 34.3%（適度な圧縮）
- **F2スコア**: 0.8688（第4位）
- **精度**: 58.7% / **再現率**: 98.7%（極めて高い再現率）
- **混同行列**: TP=682, FP=479, TN=226, FN=9

#### ruri-re310m-msmarco-small（バランス型）
- **最適閾値**: 0.4
- **圧縮率**: 24.9%（効率的な圧縮）
- **F2スコア**: 0.8693（第5位）
- **精度**: 64.7% / **再現率**: 95.1%（バランス良好）
- **混同行列**: TP=657, FP=358, TN=347, FN=34

#### ruri-re310m-full-all（高精度圧縮型）
- **最適閾値**: 0.7（高閾値でも安定）
- **圧縮率**: 32.9%（高効率圧縮）
- **F2スコア**: 0.8620（第6位）
- **精度**: 71.8% / **再現率**: 90.7%（最高精度）
- **混同行列**: TP=627, FP=246, TN=459, FN=64
- **特徴**: 高精度による効率的な圧縮を実現

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
**推奨**: jpre-base-full-small-en（閾値0.2）またはjpre-base-full-small（閾値0.1）

#### jpre-base-full-small-en（最高F2スコア）
- F2スコア: 0.8779（最高）
- 再現率: 98.4%（極めて高い、情報損失リスク最小）
- 圧縮率: 30.7%（実用的な高効率）
- 精度: 61.3%（バランス良好）

#### jpre-base-full-small（最高再現率）
- F2スコア: 0.8739（第2位）
- 再現率: 99.1%（ほぼ完璧、最高の情報保持）
- 圧縮率: 33.7%（控えめだが実用的）
- 精度: 59.3%（安定）

```python
# 最高F2スコア（推奨）
config_best_f2 = {
    'model': 'jpre-base-full-small-en',
    'threshold': 0.2,
    'expected_compression': 0.307,
    'expected_f2_score': 0.8779,
    'expected_precision': 0.613,
    'expected_recall': 0.984
}

# 最高再現率（情報損失最小）
config_best_recall = {
    'model': 'jpre-base-full-small',
    'threshold': 0.1,
    'expected_compression': 0.337,
    'expected_f2_score': 0.8739,
    'expected_precision': 0.593,
    'expected_recall': 0.991
}

# バランス型（実用性重視）
config_balanced = {
    'model': 'jpre-base-msmarco-ja-small',
    'threshold': 0.3,
    'expected_compression': 0.269,
    'expected_f2_score': 0.8739,
    'expected_precision': 0.639,
    'expected_recall': 0.962
}
```

### シナリオ2: バランス重視（一般ビジネス）
**推奨**: jpre-base-msmarco-ja-small（閾値0.3）
- F2スコア: 0.8739（同率第2位）
- 圧縮率: 26.9%（適度な効率）
- 精度: 63.9% / 再現率: 96.2%（最適バランス）
- 用途: 汎用RAGアプリケーション

### シナリオ3: 高精度圧縮重視（大規模データ処理）
**推奨**: ruri-re310m-full-all（閾値0.7）
- 圧縮率: 32.9%（高効率）
- F2スコア: 0.8620（高圧縮でも高性能）
- 精度: 71.8%（最高精度）
- 用途: ストレージ削減、高速処理が必要な場合

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

最新の統一評価スクリプトによる最終検証により、PruningEncoderモデルの実用性が確認されました：

1. **jpre-base-full-small-en**が最高のF2スコア（0.8779）を達成
2. **jpre-base-full-small**が最高再現率（99.1%）を実現
3. **jpre-base-msmarco-small**が同率2位のF2スコア（0.8739）でバランス型として最適
4. 上位モデルは26-34%の実用的な圧縮率を実現
5. すべてのモデルが90%以上の高い再現率を維持
6. 混同行列により詳細な性能分析が可能となった

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
    -m output/jpre-base-full-small-en_20250716_202531/final_model \
    --thresholds 0.2

# 3. 最高再現率モデルの評価
python scripts/pruning_exec.py \
    -m output/jpre-base-full-small_20250716_164429/final_model \
    --thresholds 0.1

# 4. バランス型モデルの評価
python scripts/pruning_exec.py \
    -m output/jpre-base-msmarco-ja-small_20250715_095147/final_model \
    --thresholds 0.3

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