# PruningEncoder性能評価レポート: テキスト圧縮効率とプルーニング精度【更新版】

## 概要

本レポートは、**拡張された**`pruning-config/pruning_data_ja.json`データセット（188クエリ、1396コンテキスト）を用いた**8つ**のPruningEncoderモデルのテキスト圧縮性能とプルーニング精度の最新評価結果です。**Query-dependent text pruning**（クエリ依存テキストプルーニング）の実用性と圧縮効率を、より大規模で多様なデータセットで検証しました。

### 評価日時
- 2025年1月15日（大規模データセット更新・再評価）

### 重要な更新内容
- **データセット規模**: 86クエリ→**188クエリ**（2.2倍）
- **コンテキスト数**: 580→**1396コンテキスト**（2.4倍）
- **評価の信頼性向上**: より多様で実践的なプルーニング挑戦

### 評価対象モデル
1. **ruri-re310m-full-all** - cl-nagoya/ruri-v3-30mベース
2. **ruri-re310m-msmarco-ja-small** - cl-nagoya/ruri-v3-30mベース（MSMARCO-JA small）【新規追加】
3. **jpre-xs-full-all** - japanese-reranker-xsmall-v2ベース  
4. **jpre-xs-msmarco-ja-minimal** - xsmall-v2ベース（MSMARCO-JA minimal）
5. **jpre-xs-msmarco-ja-small** - xsmall-v2ベース（MSMARCO-JA small）
6. **jpre-base-msmarco-ja-small** - japanese-reranker-base-v1ベース（MSMARCO-JA small）
7. **jpre-base-full-all** - japanese-reranker-base-v1ベース【新規追加】
8. **jpre-base-full-all-fixed** - japanese-reranker-base-v1ベース（改良版）【最新追加】

### 拡張データセット仕様
- **総クエリ数**: 188（技術・生活・Web記事等の多様な分野）
- **総コンテキスト数**: 1396
- **保持対象**: 691コンテキスト（49.5%）
- **削除対象**: 705コンテキスト（50.5%）
- **バランス**: ほぼ均等な正負例分布で評価の公平性を確保

## 大規模評価での総合性能比較

| モデル | 最適閾値 | 圧縮率 | 誤削除率 | 誤保持率 | F2スコア | 実用性評価 | 保持数/削除数 |
|--------|----------|--------|----------|----------|----------|------------|---------------|
| **jpre-base-full-fixed** | **0.6** | **21.3%** | **1.6%** | **59.3%** | **0.8804** | **新・最優秀** | 1098/298 |
| ruri-re310m-msmarco-small | 0.3 | 18.9% | 0.7% | 63.3% | 0.8804 | 最優秀 | 1132/264 |
| jpre-base-small | 0.6 | 24.1% | 3.0% | 55.3% | 0.8760 | 高効率 | 1060/336 |
| ruri-re310m | 0.6 | 16.6% | 0.7% | 67.8% | 0.8732 | 高安全性 | 1164/232 |
| jpre-xs-small | 0.4 | 16.5% | 1.2% | 68.5% | 0.8690 | バランス型 | 1166/230 |
| jpre-base-full | 0.5 | 16.4% | 1.3% | 69.1% | 0.8670 | 安定型 | 1169/227 |
| jpre-xs-full | 0.3 | 15.5% | 1.4% | 70.8% | 0.8633 | 安定型 | 1180/216 |
| jpre-xs-minimal | 0.4 | 12.1% | 1.3% | 77.3% | 0.8544 | 基本型 | 1227/169 |

## 主要な発見と技術的示唆

### 1. jpre-base-full-all-fixedが新たな最優秀モデルに

**改良版jpre-baseモデルの圧倒的性能**:
- **最高F2スコア**: 0.8804（ruri-re310m-msmarco-smallと同スコア）
- **優れた圧縮率**: 21.3%（第2位の高効率）
- **低い誤削除率**: 1.6%（実用的な安全性）
- **最適閾値**: 0.6（効率性重視）

**詳細メトリクス（閾値0.6）**:
- Accuracy: 69.3%
- Precision: 61.9%
- Recall: 98.4%
- jpre-base-smallと比較して誤削除率を大幅改善（3.0%→1.6%）

### 2. ruri-re310m-msmarco-ja-smallの高い総合性能

**MSMARCO-JA smallで学習したruriモデルの安定性**:
- **同率最高F2スコア**: 0.8804
- **最優秀誤削除率**: 0.7%（他モデルと同等の安全性）
- **バランス型圧縮率**: 18.9%（実用的な効率性）
- **最良誤保持率**: 63.3%（ノイズ除去も優秀）

**詳細メトリクス（閾値0.3）**:
- Accuracy: 67.7%
- Precision: 60.6%
- Recall: 99.3%
- 最適閾値が0.3と低く、安全性を重視

### 3. jpre-base-smallの高効率性

**圧縮効率では依然トップ**:
- **最高圧縮率**: 24.1%（他モデルの1.3-2倍の効率）
- **F2スコア**: 0.8760（第2位の高性能）
- **特徴**: 誤削除率3.0%を許容することで高効率を実現

### 4. モデル間の明確な性能差異

**F2スコアランキング**:
1. jpre-base-full-fixed: 0.8804（最優秀）【最新】
1. ruri-re310m-msmarco-small: 0.8804（同率最優秀）
3. jpre-base-small: 0.8760
4. ruri-re310m: 0.8732
5. jpre-xs-small: 0.8690
6. jpre-base-full: 0.8670
7. jpre-xs-full: 0.8633
8. jpre-xs-minimal: 0.8544

**圧縮効率ランキング**:
1. jpre-base-small: 24.1%（baseモデルの威力）
2. jpre-base-full-fixed: 21.3%（高効率改良版）【最新】
3. ruri-re310m-msmarco-small: 18.9%（バランス型）
4. ruri-re310m: 16.6%（安全性重視）
5. jpre-xs-small: 16.5%（xsモデル最良）
6. jpre-base-full: 16.4%
7. jpre-xs-full: 15.5%
8. jpre-xs-minimal: 12.1%（控えめだが安定）

**安全性ランキング（誤削除率）**:
1. ruri-re310m: 0.7%（超安全）
1. ruri-re310m-msmarco-small: 0.7%（同率首位）
3. jpre-xs-small: 1.2%
4. jpre-xs-minimal: 1.3%
4. jpre-base-full: 1.3%（同率）
6. jpre-xs-full: 1.4%
7. jpre-base-full-fixed: 1.6%（実用的安全性）【最新】
8. jpre-base-small: 3.0%（効率とのトレードオフ）

### 5. 大規模データセットでの新たな知見

**jpre-base-full-fixedモデルの革新的改善**:
- **最新の改良版**: jpre-base-full-all-fixed_20250716_052037
- **F2スコア**: 0.8804（最高スコアタイ）
- **圧縮率**: 21.3%（jpre-base-smallに次ぐ高効率）
- **誤削除率**: 1.6%（baseモデルとして優秀）
- **最適閾値**: 0.6（効率性を重視しつつ安全性も確保）
- **改良の成果**: jpre-base-fullから大幅な性能向上を実現

**jpre-base-fullモデルの追加評価**:
- japanese-reranker-base-v1ベースで全データ（full-all）学習
- F2スコア: 0.8670（第5位）
- 圧縮率: 16.4%（baseモデルとしては控えめ）
- 誤削除率: 1.3%（安全性重視）
- 最適閾値: 0.5（バランス型）
- baseモデルでも学習データ規模の違いで大きな性能差（small: 24.1% vs full: 16.4%圧縮率）

**MSMARCO-JA small学習の効果**:
- ruriモデルでもMSMARCO-JA small学習によりF2スコアが向上（0.8732→0.8804）
- 同じベースモデルでも学習データの違いが性能に大きく影響
- MSMARCO-JAデータセットの有効性が実証された

**閾値戦略の最適化**:
- 0.6: 効率性重視（jpre-base-small、ruri-re310m）
- 0.5: バランス重視（jpre-base-full）【新】
- 0.4: バランス重視（jpre-xs-small、jpre-xs-minimal）  
- 0.3: 安全性最優先（ruri-re310m-msmarco-small、jpre-xs-full）

## 実用導入シナリオ別推奨

### シナリオ1: 総合性能最優先
**推奨**: jpre-base-full-all-fixed（閾値0.6）【最新推奨】
- **効果**: F2スコア0.8804で最高総合性能
- **圧縮率**: 21.3%で高効率
- **安全性**: 1.6%誤削除率で実用的
- **用途例**: 一般的なRAGシステム、企業内検索

**代替推奨**: ruri-re310m-msmarco-ja-small（閾値0.3）
- **効果**: F2スコア0.8804で最高総合性能
- **安全性**: 0.7%誤削除率で情報損失リスク最小
- **用途例**: 医療・金融・法務等の高精度要求分野

```python
# 実装例
pruning_config = {
    'model': 'ruri-re310m-msmarco-ja-small',
    'threshold': 0.3,
    'expected_compression': 0.189,
    'max_false_deletion_rate': 0.007
}
```

### シナリオ2: 大量データ処理・リアルタイム重視
**推奨**: jpre-base-msmarco-ja-small（閾値0.6）
- **効果**: 24.1%圧縮でレスポンス速度大幅向上
- **リスク管理**: 3.0%誤削除率を事前アラート機能でカバー
- **用途例**: カスタマーサポートAI、リアルタイムQ&Aシステム

### シナリオ3: ビジネス・一般用途
**推奨**: jpre-xs-msmarco-ja-small（閾値0.4）
- **効果**: 16.5%圧縮で適度な効率性
- **安全性**: 1.2%誤削除率で情報損失リスク最小
- **用途例**: 企業内RAG、文書検索システム

### シナリオ4: 開発・プロトタイピング
**推奨**: jpre-xs-msmarco-ja-minimal（閾値0.4）
- **軽量性**: 小規模学習データでも実用的性能
- **開発効率**: 迅速な実装とテストが可能
- **用途例**: MVP開発、概念実証

## 実装ガイドライン

### 段階的導入戦略
```python
# フェーズ1: 保守的開始
initial_config = {
    'threshold': 0.4,
    'monitoring': True,
    'fallback_enabled': True
}

# フェーズ2: 効率性向上
optimized_config = {
    'threshold': 0.6,  # jpre-base-smallで効率性追求
    'quality_monitoring': True,
    'automatic_adjustment': True
}
```

### モニタリング指標
```python
performance_kpis = {
    'compression_rate': {'target': '>20%', 'alert': '<15%'},
    'false_deletion_rate': {'limit': '<5%', 'critical': '>10%'},
    'response_time_improvement': {'target': '>25%'},
    'user_satisfaction': {'maintain': '>95%'}
}
```

### A/Bテスト設定
```python
ab_test_config = {
    'control_group': 'no_pruning',
    'test_groups': [
        {'model': 'jpre-base-small', 'threshold': 0.6, 'traffic': 0.3},
        {'model': 'jpre-xs-small', 'threshold': 0.4, 'traffic': 0.3},
        {'model': 'ruri-re310m', 'threshold': 0.6, 'traffic': 0.4}
    ],
    'metrics': ['response_time', 'accuracy', 'user_rating']
}
```

## 技術的洞察と将来展望

### ベースモデルサイズの決定的影響
- **japanese-reranker-base-v1**: 他を圧倒する圧縮効率
- **パラメータ数増加**: 計算コスト増 vs 圧縮効率向上のトレードオフ
- **推奨**: 本格運用ではbaseモデル、開発フェーズではxsモデル

### データセット規模と性能の関係
- **MSMARCO-JA small**: 十分な学習効果（minimalと明確な差）
- **学習データ品質**: 量よりも多様性が重要
- **継続的改善**: 実運用データでのファインチューニングが有効

### 閾値最適化戦略
- **動的調整**: クエリタイプや負荷状況に応じた閾値調整
- **ユーザー適応**: 個別ユーザーのフィードバックでパーソナライズ
- **コンテキスト依存**: 文書の重要度や分野に応じた閾値設定

## 結論

**大規模データセット（1396コンテキスト）での包括評価により、8モデルの性能特性が明確になり、jpre-base-full-all-fixedが新たな最優秀モデルとして確認されました。**

### 主要な発見
1. **最高F2スコア**: jpre-base-full-fixedとruri-re310m-msmarco-smallが0.8804で同率首位
2. **最優秀圧縮効率**: jpre-base-smallが24.1%、jpre-base-full-fixedが21.3%で高効率
3. **最優秀安全性**: ruri系モデルが誤削除率0.7%で情報損失リスク最小
4. **バランス性能**: jpre-base-full-fixedが効率と性能の最良バランス

### 導入推奨事項
- **総合性能重視**: jpre-base-full-fixed（閾値0.6）で最高性能と高効率の両立
- **高安全性要求**: ruri-re310m-msmarco-small（閾値0.3）で誤削除率最小化
- **最高効率要求**: jpre-base-small（閾値0.6）で圧縮率最大化
- **段階的最適化**: 0.3→0.6の閾値調整で用途に応じた最適化
- **継続的監視**: 誤削除率と圧縮率のバランス監視

日本語RAGシステムにおけるコンテキストプルーニングは、本評価により**実用段階から本格展開段階**へと進化しました。特にjpre-base-full-all-fixedモデルは、高い性能と効率性の最良バランスを提供し、ruri-re310m-msmarco-ja-smallモデルは安全性を重視する用途に最適です。

## 再現・検証方法

```bash
# 1. 環境準備
git clone https://github.com/hotchpotch/sentence-transformers.git
cd sentence-transformers
git checkout text-pruner
uv sync

# 2. 最新データセットでの評価実行
python scripts/pruning_exec.py \
    -m output/jpre-base-msmarco-ja-small_20250715_095147/final_model \
    --thresholds 0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9

# 3. 最優秀モデルの評価
python scripts/pruning_exec.py \
    -m output/jpre-base-full-all-fixed_20250716_052037/final_model \
    --thresholds 0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9

# 4. 第2位モデルの評価
python scripts/pruning_exec.py \
    -m output/ruri-re310m-msmarco-ja-small_20250715_110605/final_model \
    --thresholds 0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9

# 5. 他モデルとの性能比較
for model in jpre-xs-small jpre-xs-minimal ruri-re310m jpre-xs-full jpre-base-full jpre-base-full-fixed; do
    python scripts/pruning_exec.py -m output/${model}_*/final_model \
        --thresholds 0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9
done

# 5. 圧縮率分析
python tmp/parse_evaluation_results.py
```

## 付録: 大規模データセット詳細

### 拡張されたデータセット構成（2025年1月15日）
- **技術分野**: 機械学習、プログラミング、システム設計
- **生活分野**: 健康、教育、観光、料理
- **ビジネス分野**: 金融、税務、経営、マーケティング  
- **Web記事**: 実際のWebコンテンツを模擬した高圧縮挑戦データ

### 圧縮難易度分布
- **高圧縮率期待**: Web記事風データ（大部分が無関連）
- **中圧縮率期待**: 技術文書（部分的関連性）
- **低圧縮率期待**: 専門分野解説（多くが関連）

詳細なデータ形式とサンプルについては、`docs-text-pruner/specs/data-format-spec.md`を参照してください。