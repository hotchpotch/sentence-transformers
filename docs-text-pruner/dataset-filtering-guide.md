# Dataset Filtering Guide for Pruning Training

このドキュメントでは、`pruning_train.py`で利用可能なデータセットフィルタリング機能について説明します。

## 概要

Pruning学習において、データセットの品質は非常に重要です。新しいフィルタリング機能により、以下が可能になりました：

1. **全ゼロrelevanceアイテムの除外**: context_spans_relevanceが全て0のアイテムを自動的に除外
2. **スマートな選択**: relevance平均値に基づいて最適なアイテムを選択
3. **柔軟な設定**: 通常/逆順ソート、先頭アイテムの保護など

## 設定パラメータ

### filter_zero_relevance_max_items
- **型**: int または None
- **デフォルト**: None（フィルタリング無効）
- **説明**: 各行で保持する最大アイテム数。設定すると自動的にフィルタリングが有効になります。

### filter_zero_relevance_max_items_reverse
- **型**: bool
- **デフォルト**: False
- **説明**: 
  - `False`（デフォルト）: relevance平均値が高い順にソート（高品質データを保持）
  - `True`: relevance平均値が低い順にソート（難しいデータを保持）

### filter_keep_first_item
- **型**: bool
- **デフォルト**: False
- **説明**: 
  - `False`（デフォルト）: 通常のソート基準で選択
  - `True`: 先頭アイテムを必ず保持（重要なデータの保護）

## 使用例

### 1. 基本的な使用（高relevanceアイテムを4つ選択）

```yaml
data_args:
  dataset_name: "hotchpotch/wip-msmarco-context-relevance"
  subset: "msmarco-ja-small"
  filter_zero_relevance_max_items: 4
```

効果：
- 各行から全ゼロのアイテムを除外
- relevance平均値が高い順に4つを選択
- 4つ未満しか残らない行は削除

### 2. 逆順ソート（低relevanceアイテムを選択）

```yaml
data_args:
  dataset_name: "hotchpotch/wip-msmarco-context-relevance"
  subset: "msmarco-ja-small"
  filter_zero_relevance_max_items: 4
  filter_zero_relevance_max_items_reverse: true
```

効果：
- relevance平均値が低い順に4つを選択
- 難しいデータでの学習に有効

### 3. 先頭アイテム保護

```yaml
data_args:
  dataset_name: "hotchpotch/wip-msmarco-context-relevance"
  subset: "msmarco-ja-small"
  filter_zero_relevance_max_items: 4
  filter_keep_first_item: true
```

効果：
- 先頭アイテムを必ず保持
- 残り3つはrelevance平均値順で選択
- Positive例が先頭にある場合に有効

### 4. 組み合わせ使用

```yaml
data_args:
  dataset_name: "hotchpotch/wip-msmarco-context-relevance"
  subset: "msmarco-ja-small"
  filter_zero_relevance_max_items: 4
  filter_zero_relevance_max_items_reverse: true
  filter_keep_first_item: true
```

効果：
- 先頭アイテム + 低relevance 3つ
- バランスの取れたデータセット構築

## フィルタリング例

元データ（8アイテム）:
```
[0] [1, 0] -> avg: 0.50 (positive)
[1] [1, 1] -> avg: 1.00
[2] [1, 0] -> avg: 0.50
[3] [1] -> avg: 1.00
[4] [1, 1, 0, 0] -> avg: 0.50
[5] [0, 0, 1, 0] -> avg: 0.25
[6] [1, 0, 0, 0] -> avg: 0.25
[7] [1, 0, 0, 0] -> avg: 0.25
```

### 通常フィルタリング（max_items=4）
結果: [0], [1], [2], [3]
- 高relevance順に選択

### 逆順フィルタリング（max_items=4, reverse=true）
結果: [5], [6], [7], [0]
- 低relevance順に選択

### 先頭保持 + 通常（max_items=4, keep_first=true）
結果: [0], [1], [3], [4]
- [0]を保持 + 高relevance 3つ

### 先頭保持 + 逆順（max_items=4, reverse=true, keep_first=true）
結果: [0], [5], [6], [7]
- [0]を保持 + 低relevance 3つ

## 実行時の表示

フィルタリング実行時、以下のような統計情報が表示されます：

```
Applying filtering to hotchpotch/wip-msmarco-context-relevance:msmarco-ja-small train set (removing zero-relevance items, max_items=4)
→ hotchpotch/wip-msmarco-context-relevance:msmarco-ja-small train: 98,000 → 86,709 samples (88.5% retained)
```

警告メッセージ：
```
⚠️  filter_zero_relevance_max_items_reverse is enabled: Keeping items with LOWER relevance scores (reverse sort)
⚠️  filter_keep_first_item is enabled: Always keeping the first item regardless of relevance
```

## 推奨設定

1. **標準的な学習**: `filter_zero_relevance_max_items: 4`
2. **Hard negative重視**: `filter_zero_relevance_max_items_reverse: true`
3. **Positive例保護**: `filter_keep_first_item: true`
4. **バランス重視**: すべてのオプションを組み合わせる

## 注意事項

- フィルタリング後のデータセットサイズに注意（学習に十分なデータが残っているか確認）
- `max_items`の値は元のデータセットの構造に応じて調整
- 評価データセットにも同じフィルタリングが適用される