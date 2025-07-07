# Query-Context Prunerデータセット（教師スコア付き）

特に指示がなければ、日本語のデータセット(ja-full, ja-small, ja-minimal)を使うこと。

## 📋 概要

このデータセットは、Query-Context Pruningタスク用の高品質な学習データセットです。japanese-reranker-xsmall-v2による教師スコアとvLLMモデルによる関連チャンクアノテーションが付与されており、効率的なチャンクプルーニングモデルの学習に最適化されています。

**リポジトリ**: `hotchpotch/wip-query-context-pruner-with-teacher-scores`（プライベート）

## 🌟 データセットラインナップ

### 🌍 **全データセット版**（多言語・多データセット統合）
**用途**: 包括的な研究・評価、マルチリンガル対応

```python
from datasets import load_dataset

# フル版（129万件）- 研究・評価用
full_dataset = load_dataset('hotchpotch/wip-query-context-pruner-with-teacher-scores', 'full')

# スモール版（10万件）- 開発・実験用
small_dataset = load_dataset('hotchpotch/wip-query-context-pruner-with-teacher-scores', 'small')

# ミニマル版（1万件）- 高速プロトタイピング用
minimal_dataset = load_dataset('hotchpotch/wip-query-context-pruner-with-teacher-scores', 'minimal')
```

**特徴**:
- **11データセット統合**: MS-MARCO（英語・日本語）、HPPRC 8種、MIRACL 18言語
- **多言語対応**: 20言語（日本語、英語、アラビア語、中国語など）
- **バランス分割**: dataset_name別に適切な比率でtrain/validation/test分割
- **教師スコア**: POS平均 0.78-0.82、NEG平均 0.15-0.19

---

### 🇺🇸 **MS-MARCO英語版**（英語特化）
**用途**: 英語専用モデル学習、MS-MARCOベンチマーク対応

```python
# フル版（50万件）- 英語モデル学習用
ms_marco_full = load_dataset('hotchpotch/wip-query-context-pruner-with-teacher-scores', 'ms-marco-full')

# スモール版（10万件）- 英語開発用
ms_marco_small = load_dataset('hotchpotch/wip-query-context-pruner-with-teacher-scores', 'ms-marco-small')

# ミニマル版（1万件）- 英語プロトタイピング用
ms_marco_minimal = load_dataset('hotchpotch/wip-query-context-pruner-with-teacher-scores', 'ms-marco-minimal')
```

**特徴**:
- **純粋英語**: MS-MARCO英語データのみ（dataset_name='ms-marco'）
- **高品質**: 教師スコア POS平均 0.73-0.80、NEG平均 0.12-0.18
- **標準ベンチマーク**: MS-MARCOデータセットの標準形式
- **効率的チャンキング**: NLTK sentence tokenizationによる最適化

---

### 🇯🇵 **日本語版**（日本語特化、MS-MARCO除外）
**用途**: 日本語専用モデル学習、日本語NLP研究

```python
# フル版（50万件）- 日本語モデル学習用
ja_full = load_dataset('hotchpotch/wip-query-context-pruner-with-teacher-scores', 'ja-full')

# スモール版（10万件）- 日本語開発用  
ja_small = load_dataset('hotchpotch/wip-query-context-pruner-with-teacher-scores', 'ja-small')

# ミニマル版（1万件）- 日本語プロトタイピング用
ja_minimal = load_dataset('hotchpotch/wip-query-context-pruner-with-teacher-scores', 'ja-minimal')
```

**特徴**:
- **純粋日本語**: ms-marco-ja、miracl-ja、HPPRC日本語系データセット
- **最高品質**: 教師スコア POS平均 0.74-0.98（全版中最高）
- **日本語最適化**: 日本語文分割（bunkai）による高精度チャンキング
- **包括的データ**: MS-MARCO日本語版 + MIRACL日本語 + HPPRC日本語系

---

## 📊 **データ構造**

すべてのsubsetで統一されたスキーマ：

```python
{
    'id': str,                                           # ユニークID
    'query': str,                                        # 検索クエリ
    'texts': List[str],                                  # 5つの候補テキスト
    'chunks_pos': List[List[List[int]]],                 # チャンク位置情報
    'labels': List[int],                                 # POS/NEGラベル [1,0,0,0,0]
    'dataset_name': str,                                 # 元データセット名
    'relevant_chunks': List[List[int]],                  # 関連チャンクインデックス（vLLM判定）
    'teacher_scores_japanese-reranker-xsmall-v2': List[float]  # 教師スコア
}
```

### フィールド詳細

- **`id`**: データセット横断でユニークな識別子
- **`query`**: ユーザーの検索クエリ（日本語または英語）
- **`texts`**: 5つの候補テキスト（1つのPOS + 4つのNEG）
- **`chunks_pos`**: 各テキストのチャンク位置情報 `[start, end]`
- **`labels`**: POSとNEGのラベル（先頭が必ずPOS=1）
- **`dataset_name`**: 元データセット名（ms-marco、ms-marco-ja、miracl-ja等）
- **`relevant_chunks`**: vLLMモデルが判定した関連チャンクの0ベースインデックス
- **`teacher_scores_*`**: japanese-reranker-xsmall-v2による教師スコア

---

## 🚀 **クイックスタート**

### 基本的な使用方法

```python
from datasets import load_dataset

# 1. 基本ロード
dataset = load_dataset('hotchpotch/wip-query-context-pruner-with-teacher-scores', 'small')

# 2. 訓練データ取得
train_data = dataset['train']
print(f"訓練データ数: {len(train_data):,}件")

# 3. サンプル確認
sample = train_data[0]
print(f"クエリ: {sample['query']}")
print(f"教師スコア: {sample['teacher_scores_japanese-reranker-xsmall-v2']}")
print(f"関連チャンク: {sample['relevant_chunks']}")
print(f"データセット: {sample['dataset_name']}")
```

### 実践的な使用例

```python
# チャンク関連性学習の例
def extract_training_pairs(sample):
    query = sample['query']
    texts = sample['texts']
    chunks_pos = sample['chunks_pos']
    relevant_chunks = sample['relevant_chunks']
    teacher_scores = sample['teacher_scores_japanese-reranker-xsmall-v2']
    
    training_pairs = []
    for i, (text, text_chunks_pos, text_relevant_chunks, teacher_score) in enumerate(
        zip(texts, chunks_pos, relevant_chunks, teacher_scores)
    ):
        # チャンクを抽出
        chunks = []
        for start, end in text_chunks_pos:
            chunk = text[start:end].strip()
            chunks.append(chunk)
        
        # 学習ペアを作成
        training_pairs.append({
            'query': query,
            'text': text,
            'chunks': chunks,
            'relevant_chunks': text_relevant_chunks,
            'teacher_score': teacher_score,
            'label': sample['labels'][i]
        })
    
    return training_pairs

# データセット処理
dataset = load_dataset('hotchpotch/wip-query-context-pruner-with-teacher-scores', 'small')
train_data = dataset['train']

all_training_pairs = []
for sample in train_data:
    pairs = extract_training_pairs(sample)
    all_training_pairs.extend(pairs)

print(f"学習ペア総数: {len(all_training_pairs):,}")
```

---

## 🎯 **用途別推奨**

| 用途 | 推奨subset | 理由 |
|------|-----------|------|
| **🔬 学術研究・評価** | `full` | 包括的なデータで堅牢な評価が可能 |
| **👨‍💻 実用開発** | `small` | 効率的な開発とテストに最適 |
| **⚡ プロトタイピング** | `minimal` | 高速な概念実証とアルゴリズム検証 |
| **🌐 英語NLPシステム** | `ms-marco-*` | 英語特化で高品質な学習 |
| **🗾 日本語NLPシステム** | `ja-*` | 日本語特化で最高品質の学習 |
| **🌍 多言語システム** | `full` | 20言語対応の汎用的な学習 |

---

## 📈 **データセット統計**

### サイズ比較

| Dataset | Train | Validation | Test | Total |
|---------|-------|------------|------|-------|
| **full** | 1,270,419 | 10,000 | 10,000 | **1,290,419** |
| **small** | 98,450 | 774 | 776 | **100,000** |
| **minimal** | 9,845 | 77 | 78 | **10,000** |
| **ms-marco-full** | 492,930 | 5,000 | 5,000 | **502,930** |
| **ms-marco-small** | 98,011 | 994 | 995 | **100,000** |
| **ms-marco-minimal** | 9,801 | 99 | 100 | **10,000** |
| **ja-full** | 500,298 | 2,999 | 2,999 | **506,296** |
| **ja-small** | 98,815 | 592 | 593 | **100,000** |
| **ja-minimal** | 9,881 | 59 | 60 | **10,000** |

### 教師スコア品質

| Dataset Category | POS平均 | NEG平均 | 品質ランク |
|-----------------|---------|---------|-----------|
| **日本語版** | 0.74-0.98 | 0.05-0.21 | ⭐⭐⭐⭐⭐ |
| **英語版** | 0.73-0.80 | 0.12-0.18 | ⭐⭐⭐⭐ |
| **全データセット版** | 0.78-0.82 | 0.15-0.19 | ⭐⭐⭐⭐ |

---

## 🔧 **技術仕様**

### データ生成パイプライン

1. **ハードネガティブサンプリング**: MS-MARCO、HPPRC、MIRACLから高品質なPOS/NEGペアを抽出
2. **多言語チャンキング**: 
   - 日本語: bunkai（高精度日本語文分割）
   - 英語: NLTK sentence_tokenize（高速・高精度）
   - その他: 言語別最適化チャンカー
3. **関連性判定**: Query-Context Pruner Multilingual（Qwen3-4B）による関連チャンク検出
4. **教師スコア付与**: japanese-reranker-xsmall-v2による高品質関連性スコア
5. **層化分割**: dataset_name別バランス考慮分割

### 品質保証

- **POS検出率**: 100%（関連チャンク検出における高精度）
- **NEG誤検出率**: 5-12%（適切な難易度設定）
- **教師スコア妥当性**: POS/NEG間の明確な分離（統計的検証済み）
- **データ整合性**: 全フィールドの妥当性チェック完了

---

## 📚 **利用可能なSubset一覧**

```python
# 全バージョン確認
available_subsets = [
    'full', 'small', 'minimal',           # 全データセット版
    'ms-marco-full', 'ms-marco-small', 'ms-marco-minimal',  # MS-MARCO英語版
    'ja-full', 'ja-small', 'ja-minimal'   # 日本語版
]

for subset in available_subsets:
    dataset = load_dataset('hotchpotch/wip-query-context-pruner-with-teacher-scores', subset)
    print(f"{subset}: {sum(len(split) for split in dataset.values()):,} 件")
```

---

## ⚠️ **注意事項**

- このデータセットは**プライベートリポジトリ**です。アクセス権限が必要です。
- 教師スコアは**japanese-reranker-xsmall-v2**による自動アノテーションです。
- 関連チャンクは**vLLMモデル**による自動判定です（人間による検証なし）。
- 研究・開発目的での使用を想定しています。

---

## 🤝 **貢献とフィードバック**

データセットの改善提案やバグレポートは、プロジェクトメンテナーまでお知らせください。特に以下の観点でのフィードバックを歓迎します：

- 教師スコアの妥当性
- 関連チャンク判定の精度
- 新しい言語やデータセットの追加要望
- 用途別最適化の提案
