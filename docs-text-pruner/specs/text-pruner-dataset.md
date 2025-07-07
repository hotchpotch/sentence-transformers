# Text-Prunerデータセット仕様

## 概要

`hotchpotch/wip-query-context-pruner`は、Provenceアプローチに基づくtext-pruner実装のための学習データセットです。このデータセットは開発中（WIP: Work In Progress）であり、フォーマットは今後改善される可能性があります。

## データセット情報

- **データセット名**: hotchpotch/wip-query-context-pruner
- **分割**: train（1,290,419サンプル）
- **元データソース**: auto-wiki-qa-nemotronなど複数のデータセットから構成

## 現在のデータ構造

### フィールド定義

| フィールド名 | 型 | 説明 |
|------------|---|------|
| `id` | string | サンプルの一意識別子（例: "auto-wiki-qa-nemotron:0"） |
| `query` | string | 検索クエリまたは質問（日本語） |
| `texts` | List[string] | 検索結果の文書リスト（複数の文書） |
| `chunks_pos` | List[List[List[int]]] | 各文書内の文（chunk）の開始・終了位置 |
| `labels` | List[int] | 各文書の関連性ラベル（0: 非関連, 1: 関連） |
| `dataset_name` | string | 元データセットの名前 |
| `relevant_chunks` | List[List[int]] | 各文書内の関連する文のインデックスリスト |

### データ例

```python
{
    "id": "auto-wiki-qa-nemotron:0",
    "query": "日本語の人称代名詞の特徴は何ですか?",
    "texts": [
        "日本語\n上の事実は、現代英語の一人称・二人称代名詞が...",
        "四人称\n以上のように、「異なるもの」を有標として...",
        # ... 複数の文書
    ],
    "chunks_pos": [
        [[0, 4], [4, 159], [159, 251], ...],  # 1つ目の文書の文分割位置
        [[0, 4], [4, 88], [88, 111], ...],    # 2つ目の文書の文分割位置
        # ...
    ],
    "labels": [1, 0, 0, 0, 0],  # 文書レベルの関連性
    "relevant_chunks": [
        [2, 3, 6, 7],  # 1つ目の文書の関連文インデックス
        [],            # 2つ目の文書は関連文なし
        # ...
    ]
}
```

## 特徴と課題

### 現在の特徴

1. **文書レベルと文レベルの両方のアノテーション**
   - `labels`: 文書全体の関連性（reranking用）
   - `relevant_chunks`: 文内の関連部分（pruning用）

2. **文分割情報の提供**
   - `chunks_pos`により、各文の正確な位置を特定可能
   - 文の再構築や部分抽出が容易

3. **複数文書の同時処理**
   - 1クエリに対して複数文書（通常5文書）を含む
   - バッチ処理に適した構造

### 改善が必要な点

1. **Provence仕様との整合性**
   - 現在の`relevant_chunks`を`pruning_labels`（バイナリラベル）に変換必要
   - 文レベルのラベルを0/1の配列として表現

2. **データ構造の簡略化**
   - `chunks_pos`と文分割済みテキストの重複を解消
   - より直感的なフィールド名への変更

3. **学習効率の向上**
   - positive/negativeの明示的な分離
   - hard negativeの追加
   - 文書の長さの正規化

## 推奨される改善案

### 改善後のデータ構造案

```python
{
    "query": "日本語の人称代名詞の特徴は何ですか?",
    "document": "日本語\n上の事実は、現代英語の...",  # 単一文書
    "sentences": [
        "日本語",
        "上の事実は、現代英語の一人称・二人称代名詞が...",
        # ... 文分割済みテキスト
    ],
    "label": 1.0,  # reranking用スコア
    "pruning_labels": [0, 1, 1, 0, 0, 1, 1, 0],  # 各文のバイナリラベル
    "metadata": {
        "dataset_name": "auto-wiki-qa-nemotron",
        "document_id": "0",
        "language": "ja"
    }
}
```

### データ変換スクリプト例

```python
def convert_to_provence_format(sample):
    """現在のフォーマットをProvence仕様に変換"""
    converted_samples = []
    
    for idx, (text, label, relevant_chunks, chunks_pos) in enumerate(
        zip(sample["texts"], sample["labels"], 
            sample["relevant_chunks"], sample["chunks_pos"])
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
        
        converted_samples.append({
            "query": sample["query"],
            "document": text,
            "sentences": sentences,
            "label": float(label),
            "pruning_labels": pruning_labels,
            "metadata": {
                "dataset_name": sample["dataset_name"],
                "document_id": f"{sample['id']}_{idx}"
            }
        })
    
    return converted_samples
```

## 使用上の注意

1. **開発中のデータセット**: フォーマットは予告なく変更される可能性があります
2. **日本語データ**: 現在は主に日本語のWikipediaベースのQAデータ
3. **ラベルの品質**: Silver labelsのため、完全な精度は保証されません
4. **データサイズ**: 約130万サンプルと大規模なため、開発時はサブセットの使用を推奨

## 今後の拡張計画

1. **多言語対応**: 英語、中国語などの追加
2. **ドメイン拡張**: 技術文書、ニュース記事などの追加
3. **Hard negative mining**: より困難な負例の追加
4. **品質向上**: Gold labelsの追加、人手によるアノテーション

## 関連資料

- [Provence実装仕様](./provence-implementation-spec.md)
- [データフォーマット仕様](./data-format-spec.md)
- [実装仕様書](./spec.md)