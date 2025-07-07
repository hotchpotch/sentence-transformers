# Provenceデータフォーマット仕様

## 概要

Provence実装で使用するデータフォーマットの詳細仕様です。Sentence Transformersの既存パターンに準拠しつつ、pruning機能に必要な拡張を定義します。

## 学習データ

### 基本フォーマット（CSV/JSON/Parquet）

```json
{
    "query": "What is the capital of France?",
    "document": "Paris is the capital and most populous city of France. Located in the north of the country, Paris has been a major European city since the Middle Ages. The city is known for its museums, monuments, and architectural landmarks.",
    "label": 1.0,
    "pruning_labels": [1, 1, 0]
}
```

### フィールド詳細

| フィールド | 型 | 必須 | 説明 |
|-----------|---|------|------|
| `query` | str | ✓ | 検索クエリまたは質問 |
| `document` | str | ✓ | プルーニング対象の文書 |
| `label` | float | ✓ | Reranking用の関連性スコア（0-1） |
| `pruning_labels` | List[int] | ✓ | 各文のバイナリラベル（0: 削除, 1: 保持） |
| `sentences` | List[str] | × | 文分割済みのテキスト（自動生成可） |
| `metadata` | Dict | × | 追加情報（データセット名、ドメイン等） |

### ラベル生成方法

1. **Silver Labels（論文準拠）**
   - LLM（Llama-3等）による自動生成
   - Citation-basedプロンプティング使用

2. **Gold Labels（高品質）**
   - 人手によるアノテーション
   - QAデータセットからの自動変換

### データセット例

```python
# HuggingFace Dataset形式
dataset = Dataset.from_dict({
    "query": [
        "What causes climate change?",
        "How do vaccines work?"
    ],
    "document": [
        "Climate change is primarily caused by human activities. The burning of fossil fuels releases greenhouse gases. These gases trap heat in the atmosphere. Deforestation also contributes to the problem.",
        "Vaccines work by training the immune system. They contain weakened or inactive parts of pathogens. The immune system learns to recognize threats. This provides protection against future infections."
    ],
    "label": [1.0, 1.0],
    "pruning_labels": [
        [1, 1, 1, 0],  # 最後の文は削除
        [1, 1, 1, 1]   # 全文保持
    ]
})
```

## 推論データ

### バッチ推論フォーマット

```python
{
    "queries": List[str],           # クエリのリスト
    "documents": List[List[str]],   # 各クエリに対する文書リスト
    "top_k": int,                   # 返すべき上位文書数
    "pruning_config": {
        "mode": "dynamic",          # "fixed", "dynamic", "threshold"
        "ratio": 0.5,               # 固定比率モード時の削減率
        "threshold": 0.7,           # 閾値モード時の保持閾値
        "min_sentences": 2          # 最小保持文数
    }
}
```

### ストリーミング推論フォーマット

```python
{
    "query": str,
    "document_stream": Iterator[str],  # 文書のストリーム
    "chunk_size": int,                 # チャンクサイズ（文数）
    "overlap": int                     # チャンク間のオーバーラップ
}
```

## 評価データ

### Reranking評価用（CrossEncoder互換）

```python
{
    "query": str,
    "positive": List[str],      # 関連文書
    "negative": List[str],      # 非関連文書
    # または
    "documents": List[str],     # 全文書
    "relevance_scores": List[float]  # 各文書の関連度
}
```

### Pruning評価用（拡張）

```python
{
    "query": str,
    "document": str,
    "ground_truth_sentences": List[str],  # 保持すべき文
    "pruned_sentences": List[str],        # 削除可能な文
    "min_acceptable_ratio": float         # 最小許容保持率
}
```

## データ変換ユーティリティ

### MS MARCOからの変換

```python
def convert_msmarco_to_provence(msmarco_data):
    """MS MARCOデータをProvence形式に変換"""
    return {
        "query": msmarco_data["query"],
        "document": msmarco_data["positive_passages"][0]["text"],
        "label": 1.0,
        "pruning_labels": generate_pruning_labels(
            query=msmarco_data["query"],
            document=msmarco_data["positive_passages"][0]["text"]
        )
    }
```

### BEIRからの変換

```python
def convert_beir_to_provence(beir_data):
    """BEIRデータをProvence形式に変換"""
    return {
        "query": beir_data["query"],
        "positive": beir_data["positive"],
        "negative": beir_data["negative"],
        "documents": beir_data["positive"] + beir_data["negative"]
    }
```

## バリデーション

### データ検証関数

```python
def validate_provence_data(data: Dict) -> bool:
    """Provenceデータの妥当性を検証"""
    
    # 必須フィールドの確認
    required = ["query", "document", "label"]
    if not all(field in data for field in required):
        return False
    
    # pruning_labelsの検証
    if "pruning_labels" in data:
        sentences = sent_tokenize(data["document"])
        if len(data["pruning_labels"]) != len(sentences):
            return False
        if not all(label in [0, 1] for label in data["pruning_labels"]):
            return False
    
    # labelの範囲確認
    if not 0 <= data["label"] <= 1:
        return False
    
    return True
```

## メモリ効率的な処理

### チャンク処理

```python
def process_in_chunks(documents: List[str], chunk_size: int = 1000):
    """大規模データセットのチャンク処理"""
    for i in range(0, len(documents), chunk_size):
        chunk = documents[i:i + chunk_size]
        yield process_chunk(chunk)
```

### ストリーミング処理

```python
def stream_process_documents(file_path: str):
    """ファイルからストリーミング処理"""
    with open(file_path, 'r') as f:
        for line in f:
            data = json.loads(line)
            yield validate_and_process(data)
```