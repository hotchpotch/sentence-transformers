# Provence Usage Guide (Updated)

## Unified Predict Method

The ProvenceEncoder now has a unified `predict` method that can handle both ranking and pruning in a single call.

## Basic Usage

### 1. Ranking Only

```python
from sentence_transformers.provence import ProvenceEncoder

# Load model
model = ProvenceEncoder.from_pretrained("./outputs/provence-ja-small/final-model")

# Query-document pairs
pairs = [
    ("What is Python?", "Python is a programming language."),
    ("What is Java?", "Java is also a programming language."),
]

# Get ranking scores only
scores = model.predict(
    pairs,
    apply_pruning=False,  # Disable pruning
    batch_size=32
)
print(f"Scores: {scores}")
```

### 2. Ranking + Pruning (Default)

```python
# Get both ranking and pruning results (default behavior)
outputs = model.predict(
    pairs,
    apply_pruning=True,  # Default
    pruning_threshold=0.3,
    return_documents=True,
    batch_size=32
)

for output in outputs:
    print(f"Score: {output.ranking_scores:.3f}")
    print(f"Compression: {output.compression_ratio:.1%}")
    print(f"Pruned: {output.pruned_documents[0]}")
```

### 3. Single Input

```python
# Single query-document pair
pair = ("Question?", "This is the answer.")

# Ranking only
score = model.predict(pair, apply_pruning=False)

# Full functionality
output = model.predict(pair, apply_pruning=True, return_documents=True)
```

## Advanced Usage

### Batch Processing with Different Thresholds

```python
# Test different pruning thresholds
thresholds = [0.1, 0.3, 0.5, 0.7]

for threshold in thresholds:
    outputs = model.predict(
        pairs,
        pruning_threshold=threshold,
        return_documents=True
    )
    
    avg_compression = sum(o.compression_ratio for o in outputs) / len(outputs)
    print(f"Threshold {threshold}: {avg_compression:.1%} compression")
```

### Large Batch Processing

```python
# Process many pairs efficiently
large_pairs = [("Query {}".format(i), "Document {}".format(i)) for i in range(1000)]

outputs = model.predict(
    large_pairs,
    batch_size=512,  # Large batch size for efficiency
    apply_pruning=True,
    pruning_threshold=0.3,
    show_progress_bar=True
)
```

## Output Format

### Ranking Only (`apply_pruning=False`)

Returns ranking scores in the requested format:
- `convert_to_numpy=True` (default): numpy array
- `convert_to_tensor=True`: PyTorch tensor  
- `convert_to_numpy=False, convert_to_tensor=False`: Python list

### Full Functionality (`apply_pruning=True`)

Returns list of `ProvenceOutput` objects with:
- `ranking_scores`: Ranking score (float)
- `compression_ratio`: Compression ratio (0.0-1.0)
- `pruned_documents`: Pruned text (if `return_documents=True`)
- `pruning_masks`: Token-level keep/prune decisions
- Other metadata

## Migration from Old API

### Before (separate methods)

```python
# Old way
scores = model.predict(pairs)  # Ranking only
outputs = model.predict_with_pruning(pairs, threshold=0.3)  # Pruning
```

### After (unified method)

```python
# New way
scores = model.predict(pairs, apply_pruning=False)  # Ranking only
outputs = model.predict(pairs, apply_pruning=True, pruning_threshold=0.3)  # Both
```

The old `predict_with_pruning` method is still available for backward compatibility.

## Performance Tips

1. **Use large batch sizes** (256-512) for better throughput
2. **Set `show_progress_bar=False`** for production use
3. **Only use `return_documents=True`** when you need the pruned text
4. **Use `apply_pruning=False`** when you only need ranking scores

## Example: RAG Pipeline Integration

```python
def rerank_and_prune(query, documents, top_k=5, pruning_threshold=0.3):
    """Rerank documents and prune irrelevant content."""
    
    # Create query-document pairs
    pairs = [(query, doc) for doc in documents]
    
    # Get rankings and pruned content
    outputs = model.predict(
        pairs,
        apply_pruning=True,
        pruning_threshold=pruning_threshold,
        return_documents=True,
        batch_size=32
    )
    
    # Sort by ranking score and select top-k
    ranked_outputs = sorted(outputs, key=lambda x: x.ranking_scores, reverse=True)
    top_outputs = ranked_outputs[:top_k]
    
    # Return pruned documents with scores
    results = []
    for output in top_outputs:
        results.append({
            'score': output.ranking_scores,
            'original': documents[outputs.index(output)],
            'pruned': output.pruned_documents[0],
            'compression': output.compression_ratio
        })
    
    return results
```