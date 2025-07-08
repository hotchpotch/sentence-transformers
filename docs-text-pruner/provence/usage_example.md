# Provence Usage Example

## Improved DataCollator with Direct Dataset Support

The new `ProvenceChunkBasedDataCollator` can work directly with HuggingFace datasets without requiring manual conversion. This provides better performance and cleaner code.

### Basic Usage

```python
from datasets import load_dataset
from sentence_transformers.provence import (
    ProvenceEncoder,
    ProvenceTrainer,
    ProvenceChunkBasedDataCollator
)
from sentence_transformers.provence.losses_chunk_based import ProvenceChunkBasedLoss

# Load dataset directly
dataset = load_dataset('hotchpotch/wip-query-context-pruner-with-teacher-scores', 'ja-minimal')

# Initialize model
model = ProvenceEncoder(
    model_name_or_path="hotchpotch/japanese-reranker-xsmall-v2",
    num_labels=1,
    max_length=512
)

# Create data collator with column mapping
data_collator = ProvenceChunkBasedDataCollator(
    tokenizer=model.tokenizer,
    max_length=512,
    # Specify column names from your dataset
    query_column="query",
    texts_column="texts", 
    labels_column="labels",
    scores_column="teacher_scores_japanese-reranker-xsmall-v2",  # Optional: for teacher distillation
    chunks_pos_column="chunks_pos",
    relevant_chunks_column="relevant_chunks"
)

# Loss function (simplified - no use_teacher_scores flag needed)
loss_fn = ProvenceChunkBasedLoss(
    model=model,
    ranking_weight=1.0,
    pruning_weight=0.8,
    is_regression=True  # True for teacher score distillation, False for hard labels
)

# Train directly with HuggingFace dataset
trainer = ProvenceTrainer(
    model=model,
    train_dataset=dataset['train'],  # Pass dataset directly
    eval_dataset=dataset['validation'],
    data_collator=data_collator,
    loss_fn=loss_fn,
    training_args={...}
)

trainer.train()
```

### Flexible Column Configuration

The DataCollator supports any column naming convention:

```python
# Example with custom column names
data_collator = ProvenceChunkBasedDataCollator(
    tokenizer=model.tokenizer,
    query_column="question",
    texts_column="documents",
    labels_column="relevance_labels",
    scores_column="model_predictions",  # Will use relevance_labels if this column doesn't exist
    chunks_pos_column="sentence_boundaries",
    relevant_chunks_column="important_sentences"
)
```

### Switching Between Teacher Scores and Hard Labels

```python
# Using teacher scores for distillation
collator_distillation = ProvenceChunkBasedDataCollator(
    tokenizer=model.tokenizer,
    scores_column="teacher_scores_model_name",  # Specify teacher scores column
    ...
)

# Using hard labels only
collator_hard_labels = ProvenceChunkBasedDataCollator(
    tokenizer=model.tokenizer,
    scores_column=None,  # Don't specify or set to None
    ...
)
```

### Benefits

1. **No Dataset Conversion**: Work directly with HuggingFace datasets
2. **Better Performance**: Leverages HuggingFace's optimized data loading
3. **Flexible Column Names**: Adapt to any dataset schema
4. **Automatic Validation**: Validates required columns on first use
5. **Graceful Fallback**: Falls back to hard labels if teacher scores are missing

### Error Handling

The DataCollator validates columns on first use:

```python
# This will raise a clear error if columns are missing
try:
    data_collator = ProvenceChunkBasedDataCollator(
        tokenizer=model.tokenizer,
        query_column="missing_column",  # This doesn't exist
        ...
    )
    batch = data_collator(dataset)
except ValueError as e:
    print(f"Column validation error: {e}")
    # Output: Missing required columns: ['missing_column']. Available columns: [...]
```