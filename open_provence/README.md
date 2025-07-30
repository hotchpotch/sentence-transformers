# OpenProvence: Query-Dependent Text Pruning

OpenProvence is a module for efficient query-dependent text pruning in RAG (Retrieval-Augmented Generation) pipelines. It combines reranking and pruning capabilities to:

1. **Rank** documents based on their relevance to a query
2. **Prune** irrelevant sentences from documents to reduce context length

## Key Features

- 🎯 **Dual-purpose model**: Performs both reranking and pruning
- 📉 **Context reduction**: Removes irrelevant sentences while preserving important information
- 🌍 **Multilingual support**: Works with multiple languages including English, Japanese, and Chinese
- 🔧 **Easy integration**: Simple API for training and inference

## Installation

OpenProvence is a standalone library for query-dependent text pruning.

## Quick Start

### Using a Pre-trained Model

```python
from open_provence import OpenProvenceEncoder

# Load a pre-trained model
encoder = OpenProvenceEncoder.from_pretrained("path/to/openprovence/model")

# Prune a document based on a query
query = "What is machine learning?"
document = "Machine learning is a subset of AI. It uses algorithms. The weather is nice today."

# Get pruned document
pruned_doc = encoder.prune(query, document, threshold=0.5)
print(pruned_doc)
# Output: "Machine learning is a subset of AI. It uses algorithms."

# Get detailed results
result = encoder.prune(query, document, threshold=0.5, return_sentences=True)
print(f"Kept {sum(result['pruning_masks'])}/{len(result['sentences'])} sentences")
print(f"Compression ratio: {result['compression_ratio']:.2%}")
```

### Training a New Model

```python
from open_provence import (
    OpenProvenceEncoder,
    OpenProvenceTrainer,
    OpenProvenceLoss,
    ProvenceDataCollator
)
from datasets import Dataset

# Initialize encoder
encoder = ProvenceEncoder(
    model_name_or_path="microsoft/mdeberta-v3-base",
    num_labels=1,
    max_length=512,
    pruning_config={
        "dropout": 0.1,
        "sentence_pooling": "mean"
    }
)

# Prepare dataset
# Dataset should contain: query, document, label, pruning_labels, teacher_score
train_dataset = Dataset.from_list([
    {
        "query": "What is AI?",
        "document": "AI is artificial intelligence. It's a field of computer science.",
        "label": 1.0,
        "pruning_labels": [1, 1],  # Keep both sentences
        "teacher_score": 0.95
    },
    # ... more examples
])

# Create trainer
trainer = ProvenceTrainer(
    model=encoder,
    train_dataset=train_dataset,
    training_args={
        "num_epochs": 3,
        "batch_size": 16,
        "learning_rate": 2e-5
    }
)

# Train
trainer.train()

# Save model
encoder.save_pretrained("./my_provence_model")
```

## API Reference

### ProvenceEncoder

The main class for query-dependent text pruning.

#### Methods

- `predict(sentences)`: Get ranking scores for query-document pairs
- `predict_with_pruning(sentences)`: Get both ranking scores and pruning decisions
- `prune(query, document)`: Prune a single document based on a query
- `save_pretrained(path)`: Save the model
- `from_pretrained(path)`: Load a saved model

#### Parameters

- `model_name_or_path`: Base model to use
- `num_labels`: Number of labels for ranking (1 for regression)
- `max_length`: Maximum sequence length
- `pruning_config`: Configuration for the pruning head
  - `dropout`: Dropout probability
  - `sentence_pooling`: Pooling method ("mean", "max", "first", "last")
  - `use_weighted_pooling`: Whether to use attention weights

### ProvenceTrainer

Trainer class for ProvenceEncoder models.

#### Parameters

- `model`: ProvenceEncoder instance
- `train_dataset`: Training dataset
- `eval_dataset`: Evaluation dataset (optional)
- `data_collator`: Data collator (optional, created automatically)
- `loss_fn`: Loss function (optional, uses PruningLoss by default)
- `training_args`: Training configuration
  - `num_epochs`: Number of training epochs
  - `batch_size`: Batch size
  - `learning_rate`: Learning rate
  - `warmup_ratio`: Warmup ratio
  - `gradient_accumulation_steps`: Gradient accumulation steps
  - `fp16`: Enable mixed precision training

### Dataset Format

Training data should include:

```python
{
    "query": str,              # Query text
    "document": str,           # Document text
    "label": float,            # Relevance label (0 or 1)
    "pruning_labels": List[int],  # Sentence-level labels (0=prune, 1=keep)
    "teacher_score": float,    # Teacher model score (optional)
    "sentence_boundaries": List[List[int]]  # Token boundaries (optional)
}
```

## Architecture

Provence uses a dual-head architecture:

1. **Ranking Head**: Standard classification head for relevance scoring
2. **Pruning Head**: Token classification head for sentence-level decisions

Both heads share the same encoder backbone, enabling efficient joint training.

## Citation

If you use Provence in your research, please cite:

```bibtex
@inproceedings{provence2024,
  title={Provence: Query-Dependent Text Pruning for Efficient RAG},
  author={...},
  booktitle={...},
  year={2024}
}
```

## License

Provence is part of the sentence-transformers library and follows the same Apache 2.0 license.