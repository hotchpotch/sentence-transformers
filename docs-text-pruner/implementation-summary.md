# Provence Implementation Summary

## Overview
This document summarizes the implementation of Provence (Query-dependent Text Pruning) functionality in Sentence Transformers.

## Implemented Components

### 1. CrossEncoder Extension
- **File**: `sentence_transformers/cross_encoder/CrossEncoder.py`
- **Key Changes**:
  - Added `enable_pruning` parameter to `__init__`
  - Added `pruning_head_config` for configuring the pruning classifier
  - Implemented `_add_pruning_head()` method to attach ProvencePruningHead
  - Modified `forward()` to support both ranking and pruning predictions
  - Added `predict_with_pruning()` for batch prediction with pruning
  - Added `prune()` for single document pruning

### 2. Data Structures
- **File**: `sentence_transformers/cross_encoder/data_structures.py`
- **Classes**:
  - `ProvenceOutput`: Dataclass for storing ranking scores and pruning masks
  - `ProvenceConfig`: Configuration for Provence functionality

### 3. Pruning Head
- **File**: `sentence_transformers/models/ProvencePruningHead.py`
- **Classes**:
  - `ProvencePruningConfig`: Configuration for the pruning head
  - `ProvencePruningHead`: PreTrainedModel for token classification
- **Features**:
  - Sentence-level pooling strategies (mean, max, first, last)
  - Compatible with AutoModelForTokenClassification
  - Supports both token-level and sentence-level predictions

### 4. Text Chunking
- **File**: `sentence_transformers/utils/text_chunking.py`
- **Classes**:
  - `BaseChunker`: Abstract base class
  - `MultilingualChunker`: Main chunker with language detection
  - `JapaneseChunker`: Bunkai-based Japanese sentence splitter
  - `EnglishChunker`: NLTK-based English sentence splitter
  - Language-specific chunkers for Chinese, etc.

### 5. Loss Function
- **File**: `sentence_transformers/cross_encoder/losses/ProvenceLoss.py`
- **Features**:
  - Joint loss for ranking and pruning objectives
  - Support for teacher score distillation
  - Sentence-level and token-level pruning loss computation
  - Configurable weights for ranking vs pruning

### 6. Data Collator
- **File**: `sentence_transformers/cross_encoder/data_collators.py`
- **Features**:
  - Prepares batch data with sentence boundaries
  - Handles ranking and pruning labels
  - Supports sentence-level pruning with boundary detection

## Usage Example

```python
from sentence_transformers import CrossEncoder

# Initialize with pruning enabled
model = CrossEncoder(
    "hotchpotch/japanese-reranker-xsmall-v2",
    enable_pruning=True,
    pruning_head_config={
        "dropout": 0.1,
        "sentence_pooling": "mean"
    }
)

# Simple pruning
query = "人工知能の最新の進歩について"
document = "長い文書..."
pruned_doc = model.prune(query, document, threshold=0.5)

# Detailed pruning with sentence info
result = model.prune(query, document, threshold=0.5, return_sentences=True)
print(f"Compression ratio: {result['compression_ratio']}")
print(f"Pruned sentences: {result['num_pruned_sentences']}")
```

## Training Example

```python
from sentence_transformers.cross_encoder.losses import ProvenceLoss
from sentence_transformers.cross_encoder.data_collators import create_provence_data_collator

# Create loss function
loss_fn = ProvenceLoss(
    model,
    ranking_weight=1.0,
    pruning_weight=0.5,
    sentence_level_pruning=True
)

# Create data collator
collator = create_provence_data_collator(model)

# Training loop
for batch in dataloader:
    loss = loss_fn(batch['sentence_features'], batch['labels'])
    loss.backward()
    optimizer.step()
```

## Architecture Support

The implementation supports various transformer architectures:
- BERT, RoBERTa, DeBERTa
- ModernBERT (as used in japanese-reranker-xsmall-v2)
- ELECTRA, XLM, ALBERT
- Other architectures with standard transformer structure

## Key Design Decisions

1. **enable_pruning Parameter**: Makes the feature opt-in, maintaining backward compatibility
2. **Dual Output**: ProvenceOutput contains both ranking scores and pruning masks
3. **Modular Design**: Pruning head is a separate module that can be used independently
4. **Language Support**: Multilingual text chunking with language-specific optimizations
5. **Flexible Training**: Supports various training scenarios (joint, pruning-only, distillation)

## Next Steps

1. Implement evaluation metrics for pruning performance
2. Add comprehensive tests
3. Create documentation for users
4. Prepare for PR submission to Sentence Transformers

## Files Modified/Created

### Modified:
- `sentence_transformers/cross_encoder/CrossEncoder.py`
- `sentence_transformers/cross_encoder/losses/__init__.py`
- `pyproject.toml` (added dependencies: bunkai, langdetect)

### Created:
- `sentence_transformers/cross_encoder/data_structures.py`
- `sentence_transformers/models/ProvencePruningHead.py`
- `sentence_transformers/utils/text_chunking.py`
- `sentence_transformers/cross_encoder/losses/ProvenceLoss.py`
- `sentence_transformers/cross_encoder/data_collators.py`

## Dependencies Added
- `bunkai>=1.5.7` - Japanese sentence segmentation
- `langdetect>=1.0.9` - Language detection for multilingual support
- `einops>=0.8.1` - For Flash Attention compatibility
- `protobuf>=6.31.1` - For sentencepiece tokenizer