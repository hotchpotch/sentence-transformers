# Transformers Library Compatibility Implementation

## Overview

Successfully implemented Transformers library compatibility for PruningEncoder models, enabling them to be loaded and used with standard `AutoModel` classes.

## Implementation Details

### 1. Created Wrapper Classes (`transformers_compat.py`)

- **PruningEncoderConfig**: Custom configuration class inheriting from `PretrainedConfig`
- **PruningEncoderForSequenceClassification**: Wrapper for reranking models
- **PruningEncoderForTokenClassification**: Wrapper for pruning-only models

### 2. Key Features

- **Dual Config System**: 
  - `pruning_encoder_config.json` for backward compatibility
  - `config.json` for Transformers compatibility
  
- **Automatic Registration**: Models are automatically registered when the module is imported

- **Seamless Integration**: Works with standard Transformers patterns

### 3. Usage Examples

#### Reranking Model (Sequence Classification)
```python
from transformers import AutoModelForSequenceClassification, AutoTokenizer
import sentence_transformers.pruning  # Required for registration

# Load model
model = AutoModelForSequenceClassification.from_pretrained("path/to/model")
tokenizer = AutoTokenizer.from_pretrained("path/to/model")

# Inference
inputs = tokenizer(query, document, return_tensors="pt")
outputs = model(**inputs)
score = torch.sigmoid(outputs.logits).item()
```

#### Pruning Model (Token Classification)
```python
from transformers import AutoModelForTokenClassification, AutoTokenizer
import sentence_transformers.pruning  # Required for registration

# Load model
model = AutoModelForTokenClassification.from_pretrained("path/to/model")
tokenizer = AutoTokenizer.from_pretrained("path/to/model")

# Inference
inputs = tokenizer(query, document, return_tensors="pt")
outputs = model(**inputs)
keep_probs = torch.softmax(outputs.logits, dim=-1)[:, :, 1]
```

## Benefits

1. **Standard API**: Use familiar Transformers patterns
2. **Pipeline Compatibility**: Works with Hugging Face pipelines
3. **Easy Integration**: Drop-in replacement for existing workflows
4. **Backward Compatible**: Original PruningEncoder API still works

## Testing

Both model types tested successfully:
- ✅ AutoModelForSequenceClassification with reranking models
- ✅ AutoModelForTokenClassification with pruning-only models
- ✅ Original PruningEncoder API remains functional

## Files Modified/Created

1. `sentence_transformers/pruning/transformers_compat.py` - New wrapper classes
2. `sentence_transformers/pruning/__init__.py` - Import and register wrappers
3. `sentence_transformers/pruning/encoder.py` - Updated save_pretrained() to save both configs
4. Test scripts demonstrating usage

## Important Notes

- Always import `sentence_transformers.pruning` before using AutoModel classes
- Models need device handling (move inputs to model's device)
- The wrapper classes handle the interface translation between Transformers and PruningEncoder