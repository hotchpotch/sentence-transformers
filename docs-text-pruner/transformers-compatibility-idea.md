# Transformers Library Compatibility for PruningEncoder

## Overview

This document outlines an idea to make PruningEncoder compatible with the standard Transformers library AutoModel classes, enabling seamless integration with existing HuggingFace workflows.

**⚠️ IMPORTANT: Delete this file after successful implementation.**

## Problem Statement

Currently, PruningEncoder models can only be loaded through our custom `PruningEncoder.from_pretrained()` method. Users cannot utilize standard Transformers patterns like:

```python
from transformers import AutoModelForSequenceClassification, AutoModelForTokenClassification

# This doesn't work with current implementation
model = AutoModelForSequenceClassification.from_pretrained("path/to/pruning/model")
```

## Proposed Solution

Create Transformers-compatible wrapper classes that expose PruningEncoder functionality through standard AutoModel interfaces.

### 1. Custom Configuration Class

```python
from transformers import PretrainedConfig

class PruningEncoderConfig(PretrainedConfig):
    model_type = "pruning_encoder"
    
    def __init__(
        self,
        mode="reranking_pruning",
        pruning_config=None,
        base_model_config=None,
        **kwargs
    ):
        super().__init__(**kwargs)
        self.mode = mode
        self.pruning_config = pruning_config or {}
        self.base_model_config = base_model_config
```

### 2. Sequence Classification Wrapper (for Reranking)

```python
from transformers import PreTrainedModel
from transformers.modeling_outputs import SequenceClassifierOutput

class PruningEncoderForSequenceClassification(PreTrainedModel):
    config_class = PruningEncoderConfig
    
    def __init__(self, config):
        super().__init__(config)
        self.pruning_encoder = PruningEncoder(
            model_name_or_path=config.base_model_name_or_path,
            mode="reranking_pruning",
            **config.pruning_config
        )
        # Expose standard classifier head
        self.classifier = self.pruning_encoder.ranking_model.classifier
        self.num_labels = config.num_labels
    
    def forward(self, input_ids, attention_mask=None, labels=None, **kwargs):
        outputs = self.pruning_encoder.forward(
            input_ids=input_ids,
            attention_mask=attention_mask,
            **kwargs
        )
        
        logits = outputs["ranking_logits"]
        loss = None
        if labels is not None:
            loss_fct = torch.nn.BCEWithLogitsLoss()
            loss = loss_fct(logits.view(-1), labels.float())
        
        return SequenceClassifierOutput(
            loss=loss,
            logits=logits,
            hidden_states=outputs.get("hidden_states")
        )
```

### 3. Token Classification Wrapper (for Pruning)

```python
from transformers.modeling_outputs import TokenClassifierOutput

class PruningEncoderForTokenClassification(PreTrainedModel):
    config_class = PruningEncoderConfig
    
    def __init__(self, config):
        super().__init__(config)
        self.pruning_encoder = PruningEncoder(
            model_name_or_path=config.base_model_name_or_path,
            mode="pruning_only",
            **config.pruning_config
        )
        # Expose standard classifier head
        self.classifier = self.pruning_encoder.pruning_head
        self.num_labels = 2  # keep/prune
    
    def forward(self, input_ids, attention_mask=None, labels=None, **kwargs):
        outputs = self.pruning_encoder.forward(
            input_ids=input_ids,
            attention_mask=attention_mask,
            **kwargs
        )
        
        logits = outputs["pruning_logits"]
        loss = None
        if labels is not None:
            loss_fct = torch.nn.CrossEntropyLoss()
            loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
        
        return TokenClassifierOutput(
            loss=loss,
            logits=logits,
            hidden_states=outputs.get("hidden_states")
        )
```

### 4. AutoModel Registration

```python
from transformers import AutoConfig, AutoModelForSequenceClassification, AutoModelForTokenClassification

# Register the config
AutoConfig.register("pruning_encoder", PruningEncoderConfig)

# Register the models
AutoModelForSequenceClassification.register(PruningEncoderConfig, PruningEncoderForSequenceClassification)
AutoModelForTokenClassification.register(PruningEncoderConfig, PruningEncoderForTokenClassification)
```

### 5. Model Configuration Examples

#### For Reranking Models
```json
{
  "model_type": "pruning_encoder",
  "mode": "reranking_pruning",
  "base_model_name_or_path": "hotchpotch/japanese-reranker-xsmall-v2",
  "pruning_config": {
    "hidden_size": 384,
    "dropout": 0.1,
    "sentence_pooling": "mean"
  },
  "num_labels": 1,
  "architectures": ["PruningEncoderForSequenceClassification"]
}
```

#### For Pruning-Only Models
```json
{
  "model_type": "pruning_encoder",
  "mode": "pruning_only",
  "base_model_name_or_path": "cl-nagoya/ruri-v3-30m",
  "pruning_config": {
    "hidden_size": 256,
    "dropout": 0.1,
    "sentence_pooling": "mean"
  },
  "num_labels": 2,
  "architectures": ["PruningEncoderForTokenClassification"]
}
```

## Usage Examples

### Reranking with Standard Transformers API

```python
from transformers import AutoModelForSequenceClassification, AutoTokenizer
import torch

# Load model using standard AutoModel
model = AutoModelForSequenceClassification.from_pretrained(
    "outputs/pruning-ja-minimal/checkpoint-412-best"
)
tokenizer = AutoTokenizer.from_pretrained(
    "outputs/pruning-ja-minimal/checkpoint-412-best"
)

# Standard HuggingFace inference
query = "機械学習について"
document = "機械学習は人工知能の一分野です。"

inputs = tokenizer(query, document, return_tensors="pt", truncation=True, padding=True)
with torch.no_grad():
    outputs = model(**inputs)
    scores = torch.sigmoid(outputs.logits)

print(f"Relevance score: {scores.item():.4f}")
```

### Token-level Pruning with Standard API

```python
from transformers import AutoModelForTokenClassification

# Load pruning model
model = AutoModelForTokenClassification.from_pretrained(
    "outputs/pruning_only_small_20250709_084354/checkpoint-1000-best"
)

inputs = tokenizer(query, document, return_tensors="pt")
with torch.no_grad():
    outputs = model(**inputs)
    pruning_probs = torch.softmax(outputs.logits, dim=-1)
    keep_probs = pruning_probs[:, :, 1]  # Probability of keeping each token

# Apply threshold for pruning decisions
threshold = 0.5
keep_mask = keep_probs > threshold
```

## Benefits

1. **Ecosystem Compatibility**: Seamless integration with existing HuggingFace workflows
2. **Ease of Use**: Standard AutoModel loading patterns
3. **Framework Integration**: Compatible with Transformers Trainer, Pipeline API, etc.
4. **Performance**: Task-specific models without unnecessary components
5. **Backward Compatibility**: Existing PruningEncoder API remains unchanged

## Implementation Plan

1. Create wrapper classes in `sentence_transformers/pruning/transformers_compat.py`
2. Add registration logic to `sentence_transformers/pruning/__init__.py`
3. Update `save_pretrained()` method to include appropriate config.json
4. Create integration tests
5. Update documentation with usage examples

## Files to Modify

- `sentence_transformers/pruning/encoder.py` - Update save_pretrained method
- `sentence_transformers/pruning/__init__.py` - Add imports and registration
- Add `sentence_transformers/pruning/transformers_compat.py` - New wrapper classes
- Update tests in `tests/pruning/`

## Testing Strategy

1. Test AutoModel loading for both sequence and token classification
2. Verify outputs match original PruningEncoder
3. Test with Transformers Pipeline API
4. Test fine-tuning with Transformers Trainer
5. Validate save/load cycle preservation

---

**Note**: This is a design document. Implementation should be done after current model training and evaluation is complete. Delete this file upon successful implementation.