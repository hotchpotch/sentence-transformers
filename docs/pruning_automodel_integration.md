# PruningEncoder AutoModel Integration Guide

## Overview

This document explains how to make PruningEncoder models compatible with Hugging Face's AutoModel classes, allowing users to load models without explicitly importing sentence_transformers.

## How AutoModel Registration Works

### 1. Static Registration (Built-in Models)
Transformers maintains mappings in:
- `CONFIG_MAPPING_NAMES`: Maps model_type → config class name
- `MODEL_MAPPING_NAMES`: Maps model_type → model class name  
- These are defined in `transformers/models/auto/*.py` files

### 2. Dynamic Registration (Custom Models)
Custom models can register themselves:
```python
from transformers import AutoConfig, AutoModelForSequenceClassification

# Register config
AutoConfig.register("model_type", ConfigClass)

# Register model
AutoModelForSequenceClassification.register(ConfigClass, ModelClass)
```

### 3. Auto-Map (Recommended for Custom Models)
The `auto_map` field in config.json tells AutoModel where to find custom classes:
```json
{
  "model_type": "pruning_encoder",
  "auto_map": {
    "AutoConfig": "module.path.ConfigClass",
    "AutoModelForSequenceClassification": "module.path.ModelClass"
  }
}
```

## Implementation for PruningEncoder

### Current Implementation

1. **Registration at Import** (`__init__.py`):
```python
from .transformers_compat import register_auto_models
register_auto_models()  # Registers with AutoConfig and AutoModel
```

2. **Config with auto_map** (in `save_pretrained`):
```python
"auto_map": {
    "AutoConfig": "sentence_transformers.pruning.transformers_compat.PruningEncoderConfig",
    "AutoModelForSequenceClassification": "sentence_transformers.pruning.transformers_compat.PruningEncoderForSequenceClassification"
}
```

3. **Loading with AutoModel**:
```python
# With sentence_transformers installed
from transformers import AutoModelForSequenceClassification
model = AutoModelForSequenceClassification.from_pretrained("path/to/model")

# Without sentence_transformers (requires trust_remote_code=True)
model = AutoModelForSequenceClassification.from_pretrained(
    "path/to/model",
    trust_remote_code=True
)
```

### Making Models Work Without sentence_transformers

For models to work without sentence_transformers installed, we need:

1. **Standalone Modeling File**: Save a self-contained Python file with the model that gets loaded when `trust_remote_code=True`:
   - Save as `modeling_pruning_encoder.py` in model directory
   - Include all necessary classes and imports
   - Update auto_map to use local references

2. **Updated auto_map**:
```json
{
  "auto_map": {
    "AutoConfig": "modeling_pruning_encoder.PruningEncoderConfig",
    "AutoModelForSequenceClassification": "modeling_pruning_encoder.PruningEncoderForSequenceClassification"
  }
}
```

## Usage Examples

### For Model Creators

When saving a PruningEncoder model:
```python
from sentence_transformers.pruning import PruningEncoder

# Train/load your model
model = PruningEncoder(...)

# Save with AutoModel compatibility
model.save_pretrained("my_model")

# The saved model will have:
# - config.json with auto_map
# - All necessary model files
# - Can be loaded with AutoModel
```

### For Model Users

#### Option 1: With sentence_transformers installed
```python
from transformers import AutoModelForSequenceClassification

model = AutoModelForSequenceClassification.from_pretrained("username/model-name")
```

#### Option 2: Without sentence_transformers (Hugging Face Hub)
```python
from transformers import AutoModelForSequenceClassification

model = AutoModelForSequenceClassification.from_pretrained(
    "username/model-name",
    trust_remote_code=True  # Required for custom models
)
```

#### Option 3: Using PruningEncoder directly
```python
from sentence_transformers.pruning import PruningEncoder

model = PruningEncoder.from_pretrained("username/model-name")
# Access all PruningEncoder-specific features
```

## Key Findings

1. **Model Type Registration**: PruningEncoder uses `model_type = "pruning_encoder"` which needs to be registered with AutoConfig.

2. **Two Loading Paths**:
   - With sentence_transformers: Uses registered classes
   - Without sentence_transformers: Uses trust_remote_code with standalone file

3. **Auto-Map Resolution**:
   - Transformers first checks if model_type is in local mappings
   - Then checks auto_map in config.json
   - With trust_remote_code=True, loads custom code from model directory

4. **CrossEncoder Compatibility**: PruningCrossEncoder wrapper allows using PruningEncoder models as drop-in replacements for CrossEncoder.

## Best Practices

1. **Always include auto_map** in config.json when saving models
2. **Test loading** with both methods (with/without sentence_transformers)
3. **Provide clear documentation** on loading methods in model cards
4. **Consider standalone files** for models shared on Hugging Face Hub

## Testing

Use the provided test script to verify AutoModel compatibility:
```bash
python scripts/test_pruning_automodel.py /path/to/saved/model
```

This will:
- Check config.json structure
- Test loading with AutoConfig
- Test loading with AutoModelForSequenceClassification  
- Verify forward pass works
- Create usage documentation