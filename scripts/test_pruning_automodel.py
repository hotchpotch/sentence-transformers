#!/usr/bin/env python3
"""
Test script to verify PruningEncoder models work with AutoModel.

This script demonstrates:
1. How models saved with auto_map can be loaded with AutoModel
2. Different loading methods (with/without sentence_transformers installed)
3. Verification of model functionality
"""

import os
import sys
import json
from pathlib import Path
import torch
from transformers import AutoConfig, AutoModelForSequenceClassification, AutoTokenizer


def test_automodel_loading(model_path: str):
    """Test loading a PruningEncoder model with AutoModel."""
    
    print(f"\n=== Testing AutoModel loading for: {model_path} ===\n")
    
    # 1. Check config.json
    config_path = Path(model_path) / "config.json"
    if config_path.exists():
        with open(config_path, 'r') as f:
            config = json.load(f)
        
        print("Config.json content:")
        print(f"  model_type: {config.get('model_type')}")
        print(f"  architectures: {config.get('architectures')}")
        print(f"  auto_map: {config.get('auto_map')}")
        print()
    else:
        print(f"ERROR: No config.json found at {config_path}")
        return False
    
    # 2. Try loading with AutoConfig
    try:
        print("Loading with AutoConfig...")
        auto_config = AutoConfig.from_pretrained(model_path, trust_remote_code=True)
        print(f"  Success! Config type: {type(auto_config).__name__}")
        print(f"  Model type: {auto_config.model_type}")
    except Exception as e:
        print(f"  ERROR: {e}")
        return False
    
    # 3. Try loading with AutoModelForSequenceClassification (with trust_remote_code)
    try:
        print("\nLoading with AutoModelForSequenceClassification (trust_remote_code=True)...")
        model = AutoModelForSequenceClassification.from_pretrained(
            model_path,
            trust_remote_code=True
        )
        print(f"  Success! Model type: {type(model).__name__}")
        
        # Test forward pass
        tokenizer = AutoTokenizer.from_pretrained(model_path)
        inputs = tokenizer("This is a test query", "This is a test document", 
                         return_tensors="pt", truncation=True, max_length=512)
        
        with torch.no_grad():
            outputs = model(**inputs)
        
        print(f"  Forward pass successful! Output shape: {outputs.logits.shape}")
        
    except Exception as e:
        print(f"  ERROR: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    # 4. Try loading without trust_remote_code (should fail but gracefully)
    try:
        print("\nLoading without trust_remote_code (expected to fail)...")
        model = AutoModelForSequenceClassification.from_pretrained(model_path)
        print("  Unexpected success!")
    except Exception as e:
        print(f"  Expected failure: {type(e).__name__}")
        print(f"  This is correct - custom models require trust_remote_code=True")
    
    return True


def create_example_usage_docs(model_path: str):
    """Create documentation for how to use the model with AutoModel."""
    
    usage_doc = f"""# Using PruningEncoder with AutoModel

This model can be loaded using Hugging Face's AutoModel classes without installing sentence_transformers.

## Requirements
- transformers >= 4.0.0
- torch >= 1.10.0

## Loading the Model

### Option 1: With AutoModelForSequenceClassification (for reranking)
```python
from transformers import AutoModelForSequenceClassification, AutoTokenizer

# Load model and tokenizer
model = AutoModelForSequenceClassification.from_pretrained(
    "{model_path}",
    trust_remote_code=True
)
tokenizer = AutoTokenizer.from_pretrained("{model_path}")

# Use for ranking
query = "What is machine learning?"
document = "Machine learning is a branch of artificial intelligence..."

inputs = tokenizer(query, document, return_tensors="pt", truncation=True)
outputs = model(**inputs)
score = outputs.logits.squeeze().item()
```

### Option 2: With sentence_transformers installed
```python
from sentence_transformers.pruning import PruningEncoder

# Load directly as PruningEncoder
model = PruningEncoder.from_pretrained("{model_path}")

# Use all PruningEncoder features
outputs = model.predict_and_prune([(query, document)])
```

## Model Architecture

This is a PruningEncoder model that combines:
- Query-document reranking
- Token-level pruning for efficient retrieval

The model outputs both ranking scores and pruning decisions for each token.
"""
    
    readme_path = Path(model_path) / "README.md"
    with open(readme_path, 'w') as f:
        f.write(usage_doc)
    
    print(f"\nCreated usage documentation at: {readme_path}")


def check_standalone_loading(model_path: str):
    """Check if the model can be loaded in a clean environment."""
    
    print("\n=== Checking standalone loading capability ===\n")
    
    # Check for required files
    required_files = [
        "config.json",
        "tokenizer_config.json", 
        "vocab.txt",  # or tokenizer.json
        "pytorch_model.bin",  # or model.safetensors
    ]
    
    missing_files = []
    for file in required_files:
        file_path = Path(model_path) / file
        # Check alternatives
        if file == "vocab.txt" and not file_path.exists():
            if (Path(model_path) / "tokenizer.json").exists():
                continue
        if file == "pytorch_model.bin" and not file_path.exists():
            if (Path(model_path) / "model.safetensors").exists():
                continue
        
        if not file_path.exists() and not any((Path(model_path) / subdir / file).exists() 
                                             for subdir in ["ranking_model", "encoder_model"]):
            missing_files.append(file)
    
    if missing_files:
        print(f"Missing files for standalone loading: {missing_files}")
        print("Note: Model might still work if base model files are in subdirectories")
    else:
        print("All required files present for standalone loading!")
    
    return len(missing_files) == 0


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python test_pruning_automodel.py <path_to_saved_model>")
        sys.exit(1)
    
    model_path = sys.argv[1]
    
    if not Path(model_path).exists():
        print(f"Error: Model path does not exist: {model_path}")
        sys.exit(1)
    
    # Run tests
    success = test_automodel_loading(model_path)
    
    if success:
        # Check standalone capability
        check_standalone_loading(model_path)
        
        # Create usage docs
        create_example_usage_docs(model_path)
        
        print("\n✅ Model is compatible with AutoModel!")
        print("Users can load it with:")
        print(f'  model = AutoModelForSequenceClassification.from_pretrained("{model_path}", trust_remote_code=True)')
    else:
        print("\n❌ Model loading failed. Check the errors above.")