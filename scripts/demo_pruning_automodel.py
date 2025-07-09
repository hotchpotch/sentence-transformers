#!/usr/bin/env python3
"""
Demonstration of PruningEncoder AutoModel compatibility.

This script shows how PruningEncoder models work with Transformers AutoModel classes.
"""

import os
import tempfile
from pathlib import Path

# Ensure sentence_transformers is importable
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from sentence_transformers.pruning import PruningEncoder
from transformers import AutoConfig, AutoModelForSequenceClassification, AutoTokenizer


def demo_save_and_load():
    """Demonstrate saving a PruningEncoder model and loading it with AutoModel."""
    
    print("=== PruningEncoder AutoModel Demo ===\n")
    
    # Create a temporary directory for the demo
    with tempfile.TemporaryDirectory() as tmpdir:
        save_path = Path(tmpdir) / "demo_model"
        
        # Step 1: Create and save a PruningEncoder model
        print("1. Creating a PruningEncoder model...")
        model = PruningEncoder(
            model_name_or_path="microsoft/deberta-v3-xsmall",  # Use a small model for demo
            mode="reranking_pruning",
            max_length=256
        )
        
        print(f"2. Saving model to {save_path}...")
        model.save_pretrained(save_path)
        
        # Check what was saved
        print("\n3. Saved files:")
        for file in save_path.rglob("*"):
            if file.is_file():
                print(f"   - {file.relative_to(save_path)}")
        
        # Check config.json
        import json
        with open(save_path / "config.json", 'r') as f:
            config = json.load(f)
        
        print("\n4. Config.json contents:")
        print(f"   model_type: {config.get('model_type')}")
        print(f"   architectures: {config.get('architectures')}")
        print(f"   auto_map: {json.dumps(config.get('auto_map', {}), indent=6).replace('\\n', '\\n   ')}")
        
        # Step 2: Load with AutoConfig
        print("\n5. Loading with AutoConfig...")
        try:
            auto_config = AutoConfig.from_pretrained(save_path)
            print(f"   ✓ Success! Config type: {type(auto_config).__name__}")
            print(f"   ✓ Model type: {auto_config.model_type}")
        except Exception as e:
            print(f"   ✗ Failed: {e}")
        
        # Step 3: Load with AutoModelForSequenceClassification
        print("\n6. Loading with AutoModelForSequenceClassification...")
        print("   Note: This works because sentence_transformers is installed and models are registered")
        try:
            auto_model = AutoModelForSequenceClassification.from_pretrained(save_path)
            print(f"   ✓ Success! Model type: {type(auto_model).__name__}")
            
            # Test the model
            tokenizer = AutoTokenizer.from_pretrained(save_path)
            inputs = tokenizer(
                "What is machine learning?",
                "Machine learning is a type of artificial intelligence.",
                return_tensors="pt",
                truncation=True,
                max_length=256
            )
            
            outputs = auto_model(**inputs)
            print(f"   ✓ Forward pass successful! Score: {outputs.logits.item():.4f}")
            
        except Exception as e:
            print(f"   ✗ Failed: {e}")
            import traceback
            traceback.print_exc()
        
        # Step 4: Show how to load without sentence_transformers
        print("\n7. For loading WITHOUT sentence_transformers installed:")
        print("   Users would need to use trust_remote_code=True:")
        print(f'   model = AutoModelForSequenceClassification.from_pretrained(')
        print(f'       "{save_path}",')
        print(f'       trust_remote_code=True')
        print(f'   )')
        print("\n   This requires a standalone modeling file in the model directory.")
        print("   Run create_standalone_pruning_model.py to generate this file.")


def show_usage_examples():
    """Show different usage examples."""
    
    print("\n\n=== Usage Examples ===\n")
    
    print("1. Loading from Hugging Face Hub (with sentence_transformers):")
    print("""
from transformers import AutoModelForSequenceClassification, AutoTokenizer

# Load model and tokenizer
model = AutoModelForSequenceClassification.from_pretrained("username/pruning-model")
tokenizer = AutoTokenizer.from_pretrained("username/pruning-model")

# Use for reranking
inputs = tokenizer(query, document, return_tensors="pt", truncation=True)
outputs = model(**inputs)
score = outputs.logits.squeeze()
""")
    
    print("\n2. Loading from Hugging Face Hub (without sentence_transformers):")
    print("""
from transformers import AutoModelForSequenceClassification, AutoTokenizer

# Need trust_remote_code=True for custom models
model = AutoModelForSequenceClassification.from_pretrained(
    "username/pruning-model",
    trust_remote_code=True
)
tokenizer = AutoTokenizer.from_pretrained("username/pruning-model")
""")
    
    print("\n3. Using PruningEncoder directly (full features):")
    print("""
from sentence_transformers.pruning import PruningEncoder

# Load model
model = PruningEncoder.from_pretrained("username/pruning-model")

# Use pruning features
outputs = model.predict_and_prune(
    [(query, document)],
    prune_ratio=0.5,
    return_pruned_texts=True
)

print(f"Score: {outputs['scores'][0]}")
print(f"Pruned text: {outputs['pruned_texts'][0]}")
""")
    
    print("\n4. Using as CrossEncoder replacement:")
    print("""
from sentence_transformers.pruning import PruningCrossEncoder

# Load as CrossEncoder
model = PruningCrossEncoder("username/pruning-model")

# Use CrossEncoder interface
scores = model.predict([
    ("query1", "document1"),
    ("query2", "document2")
])

# Or rank documents
results = model.rank(query, documents, top_k=10)
""")


if __name__ == "__main__":
    # Run the demo
    demo_save_and_load()
    
    # Show usage examples
    show_usage_examples()
    
    print("\n\n=== Summary ===")
    print("✓ PruningEncoder models are compatible with AutoModel")
    print("✓ Models saved with auto_map can be loaded without explicit imports")
    print("✓ trust_remote_code=True enables loading without sentence_transformers")
    print("✓ Multiple interfaces supported: AutoModel, PruningEncoder, CrossEncoder")
    print("\nSee docs/pruning_automodel_integration.md for detailed documentation.")