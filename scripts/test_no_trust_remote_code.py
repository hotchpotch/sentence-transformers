#!/usr/bin/env python
"""
Test loading PruningEncoder without trust_remote_code=True
"""

import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def test_without_trust_remote_code():
    """Test loading PruningEncoder without trust_remote_code=True"""
    
    print("="*60)
    print("Testing PruningEncoder without trust_remote_code=True")
    print("="*60)
    
    # Method 1: Import pruning module first to register models
    print("\n1. Testing with sentence_transformers.pruning import...")
    try:
        # Import to trigger registration
        import sentence_transformers.pruning
        
        from transformers import AutoModelForSequenceClassification, AutoTokenizer
        
        model_path = "./output/transformers_compat_test/reranking_pruning_20250709_135233/final_model"
        
        # Load without trust_remote_code
        print("   Loading model...")
        model = AutoModelForSequenceClassification.from_pretrained(model_path)
        tokenizer = AutoTokenizer.from_pretrained(model_path)
        
        print(f"   ✓ Success! Model type: {type(model).__name__}")
        
        # Test inference
        inputs = tokenizer("test query", "test document", return_tensors="pt")
        
        # Move inputs to model device
        device = next(model.parameters()).device
        inputs = {k: v.to(device) for k, v in inputs.items()}
        
        with torch.no_grad():
            outputs = model(**inputs)
        print(f"   ✓ Inference successful! Shape: {outputs.logits.shape}")
        
        return True
        
    except Exception as e:
        print(f"   ✗ Failed: {e}")
        return False


def test_different_approaches():
    """Test different approaches to avoid trust_remote_code"""
    
    print("\n" + "="*60)
    print("Testing Different Approaches")
    print("="*60)
    
    approaches = [
        ("1. Direct import + AutoModel", test_direct_import),
        ("2. Monkey patch CONFIG_MAPPING", test_monkey_patch),
        ("3. Architecture spoofing", test_architecture_spoofing),
    ]
    
    results = {}
    for name, test_func in approaches:
        print(f"\n{name}:")
        try:
            success = test_func()
            results[name] = success
            print(f"   {'✓' if success else '✗'} {'Success' if success else 'Failed'}")
        except Exception as e:
            print(f"   ✗ Error: {e}")
            results[name] = False
    
    return results


def test_direct_import():
    """Test with direct import approach"""
    import sentence_transformers.pruning
    from transformers import AutoModelForSequenceClassification
    
    model_path = "./output/transformers_compat_test/reranking_pruning_20250709_135233/final_model"
    model = AutoModelForSequenceClassification.from_pretrained(model_path)
    return True


def test_monkey_patch():
    """Test with monkey patching approach"""
    from transformers.models.auto.configuration_auto import CONFIG_MAPPING
    from transformers.models.auto.modeling_auto import MODEL_FOR_SEQUENCE_CLASSIFICATION_MAPPING
    
    # Import our classes
    from sentence_transformers.pruning.transformers_compat import (
        PruningEncoderConfig, 
        PruningEncoderForSequenceClassification
    )
    
    # Monkey patch the mappings
    CONFIG_MAPPING.register("pruning_encoder", PruningEncoderConfig)
    MODEL_FOR_SEQUENCE_CLASSIFICATION_MAPPING.register(
        PruningEncoderConfig, 
        PruningEncoderForSequenceClassification
    )
    
    from transformers import AutoModelForSequenceClassification
    model_path = "./output/transformers_compat_test/reranking_pruning_20250709_135233/final_model"
    model = AutoModelForSequenceClassification.from_pretrained(model_path)
    return True


def test_architecture_spoofing():
    """Test by making the model appear as a known architecture"""
    # This would require modifying the config.json to use a known model_type
    # like "bert" or "roberta" but this would lose custom functionality
    print("     (This approach loses custom functionality - skipping)")
    return False


def main():
    """Run all tests"""
    
    # Test 1: Basic approach
    basic_success = test_without_trust_remote_code()
    
    # Test 2: Different approaches
    approach_results = test_different_approaches()
    
    # Summary
    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    
    print(f"\nBasic approach (import + AutoModel): {'✓ PASS' if basic_success else '✗ FAIL'}")
    
    for approach, success in approach_results.items():
        print(f"{approach}: {'✓ PASS' if success else '✗ FAIL'}")
    
    print("\n" + "="*60)
    print("CONCLUSION")
    print("="*60)
    
    if basic_success:
        print("🎉 PruningEncoder works WITHOUT trust_remote_code=True!")
        print("\nRequired steps:")
        print("1. Install sentence_transformers package")
        print("2. Import sentence_transformers.pruning before using AutoModel")
        print("3. Use AutoModel normally")
        print("\nExample:")
        print("```python")
        print("import sentence_transformers.pruning")
        print("from transformers import AutoModelForSequenceClassification")
        print("model = AutoModelForSequenceClassification.from_pretrained('path/to/model')")
        print("# No trust_remote_code=True needed!")
        print("```")
    else:
        print("❌ Current approach still requires trust_remote_code=True")


if __name__ == "__main__":
    main()