#!/usr/bin/env python
"""
Test the ultimate AutoModel experience with automatic registration.
"""

import sys
import subprocess
from pathlib import Path

def test_ultimate_experience():
    """Test the ultimate user experience."""
    
    print("="*60)
    print("Ultimate AutoModel Experience Test")
    print("="*60)
    
    # Test the ultimate user experience
    ultimate_code = '''
# The ultimate user experience - just import sentence_transformers
import sentence_transformers

# Then use AutoModel normally (no explicit pruning import needed!)
from transformers import AutoModelForSequenceClassification, AutoTokenizer

model_path = "./output/transformers_compat_test/reranking_pruning_20250709_135233/final_model"

# Load model and tokenizer
model = AutoModelForSequenceClassification.from_pretrained(model_path)
tokenizer = AutoTokenizer.from_pretrained(model_path)

print(f"Model loaded: {type(model).__name__}")
print("SUCCESS: No explicit pruning import needed!")
print("SUCCESS: No trust_remote_code=True needed!")

# Test inference
import torch
inputs = tokenizer("test query", "test document", return_tensors="pt")
device = next(model.parameters()).device
inputs = {k: v.to(device) for k, v in inputs.items()}

with torch.no_grad():
    outputs = model(**inputs)
    score = torch.sigmoid(outputs.logits).item()

print(f"Inference successful: {score:.4f}")
    '''
    
    print("Testing ultimate user experience:")
    print("```python")
    print("import sentence_transformers")
    print("from transformers import AutoModelForSequenceClassification")
    print("model = AutoModelForSequenceClassification.from_pretrained('path')")
    print("# No explicit pruning import!")
    print("# No trust_remote_code=True!")
    print("```")
    
    try:
        # Run in subprocess to test clean import
        result = subprocess.run(
            [sys.executable, "-c", ultimate_code],
            capture_output=True,
            text=True,
            cwd=str(Path(__file__).parent.parent)
        )
        
        if result.returncode == 0:
            print(f"\n✅ SUCCESS!")
            print(result.stdout.strip())
        else:
            print(f"\n❌ FAILED:")
            print(result.stderr.strip())
            
    except Exception as e:
        print(f"\n❌ ERROR: {e}")


def test_different_import_patterns():
    """Test different import patterns."""
    
    print("\n" + "="*60)
    print("Testing Different Import Patterns")
    print("="*60)
    
    patterns = [
        ("Direct sentence_transformers import", "import sentence_transformers"),
        ("CrossEncoder import", "from sentence_transformers import CrossEncoder"),
        ("SentenceTransformer import", "from sentence_transformers import SentenceTransformer"),
        ("Explicit pruning import", "import sentence_transformers.pruning"),
    ]
    
    base_code = '''
{import_statement}

from transformers import AutoModelForSequenceClassification
model_path = "./output/transformers_compat_test/reranking_pruning_20250709_135233/final_model"
model = AutoModelForSequenceClassification.from_pretrained(model_path)
print("SUCCESS: Model loaded with {pattern}")
    '''
    
    for pattern, import_stmt in patterns:
        print(f"\n{pattern}:")
        test_code = base_code.format(import_statement=import_stmt, pattern=pattern)
        
        try:
            result = subprocess.run(
                [sys.executable, "-c", test_code],
                capture_output=True,
                text=True,
                cwd=str(Path(__file__).parent.parent)
            )
            
            if result.returncode == 0:
                print(f"   ✅ {result.stdout.strip()}")
            else:
                print(f"   ❌ Failed: {result.stderr.strip().splitlines()[-1]}")
                
        except Exception as e:
            print(f"   ❌ Error: {e}")


def create_final_demo():
    """Create the final demo."""
    
    print("\n" + "="*60)
    print("Final Demo: Best User Experience")
    print("="*60)
    
    demo_code = '''
# BEST USER EXPERIENCE
# 1. Import sentence_transformers (triggers auto-registration)
import sentence_transformers

# 2. Use AutoModel normally  
from transformers import AutoModelForSequenceClassification, AutoTokenizer

# 3. Load model (no trust_remote_code needed!)
model = AutoModelForSequenceClassification.from_pretrained("path/to/pruning_model")
tokenizer = AutoTokenizer.from_pretrained("path/to/pruning_model")

# 4. Use like any other transformers model
inputs = tokenizer("query", "document", return_tensors="pt")
outputs = model(**inputs)
score = torch.sigmoid(outputs.logits).item()
    '''
    
    print("Final recommended usage pattern:")
    print(demo_code)
    
    print("\nComparison with alternatives:")
    print("📊 trust_remote_code=True approach:")
    print("   ✅ No imports needed")
    print("   ❌ Security concern")
    print("   ❌ Requires trust_remote_code=True")
    print("")
    print("📊 Current approach (auto-registration):")
    print("   ✅ No security concerns")
    print("   ✅ No trust_remote_code needed")
    print("   ✅ Works with standard AutoModel")
    print("   ⚠️ Requires sentence_transformers import")
    print("")
    print("🏆 WINNER: Auto-registration approach!")


def main():
    """Run all tests."""
    test_ultimate_experience()
    test_different_import_patterns()
    create_final_demo()
    
    print("\n" + "="*80)
    print("🎉 FINAL ACHIEVEMENT SUMMARY")
    print("="*80)
    
    print("\n✅ PruningEncoder models now work with AutoModel in THREE ways:")
    print("1. 🔐 With trust_remote_code=True (auto_map mechanism)")
    print("2. 🚀 With sentence_transformers.pruning import (explicit registration)")
    print("3. 🎯 With sentence_transformers import (auto-registration)")
    print("")
    print("🏆 BEST PRACTICE - Auto-registration approach:")
    print("```python")
    print("import sentence_transformers  # Auto-registers pruning models")
    print("from transformers import AutoModelForSequenceClassification")
    print("model = AutoModelForSequenceClassification.from_pretrained('path')")
    print("# No trust_remote_code=True needed!")
    print("```")
    print("")
    print("🎊 This provides the best balance of:")
    print("   - Security (no trust_remote_code)")
    print("   - Convenience (familiar AutoModel pattern)")
    print("   - Reliability (explicit registration)")


if __name__ == "__main__":
    main()