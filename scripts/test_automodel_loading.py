#!/usr/bin/env python
"""
Test script to verify AutoModel loading works correctly for PruningEncoder.
"""

import tempfile
import shutil
from pathlib import Path
from transformers import AutoConfig, AutoModelForSequenceClassification, AutoModelForTokenClassification
from sentence_transformers.pruning import PruningEncoder


def test_automodel_loading():
    """Test that models can be loaded via AutoModel after saving."""
    
    print("Testing AutoModel loading for PruningEncoder...")
    
    # Create a temporary directory for testing
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir)
        
        # Test 1: Reranking + Pruning mode
        print("\n1. Testing reranking_pruning mode...")
        model_path = temp_path / "test_reranking_pruning"
        
        # Create and save a model
        encoder = PruningEncoder(
            model_name_or_path="hotchpotch/japanese-reranker-xsmall-v2",
            mode="reranking_pruning",
            max_length=512
        )
        encoder.save_pretrained(model_path)
        
        # Check that the required files exist
        assert (model_path / "config.json").exists(), "config.json not found"
        assert (model_path / "modeling_pruning_encoder.py").exists(), "modeling file not found"
        
        # Load config via AutoConfig
        try:
            config = AutoConfig.from_pretrained(model_path, trust_remote_code=True)
            print(f"✓ AutoConfig loaded successfully: {config.model_type}")
        except Exception as e:
            print(f"✗ AutoConfig loading failed: {e}")
            raise
        
        # Load model via AutoModelForSequenceClassification
        try:
            model = AutoModelForSequenceClassification.from_pretrained(
                model_path, 
                trust_remote_code=True
            )
            print(f"✓ AutoModelForSequenceClassification loaded successfully")
        except Exception as e:
            print(f"✗ AutoModelForSequenceClassification loading failed: {e}")
            raise
        
        # Test 2: Pruning only mode
        print("\n2. Testing pruning_only mode...")
        model_path2 = temp_path / "test_pruning_only"
        
        # Create and save a model
        encoder2 = PruningEncoder(
            model_name_or_path="hotchpotch/japanese-reranker-xsmall-v2",
            mode="pruning_only",
            max_length=512
        )
        encoder2.save_pretrained(model_path2)
        
        # Load model via AutoModelForTokenClassification
        try:
            model2 = AutoModelForTokenClassification.from_pretrained(
                model_path2,
                trust_remote_code=True
            )
            print(f"✓ AutoModelForTokenClassification loaded successfully")
        except Exception as e:
            print(f"✗ AutoModelForTokenClassification loading failed: {e}")
            raise
        
        print("\n✅ All tests passed! AutoModel loading works correctly.")
        
        # Test 3: Check the config content
        print("\n3. Checking config.json content...")
        import json
        with open(model_path / "config.json", 'r') as f:
            config_dict = json.load(f)
        
        print(f"Model type: {config_dict.get('model_type')}")
        print(f"Auto map entries:")
        for key, value in config_dict.get('auto_map', {}).items():
            print(f"  {key}: {value}")


if __name__ == "__main__":
    test_automodel_loading()