#!/usr/bin/env python3
"""
Test unified predict method for ProvenceEncoder
"""

import pytest
import torch
import numpy as np
from sentence_transformers.provence import ProvenceEncoder, ProvenceOutput


def test_predict_ranking_only():
    """Test predict with apply_pruning=False (ranking only)"""
    model = ProvenceEncoder(
        model_name_or_path="hotchpotch/japanese-reranker-xsmall-v2",
        num_labels=1,
        max_length=128,
    )
    model.eval()
    
    # Test data
    pairs = [
        ("What is Python?", "Python is a programming language."),
        ("What is Java?", "Java is also a programming language."),
    ]
    
    # Test with apply_pruning=False
    scores = model.predict(
        pairs,
        apply_pruning=False,
        convert_to_numpy=True
    )
    
    assert isinstance(scores, np.ndarray)
    assert len(scores) == 2
    assert all(0 <= s <= 1 for s in scores)
    
    # Test with convert_to_tensor
    scores_tensor = model.predict(
        pairs,
        apply_pruning=False,
        convert_to_tensor=True,
        convert_to_numpy=False
    )
    
    assert isinstance(scores_tensor, torch.Tensor)
    assert scores_tensor.shape[0] == 2


def test_predict_with_pruning():
    """Test predict with apply_pruning=True (full functionality)"""
    model = ProvenceEncoder(
        model_name_or_path="hotchpotch/japanese-reranker-xsmall-v2",
        num_labels=1,
        max_length=128,
    )
    model.eval()
    
    # Test data
    pairs = [
        ("What is Python?", "Python is a programming language. It is widely used for web development. Python has simple syntax."),
        ("What is Java?", "Java is also a programming language. It runs on JVM. Java is object-oriented."),
    ]
    
    # Test with apply_pruning=True
    outputs = model.predict(
        pairs,
        apply_pruning=True,
        pruning_threshold=0.5,
        return_documents=True
    )
    
    assert isinstance(outputs, list)
    assert len(outputs) == 2
    assert all(isinstance(o, ProvenceOutput) for o in outputs)
    
    # Check each output
    for output in outputs:
        assert output.ranking_scores is not None
        assert isinstance(output.ranking_scores, (float, np.float32, np.float64))
        assert 0 <= output.ranking_scores <= 1
        
        assert output.compression_ratio is not None
        assert 0 <= output.compression_ratio <= 1
        
        assert output.pruned_documents is not None
        assert isinstance(output.pruned_documents, list)


def test_predict_single_input():
    """Test predict with single input"""
    model = ProvenceEncoder(
        model_name_or_path="hotchpotch/japanese-reranker-xsmall-v2",
        num_labels=1,
        max_length=128,
    )
    model.eval()
    
    # Single pair
    pair = ("What is Python?", "Python is a programming language.")
    
    # Test ranking only
    score = model.predict(
        pair,
        apply_pruning=False,
        convert_to_numpy=False
    )
    assert isinstance(score, list)
    assert len(score) == 1
    
    # Test with pruning
    output = model.predict(
        pair,
        apply_pruning=True,
        return_documents=True
    )
    assert isinstance(output, ProvenceOutput)
    assert output.ranking_scores is not None


def test_predict_batch_processing():
    """Test batch processing in predict"""
    model = ProvenceEncoder(
        model_name_or_path="hotchpotch/japanese-reranker-xsmall-v2",
        num_labels=1,
        max_length=128,
    )
    model.eval()
    
    # Create many pairs
    pairs = [
        (f"Question {i}?", f"Answer {i}.")
        for i in range(10)
    ]
    
    # Test with small batch size
    outputs = model.predict(
        pairs,
        batch_size=3,
        apply_pruning=True
    )
    
    assert len(outputs) == 10
    assert all(isinstance(o, ProvenceOutput) for o in outputs)


def test_predict_different_thresholds():
    """Test different pruning thresholds"""
    model = ProvenceEncoder(
        model_name_or_path="hotchpotch/japanese-reranker-xsmall-v2",
        num_labels=1,
        max_length=128,
    )
    model.eval()
    
    pair = ("What is Python?", "Python is a programming language. " * 10)
    
    # Test different thresholds
    thresholds = [0.1, 0.3, 0.5, 0.7, 0.9]
    compression_ratios = []
    
    for threshold in thresholds:
        output = model.predict(
            pair,
            apply_pruning=True,
            pruning_threshold=threshold,
            return_documents=True
        )
        compression_ratios.append(output.compression_ratio)
    
    # Higher thresholds should generally lead to more compression
    # (though not strictly monotonic due to token boundaries)
    assert compression_ratios[0] <= compression_ratios[-1] + 0.1  # Allow small variance


if __name__ == "__main__":
    pytest.main([__file__, "-v"])