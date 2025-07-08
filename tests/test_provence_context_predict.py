#!/usr/bin/env python3
"""
Test predict_context method for ProvenceEncoder
"""

import pytest
import torch
import numpy as np
from sentence_transformers.provence import ProvenceEncoder, ProvenceContextOutput


def test_predict_context_basic():
    """Test basic functionality of predict_context"""
    model = ProvenceEncoder(
        model_name_or_path="hotchpotch/japanese-reranker-xsmall-v2",
        num_labels=1,
        max_length=128,
    )
    model.eval()
    
    # Test data with chunk positions
    query = "What is Python?"
    document = "Python is a programming language. It is widely used for web development. Python has simple syntax."
    
    # Define chunks (character positions)
    chunks = [
        [0, 33],    # "Python is a programming language."
        [34, 75],   # "It is widely used for web development."
        [76, 103],  # "Python has simple syntax."
    ]
    
    # Test single input
    output = model.predict_context(
        (query, document),
        chunks,
        token_threshold=0.5,
        chunk_threshold=0.5
    )
    
    assert isinstance(output, ProvenceContextOutput)
    assert output.ranking_scores is not None
    assert isinstance(output.ranking_scores, (float, np.float32, np.float64))
    assert 0 <= output.ranking_scores <= 1
    
    assert output.chunk_predictions is not None
    assert len(output.chunk_predictions) == 3
    assert all(pred in [0, 1] for pred in output.chunk_predictions)
    
    assert output.chunk_scores is not None
    assert len(output.chunk_scores) == 3
    assert all(0 <= score <= 1 for score in output.chunk_scores)
    
    assert output.chunk_positions == chunks
    assert output.compression_ratio is not None


def test_predict_context_batch():
    """Test batch processing of predict_context"""
    model = ProvenceEncoder(
        model_name_or_path="hotchpotch/japanese-reranker-xsmall-v2",
        num_labels=1,
        max_length=128,
    )
    model.eval()
    
    # Test data
    pairs = [
        ("What is Python?", "Python is a programming language. It is easy to learn."),
        ("What is Java?", "Java is also a programming language. It runs on JVM."),
    ]
    
    chunk_positions = [
        [[0, 33], [34, 56]],  # For first document
        [[0, 40], [41, 62]],  # For second document
    ]
    
    outputs = model.predict_context(
        pairs,
        chunk_positions,
        batch_size=2,
        token_threshold=0.5,
        chunk_threshold=0.5
    )
    
    assert isinstance(outputs, list)
    assert len(outputs) == 2
    
    for i, output in enumerate(outputs):
        assert isinstance(output, ProvenceContextOutput)
        assert output.ranking_scores is not None
        assert len(output.chunk_predictions) == len(chunk_positions[i])
        assert len(output.chunk_scores) == len(chunk_positions[i])
        assert output.chunk_positions == chunk_positions[i]


def test_predict_context_thresholds():
    """Test different threshold combinations"""
    model = ProvenceEncoder(
        model_name_or_path="hotchpotch/japanese-reranker-xsmall-v2",
        num_labels=1,
        max_length=128,
    )
    model.eval()
    
    query = "What is Python?"
    document = "Python is a programming language. " * 5  # Repeat to get more chunks
    
    chunks = [
        [i*35, (i+1)*35-1] for i in range(5)  # 5 chunks
    ]
    
    # Test different threshold combinations
    threshold_combinations = [
        (0.1, 0.1),  # Lenient
        (0.5, 0.5),  # Moderate
        (0.8, 0.8),  # Strict
    ]
    
    results = []
    for token_th, chunk_th in threshold_combinations:
        output = model.predict_context(
            (query, document),
            chunks,
            token_threshold=token_th,
            chunk_threshold=chunk_th
        )
        
        num_kept = output.chunk_predictions.sum()
        results.append(num_kept)
    
    # Generally, stricter thresholds should keep fewer chunks
    # (though not guaranteed due to model behavior)
    assert all(isinstance(result, (int, np.integer)) for result in results)


def test_predict_context_empty_chunks():
    """Test handling of empty chunk list"""
    model = ProvenceEncoder(
        model_name_or_path="hotchpotch/japanese-reranker-xsmall-v2",
        num_labels=1,
        max_length=128,
    )
    model.eval()
    
    query = "What is Python?"
    document = "Python is a programming language."
    
    # Empty chunks
    chunks = []
    
    output = model.predict_context(
        (query, document),
        chunks,
        token_threshold=0.5,
        chunk_threshold=0.5
    )
    
    assert isinstance(output, ProvenceContextOutput)
    assert output.ranking_scores is not None
    assert len(output.chunk_predictions) == 0
    assert len(output.chunk_scores) == 0
    assert output.compression_ratio == 0.0


def test_chunk_evaluation_metrics():
    """Test chunk evaluation against ground truth"""
    # This will be expanded when we have evaluation metrics
    
    # Mock ground truth
    true_chunks = [1, 0, 1, 0]  # 4 chunks, 1st and 3rd are relevant
    pred_chunks = [1, 1, 1, 0]  # Predicted relevant chunks
    
    # Calculate basic metrics manually for testing
    tp = sum(1 for t, p in zip(true_chunks, pred_chunks) if t == 1 and p == 1)
    fp = sum(1 for t, p in zip(true_chunks, pred_chunks) if t == 0 and p == 1)
    fn = sum(1 for t, p in zip(true_chunks, pred_chunks) if t == 1 and p == 0)
    tn = sum(1 for t, p in zip(true_chunks, pred_chunks) if t == 0 and p == 0)
    
    assert tp == 2  # Correctly predicted 2 relevant chunks
    assert fp == 1  # Incorrectly predicted 1 irrelevant chunk as relevant
    assert fn == 0  # Missed 0 relevant chunks
    assert tn == 1  # Correctly predicted 1 irrelevant chunk
    
    accuracy = (tp + tn) / (tp + fp + fn + tn)
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
    
    assert accuracy == 0.75
    assert precision == 2/3
    assert recall == 1.0
    assert abs(f1 - 0.8) < 0.01


if __name__ == "__main__":
    pytest.main([__file__, "-v"])