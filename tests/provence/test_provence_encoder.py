"""
Unit tests for ProvenceEncoder.
"""

import tempfile
from pathlib import Path

import numpy as np
import pytest
import torch

from sentence_transformers.provence import ProvenceEncoder, ProvenceOutput


class TestProvenceEncoder:
    """Test ProvenceEncoder functionality."""
    
    @pytest.fixture
    def small_model_name(self):
        """Use a small model for testing."""
        return "sentence-transformers-testing/stsb-bert-tiny-safetensors"
    
    @pytest.fixture
    def encoder(self, small_model_name):
        """Create a ProvenceEncoder instance."""
        return ProvenceEncoder(
            model_name_or_path=small_model_name,
            num_labels=1,
            max_length=128,
            pruning_config={
                "dropout": 0.1,
                "sentence_pooling": "mean"
            }
        )
    
    def test_encoder_initialization(self, encoder):
        """Test ProvenceEncoder initialization."""
        assert encoder is not None
        assert encoder.num_labels == 1
        assert encoder.max_length == 128
        assert encoder.ranking_model is not None
        assert encoder.pruning_head is not None
        assert encoder.tokenizer is not None
        assert encoder.text_chunker is not None
        assert hasattr(encoder, 'activation_fn')
    
    def test_forward_pass(self, encoder):
        """Test forward pass through the encoder."""
        # Create dummy inputs
        batch_size = 2
        seq_length = 20
        
        input_ids = torch.randint(0, 1000, (batch_size, seq_length))
        attention_mask = torch.ones(batch_size, seq_length)
        
        # Move to device
        input_ids = input_ids.to(encoder.device)
        attention_mask = attention_mask.to(encoder.device)
        
        # Forward pass
        outputs = encoder.forward(
            input_ids=input_ids,
            attention_mask=attention_mask
        )
        
        # Check outputs
        assert "ranking_logits" in outputs
        assert "pruning_logits" in outputs
        assert "hidden_states" in outputs
        
        assert outputs["ranking_logits"].shape == (batch_size, 1)
        assert outputs["pruning_logits"].shape == (batch_size, seq_length, 2)
        assert outputs["hidden_states"].shape == (batch_size, seq_length, encoder.config.hidden_size)
    
    def test_predict_ranking(self, encoder):
        """Test ranking prediction."""
        sentences = [
            ("What is AI?", "AI stands for Artificial Intelligence."),
            ("What is ML?", "ML is Machine Learning.")
        ]
        
        scores = encoder.predict(sentences, batch_size=2)
        
        assert len(scores) == 2
        assert all(isinstance(score, float) for score in scores)
        assert all(0 <= score <= 1 for score in scores)  # Sigmoid activation
    
    def test_predict_single_pair(self, encoder):
        """Test prediction with single pair."""
        query = "What is Python?"
        document = "Python is a programming language."
        
        score = encoder.predict((query, document))
        
        assert isinstance(score, (list, np.ndarray))
        if isinstance(score, list):
            assert len(score) == 1
            score = score[0]
        else:
            assert score.shape == () or score.shape == (1,)
            score = score.item() if hasattr(score, 'item') else float(score)
        
        assert 0 <= score <= 1
    
    def test_predict_with_pruning(self, encoder):
        """Test prediction with pruning."""
        sentences = [
            ("What is AI?", "AI stands for Artificial Intelligence. It is a field of computer science. The weather is nice today."),
            ("What is ML?", "Machine Learning is a subset of AI. It uses algorithms to learn from data.")
        ]
        
        outputs = encoder.predict_with_pruning(
            sentences,
            batch_size=2,
            pruning_threshold=0.5,
            return_documents=True
        )
        
        assert len(outputs) == 2
        assert all(isinstance(output, ProvenceOutput) for output in outputs)
        
        for output in outputs:
            assert hasattr(output, 'ranking_scores')
            assert hasattr(output, 'pruning_masks')
            assert hasattr(output, 'sentences')
            assert hasattr(output, 'compression_ratio')
            assert hasattr(output, 'pruned_documents')
            
            assert 0 <= output.compression_ratio <= 1
            assert len(output.sentences[0]) == len(output.pruning_masks[0])
    
    def test_prune_method(self, encoder):
        """Test the prune method."""
        query = "What is machine learning?"
        document = "Machine learning is a subset of AI. It allows computers to learn from data. The sky is blue. Birds can fly."
        
        # Test basic pruning
        pruned = encoder.prune(query, document, threshold=0.5)
        assert isinstance(pruned, str)
        assert len(pruned) <= len(document)
        
        # Test detailed pruning
        result = encoder.prune(
            query, 
            document, 
            threshold=0.5, 
            return_sentences=True
        )
        
        assert isinstance(result, dict)
        assert "pruned_document" in result
        assert "sentences" in result
        assert "pruning_masks" in result
        assert "ranking_score" in result
        assert "compression_ratio" in result
        assert "num_pruned_sentences" in result
        
        assert len(result["sentences"]) == len(result["pruning_masks"])
        assert 0 <= result["compression_ratio"] <= 1
        assert result["num_pruned_sentences"] >= 0
    
    def test_save_load(self, encoder):
        """Test saving and loading the encoder."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            save_path = Path(tmp_dir) / "test_provence_encoder"
            
            # Save
            encoder.save_pretrained(save_path)
            
            # Check files exist
            assert (save_path / "config.json").exists()
            assert (save_path / "ranking_model").exists()
            assert (save_path / "pruning_head").exists()
            
            # Load
            loaded_encoder = ProvenceEncoder.from_pretrained(save_path)
            
            # Check loaded model works
            query = "Test query"
            document = "Test document"
            
            original_score = encoder.predict((query, document))[0]
            loaded_score = loaded_encoder.predict((query, document))[0]
            
            # Scores should be very close (allowing for floating point differences)
            assert abs(original_score - loaded_score) < 1e-4
    
    def test_device_handling(self):
        """Test device handling."""
        device = "cuda" if torch.cuda.is_available() else "cpu"
        
        encoder = ProvenceEncoder(
            model_name_or_path="sentence-transformers-testing/stsb-bert-tiny-safetensors",
            device=device
        )
        
        assert str(encoder.device) == device
        assert next(encoder.ranking_model.parameters()).device.type == device
        assert next(encoder.pruning_head.parameters()).device.type == device
    
    def test_batch_processing(self, encoder):
        """Test batch processing with different batch sizes."""
        # Create multiple sentence pairs
        sentences = [
            (f"Query {i}", f"Document {i} with some content.")
            for i in range(10)
        ]
        
        # Test with different batch sizes
        for batch_size in [1, 3, 10]:
            scores = encoder.predict(
                sentences,
                batch_size=batch_size,
                show_progress_bar=False
            )
            
            assert len(scores) == 10
            assert all(isinstance(score, float) for score in scores)
    
    def test_multilingual_support(self, encoder):
        """Test multilingual text support."""
        sentences = [
            ("What is AI?", "AI stands for Artificial Intelligence."),
            ("AIとは何ですか？", "AIは人工知能の略です。"),
            ("什么是AI？", "AI是人工智能的缩写。")
        ]
        
        outputs = encoder.predict_with_pruning(
            sentences,
            return_documents=True
        )
        
        assert len(outputs) == 3
        for output in outputs:
            assert len(output.sentences[0]) > 0
            assert len(output.pruning_masks[0]) > 0


if __name__ == "__main__":
    pytest.main([__file__])