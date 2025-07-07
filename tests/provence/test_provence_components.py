"""
Unit tests for Provence components.
"""

from __future__ import annotations

import tempfile
from pathlib import Path

import numpy as np
import pytest
import torch

from sentence_transformers.provence.data_structures import ProvenceOutput, ProvenceConfig
from sentence_transformers.provence.models import ProvencePruningHead, ProvencePruningConfig
from sentence_transformers.utils.text_chunking import MultilingualChunker
from tests.utils import SafeTemporaryDirectory


class TestProvenceConfig:
    """Test ProvenceConfig data structure."""
    
    def test_provence_config_default_values(self):
        """Test default values of ProvenceConfig."""
        config = ProvenceConfig()
        assert config.pruning_hidden_size is None
        assert config.pruning_num_labels == 2
        assert config.pruning_dropout == 0.1
        assert config.chunker_type == "multilingual"
        assert config.max_sentences == 64
        assert config.min_sentence_length == 5
        assert config.max_sentence_length == 500
        assert config.pruning_mode == "sentence"
        assert config.default_pruning_threshold == 0.5
        assert config.min_sentences_to_keep == 1
        assert config.use_cache is True
        assert config.batch_size == 32
    
    def test_provence_config_to_dict(self):
        """Test converting ProvenceConfig to dictionary."""
        config = ProvenceConfig(
            pruning_hidden_size=768,
            pruning_dropout=0.2,
            chunker_type="simple"
        )
        config_dict = config.to_dict()
        assert config_dict["pruning_hidden_size"] == 768
        assert config_dict["pruning_dropout"] == 0.2
        assert config_dict["chunker_type"] == "simple"


class TestProvenceOutput:
    """Test ProvenceOutput data structure."""
    
    def test_provence_output_creation(self):
        """Test creating ProvenceOutput with various fields."""
        output = ProvenceOutput(
            ranking_scores=np.array([0.8, 0.6]),
            pruning_masks=np.array([[True, False, True], [True, True, False]]),
            sentences=[["Sentence 1.", "Sentence 2.", "Sentence 3."]],
            compression_ratio=0.67,
            num_pruned_sentences=1
        )
        assert output.ranking_scores.shape == (2,)
        assert output.pruning_masks.shape == (2, 3)
        assert len(output.sentences) == 1
        assert output.compression_ratio == 0.67
        assert output.num_pruned_sentences == 1
    
    def test_provence_output_to_dict(self):
        """Test converting ProvenceOutput to dictionary."""
        output = ProvenceOutput(
            ranking_scores=np.array([0.9]),
            pruning_masks=np.array([[True, False]]),
            compression_ratio=0.5
        )
        output_dict = output.to_dict()
        assert "ranking_scores" in output_dict
        assert output_dict["ranking_scores"] == [0.9]  # Converted to list (1D array)
        assert output_dict["pruning_masks"] == [[True, False]]  # 2D array
        assert output_dict["compression_ratio"] == 0.5
    
    def test_provence_output_repr(self):
        """Test string representation of ProvenceOutput."""
        output = ProvenceOutput(
            ranking_scores=np.array([0.9]),
            pruning_masks=np.array([[True, False]]),
            compression_ratio=0.5
        )
        repr_str = repr(output)
        assert "ranking_scores=(1,)" in repr_str
        assert "pruning_masks=(1, 2)" in repr_str
        assert "compression_ratio=0.50" in repr_str


class TestProvencePruningHead:
    """Test ProvencePruningHead module."""
    
    def test_pruning_head_config(self):
        """Test ProvencePruningConfig creation and properties."""
        config = ProvencePruningConfig(
            hidden_size=768,
            num_labels=2,
            classifier_dropout=0.1,
            sentence_pooling="mean",
            use_weighted_pooling=False
        )
        assert config.hidden_size == 768
        assert config.num_labels == 2
        assert config.classifier_dropout == 0.1
        assert config.sentence_pooling == "mean"
        assert config.use_weighted_pooling is False
    
    def test_pruning_head_initialization(self):
        """Test ProvencePruningHead initialization."""
        config = ProvencePruningConfig(hidden_size=256)
        head = ProvencePruningHead(config)
        
        assert isinstance(head.dropout, torch.nn.Dropout)
        assert head.dropout.p == 0.1
        assert isinstance(head.classifier, torch.nn.Linear)
        assert head.classifier.in_features == 256
        assert head.classifier.out_features == 2
        assert head.num_labels == 2
        assert head.sentence_pooling == "mean"
    
    def test_pruning_head_forward_token_level(self):
        """Test forward pass without sentence boundaries (token-level)."""
        config = ProvencePruningConfig(hidden_size=256)
        head = ProvencePruningHead(config)
        
        batch_size = 2
        seq_len = 10
        hidden_size = 256
        
        # Create dummy inputs
        hidden_states = torch.randn(batch_size, seq_len, hidden_size)
        attention_mask = torch.ones(batch_size, seq_len)
        
        # Forward pass
        outputs = head(hidden_states=hidden_states, attention_mask=attention_mask)
        
        assert hasattr(outputs, "logits")
        assert outputs.logits.shape == (batch_size, seq_len, 2)
        assert outputs.loss is None  # No labels provided
    
    def test_pruning_head_forward_with_labels(self):
        """Test forward pass with labels (token-level loss)."""
        config = ProvencePruningConfig(hidden_size=256)
        head = ProvencePruningHead(config)
        
        batch_size = 2
        seq_len = 10
        hidden_size = 256
        
        # Create dummy inputs
        hidden_states = torch.randn(batch_size, seq_len, hidden_size)
        attention_mask = torch.ones(batch_size, seq_len)
        labels = torch.randint(0, 2, (batch_size, seq_len))
        
        # Forward pass
        outputs = head(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            labels=labels
        )
        
        assert outputs.loss is not None
        assert outputs.loss.requires_grad
        assert outputs.logits.shape == (batch_size, seq_len, 2)
    
    def test_pruning_head_save_load(self):
        """Test saving and loading ProvencePruningHead."""
        config = ProvencePruningConfig(hidden_size=256, sentence_pooling="max")
        head = ProvencePruningHead(config)
        
        with SafeTemporaryDirectory() as tmp_dir:
            # Save
            head.save_pretrained(tmp_dir)
            
            # Check files exist
            assert Path(tmp_dir, "config.json").exists()
            assert Path(tmp_dir, "pytorch_model.bin").exists()
            
            # Load
            loaded_head = ProvencePruningHead.from_pretrained(tmp_dir)
            
            # Check config is preserved
            assert loaded_head.config.hidden_size == 256
            assert loaded_head.config.sentence_pooling == "max"
            
            # Check weights are the same
            for (name1, param1), (name2, param2) in zip(
                head.named_parameters(), loaded_head.named_parameters()
            ):
                assert name1 == name2
                assert torch.allclose(param1, param2)


class TestTextChunking:
    """Test text chunking utilities."""
    
    def test_multilingual_chunker_creation(self):
        """Test creating MultilingualChunker."""
        chunker = MultilingualChunker()
        assert chunker is not None
        assert chunker._default_chunker is not None
        assert len(chunker._chunkers) == 0  # Lazy loading
    
    def test_chunker_english(self):
        """Test chunking English text."""
        chunker = MultilingualChunker()
        text = "Hello world. This is a test. How are you?"
        
        result = chunker.chunk_text(text, language="en")
        assert len(result) == 3
        
        # Check sentences
        sentences = [chunk for chunk, _ in result]
        assert "Hello world." in sentences[0]
        assert "This is a test." in sentences[1]
        assert "How are you?" in sentences[2]
        
        # Check positions
        for chunk, (start, end) in result:
            assert text[start:end].strip() == chunk.strip()
    
    def test_chunker_japanese(self):
        """Test chunking Japanese text."""
        chunker = MultilingualChunker()
        text = "こんにちは。これはテストです。元気ですか？"
        
        result = chunker.chunk_text(text, language="ja")
        assert len(result) >= 2  # At least 2 sentences
        
        # Check that positions are valid
        for chunk, (start, end) in result:
            assert start >= 0
            assert end <= len(text)
            assert start < end
    
    def test_chunker_auto_detect(self):
        """Test automatic language detection."""
        chunker = MultilingualChunker()
        
        # English text
        en_text = "Hello world. This is a test."
        result = chunker.chunk_text(en_text, language="auto")
        assert len(result) >= 2
        
        # Japanese text
        ja_text = "こんにちは。これはテストです。"
        result = chunker.chunk_text(ja_text, language="auto")
        assert len(result) >= 1
    
    def test_chunker_reconstruct_text(self):
        """Test reconstructing text from chunks and masks."""
        chunker = MultilingualChunker()
        
        # Test simple reconstruction
        sentences = ["Hello world.", "This is a test.", "How are you?"]
        masks = [True, False, True]
        
        result = chunker.reconstruct_text(sentences, masks)
        assert "Hello world." in result
        assert "This is a test." not in result
        assert "How are you?" in result
    
    def test_chunker_supported_languages(self):
        """Test getting supported languages."""
        supported = MultilingualChunker.get_supported_languages()
        assert "ja" in supported
        assert "en" in supported
        assert "zh" in supported


# Note: CrossEncoder integration tests have been removed
# as Provence is now implemented as a standalone ProvenceEncoder


if __name__ == "__main__":
    pytest.main([__file__])